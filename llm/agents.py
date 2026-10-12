"""AI 玩家系统：人设卡、四层 prompt、私有策略和公开记录校验。

分层（设计 D3/D4/D5）：
- 每次调用返回结构化 JSON（pydantic 校验），解析失败携带错误重试最多 2 次，
  仍失败降级为安全行为，对局绝不因 LLM 输出异常而中断
- prompt = 身份层 + 记忆层（视角过滤事件流 + 发言编号）+ 情境层 + 输出层
- 格式、目标与事实校验共享三次尝试；未验证候选不公开
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from config import DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS_LIMIT
from engine.events import Event, EventKind
from engine.game import SpeechContext
from engine.roles import Role
from llm.client import OutputTruncatedError
from llm.grounding import SpeechReference, speech_index, validate_speech_facts
from llm.strategies import get_strategy_advice, public_dead_seats

logger = logging.getLogger("llm_murder")

MAX_SPEECH_CHARS = 240  # 完整首轮发言上限；超长要求缩写，不裁切正文。
MAX_REPLY_CHARS = 150


# ================================================================ 人设卡系统


@dataclass
class Persona:
    """人设卡：姓名 + 性格 + 口癖，每局随机分配且局内不变。"""

    name: str
    personality: str
    style: str


PERSONA_POOL: list[Persona] = [
    Persona("老猫", "谨慎多疑，凡事都要抠细节，说话慢条斯理", "口癖是「我总觉得哪里不对」"),
    Persona("阿飞", "冲动热血，心直口快，容易被带节奏也容易带别人节奏", "口癖是「我跟你说」"),
    Persona("学姐", "理性冷静的分析派，喜欢梳理论点与时间线", "口癖是「从逻辑上讲」"),
    Persona("大树", "憨厚耿直，不太会拐弯抹角，容易紧张", "口癖是「俺寻思」"),
    Persona("小蝶", "活泼机灵，爱调侃，观察力意外地敏锐", "口癖是「嘿嘿，有意思」"),
    Persona("铁叔", "沉稳老练，话不多但句句戳要害", "口癖是「我话讲完」"),
]


def assign_personas(rng: random.Random, count: int) -> list[Persona]:
    """从人设池随机抽取 count 套互不重复的人设（每局分配后局内不变）。"""
    return rng.sample(PERSONA_POOL, k=count)


# ================================================================ 输出模型


class PrivateStrategy(BaseModel):
    assessment: str = Field(min_length=1, max_length=120)
    goal: str = Field(min_length=1, max_length=60)
    claim: str = Field(min_length=1, max_length=120)


class SpeechPack(BaseModel):
    """公开发言与立场、引用依据，以及不公开的策略摘要。"""

    speech: str = Field(min_length=1)
    suspect: int = Field(default=0, ge=0, le=6, description="公开怀疑的座位号(1-6)，0=未定")
    mood: str = Field(default="平静")
    references: list[SpeechReference] = Field(default_factory=list)
    private_strategy: PrivateStrategy | None = None


@dataclass(frozen=True)
class ValidationIssue:
    reason: str
    kind: str


class VotePack(BaseModel):
    """投票包：vote 为 1-based 座位号，0=弃票。"""

    vote: int = Field(ge=0, le=6)
    private_strategy: PrivateStrategy | None = None


class TargetPack(BaseModel):
    """技能目标包（狼刀/查验共用）：target 为 1-based 座位号。"""

    target: int = Field(ge=1, le=6)
    private_strategy: PrivateStrategy | None = None


class WolfPack(TargetPack):
    team_plan: str | None = Field(default=None, min_length=1, max_length=240)


class WitchPack(BaseModel):
    """女巫决策包：save=用解药；poison 为 1-based 座位号，0=不用毒。"""

    save: bool
    poison: int = Field(default=0, ge=0, le=6)
    private_strategy: PrivateStrategy | None = None


# ================================================================ JSON 提取


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_block(text: str) -> tuple[dict | None, str | None]:
    """从 LLM 文本中提取 JSON 对象，容错围栏/前后缀噪声。

    返回 (dict, None) 或 (None, 错误描述)。
    """
    candidates = [text.strip()]
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    brace = _JSON_BRACE_RE.search(text)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, None
        except (json.JSONDecodeError, ValueError):
            continue
    return None, "未找到合法 JSON 对象"


def parse_pack(
    text: str, model_cls: type[BaseModel]
) -> tuple[BaseModel | None, str | None]:
    """提取 + schema 校验。返回 (模型实例, None) 或 (None, 错误描述)。"""
    obj, err = extract_json_block(text)
    if obj is None:
        return None, err
    try:
        return model_cls.model_validate(obj), None
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "未知字段"
        return (
            None,
            f"字段校验失败：{loc} {first['msg']}（收到键：{list(obj.keys())}）",
        )


SAFE_SPEECH = "目前依据不足，我暂时保留判断。"


# ================================================================ prompt 拼装


def slabel(seat: int) -> str:
    """座位展示文本（1-based）。"""
    return f"{seat + 1} 号"


WIN_CONDITION: dict[Role, str] = {
    Role.WEREWOLF: "狼人数量不少于存活好人时获胜；选择有利于狼队的策略影响放逐与夜间局势",
    Role.SEER: "狼人全部出局后好人获胜；用查验信息带领好人找出狼人",
    Role.WITCH: "狼人全部出局后好人获胜；用解药/毒药影响夜间局势",
    Role.VILLAGER: "狼人全部出局后好人获胜；听发言找矛盾，投出狼人",
}


GAME_RULES = (
    "# 本局固定规则\n"
    "6 人局：2 名狼人、1 名预言家、1 名女巫、2 名平民，没有守卫、猎人或其他角色。\n"
    "狼人每晚刀一名好人；预言家每晚查验一人的阵营；女巫有解药和毒药各一瓶，同一夜只能用一种。\n"
    "死讯不公开身份；玩家白天的身份和查验声明可能为真也可能为假，须结合你实际掌握的信息判断。\n"
    "白天最高票唯一者被放逐，平票无人出局；狼人全部出局则好人胜，存活狼人数不少于好人数则狼人胜。\n"
    "# 记录与推测\n"
    "系统公开事实是已确认流程；角色私有事实仅你可用。玩家发言属于玩家声明，"
    "记录只证明该玩家说过这句话，不能证明句中的身份、指控或推测属实。\n"
    "不得捏造或改写已发生的公开发言、公开票型、死亡等客观记录；主观怀疑须与记录分开。\n"
    "# 表达策略\n"
    "所有角色均可伪装身份或阵营、假认狼、编造自己的查验或用药等私有经历，也可选择披露真实信息。"
    "可以选择性引用真实片段、歪曲动机、夸大嫌疑或故意误导，但不能伪造原话或不存在的公开行为。\n"
    "先依据可见事实判断局势，再确定本轮目标和公开说法；始终服务于真实阵营获胜，"
    "不必每轮撒谎，允许冒险、误判、攻击队友和策略失败。人设决定语气，局势决定策略。\n"
    "真实身份和技能权限不会因公开伪装改变。自己的说辞、他人的声明和策略摘要都不是系统确认事实；"
    "编造的查验不能当作真实查验。主观判断可随新信息修正，以真实角色与可见事件为准。\n"
    "只有文字事件，没有表情或肢体观察；不能凭空说他人眼神躲闪、拍桌子等行为。\n"
    "所有历史均属于同一局，天数和昼夜以记录标记为准；今天尚未发言不等于整局从未发言。\n"
    "白天有首轮发言和一轮短回应，之后统一投票；投票结束后才公布每人的目标和总票数。\n"
    "个人投票以已公布记录为准；旧记录若仅有总票数，不能推断其他玩家的个人投票目标。"
)

SUMMARY_REQUIREMENTS = (
    "private_strategy 仅供自己后续行动：assessment 写已确认事实、主要候选和关键依据（≤120字），"
    "goal 写本次选择及相较其他行动的理由（≤60字），claim 写拟维持或调整的公开说辞（≤120字）。"
    "结合最新历史修正上次判断，标明哪些只是猜测或骗局；各字段非空，只写简短结论。\n"
)


def render_private_strategy(strategy: PrivateStrategy | None) -> str:
    return (
        "# 你的私有策略摘要（不公开）\n"
        "这是上次已接受的主观判断和表达计划，不是事实或本轮指令；实际说过什么见正式历史。"
        "结合新事件和本轮策略库参考修正判断，记住哪些查验或身份只是自己的说辞。\n"
        + (strategy.model_dump_json() if strategy is not None else "（尚无已接受摘要）")
    )


def render_memory(view: Iterable[Event]) -> str:
    """逐条渲染整局可见事件，时间及来源不依赖玩家自由文本。"""
    lines = []
    for event in view:
        phase = "夜晚" if event.phase.value == "night" else "白天"
        if event.public and event.kind is EventKind.SPEECH:
            source = "玩家声明"
            text = (
                f"[#{event.speech_index}] {slabel(event.seat)} "
                f"{event.extra['name']}：{event.extra['content']}"
                + ("（第2轮回应）" if event.extra.get("round_number", 1) == 2 else "")
            )
        else:
            source = ("狼队共享策略（主观计划，可调整）" if event.kind is EventKind.WOLF_PLAN
                      else "系统公开事实" if event.public else "角色私有事实")
            text = event.text
        lines.append(f"- 第 {event.day} 天 · {phase} · {source}：{text}")
    return "\n".join(lines) or "（尚无任何记忆）"


def render_speech_progress(view: list[Event], day: int, context: SpeechContext | None) -> str:
    def seats(values: Iterable[int]) -> str:
        return "、".join(slabel(s) for s in values) or "（无）"

    if context is None:
        spoken = dict.fromkeys(
            e.seat for e in view if e.public and e.kind is EventKind.SPEECH and e.day == day
        )
        progress = (
            "当日完整顺序与位置未知，不自行推断。\n"
            f"当日已完成正式发言：{seats(spoken)}。\n"
        )
    else:
        progress = (
            f"当前第 {context.round_number} 轮（{'首轮发言' if context.round_number == 1 else '简短回应'}）。\n"
            f"引擎当日实际顺序：{seats(context.order)}。\n"
            f"当前发言人：{slabel(context.speaker)}，第 {context.position} 位。\n"
            f"当日已完成正式发言：{seats(context.spoken)}。\n"
            f"当前发言人之后尚未轮到：{seats(context.remaining)}。\n"
            f"公开座位姓名：{'、'.join(f'{slabel(s)} {name}' for s, name in context.names)}。\n"
        )
    history = any(e.public and e.kind is EventKind.SPEECH for e in view)
    progress += "整局已有公开发言，见完整记忆。\n" if history else "整局尚无公开发言记录。\n"
    return progress + "正常轮候不是故意沉默的证据；今日进度不能否定往日发言。"


def build_speech_prompt(
    *,
    seat: int,
    name: str,
    role: Role,
    persona: Persona,
    teammate_seats: list[int] | None,
    view: list[Event],
    day: int,
    alive_seats: list[int],
    my_votes: list[tuple[int, int | None]],
    last_suspect: int | None,
    private_strategy: PrivateStrategy | None = None,
    context: SpeechContext | None = None,
    retry_warning: str | None = None,
) -> list[dict]:
    """四层拼装白天发言 prompt。返回 openai messages。"""
    reply_round = context is not None and context.round_number == 2
    char_limit = MAX_REPLY_CHARS if reply_round else MAX_SPEECH_CHARS
    # ---------- 身份层
    identity = (
        f"{GAME_RULES}\n\n"
        f"# 你的真实身份（决定阵营目标与技能权限）\n"
        f"你是 6 人局狼人杀玩家：{name}，坐在 {slabel(seat)}。\n"
        f"你的角色：{role.value}。胜利条件：{WIN_CONDITION[role]}\n"
        f"{ROLE_SECRET(role, teammate_seats)}"
        f"你的人设：{persona.personality}；{persona.style}。请全程保持人设语气。"
    )

    # ---------- 记忆层
    memory = f"# 对局记忆（仅你视角内的信息，按时间顺序）\n{render_memory(view)}"

    # ---------- 情境层
    alive_desc = "、".join(slabel(s) for s in alive_seats)
    my_speeches = [
        f"第 {e.day} 天 [#{e.speech_index}]：{e.extra['content']}"
        for e in view
        if e.kind is EventKind.SPEECH and e.seat == seat
    ]
    my_speech_desc = "；".join(my_speeches[-3:]) or "（你还没发过言）"
    vote_desc = "、".join(
        f"第{d}天投给{slabel(t)}" if t is not None else f"第{d}天弃票"
        for d, t in my_votes
    ) or "（你还没投过票）"
    suspect_desc = slabel(last_suspect) if last_suspect is not None else "还没明确的怀疑对象"
    context = (
        f"# 当前情境\n"
        f"现在是第 {day} 天，白天发言阶段。轮到你（{slabel(seat)}，{name}）发言。\n"
        f"存活玩家：{alive_desc}。\n"
        f"{render_speech_progress(view, day, context)}\n"
        f"你近期的发言摘要（最近三条，完整历史见记忆；考虑如何解释公开改口，解释可服务于策略）：{my_speech_desc}\n"
        f"你此前的投票：{vote_desc}\n"
        f"你上轮公开怀疑对象（不等于真实判断）：{suspect_desc}\n\n"
        f"{render_private_strategy(private_strategy)}"
    )
    advice = get_strategy_advice(
        role=role, seat=seat, action="speech", view=view, day=day,
        alive_seats=alive_seats, teammate_seats=teammate_seats,
    )
    context += "\n\n" + advice.text
    tasks = {
        Role.SEER: "比较公布真实结果和继续隐藏的后果；关键轮或有查杀时发布真实结果，争取明确处理目标。",
        Role.WITCH: "用真实刀口及资源纠正错误指控，比较披露和隐藏的收益，帮助好人集中关键票。",
        Role.VILLAGER: "比较具体声明及其依据，给出暂定站边和投票方向；无证据时提出可回答的具体问题。",
        Role.WEREWOLF: "结合狼队共享计划，确定本轮争取谁的票、公开说辞及推动的放逐目标。",
    }
    context += f"\n# 本轮发言任务\n{tasks[role]}\n"
    context += ("你已读到全部首轮发言：回应针对自己的关键指控或新证据，说明站边是否改变并提出归票目标。"
                "这是投票前最后一次发言，不把行动留到不存在的下一轮。"
                if reply_round else "贡献一个具体信息、判断或问题；后面还有一次短回应可根据新信息调整。")

    # ---------- 输出层：引用示例从本局真实记录生成。
    example = {
        "private_strategy": {"assessment": f"真实身份是{role.value}；以已确认阵营和公开声明区分事实与猜测。",
                             "goal": "比较各目标和等待的后果，推动有利于本阵营的行动。", "claim": "提出具体依据、站边或待确认问题。"},
        "speech": "请预言家交代查验目标及结果，大家说明支持谁和理由，我会据此选择投票方向。",
        "suspect": 0, "mood": "平静", "references": [],
    }
    past = next((e for e in reversed(view) if e.public and e.kind is EventKind.SPEECH and e.seat != seat), None)
    if past is not None and past.extra["content"]:
        quote = past.extra["content"][:12]
        example["speech"] = f"第 {past.day} 天 {slabel(past.seat)} 在 [#{past.speech_index}] 说过“{quote}”，请说明这个判断的具体依据和投票目标。"
        example["references"] = [{"speech_index": past.speech_index, "seat": past.seat + 1, "day": past.day, "quote": quote}]
    check = next((e for e in reversed(view) if e.kind is EventKind.SEER_CHECK and seat in e.viewers), None)
    if role is Role.SEER and check is not None:
        result = "狼人" if check.extra["is_wolf"] else "好人"
        example["speech"] = f"我是预言家，第{check.day}夜验{slabel(check.extra['target'])}是{result}。请大家对照真实结果说明站边。"
        example["references"] = []
        if check.extra["is_wolf"] and check.extra["target"] in alive_seats:
            example["suspect"] = check.extra["target"] + 1
    speech_schema = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        f"{SUMMARY_REQUIREMENTS}"
        "suspect 填你对外声称怀疑的存活座位号(1-6)，未定填0；可以不同于真实判断或实际投票。"
        "mood 填简短情绪词，speech 只写公开发言，不透露私有摘要。\n"
        "引用时 references 每项填写 speech_index、seat（1-based）、day、quote（非空原文片段）；"
        "正文须使用同一真实 [#编号]，人物和时间须一致，无引用填空列表。\n"
        "引用往日发言请明确昨天、第 N 天或历史时间，不要写成今天刚才。\n"
        "\n## 行为铁律\n"
        "1. 真实认知仅来自你的角色与可见事件；可以编造自身私有经历作为对外说辞，不能将其记为真实经历。\n"
        "2. 根据真实阵营目标选择诚实、隐瞒或欺骗；允许假跳、认狼、改口及策略性动机解释。\n"
        f"3. 口语化、像真人说话，禁止列表体与书面语，不超过 {char_limit} 字符（含标点、空格和引用编号）。"
        "超长会要求重新输出，请保留完整语句、关键结论和对应引用。\n"
        "4. 引用别人的观点必须用真实存在的发言编号（如 [#3]），不确定编号就不要引用。\n"
        "5. 不得捏造他人已发生的公开行为，区分记录与解读，明确今日或历史时间范围。"
        "尚未发言不能说成长篇大论，正常轮候不能说成故意沉默。\n"
        "6. 只输出 JSON 本身。"
    )
    messages = [
        {"role": "system", "content": f"{identity}\n\n{memory}\n\n{context}"},
        {
            "role": "user",
            "content": speech_schema
            + (
                f"\n\n警告：{retry_warning}，请重新组织发言。"
                if retry_warning
                else ""
            ),
        },
    ]
    return messages


def ROLE_SECRET(role: Role, teammate_seats: list[int] | None) -> str:
    if role is Role.WEREWOLF and teammate_seats:
        mates = "、".join(slabel(s) for s in teammate_seats if s is not None)
        return f"你的狼人队友：{mates}。夜晚你们共同确认一个刀口目标。\n"
    if role is Role.WITCH:
        return "你有一瓶解药与一瓶毒药，各限用一次，同一夜不能同时使用；每晚你会得知刀口。\n"
    if role is Role.SEER:
        return "你每晚查验一名玩家的阵营（狼人/好人），只有你自己知道结果。\n"
    return ""


def build_vote_prompt(
    *,
    seat: int,
    name: str,
    role: Role,
    persona: Persona,
    view: list[Event],
    day: int,
    alive_seats: list[int],
    my_votes: list[tuple[int, int | None]],
    private_strategy: PrivateStrategy | None = None,
    teammate_seats: list[int] | None = None,
) -> list[dict]:
    """投票 prompt：基于全部发言后的独立调用。"""
    identity = (
        f"{GAME_RULES}\n\n"
        f"# 你的真实身份\n"
        f"你是 6 人局狼人杀玩家：{name}，坐在 {slabel(seat)}，角色：{role.value}。\n"
        f"胜利条件：{WIN_CONDITION[role]}"
    )
    memory = f"# 对局记忆（仅你视角内的信息）\n{render_memory(view)}"
    context = (
        f"# 当前情境\n"
        f"现在是第 {day} 天，全部发言结束，进入投票阶段。你（{slabel(seat)}，{name}）要投出一票。\n"
        f"存活玩家：{'、'.join(slabel(s) for s in alive_seats)}。\n"
        f"你此前的投票：{('、'.join(f'第{d}天投给{slabel(t)}' if t is not None else f'第{d}天弃票' for d, t in my_votes) or '还没有投过票')}\n\n"
        f"{render_private_strategy(private_strategy)}"
    )
    advice = get_strategy_advice(
        role=role, seat=seat, action="vote", view=view, day=day,
        alive_seats=alive_seats, teammate_seats=teammate_seats,
    )
    context += "\n\n" + advice.text
    living = advice.situation.alive
    targets = (advice.situation.known_good if role is Role.WEREWOLF else advice.situation.known_wolves) & living
    if not targets:
        targets = living - advice.situation.known_good - advice.situation.known_wolves
    target = next(iter(sorted(targets - {seat})), None)
    example = {"vote": target + 1 if target is not None else 0,
               "private_strategy": {"assessment": "根据全部发言比较候选及关键证据。", "goal": "选最有利于本阵营的放逐目标。", "claim": "解释公开站边。"}}
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        f"{SUMMARY_REQUIREMENTS}"
        "明确比较投各候选、分票和弃票的后果；不能因证据不完全而在等待必输时放弃行动。\n"
        "vote 填你要放逐的存活座位号(1-6)。\n"
        "可以弃票（填0）；依据真实阵营获胜选择目标，可以与公开攻击对象不同，也可以策略性投队友。\n"
        "## 铁律\n"
        f"1. 依据真实可见事实和主观策略行动，自己的假查验不是实际技能结果。\n"
        f"2. 只输出 JSON 本身。"
    )
    return [
        {"role": "system", "content": f"{identity}\n\n{memory}\n\n{context}"},
        {"role": "user", "content": output},
    ]


def build_wolf_den_prompt(
    *,
    wolf_seats: list[int],
    persona: Persona,
    view: list[Event],
    day: int,
    killable: list[int],
    private_strategy: PrivateStrategy | None = None,
    seat: int | None = None,
) -> list[dict]:
    """狼窝会话 prompt：两狼共享一次调用，统一产出刀口。"""
    living_wolves = sorted(set(wolf_seats) - public_dead_seats(view))
    wolves_desc = "、".join(slabel(s) for s in living_wolves)
    caller = seat if seat is not None else living_wolves[0]
    identity = (
        f"{GAME_RULES}\n\n"
        f"# 狼队内部密谈（绝密）\n"
        f"现在是第 {day} 夜。存活狼队成员：{wolves_desc}。\n"
        f"这次调用代表整个狼队做决策（由「{persona.name}」主导口吻）。\n"
        f"胜利条件：{WIN_CONDITION[Role.WEREWOLF]}"
    )
    memory = f"# 狼队视角的记忆\n{render_memory(view)}"
    context = (
        f"# 今夜决策\n"
        f"可刀的存活好人：{'、'.join(slabel(s) for s in killable)}。\n"
        f"综合考虑：谁对狼队威胁最大（可能身份是预言家/女巫）、白天局势走向。\n\n"
        f"{render_private_strategy(private_strategy)}"
    )
    context += "\n\n" + get_strategy_advice(
        role=Role.WEREWOLF, seat=caller, action="wolf_kill", view=view, day=day,
        alive_seats=sorted(set(killable) | set(living_wolves)), teammate_seats=wolf_seats,
    ).text
    example = {"target": next(iter(killable), 0) + 1,
               "team_plan": "写明谁主导说辞、队友怎样配合、争取谁及何时改计划。",
               "private_strategy": {"assessment": "比较技能威胁和白天可利用的分歧。", "goal": "选刀口并协调白天行动。", "claim": "维持公开伪装。"}}
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        f"{SUMMARY_REQUIREMENTS}"
        "team_plan 是最多240字的狼队共享计划，只使用狼队共同可见信息；明确分工、归票和退出条件。\n"
        "target 必须填可刀对象中的一个座位号(1-6)。\n"
        "## 铁律\n只输出 JSON 本身。"
    )
    return [
        {"role": "system", "content": f"{identity}\n\n{memory}\n\n{context}"},
        {"role": "user", "content": output},
    ]


def build_seer_prompt(
    *,
    seat: int,
    persona: Persona,
    view: list[Event],
    day: int,
    checkable: list[int],
    private_strategy: PrivateStrategy | None = None,
) -> list[dict]:
    identity = (
        f"{GAME_RULES}\n\n"
        f"# 你的真实身份\n"
        f"你是预言家（{slabel(seat)}，人设：{persona.name}）。"
        f"每晚查验一名玩家，获知他是狼人还是好人。结果只有你自己知道。\n"
        f"胜利条件：{WIN_CONDITION[Role.SEER]}"
    )
    memory = f"# 你的记忆（含既往查验结果）\n{render_memory(view)}"
    context = (
        f"# 今夜决策\n"
        f"第 {day} 夜，可查验对象：{'、'.join(slabel(s) for s in checkable)}。\n"
        "比较查到好人或狼人能如何改变明天决策，优先查验有区分力的未知对象；"
        "已确认阵营通常无需重复查验。\n\n"
        f"{render_private_strategy(private_strategy)}"
    )
    context += "\n\n" + get_strategy_advice(
        role=Role.SEER, seat=seat, action="seer_check", view=view, day=day,
        alive_seats=sorted(set(checkable) | {seat}),
    ).text
    checked = {e.extra["target"] for e in view if e.kind is EventKind.SEER_CHECK and seat in e.viewers}
    target = next((s for s in checkable if s not in checked), next(iter(checkable), 0))
    example = {"target": target + 1,
               "private_strategy": {"assessment": "比较未知候选与已确认阵营。", "goal": "查验能改变明天归票的未知对象。", "claim": "准备发布真实结果。"}}
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        f"{SUMMARY_REQUIREMENTS}"
        "target 必须填可查验对象中的一个座位号(1-6)。\n"
        "## 铁律\n只输出 JSON 本身。"
    )
    return [
        {"role": "system", "content": f"{identity}\n\n{memory}\n\n{context}"},
        {"role": "user", "content": output},
    ]


def build_witch_prompt(
    *,
    seat: int,
    persona: Persona,
    view: list[Event],
    day: int,
    kill_target: int,
    can_save: bool,
    can_poison: bool,
    alive_seats: list[int],
    private_strategy: PrivateStrategy | None = None,
) -> list[dict]:
    identity = (
        f"{GAME_RULES}\n\n"
        f"# 你的真实身份\n"
        f"你是女巫（{slabel(seat)}，人设：{persona.name}）。"
        f"拥有一瓶解药与一瓶毒药，各限用一次，同一夜不得同时使用。\n"
        f"胜利条件：{WIN_CONDITION[Role.WITCH]}"
    )
    memory = f"# 你的记忆（含用药记录）\n{render_memory(view)}"
    context = (
        f"# 今夜决策\n"
        f"第 {day} 夜，今晚的刀口是 {slabel(kill_target)}。\n"
        f"解药{'还在' if can_save else '已用完'}；毒药{'还在' if can_poison else '已用完'}。\n"
        f"存活玩家：{'、'.join(slabel(s) for s in alive_seats)}。\n"
        f"权衡：救刀口（保人）还是毒掉你实际怀疑的人，或都不用。注意：解药和毒药同一夜只能用一个。\n\n"
        f"{render_private_strategy(private_strategy)}"
    )
    context += "\n\n" + get_strategy_advice(
        role=Role.WITCH, seat=seat, action="witch_act", view=view, day=day,
        alive_seats=alive_seats, kill_target=kill_target,
        can_save=can_save, can_poison=can_poison,
    ).text
    example = {"save": can_save, "poison": 0,
               "private_strategy": {"assessment": "刀口在本局是好人；比较药量和人数危险。", "goal": "比较救人、毒人和不用药的结算后果。", "claim": "必要时用真实刀口信息纠偏。"}}
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        f"{SUMMARY_REQUIREMENTS}"
        "save 为 true 表示用解药救刀口，false 表示不用；poison 填毒杀的存活座位号(1-6)，不用毒填0。\n"
        "## 铁律\n"
        f"1. save 与 poison 不得同时使用（要么救，要么毒，要么都不用）。\n"
        f"2. 只输出 JSON 本身。"
    )
    return [
        {"role": "system", "content": f"{identity}\n\n{memory}\n\n{context}"},
        {"role": "user", "content": output},
    ]


# ================================================================ AI 玩家


class AIPlayer:
    """单个 AI 玩家：prompt 拼装 → 调用 → 校验 → 提交公开立场和私有策略。

    所有方法只依赖传入的视角事件流（view），与引擎解耦；
    决策失败一律降级为安全行为，绝不抛出 LLM 异常打断对局。
    """

    def __init__(
        self,
        client,  # noqa: ANN001 - LLMClient
        *,
        seat: int,
        name: str,
        role: Role,
        persona: Persona,
        teammate_seats: list[int] | None = None,
    ) -> None:
        self.client = client
        self.seat = seat
        self.name = name
        self.role = role
        self.persona = persona
        self.teammate_seats = teammate_seats or []
        self.my_votes: list[tuple[int, int | None]] = []
        self.last_suspect: int | None = None  # 上轮公开立场，可以与真实判断不同。
        self.private_strategy: PrivateStrategy | None = None
        self.team_plan: str | None = None

    # ------------------------------------------------------------ 校验工具

    @staticmethod
    def _alive_target_error(
        seat_1based: int, alive_seats: list[int], *, allow_zero: bool
    ) -> str | None:
        """目标合法性校验：返回错误描述或 None。"""
        if seat_1based == 0:
            return None if allow_zero else "目标不能为 0，必须是存活玩家座位号"
        if (seat_1based - 1) not in alive_seats:
            return (
                f"目标 {seat_1based} 不在存活玩家中，"
                f"可选：{'、'.join(slabel(s) for s in alive_seats)}"
            )
        return None

    # ------------------------------------------------------------ 白天发言

    def day_speech(
        self,
        view: list[Event],
        day: int,
        alive_seats: list[int],
        on_delta: Callable[[str], None] | None = None,
        *,
        context: SpeechContext | None = None,
    ) -> tuple[str, int | None, str]:
        """返回已校验发言、怀疑座位（0-based）及情绪。

        on_delta 保留以兼容直接调用；候选仅内部收集，公开展示走正式事件。
        """
        kwargs = dict(
            seat=self.seat, name=self.name, role=self.role,
            persona=self.persona, teammate_seats=self.teammate_seats,
            view=view, day=day, alive_seats=alive_seats,
            my_votes=self.my_votes, last_suspect=self.last_suspect,
            private_strategy=self.private_strategy,
            context=context,
        )

        records = speech_index(view)
        char_limit = MAX_REPLY_CHARS if context is not None and context.round_number == 2 else MAX_SPEECH_CHARS
        def validate(pack: SpeechPack) -> ValidationIssue | None:
            # 所有校验与最终公开文本使用同一规范化结果。
            pack.speech = pack.speech.strip()
            if not pack.speech:
                return ValidationIssue("发言不能为空", "format")
            if len(pack.speech) > char_limit:
                return ValidationIssue(
                    f"发言超过 {char_limit} 字符（当前 {len(pack.speech)} 字符）。"
                    f"请缩写为不超过 {char_limit} 字符的完整发言，保留关键结论，"
                    "同步调整正文引用和 references，不要截断句子。",
                    "length",
                )
            target_error = self._alive_target_error(pack.suspect, alive_seats, allow_zero=True)
            if target_error:
                return ValidationIssue(target_error, "target")
            fact_error = validate_speech_facts(
                pack.speech, pack.references, records=records, day=day, context=context, view=view,
            )
            if fact_error:
                return ValidationIssue(fact_error, "fact")
            return None

        def speech_mock() -> str:
            # mock 模式合法发言包：怀疑对象取第一个非自己的存活座位
            target = next((s for s in alive_seats if s != self.seat), None)
            return json.dumps(
                {
                    "speech": "（离线模式）我先听听大家的看法，保留意见。",
                    "suspect": (target + 1) if target is not None else 0,
                    "mood": "平静",
                    "references": [],
                    "private_strategy": {"assessment": "离线模式没有真实推理。", "goal": "演示合法流程。", "claim": "保留判断。"},
                },
                ensure_ascii=False,
            )

        pack, err = call_structured(
            self.client, build_speech_prompt(**kwargs), SpeechPack,
            # 客户端可流式收集，原始 delta 不能传给公开回调。
            stream=True, validate=validate,
            mock_reply=speech_mock,
        )
        if pack is None:
            logger.warning("AI「%s」发言降级（输出不可用）：%s", self.name, err)
            self.last_suspect = None
            return SAFE_SPEECH, None, "平静"

        suspect = pack.suspect - 1 if pack.suspect > 0 else None
        self.last_suspect = suspect
        if pack.private_strategy is not None:
            self.private_strategy = pack.private_strategy
        return pack.speech, suspect, pack.mood

    # ------------------------------------------------------------ 投票

    def day_vote(
        self, view: list[Event], day: int, alive_seats: list[int]
    ) -> int | None:
        """投票（看到全部发言后）。返回目标座位 0-based；None=弃票。"""
        def validate(pack: VotePack) -> str | None:
            return self._alive_target_error(pack.vote, alive_seats, allow_zero=True)

        def vote_mock() -> str:
            # mock 模式合法投票包：投第一个非自己的存活座位
            target = next((s for s in alive_seats if s != self.seat), None)
            return json.dumps({"vote": (target + 1) if target is not None else 0})

        pack, err = call_structured(
            self.client,
            build_vote_prompt(
                seat=self.seat, name=self.name, role=self.role,
                persona=self.persona, view=view, day=day,
                alive_seats=alive_seats, my_votes=self.my_votes,
                private_strategy=self.private_strategy,
                teammate_seats=self.teammate_seats,
            ),
            VotePack,
            validate=validate, temperature=0.3,
            mock_reply=vote_mock,
        )
        target: int | None = (
            pack.vote - 1 if pack is not None and pack.vote > 0 else None
        )
        if pack is None:
            logger.warning("AI「%s」投票降级为弃票：%s", self.name, err)
        elif pack.private_strategy is not None:
            self.private_strategy = pack.private_strategy
        self.my_votes.append((day, target))
        return target

    # ------------------------------------------------------------ 技能决策

    def wolf_den_decision(
        self, view: list[Event], day: int, killable: list[int]
    ) -> int:
        """狼窝会话：两狼共享一次调用统一刀口。返回 0-based 目标。"""
        self.team_plan = None
        def validate(pack: TargetPack) -> str | None:
            return self._alive_target_error(pack.target, killable, allow_zero=False)

        pack, err = call_structured(
            self.client,
            build_wolf_den_prompt(
                wolf_seats=self.teammate_seats + [self.seat]
                if self.seat not in self.teammate_seats
                else sorted(set(self.teammate_seats) | {self.seat}),
                persona=self.persona, view=view, day=day, killable=killable,
                private_strategy=self.private_strategy,
                seat=self.seat,
            ),
            WolfPack,
            validate=validate, temperature=0.3,
            mock_reply=lambda: json.dumps({"target": killable[0] + 1}),
        )
        if pack is None:
            logger.warning("狼窝决策降级为刀首名好人：%s", err)
            return killable[0]
        if pack.private_strategy is not None:
            self.private_strategy = pack.private_strategy
        self.team_plan = pack.team_plan
        return pack.target - 1

    def seer_decision(
        self, view: list[Event], day: int, checkable: list[int]
    ) -> int:
        """预言家查验目标。返回 0-based 目标。"""
        def validate(pack: TargetPack) -> str | None:
            return self._alive_target_error(pack.target, checkable, allow_zero=False)

        pack, err = call_structured(
            self.client,
            build_seer_prompt(
                seat=self.seat, persona=self.persona, view=view,
                day=day, checkable=checkable,
                private_strategy=self.private_strategy,
            ),
            TargetPack,
            validate=validate, temperature=0.3,
            mock_reply=lambda: json.dumps({"target": checkable[0] + 1}),
        )
        if pack is None:
            logger.warning("预言家决策降级为查验首位：%s", err)
            return checkable[0]
        if pack.private_strategy is not None:
            self.private_strategy = pack.private_strategy
        return pack.target - 1

    def witch_decision(
        self,
        view: list[Event],
        day: int,
        kill_target: int,
        can_save: bool,
        can_poison: bool,
        alive_seats: list[int],
    ) -> tuple[bool, int | None]:
        """女巫用药决策。返回 (用解药, 毒杀座位 0-based 或 None)。"""
        def validate(pack: WitchPack) -> str | None:
            if pack.save and not can_save:
                return "解药已用完，不能再用"
            if pack.poison > 0:
                if not can_poison:
                    return "毒药已用完，不能再用"
                if pack.save:
                    return "解药与毒药同一夜不得同时使用"
                if (pack.poison - 1) not in alive_seats:
                    return (
                        f"毒药目标 {pack.poison} 不在存活玩家中，"
                        f"可选：{'、'.join(slabel(s) for s in alive_seats)}"
                    )
            return None

        pack, err = call_structured(
            self.client,
            build_witch_prompt(
                seat=self.seat, persona=self.persona, view=view, day=day,
                kill_target=kill_target, can_save=can_save,
                can_poison=can_poison, alive_seats=alive_seats,
                private_strategy=self.private_strategy,
            ),
            WitchPack,
            validate=validate, temperature=0.3,
            mock_reply=lambda: json.dumps({"save": False, "poison": 0}),
        )
        if pack is None:
            logger.warning("女巫决策降级为不用药：%s", err)
            return False, None
        if pack.private_strategy is not None:
            self.private_strategy = pack.private_strategy
        poison = pack.poison - 1 if pack.poison > 0 else None
        return pack.save, poison


# ================================================================ 结构化调用管道


def call_structured(
    client,  # noqa: ANN001 - LLMClient
    messages: list[dict],
    model_cls: type[BaseModel],
    *,
    max_retries: int = 2,
    stream: bool = False,
    on_delta: Callable[[str], None] | None = None,
    validate: Callable[[BaseModel], str | ValidationIssue | None] | None = None,
    mock_reply: Callable[[], str] | None = None,
    temperature: float = 0.8,
) -> tuple[BaseModel | None, str | None]:
    """最多重试 2 次：格式/业务错误反馈给模型，截断则在上限内扩大预算。

    mock_reply：mock 模式下按请求 schema 生成合法回复（避免 mock 误触发降级）。
    返回 (模型实例, None) 或 (None, 最终错误)。
    """
    convo = list(messages)
    last_err = "未知错误"
    config = getattr(client, "config", None)
    budget = getattr(config, "max_tokens", DEFAULT_MAX_TOKENS)
    budget_limit = getattr(config, "max_tokens_limit", DEFAULT_MAX_TOKENS_LIMIT)
    usage = getattr(client, "usage", None)
    for attempt in range(max_retries + 1):
        if attempt and usage is not None:
            usage.structured_retries += 1
        try:
            text = client.chat(
                convo, stream=stream, on_delta=on_delta,
                max_tokens=budget, mock_reply=mock_reply, temperature=temperature,
            )
        except OutputTruncatedError as exc:
            last_err = str(exc)
            next_budget = min(budget * 2, budget_limit)
            logger.warning("结构化输出被截断（第 %d 次）：%s", attempt + 1, last_err)
            if attempt == max_retries or next_budget <= budget:
                break
            logger.info("扩大输出预算重试：%d → %d tokens", budget, next_budget)
            budget = next_budget
            # 截断不是格式错误，不将半截 JSON 放回对话。
            continue
        except RuntimeError as exc:
            # 客户端的网络重试已耗尽，该行动交给调用方安全降级。
            if usage is not None:
                usage.fallback_actions += 1
            return None, str(exc)
        pack, err = parse_pack(text, model_cls)
        if pack is not None and validate is not None:
            err = validate(pack)
        if pack is not None and err is None:
            return pack, None
        issue = err if isinstance(err, ValidationIssue) else None
        last_err = issue.reason if issue is not None else err or "未知错误"
        logger.warning(
            "结构化输出%s失败（第 %d 次）：%s",
            f"校验/{issue.kind}" if issue is not None else "解析",
            attempt + 1, last_err,
        )
        convo = convo + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    f"你的输出无法使用：{last_err}\n"
                    f"请严格按格式重新输出，只输出一个合法 JSON 对象，不要任何其他文字。"
                ),
            },
        ]
    if usage is not None:
        usage.fallback_actions += 1
    return None, last_err
