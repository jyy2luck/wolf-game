"""AI 玩家系统：人设卡、四层 prompt 拼装、结构化输出解析、穿帮检测。

分层（设计 D3/D4/D5）：
- 每次调用返回结构化 JSON（pydantic 校验），解析失败携带错误重试最多 2 次，
  仍失败降级为安全行为，对局绝不因 LLM 输出异常而中断
- prompt = 身份层 + 记忆层（视角过滤事件流 + 发言编号）+ 情境层 + 输出层
- 穿帮检测：确定性关键词规则，检出携带警告重试一次，二次穿帮替换安全发言
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from engine.events import Event, EventKind
from engine.roles import Role

logger = logging.getLogger("llm_murder")

MAX_SPEECH_CHARS = 150  # 铁律：发言 ≤150 字，超长截断


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


class SpeechPack(BaseModel):
    """白天发言包：发言 + 怀疑对象 + 情绪（同包返回，省调用）。"""

    speech: str = Field(min_length=1)
    suspect: int = Field(default=0, ge=0, le=6, description="怀疑的座位号(1-6)，0=未定")
    mood: str = Field(default="平静")


class VotePack(BaseModel):
    """投票包：vote 为 1-based 座位号，0=弃票。"""

    vote: int = Field(ge=0, le=6)


class TargetPack(BaseModel):
    """技能目标包（狼刀/查验共用）：target 为 1-based 座位号。"""

    target: int = Field(ge=1, le=6)


class WitchPack(BaseModel):
    """女巫决策包：save=用解药；poison 为 1-based 座位号，0=不用毒。"""

    save: bool
    poison: int = Field(default=0, ge=0, le=6)


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


# ================================================================ 穿帮检测器

# 关键词组：狼身份自认 / 队友指称 —— 任何角色说出即穿帮
LEAK_WOLF_ID = re.compile(
    r"我的?狼队友|我们狼(人)?|作为狼人|我是狼|狼刀|刀口|今晚(我们)?刀"
)
# 关键词组：队友指称 —— 任何角色说出即穿帮（好人没有"队友"概念）
LEAK_TEAMMATE = re.compile(r"我的?队友")
# 关键词组：查验指称 —— 仅预言家可提（跳身份策略），其他人说出即穿帮
LEAK_CHECK = re.compile(r"我(验过|查验|查了?|的查验)")
# 关键词组：用药指称 —— 仅女巫可提（跳身份策略），其他人说出即穿帮
LEAK_POTION = re.compile(r"我(救了?|毒了?|用药|用了?解药|用了?毒药)|我的解药|我的毒药")


def detect_leak(speech: str, role: Role) -> str | None:
    """确定性穿帮检测：返回违规描述，无违规返回 None。

    规则（视角越权 + 表演纪律）：
    - 狼身份/队友/刀口字眼：任何角色都不得出现在 speech
    - 查验字眼：仅预言家可提；用药字眼：仅女巫可提
    """
    if (m := LEAK_WOLF_ID.search(speech)) or (m := LEAK_TEAMMATE.search(speech)):
        return f"出现视角越权表述「{m.group(0)}」"
    if role is not Role.SEER and (m := LEAK_CHECK.search(speech)):
        return f"非预言家提及查验「{m.group(0)}」"
    if role is not Role.WITCH and (m := LEAK_POTION.search(speech)):
        return f"非女巫提及用药「{m.group(0)}」"
    return None


SAFE_SPEECH = "我保留意见，先听大家的。"


# ================================================================ prompt 拼装


def slabel(seat: int) -> str:
    """座位展示文本（1-based）。"""
    return f"{seat + 1} 号"


WIN_CONDITION: dict[Role, str] = {
    Role.WEREWOLF: "狼人数量不少于存活好人时获胜；白天伪装好人、误导投票放逐好人",
    Role.SEER: "放逐全部狼人后好人获胜；用查验信息带领好人找出狼人",
    Role.WITCH: "放逐全部狼人后好人获胜；用解药/毒药影响夜间局势",
    Role.VILLAGER: "放逐全部狼人后好人获胜；听发言找矛盾，投出狼人",
}


def render_memory(view: Iterable[Event]) -> str:
    """记忆层：视角过滤事件流逐条渲染（发言编号已内嵌于 text）。"""
    return "\n".join(f"- {e.text}" for e in view) or "（尚无任何记忆）"


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
    retry_warning: str | None = None,
) -> list[dict]:
    """四层拼装白天发言 prompt。返回 openai messages。"""
    # ---------- 身份层
    identity = (
        f"# 你的身份（绝密，游戏结束时才公开）\n"
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
        e.extra.get("content", e.text)
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
        f"你此前的发言（保持立场连贯，若转变怀疑对象必须说明理由）：{my_speech_desc}\n"
        f"你此前的投票：{vote_desc}\n"
        f"你当前的怀疑对象：{suspect_desc}"
    )

    # ---------- 输出层
    speech_schema = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        '{"speech": "你的口语化发言，不超过150字", '
        '"suspect": <你当前最怀疑的座位号(1-6，不确定填0)>, '
        '"mood": <一到两个字的情绪词>}\n'
        "\n## 行为铁律\n"
        "1. 绝不泄露视角之外的信息；夜晚发生但你不知情的事一个字都不能提。\n"
        "2. 不提「队友」「狼刀/刀口」这类词。\n"
        "3. 口语化、像真人说话，禁止列表体与书面语，不超过 150 字。\n"
        "4. 引用别人的观点必须用真实存在的发言编号（如 [#3]），不确定编号就不要引用。\n"
        "5. 只输出 JSON 本身。"
    )
    if role is Role.WEREWOLF:
        speech_schema = speech_schema.replace(
            "2. 不提",
            "2. 你是狼人，发言必须完全站在好人视角分析；绝不承认或暗示自己是狼人，不提",
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
) -> list[dict]:
    """投票 prompt：基于全部发言后的独立调用。"""
    identity = (
        f"# 你的身份（绝密）\n"
        f"你是 6 人局狼人杀玩家：{name}，坐在 {slabel(seat)}，角色：{role.value}。\n"
        f"胜利条件：{WIN_CONDITION[role]}"
    )
    memory = f"# 对局记忆（仅你视角内的信息）\n{render_memory(view)}"
    context = (
        f"# 当前情境\n"
        f"现在是第 {day} 天，全部发言结束，进入投票阶段。你（{slabel(seat)}，{name}）要投出一票。\n"
        f"存活玩家：{'、'.join(slabel(s) for s in alive_seats)}。\n"
        f"你此前的投票：{('、'.join(f'第{d}天投给{slabel(t)}' if t is not None else f'第{d}天弃票' for d, t in my_votes) or '还没有投过票')}"
    )
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        '{"vote": <你要投票放逐的座位号(1-6)>}\n'
        "可以投弃票（填 0），但投出你最怀疑的人更符合你的胜利条件。\n"
        "## 铁律\n"
        f"1. 绝不泄露视角之外的信息。\n"
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
) -> list[dict]:
    """狼窝会话 prompt：两狼共享一次调用，统一产出刀口。"""
    wolves_desc = "、".join(slabel(s) for s in wolf_seats)
    identity = (
        f"# 狼队内部密谈（绝密）\n"
        f"现在是第 {day} 夜。你们是狼队：{wolves_desc}。\n"
        f"这次调用代表整个狼队做决策（由「{persona.name}」主导口吻）。"
    )
    memory = f"# 狼队视角的记忆\n{render_memory(view)}"
    context = (
        f"# 今夜决策\n"
        f"可刀的存活好人：{'、'.join(slabel(s) for s in killable)}。\n"
        f"综合考虑：谁对狼队威胁最大（可能身份是预言家/女巫）、白天局势走向。"
    )
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        '{"target": <你们决定刀杀的座位号(1-6)>}\n'
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
) -> list[dict]:
    identity = (
        f"# 你的身份（绝密）\n"
        f"你是预言家（{slabel(seat)}，人设：{persona.name}）。"
        f"每晚查验一名玩家，获知他是狼人还是好人。结果只有你自己知道。"
    )
    memory = f"# 你的记忆（含既往查验结果）\n{render_memory(view)}"
    context = (
        f"# 今夜决策\n"
        f"第 {day} 夜，可查验对象：{'、'.join(slabel(s) for s in checkable)}。\n"
        f"优先查验你最怀疑的人，避免浪费在已确认的人身上。"
    )
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        '{"target": <你要查验的座位号(1-6)>}\n'
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
) -> list[dict]:
    identity = (
        f"# 你的身份（绝密）\n"
        f"你是女巫（{slabel(seat)}，人设：{persona.name}）。"
        f"拥有一瓶解药与一瓶毒药，各限用一次，同一夜不得同时使用。"
    )
    memory = f"# 你的记忆（含用药记录）\n{render_memory(view)}"
    context = (
        f"# 今夜决策\n"
        f"第 {day} 夜，今晚的刀口是 {slabel(kill_target)}。\n"
        f"解药{'还在' if can_save else '已用完'}；毒药{'还在' if can_poison else '已用完'}。\n"
        f"存活玩家：{'、'.join(slabel(s) for s in alive_seats)}。\n"
        f"权衡：救刀口（保人）还是毒掉你最怀疑的人，或都不用。注意：解药和毒药同一夜只能用一个。"
    )
    output = (
        "# 输出要求\n"
        "只输出一个 JSON 对象（不要输出任何其他文字）：\n"
        '{"save": <true=用解药救刀口/false=不用>, "poison": <毒杀的座位号(1-6)，不用毒填0>}\n'
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
    """单个 AI 玩家：prompt 拼装 → 调用 → 解析 → 穿帮检测 → 回填。

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
        self.last_suspect: int | None = None
        self.leak_count = 0  # 穿帮次数（日志观察用）

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
    ) -> tuple[str, int | None, str]:
        """白天发言。返回 (发言文本, 怀疑座位 0-based 或 None, 情绪)。"""
        kwargs = dict(
            seat=self.seat, name=self.name, role=self.role,
            persona=self.persona, teammate_seats=self.teammate_seats,
            view=view, day=day, alive_seats=alive_seats,
            my_votes=self.my_votes, last_suspect=self.last_suspect,
        )

        def validate(pack: SpeechPack) -> str | None:
            return self._alive_target_error(pack.suspect, alive_seats, allow_zero=True)

        def speech_mock() -> str:
            # mock 模式合法发言包：怀疑对象取第一个非自己的存活座位
            target = next((s for s in alive_seats if s != self.seat), None)
            return json.dumps(
                {
                    "speech": "（离线模式）我先听听大家的看法，保留意见。",
                    "suspect": (target + 1) if target is not None else 0,
                    "mood": "平静",
                },
                ensure_ascii=False,
            )

        pack, err = call_structured(
            self.client, build_speech_prompt(**kwargs), SpeechPack,
            stream=True, on_delta=on_delta, validate=validate,
            mock_reply=speech_mock,
        )
        if pack is None:
            logger.warning("AI「%s」发言降级（解析失败）：%s", self.name, err)
            return SAFE_SPEECH, None, "平静"

        speech = pack.speech.strip()[:MAX_SPEECH_CHARS]
        leak = detect_leak(speech, self.role)
        if leak is not None:
            # 携带警告重试一次
            retry_kwargs = dict(kwargs, retry_warning=leak)
            pack2, err2 = call_structured(
                self.client, build_speech_prompt(**retry_kwargs), SpeechPack,
                stream=True, on_delta=on_delta, validate=validate,
                mock_reply=speech_mock,
            )
            if pack2 is not None and detect_leak(pack2.speech.strip()[:MAX_SPEECH_CHARS], self.role) is None:
                pack, speech, leak = pack2, pack2.speech.strip()[:MAX_SPEECH_CHARS], None
            if leak is not None:
                self.leak_count += 1
                logger.warning(
                    "AI「%s」二次穿帮，替换为安全发言：%s", self.name, leak
                )
                speech = SAFE_SPEECH

        suspect = pack.suspect - 1 if pack.suspect > 0 else None
        self.last_suspect = suspect
        return speech, suspect, pack.mood

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
            ),
            VotePack,
            validate=validate,
            mock_reply=vote_mock,
        )
        target: int | None = (
            pack.vote - 1 if pack is not None and pack.vote > 0 else None
        )
        if pack is None:
            logger.warning("AI「%s」投票降级为弃票：%s", self.name, err)
        self.my_votes.append((day, target))
        return target

    # ------------------------------------------------------------ 技能决策

    def wolf_den_decision(
        self, view: list[Event], day: int, killable: list[int]
    ) -> int:
        """狼窝会话：两狼共享一次调用统一刀口。返回 0-based 目标。"""
        def validate(pack: TargetPack) -> str | None:
            return self._alive_target_error(pack.target, killable, allow_zero=False)

        pack, err = call_structured(
            self.client,
            build_wolf_den_prompt(
                wolf_seats=self.teammate_seats + [self.seat]
                if self.seat not in self.teammate_seats
                else sorted(set(self.teammate_seats) | {self.seat}),
                persona=self.persona, view=view, day=day, killable=killable,
            ),
            TargetPack,
            validate=validate,
            mock_reply=lambda: json.dumps({"target": killable[0] + 1}),
        )
        if pack is None:
            logger.warning("狼窝决策降级为刀首名好人：%s", err)
            return killable[0]
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
            ),
            TargetPack,
            validate=validate,
            mock_reply=lambda: json.dumps({"target": checkable[0] + 1}),
        )
        if pack is None:
            logger.warning("预言家决策降级为查验首位：%s", err)
            return checkable[0]
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
            ),
            WitchPack,
            validate=validate,
            mock_reply=lambda: json.dumps({"save": False, "poison": 0}),
        )
        if pack is None:
            logger.warning("女巫决策降级为不用药：%s", err)
            return False, None
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
    validate: Callable[[BaseModel], str | None] | None = None,
    mock_reply: Callable[[], str] | None = None,
) -> tuple[BaseModel | None, str | None]:
    """带重试的结构化调用：解析失败/业务校验失败时携带错误重试（最多 2 次）。

    mock_reply：mock 模式下按请求 schema 生成合法回复（避免 mock 误触发降级）。
    返回 (模型实例, None) 或 (None, 最终错误)。
    """
    convo = list(messages)
    last_err = "未知错误"
    for attempt in range(max_retries + 1):
        text = client.chat(
            convo, stream=stream, on_delta=on_delta, mock_reply=mock_reply
        )
        pack, err = parse_pack(text, model_cls)
        if pack is not None and validate is not None:
            err = validate(pack)
        if pack is not None and err is None:
            return pack, None
        last_err = err or "未知错误"
        logger.warning(
            "结构化输出解析失败（第 %d 次）：%s", attempt + 1, last_err
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
    return None, last_err