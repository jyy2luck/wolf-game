"""按真实角色、当前规则和合法可见局面检索策略；不调用模型或读取全知状态。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.events import Event, EventKind, visible_events
from engine.roles import Role
from engine.boards import Board, SIX_CLASSIC

logger = logging.getLogger("llm_murder")

Action = Literal[
    "speech", "vote", "wolf_kill", "seer_check", "witch_act",
    "hunter_shot", "guard", "link", "special", "last_words",
    "sheriff_run", "sheriff_withdraw", "sheriff_vote", "sheriff_direction", "badge",
]
MAX_CARDS = 8
MAX_REFERENCE_CHARS = 4000
WOLF_ROLES = frozenset({"狼人", "白狼王", "狼美人"})


@dataclass(frozen=True)
class RuleProfile:
    """只能提供公开配置和已经明确的规则能力；未知能力不默认存在。"""

    name: str
    initial_players: int
    initial_wolves: int | None
    roles: frozenset[str]
    capabilities: frozenset[str]
    victory: str = "parity"
    role_counts: tuple[tuple[str, int], ...] = ()


def rule_profile(board: Board) -> RuleProfile:
    return RuleProfile(board.id + "-v" + board.version, board.player_count,
                       dict(board.role_counts)[Role.WEREWOLF],
                       frozenset(r.value for r, _ in board.role_counts), board.capabilities,
                       board.victory, tuple((r.value, n) for r, n in board.role_counts))


CURRENT_RULES = RuleProfile(
    name="six-player-v1",
    initial_players=6,
    initial_wolves=2,
    roles=frozenset(role.value for role in SIX_CLASSIC.setup),
    capabilities=frozenset({
        "stable_two_team_parity", "stable_alignment_checks", "wolves_kill_only_good",
        "single_night_death_good", "no_other_night_protection", "no_death_counteraction", "witch_save",
        "witch_self_save", "witch_one_of_each", "witch_acts_before_dawn",
        "repeat_daily_speech", "individual_votes",
    }),
)


class PopulationGuide(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_alive: int = Field(ge=0)
    max_alive: int | None = Field(default=None, ge=0)
    guidance: str = Field(min_length=1, max_length=1000)
    victory_rules: frozenset[str] = frozenset({"parity"})

    @model_validator(mode="after")
    def valid_range(self):
        if self.max_alive is not None and self.max_alive < self.min_alive:
            raise ValueError("人数范围倒置")
        return self

    def matches(self, count: int) -> bool:
        return count >= self.min_alive and (self.max_alive is None or count <= self.max_alive)


class StrategyCard(PopulationGuide):
    id: str = Field(pattern=r"^[A-Z]{1,2}\d{2}$")
    role: str
    name: str = Field(min_length=1, max_length=60)
    adjustment: str = Field(min_length=1, max_length=1000)
    actions: frozenset[Action] = Field(min_length=1)
    priority: int = Field(ge=0, le=1000)
    required_rules: frozenset[str] = frozenset()
    required_flags: frozenset[str] = frozenset()
    excluded_flags: frozenset[str] = frozenset()
    boost_flags: frozenset[str] = frozenset()
    uses: frozenset[Literal["own", "other"]] = frozenset({"own", "other"})
    required_skills: frozenset[str] = frozenset()
    related_windows: frozenset[Action] = frozenset()
    clues: tuple[str, ...] = ()
    responses: tuple[str, ...] = ()
    failure_cost: str = ""
    counters: tuple[str, ...] = ()
    exit_conditions: tuple[str, ...] = ()
    example: str = ""
    counterexample: str = ""


class StrategyLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1, 2]
    source: str
    roles: dict[str, tuple[PopulationGuide, ...]]
    cards: tuple[StrategyCard, ...]

    @model_validator(mode="after")
    def complete_library(self):
        identifiers = [card.id for card in self.cards]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("策略编号重复")
        for identifier in ("C01", "C05"):
            if not any(card.id == identifier and card.role == "*" for card in self.cards):
                raise ValueError("缺少基础判断卡")
        if not self.roles:
            raise ValueError("缺少身份人数指导")
        for role, guides in self.roles.items():
            next_minimum = 0
            for guide in sorted(guides, key=lambda row: row.min_alive):
                if guide.min_alive != next_minimum:
                    raise ValueError(f"{role}人数指导有空缺或重叠")
                next_minimum = guide.max_alive + 1 if guide.max_alive is not None else -1
            if next_minimum != -1:
                raise ValueError(f"{role}人数指导缺少开放上界")
        if any(card.role != "*" and card.role not in self.roles for card in self.cards):
            raise ValueError("策略引用未定义身份")
        if any(set(card.counters) - set(identifiers) for card in self.cards):
            raise ValueError("反制策略关联编号无效")
        return self


@lru_cache(maxsize=1)
def load_strategy_library() -> StrategyLibrary | None:
    """包数据局内只加载一次，查询仍在每次行动按新局面计算。"""
    try:
        raw = files("llm").joinpath("strategy_library.json").read_text(encoding="utf-8")
        return StrategyLibrary.model_validate_json(raw)
    except (OSError, ValueError) as exc:
        logger.warning("策略库不可用，使用基础参考（%s）", type(exc).__name__)
        return None


def public_dead_seats(view: list[Event]) -> frozenset[int]:
    dead: set[int] = set()
    for event in view:
        if not event.public:
            continue
        if event.kind is EventKind.DEATH:
            dead.update(event.extra.get("seats", ()))
        elif event.kind is EventKind.EXILE and event.seat is not None:
            dead.add(event.seat)
    return frozenset(dead)


@dataclass(frozen=True)
class Situation:
    role: str
    action: Action
    alive: frozenset[int]
    known_good: frozenset[int]
    known_wolves: frozenset[int]
    wolf_counts: tuple[int, ...]
    flags: frozenset[str]
    settled: bool
    claims: tuple[str, ...] = ()
    inference_complete: bool = True
    villager_counts: tuple[int, ...] = ()
    god_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class PublicClaim:
    event: Event
    role: str
    target: int | None = None
    is_wolf: bool | None = None


# shortcut: 仅提取明确第一人称身份及查验句式，歧义仍交由模型读原文，新增句式时补成对用例。
_CLAIM_TIME = r"(?:昨晚|昨夜|昨天|第\s*\d+\s*[夜天晚])"
_CLAIM_START = r"(?:^|[，,。！？!?；;\n])\s*(?:" + _CLAIM_TIME + r"\s*)?"
_ROLE_CLAIM = re.compile(_CLAIM_START + r"(?:我是|我跳|本人是)(预言家|女巫|平民|狼人|猎人|白痴|守卫)(?=$|[\s，,。！？!?；;])")
_CHECK_CLAIM = re.compile(
    _CLAIM_START + r"(?P<speaker>我)?(?:" + _CLAIM_TIME + r"\s*)?(?:查验|验)(?:了|过)?\s*(?P<target>[1-9]\d*)\s*号"
    r"\s*[，,：:]?\s*(?:是|为|结果是|结果为)?\s*(?P<result>狼人|好人|金水|查杀|狼)(?=$|[\s，,。！？!?；;])"
)
_QUOTED_CLAIM = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|‘[^’]*’|"[^"]*"')
_CONDITIONAL_OR_REPORT = re.compile(r"如果|假如|假设|要是|倘若|(?:[1-9]\d*\s*号|他|她|有人)(?:说|声称|表示)|转述|复述")


def public_claims(view: list[Event], seats: set[int] | None = None) -> tuple[PublicClaim, ...]:
    claims = []
    declared_roles = {}
    for event in view:
        if not event.public or event.kind is not EventKind.SPEECH:
            continue
        text = _QUOTED_CLAIM.sub("", event.extra["content"])
        for sentence in re.split(r"[。！？!?；;\n]", text):
            if _CONDITIONAL_OR_REPORT.search(sentence):
                continue
            for match in _ROLE_CLAIM.finditer(sentence):
                claims.append(PublicClaim(event, match[1]))
                declared_roles[event.seat] = match[1]
            for match in _CHECK_CLAIM.finditer(sentence):
                if (match["speaker"] or declared_roles.get(event.seat) == "预言家") and (seats is None or int(match["target"]) - 1 in seats):
                    claims.append(PublicClaim(event, "预言家", int(match["target"]) - 1,
                                              match["result"] in {"狼", "狼人", "查杀"}))
    return tuple(claims)


MAX_INFERENCE_BRANCHES = 5000


def _infer_counts(rules, alive, good, wolves, known_roles, settled):
    """只遍历狼座位组合，神民用容量区间检查；不枚举身份/私有经历。"""
    all_seats = set(range(rules.initial_players))
    initial_wolves = rules.initial_wolves
    if initial_wolves is None:
        lower = max(len(wolves & alive), 1 if settled else 0)
        upper = len(alive - good)
        if rules.victory == "parity" and settled:
            upper = min(upper, (len(alive) - 1) // 2)
        return tuple(range(lower, upper + 1)), (), (), set(), set(), True
    total_roles = dict(rules.role_counts)
    if total_roles and any(sum(r == value for r in known_roles.values()) > number
                           for value, number in total_roles.items()):
        return (), (), (), set(), set(), True
    remaining = initial_wolves - len(wolves)
    unknown = sorted(all_seats - good - wolves)
    if remaining < 0 or remaining > len(unknown):
        return (), (), (), set(), set(), True
    union, intersection, counts, villager_counts, god_counts = set(), None, set(), set(), set()
    village_total = total_roles.get("平民")
    forced_village = {s for s, r in known_roles.items() if r == "平民"}
    forced_god = {s for s, r in known_roles.items() if r not in WOLF_ROLES and r != "平民"}
    for i, group in enumerate(combinations(unknown, remaining)):
        if i >= MAX_INFERENCE_BRANCHES:
            # 截断交集绝不能用于身份认证，也不呈现不完备计数为精确范围。
            return (), (), (), set(), set(), False
        branch = wolves | set(group)
        w = len(branch & alive)
        if settled and (w == 0 or (rules.victory == "parity" and w * 2 >= len(alive))):
            continue
        living_good = alive - branch
        vg_range = ()
        if village_total is not None:
            if branch & (forced_village | forced_god):
                continue
            other_good = all_seats - branch
            v_min = max(len(forced_village & alive), village_total - len(other_good - alive - forced_god))
            v_max = min(len(living_good - forced_god), village_total - len(forced_village - alive))
            if settled and rules.victory == "edge":
                v_min, v_max = max(1, v_min), min(len(living_good) - 1, v_max)
            if v_min > v_max:
                continue
            vg_range = range(v_min, v_max + 1)
        counts.add(w)
        villager_counts.update(vg_range)
        god_counts.update(len(living_good) - v for v in vg_range)
        union.update(branch)
        intersection = branch.copy() if intersection is None else intersection & branch
    return (tuple(sorted(counts)), tuple(sorted(villager_counts)), tuple(sorted(god_counts)),
            all_seats - union if intersection is not None else set(), intersection or set(), True)


def build_situation(
    *, role: Role | str, seat: int, action: Action, view: list[Event], day: int,
    alive_seats: list[int], rules: RuleProfile | None = None, board: Board | None = None,
    teammate_seats: list[int] | None = None, kill_target: int | None = None,
    can_save: bool | None = None, can_poison: bool | None = None,
    settled: bool = True,
) -> Situation:
    """settled 表示当前人数已经经过一次完整判胜，而非待枪等过渡人数。"""
    rules = rules or rule_profile(board or SIX_CLASSIC)
    role_name = role.value if isinstance(role, Role) else role
    alive = frozenset(alive_seats)
    # 调用者本应已过滤；再次过滤，防止直接调用误传其他人的私有事件。
    own_view = visible_events(view, seat)
    good: set[int] = set()
    wolves: set[int] = set()
    flags: set[str] = set()
    caps = rules.capabilities
    fixed_parity = "stable_two_team_parity" in caps
    two_team = fixed_parity or "edge_victory" in caps
    known_roles = {seat: role_name} if role_name in rules.roles else {}
    if settled:
        flags.add("settled")
    is_wolf = role_name in WOLF_ROLES
    if two_team and role_name in rules.roles:
        (wolves if is_wolf else good).add(seat)
    if is_wolf:
        if teammate_seats is not None:
            wolves.update(teammate_seats)
        for event in own_view:
            if event.kind is EventKind.WOLF_REVEAL:
                wolves.update(event.viewers)
        if two_team and rules.initial_wolves is not None and len(wolves) == rules.initial_wolves:
            good.update(set(range(rules.initial_players)) - wolves)
    for event in own_view:
        if event.kind is EventKind.SPEECH and event.public:
            flags.add("has_public_speech")
        if two_team and event.public and event.kind in {EventKind.ROLE_ASSIGNED, EventKind.HUNTER_SHOT, EventKind.IDIOT_REVEAL}:
            confirmed_role = event.extra.get("role")
            if confirmed_role in rules.roles:
                (wolves if confirmed_role in WOLF_ROLES else good).add(event.seat)
                known_roles[event.seat] = confirmed_role
        if (event.kind is EventKind.SEER_CHECK and role_name == "预言家"
                and seat in event.viewers and "stable_alignment_checks" in caps):
            target = event.extra.get("target")
            result = event.extra.get("is_wolf")
            if isinstance(target, int) and isinstance(result, bool):
                flags.add("has_check")
                flags.add("has_wolf_check" if result else "has_good_check")
                (wolves if result else good).add(target)
        if (event.kind is EventKind.WITCH_KNEW and role_name == "女巫"
                and seat in event.viewers):
            flags.add("has_knife")
            target = event.extra.get("target")
            if fixed_parity and "wolves_kill_only_good" in caps and isinstance(target, int):
                good.add(target)
            if event.day == day and kill_target is None:
                kill_target = target
        if (fixed_parity and "single_night_death_good" in caps and event.public
                and event.kind is EventKind.DEATH and len(event.extra.get("seats", ())) == 1):
            # 此能力只在普通狼刀+同夜不可双药、无其他死因的板子中声明。
            good.update(event.extra["seats"])
    if role_name == "女巫":
        if kill_target is not None:
            flags.add("has_knife")
            if fixed_parity and "wolves_kill_only_good" in caps:
                good.add(kill_target)
            if kill_target == seat:
                flags.add("knife_is_self")
        if "witch_one_of_each" in caps:
            used = {event.extra.get("potion") for event in own_view
                    if event.kind is EventKind.POTION_USED and seat in event.viewers}
            if can_save is None:
                can_save = "解药" not in used
            if can_poison is None:
                can_poison = "毒药" not in used
        if can_save is not None:
            flags.add("antidote_available" if can_save else "antidote_unavailable")
        if can_poison is True:
            flags.add("poison_available")
    if day == 1 and action in {"wolf_kill", "seer_check", "witch_act", "guard"}:
        flags.add("first_night")
    if len(alive) <= 4:
        flags.add("small_population")
    if len(wolves & alive) >= 2:
        flags.add("multiple_wolves")

    all_seats = set(range(rules.initial_players))
    consistent = (bool(alive) and (seat in alive or action in {"hunter_shot", "last_words", "badge"})
                  and alive <= all_seats and not good & wolves and (good | wolves) <= all_seats)
    counts, villagers, gods, complete = (), (), (), True
    if two_team and consistent:
        counts, villagers, gods, inferred_good, inferred_wolves, complete = _infer_counts(
            rules, alive, good, wolves, known_roles, settled)
        if complete and counts:
            good.update(inferred_good)
            wolves.update(inferred_wolves)
        elif complete and not counts:
            # 矛盾输入不产生认证结论。
            good, wolves = set(), set()
            flags.add("contradictory_facts")
        if fixed_parity and settled and counts:
            dangers = [2 * count >= len(alive) - 1 for count in counts]
            if any(dangers): flags.add("critical_possible")
            if all(dangers): flags.add("critical_certain")
    if not consistent:
        good, wolves = set(), set()
        flags.add("contradictory_facts")
    if "edge_victory" in caps:
        flags.add("edge_victory")
    if any(e.public and e.kind is EventKind.IDIOT_REVEAL for e in own_view):
        flags.add("idiot_revealed")
    if any(e.public and e.kind is EventKind.SHERIFF_ELECTED and e.seat is not None for e in own_view):
        flags.add("sheriff_active")
    if len(wolves & alive) >= 2:
        flags.add("multiple_wolves")
    claims = public_claims(own_view, all_seats)
    latest_roles = {claim.event.seat: claim.role for claim in claims}
    seer_claimants = {author for author, claimed_role in latest_roles.items()
                     if claimed_role == "预言家" and author in alive}
    if len(seer_claimants) >= 2:
        flags.add("seer_contested")
    if role_name == "预言家" and seer_claimants - {seat}:
        flags.add("seer_counterclaim")
    latest_checks = {(c.event.seat, c.target): c for c in claims if c.target is not None}
    active_checks = [c for c in latest_checks.values() if c.event.seat in seer_claimants]
    if any(c.is_wolf and c.target == seat and c.event.seat != seat for c in active_checks):
        flags.add("self_accused")
    if any(c.is_wolf and c.target in good and c.event.seat != seat for c in active_checks):
        flags.add("known_good_accused")
    descriptions = tuple(
        f"第{c.event.day}天 [#{c.event.speech_index}] {c.event.seat + 1}号声明"
        + (f"自己是{c.role}" if c.target is None else f"查验{c.target + 1}号为{'狼人' if c.is_wolf else '好人'}")
        for c in claims
    )
    return Situation(role_name, action, alive, frozenset(good), frozenset(wolves),
                     counts, frozenset(flags), settled, descriptions, complete, villagers, gods)


def select_strategy_cards(
    library: StrategyLibrary, situation: Situation, rules: RuleProfile,
) -> tuple[StrategyCard, ...]:
    skills = {"狼人": "wolf_kill", "预言家": "seer_check", "女巫": "witch_act",
              "猎人": "hunter_shot", "守卫": "guard"}
    own_skills = {skills[situation.role]} if situation.role in skills else set()
    if "sheriff" in rules.capabilities:
        own_skills.add("sheriff")  # 身份资格仍以当前引擎窗口为准。
    candidates = [card for card in library.cards
                  if (card.role == "*" or card.role in rules.roles) and "own" in card.uses
                  and card.required_skills <= own_skills and situation.action in card.actions
                  and card.matches(len(situation.alive))
                  and card.required_rules <= rules.capabilities
                  and card.required_flags <= situation.flags
                  and not card.excluded_flags & situation.flags]
    foundation = [card for card in candidates if card.id in {"C01", "C05"}]
    remaining = [card for card in candidates if card.id not in {"C01", "C05"}]
    remaining.sort(key=lambda card: (
        -(card.priority + 40 * len(card.boost_flags & situation.flags)
          + (25 if card.role == situation.role else 0)), card.id,
    ))
    return tuple((foundation + remaining)[:5])


def select_other_cards(library, situation, rules):
    # 他方药量/查验未知；不把调用者的资源或真实查验赋给他方。
    flags = set(situation.flags) - {"has_check", "has_wolf_check", "has_good_check", "has_knife",
                                 "knife_is_self", "antidote_available", "antidote_unavailable", "poison_available"}
    flags.update({"antidote_available", "poison_available", "has_knife"})
    rows = [c for c in library.cards if c.role in rules.roles and "other" in c.uses
            and c.role != situation.role and (situation.action in c.actions or situation.action in c.related_windows)
            and c.matches(len(situation.alive)) and c.required_rules <= rules.capabilities
            and c.required_flags <= flags and not c.excluded_flags & flags]
    rows.sort(key=lambda c: (-(c.priority + 40 * len(c.boost_flags & flags)), c.id))
    # 为相关技能各保留空间，防止一种身份占满回应参考。
    chosen, roles = [], set()
    for card in rows:
        if card.role not in roles:
            chosen.append(card)
            roles.add(card.role)
        if len(chosen) == 3:
            break
    return tuple(chosen)


def _seat_labels(seats: frozenset[int]) -> str:
    ordered = sorted(seats)
    return "、".join(f"{seat + 1}号" for seat in ordered[:20]) + ("等" if len(ordered) > 20 else "")


def _situation_text(situation: Situation) -> str:
    n = len(situation.alive)
    lines = [f"当前存活 {n} 人；参考真实身份：{situation.role}；行动：{situation.action}。"]
    if situation.wolf_counts:
        branches = " / ".join(f"{w}狼{n - w}好人" for w in situation.wolf_counts)
        lines.append(f"仅按本局规则及你可见的事实推导，存活阵营人数候选：{branches}。")
        if "critical_possible" in situation.flags:
            qualifier = "全部候选" if "critical_certain" in situation.flags else "部分候选"
            lines.append(f"{qualifier}中，再净减员一名好人会到人数胜利线；技能与防护分支仍需单独比较。")
    else:
        lines.append("当前资料或规则不足以确定普通人数局的狼数候选；保留未知，按真实结算规则判断。")
    if not situation.inference_complete:
        lines.append("计算达到上限，推断非完备；截断分支的交集不能认证任何身份或资源。")
    if "edge_victory" in situation.flags:
        lines.append(f"屠边继续游戏约束：存活平民候选{list(situation.villager_counts)}，神职候选{list(situation.god_counts)}；未知身份不强行归类，狼好人数相等不代表终局。")
    for label, values in (("好人", situation.known_good), ("狼人", situation.known_wolves)):
        living = values & situation.alive
        if living:
            lines.append(f"你根据自身身份或合法可见硬事实确认的存活{label}：{_seat_labels(living)}。")
    unknown = situation.alive - situation.known_good - situation.known_wolves
    if unknown:
        lines.append(f"阵营仍未知：{_seat_labels(unknown)}；声明仅作待验证线索。")
    if "critical_certain" in situation.flags:
        lines.append("错放逐一名好人会立即达到狼人人数胜利线；平票后若狼刀成功且无技能扭转也会输。")
    if situation.claims:
        lines.append("近期公开声明对照（不是系统认证，历史说法可能已撤回；核对完整原文）：\n" + "\n".join(situation.claims[-12:]))
    return "\n".join(lines)


@dataclass(frozen=True)
class StrategyAdvice:
    situation: Situation
    cards: tuple[StrategyCard, ...]
    text: str
    other_cards: tuple[StrategyCard, ...] = ()


def get_strategy_advice(
    *, role: Role | str, seat: int, action: Action, view: list[Event], day: int,
    alive_seats: list[int], rules: RuleProfile | None = None, board: Board | None = None,
    teammate_seats: list[int] | None = None, kill_target: int | None = None,
    can_save: bool | None = None, can_poison: bool | None = None,
    settled: bool = True,
) -> StrategyAdvice:
    rules = rules or rule_profile(board or SIX_CLASSIC)
    situation = build_situation(
        role=role, seat=seat, action=action, view=view, day=day, alive_seats=alive_seats,
        rules=rules, teammate_seats=teammate_seats, kill_target=kill_target,
        can_save=can_save, can_poison=can_poison, settled=settled,
    )
    header = (
        "# 本轮策略库参考（不公开）\n"
        "依据最新可见历史比较这些打法，选本轮目标并随新信息调整；策略是建议，不是已发生事实。\n"
        "保持人设风格；真实阵营、合法技能与客观记录优先。自己的骗局和他人声明不变成真实信息。\n"
        "仅使用本局启用的角色和合法技能；参考文字提及的其他机制不适用于本局。\n"
        "投票使用全部发言后的判断；私有摘要可简短注明所用打法和切换条件，公开发言无需背诵攻略。\n"
    )
    text = header + f"规则配置：{rules.name}；启用角色：{'、'.join(sorted(rules.roles))}。\n"
    text += _situation_text(situation)
    library = load_strategy_library()
    if library is None:
        text += "\n策略库暂不可用：比较已确认事实与声明、人数风险和行动后果，选择当前有依据的行动。"
        return StrategyAdvice(situation, (), text[:MAX_REFERENCE_CHARS])
    guides = library.roles.get(situation.role, ()) if situation.role in rules.roles else ()
    guide = next((row for row in guides if row.matches(len(situation.alive))), None)
    if (guide is not None and rules.victory in guide.victory_rules and settled
            and (rules.victory != "parity" or "stable_two_team_parity" in rules.capabilities)):
        text += f"\n该身份在当前人数下的参考：{guide.guidance}"
    else:
        text += "\n按当局胜利条件、实际票权、资源和完整技能结算选择；不直接套用普通人数残局表。"
    included: list[StrategyCard] = []
    text += "\n共同依据与本人候选方法（跨角色表达可借用，真实技能权限不变）："
    for card in select_strategy_cards(library, situation, rules):
        line = (f"\n[{card.id}] {card.name}：{card.guidance} 调整/退出：{card.adjustment}"
                f" 线索：{'；'.join(card.clues)} 失败代价：{card.failure_cost}")
        if len(text) + len(line) <= MAX_REFERENCE_CHARS - 1100:
            text += line
            included.append(card)
    others = []
    text += "\n他方可能打法与可用应对（角色、私有结果和资源均为条件假设，不是认证）："
    for card in select_other_cards(library, situation, rules):
        line = (f"\n[{card.id}] 假设他方为{card.role}且资源可用：{card.name}。{card.guidance}"
                f" 可能回应：{'；'.join(card.responses)} 反制/退出：{card.adjustment}")
        if len(text) + len(line) <= MAX_REFERENCE_CHARS:
            text += line
            others.append(card)
    return StrategyAdvice(situation, tuple(included), text, tuple(others))
