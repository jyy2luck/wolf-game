"""按真实角色、当前规则和合法可见局面检索策略；不调用模型或读取全知状态。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.events import Event, EventKind, visible_events
from engine.roles import Role

logger = logging.getLogger("llm_murder")

Action = Literal[
    "speech", "vote", "wolf_kill", "seer_check", "witch_act",
    "hunter_shot", "guard", "link", "special", "last_words",
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


CURRENT_RULES = RuleProfile(
    name="six-player-v1",
    initial_players=6,
    initial_wolves=2,
    roles=frozenset(role.value for role in Role),
    capabilities=frozenset({
        "stable_two_team_parity", "stable_alignment_checks", "wolves_kill_only_good",
        "single_night_death_good", "no_other_night_protection", "no_death_counteraction", "witch_save",
        "witch_self_save", "witch_one_of_each", "witch_acts_before_dawn",
        "single_daily_speech", "aggregate_votes",
    }),
)


class PopulationGuide(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_alive: int = Field(ge=0)
    max_alive: int | None = Field(default=None, ge=0)
    guidance: str = Field(min_length=1, max_length=1000)

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


class StrategyLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
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


def build_situation(
    *, role: Role | str, seat: int, action: Action, view: list[Event], day: int,
    alive_seats: list[int], rules: RuleProfile = CURRENT_RULES,
    teammate_seats: list[int] | None = None, kill_target: int | None = None,
    can_save: bool | None = None, can_poison: bool | None = None,
    settled: bool = True,
) -> Situation:
    """settled 表示当前人数已经经过一次完整判胜，而非待枪等过渡人数。"""
    role_name = role.value if isinstance(role, Role) else role
    alive = frozenset(alive_seats)
    # 调用者本应已过滤；再次过滤，防止直接调用误传其他人的私有事件。
    own_view = visible_events(view, seat)
    good: set[int] = set()
    wolves: set[int] = set()
    flags: set[str] = set()
    caps = rules.capabilities
    fixed_parity = "stable_two_team_parity" in caps
    if settled:
        flags.add("settled")
    is_wolf = role_name in WOLF_ROLES
    if fixed_parity and role_name in rules.roles:
        (wolves if is_wolf else good).add(seat)
    if is_wolf:
        if teammate_seats is not None:
            wolves.update(teammate_seats)
        for event in own_view:
            if event.kind is EventKind.WOLF_REVEAL:
                wolves.update(event.viewers)
        if fixed_parity and rules.initial_wolves is not None and len(wolves) == rules.initial_wolves:
            good.update(set(range(rules.initial_players)) - wolves)
    for event in own_view:
        if event.kind is EventKind.SPEECH and event.public:
            flags.add("has_public_speech")
        if fixed_parity and event.public and event.kind is EventKind.ROLE_ASSIGNED:
            confirmed_role = event.extra.get("role")
            if confirmed_role in rules.roles:
                (wolves if confirmed_role in WOLF_ROLES else good).add(event.seat)
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
    if day == 1 and action in {"wolf_kill", "seer_check", "witch_act"}:
        flags.add("first_night")
    if len(alive) <= 4:
        flags.add("small_population")
    if len(wolves & alive) >= 2:
        flags.add("multiple_wolves")

    counts: tuple[int, ...] = ()
    all_seats = set(range(rules.initial_players))
    consistent = (bool(alive) and seat in alive and alive <= all_seats
                  and not good & wolves and (good | wolves) <= all_seats)
    if fixed_parity and consistent:
        dead = all_seats - alive
        lower = max(len(wolves & alive), 1 if settled else 0)
        upper = len(alive - good)
        if rules.initial_wolves is not None:
            lower = max(lower, rules.initial_wolves - len(dead - good))
            upper = min(upper, rules.initial_wolves - len(wolves & dead))
        if settled:
            upper = min(upper, (len(alive) - 1) // 2)
        counts = tuple(range(lower, upper + 1))
        if counts:
            # 只描述普通成功刀一名好人的条件分支；不把其他资源未知写成已无保护。
            dangers = [2 * count >= len(alive) - 1 for count in counts]
            if any(dangers):
                flags.add("critical_possible")
            if all(dangers):
                flags.add("critical_certain")
    if not consistent:
        disputed = good & wolves
        good = (good - disputed) & all_seats
        wolves = (wolves - disputed) & all_seats
    return Situation(role_name, action, alive, frozenset(good), frozenset(wolves),
                     counts, frozenset(flags), settled)


def select_strategy_cards(
    library: StrategyLibrary, situation: Situation, rules: RuleProfile,
) -> tuple[StrategyCard, ...]:
    if situation.role not in rules.roles:
        eligible_roles = {"*"}
    else:
        eligible_roles = {"*", situation.role}
        if situation.role in WOLF_ROLES:
            eligible_roles.add("狼人")
    candidates = [card for card in library.cards
                  if card.role in eligible_roles and situation.action in card.actions
                  and card.matches(len(situation.alive))
                  and card.required_rules <= rules.capabilities
                  and card.required_flags <= situation.flags
                  and not card.excluded_flags & situation.flags]
    foundation = [card for card in candidates if card.id in {"C01", "C05"}]
    remaining = [card for card in candidates if card.id not in {"C01", "C05"}]
    remaining.sort(key=lambda card: (
        -(card.priority + 40 * len(card.boost_flags & situation.flags)), card.id,
    ))
    return tuple((foundation + remaining)[:MAX_CARDS])


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
    for label, values in (("好人", situation.known_good), ("狼人", situation.known_wolves)):
        living = values & situation.alive
        if living:
            lines.append(f"你根据自身身份或合法可见硬事实确认的存活{label}：{_seat_labels(living)}。")
    return "\n".join(lines)


@dataclass(frozen=True)
class StrategyAdvice:
    situation: Situation
    cards: tuple[StrategyCard, ...]
    text: str


def get_strategy_advice(
    *, role: Role | str, seat: int, action: Action, view: list[Event], day: int,
    alive_seats: list[int], rules: RuleProfile = CURRENT_RULES,
    teammate_seats: list[int] | None = None, kill_target: int | None = None,
    can_save: bool | None = None, can_poison: bool | None = None,
    settled: bool = True,
) -> StrategyAdvice:
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
    if guide is not None and "stable_two_team_parity" in rules.capabilities and settled:
        text += f"\n该身份在当前人数下的参考：{guide.guidance}"
    else:
        text += "\n按当局胜利条件、实际票权、资源和完整技能结算选择；不直接套用普通人数残局表。"
    included: list[StrategyCard] = []
    for card in select_strategy_cards(library, situation, rules):
        line = f"\n[{card.id}] {card.name}：{card.guidance} 调整/退出：{card.adjustment}"
        if len(text) + len(line) <= MAX_REFERENCE_CHARS:
            text += line
            included.append(card)
    return StrategyAdvice(situation, tuple(included), text)
