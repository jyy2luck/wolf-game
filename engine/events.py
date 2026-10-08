"""事件溯源：对局一切信息以事件流表示。

- 公开事件：全员可见（发言、死讯、票型、放逐、终局）
- 私有事件：仅特定视角可见（自身角色、狼队友、狼刀目标、查验结果、用药）
- `visible_events(events, seat)` 为纯函数：人类与 AI 共用同一视角过滤机制

座位号内部为 0-based，渲染/文本中均为 1-based（座位号 = seat + 1）。
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field


class Phase(str, enum.Enum):
    NIGHT = "night"
    DAY = "day"


class EventKind(str, enum.Enum):
    # 私有事件
    ROLE_ASSIGNED = "role_assigned"      # 本人得知自己的角色
    WOLF_REVEAL = "wolf_reveal"          # 狼人得知队友（开局）
    WOLF_KILL = "wolf_kill"              # 狼人确认刀口目标（夜）
    SEER_CHECK = "seer_check"            # 预言家查验结果（夜）
    WITCH_KNEW = "witch_knew"            # 女巫得知当晚刀口（夜）
    POTION_USED = "potion_used"          # 女巫用药记录（夜）
    # 公开事件
    NIGHT_FALLS = "night_falls"          # 天黑
    DAWN = "dawn"                        # 天亮
    DEATH = "death"                      # 死讯（不翻牌）
    PEACEFUL_NIGHT = "peaceful_night"    # 平安夜
    SPEECH = "speech"                    # 白天发言（带全局编号）
    VOTE_RESULT = "vote_result"          # 票型公布
    EXILE = "exile"                      # 放逐结果（平票无人）
    GAME_OVER = "game_over"              # 终局（胜方与原因）


@dataclass
class Event:
    """单条对局事件。

    public=True 时全员可见；public=False 时仅 viewers（座位集合）可见。
    text 为面向玩家的中文文本（UI 与 prompt 记忆层共用）。
    extra 存放结构化细节（如查验结果、票型），供 UI/结算使用。
    """

    kind: EventKind
    day: int
    phase: Phase
    text: str
    public: bool = True
    viewers: frozenset[int] = field(default_factory=frozenset)
    seat: int | None = None          # 事件主体座位（0-based），无主体为 None
    speech_index: int | None = None  # SPEECH 事件的全局发言编号（1-based）
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ 构造辅助


def _make(
    kind: EventKind,
    day: int,
    phase: Phase,
    text: str,
    *,
    public: bool,
    viewers: Iterable[int] = (),
    seat: int | None = None,
    speech_index: int | None = None,
    **extra,
) -> Event:
    return Event(
        kind=kind,
        day=day,
        phase=phase,
        text=text,
        public=public,
        viewers=frozenset(viewers),
        seat=seat,
        speech_index=speech_index,
        extra=dict(extra),
    )


def role_assigned(day: int, seat: int, role_name: str) -> Event:
    return _make(
        EventKind.ROLE_ASSIGNED, day, Phase.NIGHT,
        f"你的身份是「{role_name}」。",
        public=False, viewers=[seat], seat=seat, role=role_name,
    )


def wolf_reveal(day: int, wolf_seats: Iterable[int], teammate_names: str) -> Event:
    seats = list(wolf_seats)
    return _make(
        EventKind.WOLF_REVEAL, day, Phase.NIGHT,
        f"你的狼人队友：{teammate_names}。夜晚共同确认刀口目标。",
        public=False, viewers=seats, extra={"wolf_seats": seats},
    )


def night_falls(day: int) -> Event:
    return _make(
        EventKind.NIGHT_FALLS, day, Phase.NIGHT,
        f"—— 第 {day} 夜降临，全员闭眼 ——",
        public=True,
    )


def wolf_kill(day: int, wolf_seats: Iterable[int], target: int) -> Event:
    return _make(
        EventKind.WOLF_KILL, day, Phase.NIGHT,
        f"狼人确认刀口：{target + 1} 号。",
        public=False, viewers=wolf_seats, seat=target, target=target,
    )


def seer_check(day: int, seer_seat: int, target: int, is_wolf: bool) -> Event:
    result = "狼人" if is_wolf else "好人"
    return _make(
        EventKind.SEER_CHECK, day, Phase.NIGHT,
        f"查验 {target + 1} 号：[{result}]。",
        public=False, viewers=[seer_seat], seat=target,
        target=target, is_wolf=is_wolf,
    )


def witch_knew(day: int, witch_seat: int, target: int) -> Event:
    if target == witch_seat:
        text = f"今晚的刀口是你自己（{target + 1} 号）。"
    else:
        text = f"今晚的刀口是 {target + 1} 号。"
    return _make(
        EventKind.WITCH_KNEW, day, Phase.NIGHT,
        text,
        public=False, viewers=[witch_seat], seat=target, target=target,
    )


def potion_used(
    day: int, witch_seat: int, potion: str, target: int | None
) -> Event:
    desc = f"{potion} → {target + 1} 号" if target is not None else f"{potion} → 未使用"
    return _make(
        EventKind.POTION_USED, day, Phase.NIGHT,
        f"用药记录：{desc}。",
        public=False, viewers=[witch_seat], seat=witch_seat,
        potion=potion, target=target,
    )


def dawn(day: int) -> Event:
    return _make(
        EventKind.DAWN, day, Phase.DAY,
        f"—— 第 {day} 天亮了 ——",
        public=True,
    )


def death(day: int, seats: Iterable[int]) -> Event:
    seat_list = list(seats)
    if not seat_list:
        raise ValueError("death 需要至少一个座位")
    who = "、".join(f"{s + 1} 号" for s in seat_list)
    return _make(
        EventKind.DEATH, day, Phase.DAY,
        f"{who} 死了。（身份暂不揭示）",
        public=True, seats=seat_list,
    )


def peaceful_night(day: int) -> Event:
    return _make(
        EventKind.PEACEFUL_NIGHT, day, Phase.DAY,
        "昨夜是平安夜，无人死亡。",
        public=True,
    )


def speech(day: int, seat: int, name: str, index: int, content: str) -> Event:
    return _make(
        EventKind.SPEECH, day, Phase.DAY,
        f"[#{index}] {name}（{seat + 1} 号）：{content}",
        public=True, seat=seat, speech_index=index, name=name, content=content,
    )


def vote_result(day: int, tally: dict[int, int]) -> Event:
    votes = "，".join(
        f"{seat + 1} 号 {count} 票" for seat, count in sorted(tally.items())
    )
    return _make(
        EventKind.VOTE_RESULT, day, Phase.DAY,
        f"票型：{votes}",
        public=True, tally=dict(tally),
    )


def exile(day: int, seat: int | None) -> Event:
    if seat is None:
        return _make(
            EventKind.EXILE, day, Phase.DAY,
            "平票，今日无人被放逐。",
            public=True, seat=None,
        )
    return _make(
        EventKind.EXILE, day, Phase.DAY,
        f"{seat + 1} 号被放逐出局。（身份暂不揭示）",
        public=True, seat=seat, target=seat,
    )


def game_over(day: int, winner: str, reason: str) -> Event:
    return _make(
        EventKind.GAME_OVER, day, Phase.DAY,
        f"游戏结束：{winner} 获胜！{reason}",
        public=True, winner=winner, reason=reason,
    )


# ------------------------------------------------------------------ 视角过滤


def visible_events(events: Iterable[Event], seat: int) -> list[Event]:
    """视角过滤纯函数：返回 seat 玩家可见的事件子集。

    公开事件全员可见；私有事件仅 viewers 内的座位可见。
    人类玩家与 AI 玩家共用本函数，保证信息公平。
    """
    return [e for e in events if e.public or seat in e.viewers]