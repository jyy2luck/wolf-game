"""确定性对局状态机：角色分配、昼夜循环、技能结算、投票计数、胜负判定。

引擎全知且确定性（设计 D1/D2）：所有规则判定由代码完成，LLM 不参与裁决。
决策通过 DecisionHooks 回调由外层提供（人类 UI / AI agents / 测试桩），
引擎对 hook 返回值做合法性校验，非法目标直接抛错。

白天发言顺序：从上一位死者的下家开始按座位顺序（环状），仅存活玩家发言。
投票：存活玩家各投一名存活玩家或弃票；最高票并列（含全弃票）→ 无人放逐。
胜负：狼全灭 = 好人胜；存活狼数 ≥ 存活好人数 = 狼人胜。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from engine import events as ev
from engine.roles import Role, Team, deal_roles, team_of


@dataclass
class Player:
    seat: int          # 0-based
    name: str
    role: Role
    alive: bool = True

    @property
    def label(self) -> str:
        return f"{self.name}（{self.seat + 1} 号）"


@dataclass(frozen=True)
class SpeechContext:
    """引擎实际调度的公开进度。座位为 0-based，position 为 1-based。"""

    day: int
    speaker: int
    order: tuple[int, ...]
    position: int
    spoken: tuple[int, ...]
    remaining: tuple[int, ...]
    names: tuple[tuple[int, str], ...]
    round_number: int = 1


# 决策回调类型 -----------------------------------------------------------------
# wolf_kill: 由狼队统一决策一个刀口目标
WolfKillFn = Callable[[Sequence[int], Sequence[int], list[ev.Event], int], int]
# seer_check: 预言家选一名查验目标
SeerCheckFn = Callable[[int, Sequence[int], list[ev.Event], int], int]
# witch_act: 返回 (用解药, 毒药目标或 None)，同一夜两者不得同时使用
WitchActFn = Callable[
    [int, int | None, bool, bool, list[ev.Event], int], tuple[bool, int | None]
]
# speech: 返回发言文本
SpeechFn = Callable[[int, list[ev.Event], int, SpeechContext], str]
# vote: 返回目标座位或 None（弃票）
VoteFn = Callable[[int, Sequence[int], list[ev.Event], int], int | None]


@dataclass
class DecisionHooks:
    """外层决策接口：main.py 中的控制器把人类 UI 与 AI agents 路由到这里。"""

    wolf_kill: WolfKillFn
    seer_check: SeerCheckFn
    witch_act: WitchActFn
    speech: SpeechFn
    vote: VoteFn


@dataclass
class Game:
    """6 人局状态机。rng 可注入种子以便测试复现。"""

    names: list[str]
    hooks: DecisionHooks
    rng: random.Random = field(default_factory=random.Random)
    roles: list[Role] = field(init=False)
    players: list[Player] = field(init=False)
    events: list[ev.Event] = field(default_factory=list, init=False)
    day: int = 0
    speech_counter: int = 0
    last_dead_seat: int | None = None
    save_available: bool = True    # 女巫解药
    poison_available: bool = True  # 女巫毒药
    winner: str | None = None      # Team 值（字符串）
    win_reason: str = ""

    # ------------------------------------------------------------ 开局

    def __post_init__(self) -> None:
        assert len(self.names) == 6, "6 人局需要 6 个名字"
        self.roles = deal_roles(rng=self.rng)
        self.players = [
            Player(seat=i, name=self.names[i], role=self.roles[i])
            for i in range(6)
        ]
        # 每人私有：自己的角色
        for p in self.players:
            self.events.append(ev.role_assigned(self.day, p.seat, p.role.value))
        # 狼人私有：队友揭示
        wolves = [p.seat for p in self.players if p.role is Role.WEREWOLF]
        teammate = "、".join(
            f"{s + 1} 号 {self.players[s].name}" for s in wolves
        )
        self.events.append(ev.wolf_reveal(self.day, wolves, teammate))

    # ------------------------------------------------------------ 状态查询

    def player(self, seat: int) -> Player:
        return self.players[seat]

    def role_of(self, seat: int) -> Role:
        return self.players[seat].role

    def alive_seats(self) -> list[int]:
        return [p.seat for p in self.players if p.alive]

    def wolves_alive(self) -> list[int]:
        return [
            p.seat for p in self.players
            if p.alive and p.role is Role.WEREWOLF
        ]

    def view(self, seat: int) -> list[ev.Event]:
        """该座位玩家的视角事件流（人类与 AI 共用）。"""
        return ev.visible_events(self.events, seat)

    # ------------------------------------------------------------ 主循环

    def run(self) -> str:
        """驱动昼夜循环直到胜负判定，返回获胜阵营。"""
        while self.winner is None:
            self.day += 1
            self._play_night()
            if self.winner is not None:
                break
            self._play_day()
        self.events.append(
            ev.game_over(self.day, self.winner or "", self.win_reason)
        )
        return self.winner or ""

    # ------------------------------------------------------------ 夜晚

    def _play_night(self) -> None:
        self.events.append(ev.night_falls(self.day))

        deaths: list[int] = []
        kill_target: int | None = None
        saved = False
        poisoned: int | None = None

        # 1) 狼刀：两狼共享统一目标（hook 由控制器路由到人类确认或狼窝会话）
        wolves = self.wolves_alive()
        if wolves:
            killable = [s for s in self.alive_seats() if self.role_of(s) is not Role.WEREWOLF]
            target = self.hooks.wolf_kill(wolves, killable, self.view(wolves[0]), self.day)
            target = self._validate_target(target, killable, "狼刀")
            kill_target = target
            self.events.append(ev.wolf_kill(self.day, wolves, target))

        # 2) 预言家查验：结果写入私有事件
        seer = next(
            (p for p in self.players
             if p.alive and p.role is Role.SEER), None
        )
        if seer is not None:
            checkable = [s for s in self.alive_seats() if s != seer.seat]
            target = self.hooks.seer_check(seer.seat, checkable, self.view(seer.seat), self.day)
            target = self._validate_target(target, checkable, "查验")
            is_wolf = self.role_of(target) is Role.WEREWOLF
            self.events.append(ev.seer_check(self.day, seer.seat, target, is_wolf))

        # 3) 女巫用药：解药/毒药各限一次、同夜不同用
        witch = next(
            (p for p in self.players
             if p.alive and p.role is Role.WITCH), None
        )
        if witch is not None and kill_target is not None:
            self.events.append(ev.witch_knew(self.day, witch.seat, kill_target))
            use_save, poison_target = self.hooks.witch_act(
                witch.seat, kill_target,
                self.save_available, self.poison_available,
                self.view(witch.seat), self.day,
            )
            if use_save and poison_target is not None:
                raise ValueError("解药与毒药同一夜不得同时使用")
            if use_save:
                if not self.save_available:
                    raise ValueError("解药已用完")
                self.save_available = False
                saved = True
                self.events.append(ev.potion_used(self.day, witch.seat, "解药", kill_target))
            if poison_target is not None:
                if not self.poison_available:
                    raise ValueError("毒药已用完")
                self.poison_available = False
                poison_target = self._validate_target(
                    poison_target, self.alive_seats(), "毒药"
                )
                poisoned = poison_target
                self.events.append(ev.potion_used(self.day, witch.seat, "毒药", poison_target))
            elif not use_save:
                self.events.append(ev.potion_used(self.day, witch.seat, "未用药", None))

        # 4) 天亮结算死讯
        if kill_target is not None and not saved:
            deaths.append(kill_target)
        if poisoned is not None and poisoned not in deaths:
            deaths.append(poisoned)

        self.events.append(ev.dawn(self.day))
        if deaths:
            for s in deaths:
                self.players[s].alive = False
            self.last_dead_seat = deaths[0]
            self.events.append(ev.death(self.day, deaths))
        else:
            self.events.append(ev.peaceful_night(self.day))

        self._check_game_over()

    # ------------------------------------------------------------ 白天

    def _play_day(self) -> None:
        alive = self.alive_seats()
        # 发言顺序：从上一位死者下家开始（环状座位顺序），仅存活者发言
        start = 0 if self.last_dead_seat is None else self.last_dead_seat + 1
        order = [s for s in range(start, 6)] + [s for s in range(0, start)]
        speakers = [s for s in order if s in alive]

        for round_number in (1, 2):
            for position, seat in enumerate(speakers):
                context = SpeechContext(
                    day=self.day, speaker=seat, order=tuple(speakers),
                    position=position + 1, spoken=tuple(speakers[:position]),
                    remaining=tuple(speakers[position + 1:]),
                    names=tuple((p.seat, p.name) for p in self.players), round_number=round_number,
                )
                text = self.hooks.speech(seat, self.view(seat), self.day, context)
                self.speech_counter += 1
                self.events.append(
                    ev.speech(self.day, seat, self.players[seat].name,
                              self.speech_counter, text, round_number=round_number)
                )

        # 统一投票：存活玩家各投一名存活玩家或弃票
        votes: list[tuple[int, int | None]] = []
        for seat in alive:
            target = self.hooks.vote(seat, alive, self.view(seat), self.day)
            if target is not None:
                target = self._validate_target(target, alive, "投票")
            votes.append((seat, target))

        tally: dict[int, int] = {}
        for _, target in votes:
            if target is not None:
                tally[target] = tally.get(target, 0) + 1
        self.events.append(ev.vote_result(self.day, tally, votes))

        # 计票放逐：最高票并列（含全弃票）→ 无人出局
        exiled: int | None = None
        if tally:
            top = max(tally.values())
            leaders = [s for s, c in tally.items() if c == top]
            if len(leaders) == 1:
                exiled = leaders[0]
        self.events.append(ev.exile(self.day, exiled))
        if exiled is not None:
            self.players[exiled].alive = False
            self.last_dead_seat = exiled

        self._check_game_over()

    # ------------------------------------------------------------ 胜负

    def _check_game_over(self) -> None:
        if self.winner is not None:
            return
        wolves = len(self.wolves_alive())
        good = len([p for p in self.players
                    if p.alive and team_of(p.role) is Team.GOOD])
        if wolves == 0:
            self.winner = Team.GOOD.value
            self.win_reason = "狼人全部出局"
        elif wolves >= good:
            self.winner = Team.WOLF.value
            self.win_reason = f"存活狼人（{wolves}）不少于存活好人（{good}）"

    # ------------------------------------------------------------ 校验

    def _validate_target(
        self, target: int | None, legal: Sequence[int], action: str
    ) -> int:
        if target is None or target not in legal:
            raise ValueError(f"{action} 非法目标：{target}，合法集合 {list(legal)}")
        return target
