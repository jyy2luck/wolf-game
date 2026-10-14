"""配置驱动的确定性状态机；规则、资格、屏障和完整判胜均由引擎裁决。"""
from __future__ import annotations

import enum
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from engine import events as ev
from engine.boards import Board, SIX_CLASSIC
from engine.roles import Role, Team, deal_roles, is_god, team_of


class DeathCause(str, enum.Enum):
    KNIFE = "knife"
    POISON = "poison"
    EXILE = "exile"
    SHOT = "shot"


@dataclass
class Player:
    seat: int
    name: str
    role: Role
    alive: bool = True
    can_vote: bool = True
    revealed: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}（{self.seat + 1} 号）"


@dataclass(frozen=True)
class SpeechContext:
    day: int
    speaker: int
    order: tuple[int, ...]
    position: int
    spoken: tuple[int, ...]
    remaining: tuple[int, ...]
    names: tuple[tuple[int, str], ...]
    round_number: int = 1
    stage: str = "speech"
    phase: ev.Phase = ev.Phase.DAY
    future_responders: tuple[int, ...] = ()
    settled: bool = True


@dataclass(frozen=True)
class ActionWindow:
    action: str
    actor: int
    day: int
    phase: ev.Phase
    alive: tuple[int, ...]
    targets: tuple[int, ...] = ()
    allow_empty: bool = False
    settled: bool = True
    vote_weight: int = 2
    options: tuple[str, ...] = ()


WolfKillFn = Callable[[Sequence[int], Sequence[int], list[ev.Event], int], int | None]
SeerCheckFn = Callable[[int, Sequence[int], list[ev.Event], int], int]
WitchActFn = Callable[[int, int | None, bool, bool, list[ev.Event], int], tuple[bool, int | None]]
SpeechFn = Callable[[int, list[ev.Event], int, SpeechContext], str]
VoteFn = Callable[[int, Sequence[int], list[ev.Event], int], int | None]
ActionFn = Callable[[ActionWindow, list[ev.Event]], int | str | bool | None]


@dataclass
class DecisionHooks:
    wolf_kill: WolfKillFn
    seer_check: SeerCheckFn
    witch_act: WitchActFn
    speech: SpeechFn
    vote: VoteFn
    action: ActionFn | None = None


@dataclass
class Game:
    names: list[str]
    hooks: DecisionHooks
    rng: random.Random = field(default_factory=random.Random)
    board: Board = SIX_CLASSIC
    roles: list[Role] = field(init=False)
    players: list[Player] = field(init=False)
    events: list[ev.Event] = field(default_factory=list, init=False)
    day: int = 0
    speech_counter: int = 0
    last_dead_seat: int | None = None
    save_available: bool = True
    poison_available: bool = True
    winner: str | None = None
    win_reason: str = ""
    phase: ev.Phase = ev.Phase.NIGHT
    stage: str = "night"
    sheriff: int | None = None
    guard_previous: int | None = None
    pending_deaths: dict[int, set[DeathCause]] = field(default_factory=dict, init=False)
    death_causes: dict[int, frozenset[DeathCause]] = field(default_factory=dict, init=False)
    _settling: bool = field(default=False, init=False)
    _election_records: list[ev.Event] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.board.validate()
        for name in ("wolf_kill", "seer_check", "witch_act", "speech", "vote"):
            if not callable(getattr(self.hooks, name)):
                raise ValueError(f"缺少 {name} 决策处理器")
        if len(self.names) != self.board.player_count:
            raise ValueError(f"{self.board.name}需要 {self.board.player_count} 个名字")
        if (self.board.sheriff or any(r in self.board.setup for r in (Role.GUARD, Role.HUNTER))) and not callable(self.hooks.action):
            raise ValueError("此板子需要新技能/警长 action 决策处理器")
        self.roles = deal_roles(rng=self.rng, board=self.board)
        self.players = [Player(i, name, role) for i, (name, role) in enumerate(zip(self.names, self.roles))]
        self.events.append(ev.Event(ev.EventKind.BOARD, 0, ev.Phase.NIGHT, self.board.rules_text(),
                                    extra=self.board.public_projection()))
        for p in self.players:
            self.events.append(ev.role_assigned(0, p.seat, p.role.value))
        wolves = [p.seat for p in self.players if p.role is Role.WEREWOLF]
        teammate = "、".join(f"{s + 1} 号 {self.players[s].name}" for s in wolves)
        self.events.append(ev.wolf_reveal(0, wolves, teammate))

    def player(self, seat: int) -> Player:
        return self.players[seat]

    def role_of(self, seat: int) -> Role:
        return self.player(seat).role

    def alive_seats(self) -> list[int]:
        # 夜间待死亡者尚未公开，保持公开存活名单直至统一公布。
        return [p.seat for p in self.players if p.alive]

    def wolves_alive(self) -> list[int]:
        return [p.seat for p in self.players if p.alive and p.role is Role.WEREWOLF]

    def view(self, seat: int) -> list[ev.Event]:
        return ev.visible_events(self.events, seat)

    def _public(self, kind, text, *, seat=None, **extra):
        event = ev.Event(kind, self.day, self.phase, text, seat=seat, extra=extra)
        self.events.append(event)
        return event

    def _stage(self, stage: str, phase: ev.Phase, **extra):
        self.stage, self.phase = stage, phase
        self._public(ev.EventKind.STAGE, f"阶段：{ev.STAGE_LABELS.get(stage, stage)}", stage=stage, settled=not self._settling, **extra)

    def _action(self, action: str, actor: int, targets=(), *, empty=False, options=()):
        if self.hooks.action is None:
            raise ValueError(f"缺少 {action} 处理器")
        window = ActionWindow(action, actor, self.day, self.phase, tuple(self.alive_seats()),
                              tuple(targets), empty, not self._settling,
                              3 if actor == self.sheriff else 2, tuple(options))
        result = self.hooks.action(window, self.view(actor))
        if options:
            if result not in options:
                raise ValueError(f"{action} 非法选项 {result}")
        elif action in {"sheriff_run", "sheriff_withdraw"}:
            if type(result) is not bool:
                raise ValueError(f"{action} 必须是布尔值")
        elif result is None and empty:
            pass
        else:
            self._validate_target(result, targets, action)
        return result

    def run(self) -> str:
        while self.winner is None:
            self.day += 1
            self._play_night()
            if self.winner is None:
                self._play_day()
        self.events.append(ev.game_over(self.day, self.winner, self.win_reason))
        return self.winner

    def _play_night(self):
        self.phase, self.stage = ev.Phase.NIGHT, "night"
        self.events.append(ev.night_falls(self.day))
        initial_alive = self.alive_seats()
        kill_target = guarded = poisoned = None
        saved = False
        guard = next((p for p in self.players if p.alive and p.role is Role.GUARD), None)
        if guard:
            guarded = self._action("guard", guard.seat, [s for s in initial_alive if s != self.guard_previous], empty=True)
            self.guard_previous = guarded
            self.events.append(ev.Event(ev.EventKind.GUARD, self.day, self.phase,
                f"守护：{guarded + 1}号。" if guarded is not None else "空守。",
                public=False, viewers=frozenset({guard.seat}), seat=guard.seat, extra={"target": guarded}))
        wolves = self.wolves_alive()
        if wolves:
            killable = [s for s in initial_alive if self.board.wolf_friendly_fire or self.role_of(s) is not Role.WEREWOLF]
            kill_target = self.hooks.wolf_kill(wolves, killable, self.view(wolves[0]), self.day)
            if kill_target is not None or not self.board.empty_kill:
                self._validate_target(kill_target, killable, "狼刀")
            self.events.append(ev.wolf_kill(self.day, wolves, kill_target))
        seer = next((p for p in self.players if p.alive and p.role is Role.SEER), None)
        if seer:
            checkable = [s for s in initial_alive if s != seer.seat]
            target = self.hooks.seer_check(seer.seat, checkable, self.view(seer.seat), self.day)
            self._validate_target(target, checkable, "查验")
            self.events.append(ev.seer_check(self.day, seer.seat, target, self.role_of(target) is Role.WEREWOLF))
        witch = next((p for p in self.players if p.alive and p.role is Role.WITCH), None)
        if witch and (kill_target is not None or self.board.empty_kill):
            visible_knife = kill_target if self.save_available or self.board.knife_after_save else None
            if visible_knife is not None:
                self.events.append(ev.witch_knew(self.day, witch.seat, visible_knife))
            can_save = self.save_available and visible_knife is not None and (
                visible_knife != witch.seat or self.board.witch_self_save == "always" or self.day == 1)
            use_save, poison = self.hooks.witch_act(witch.seat, visible_knife, can_save,
                self.poison_available, self.view(witch.seat), self.day)
            if use_save and poison is not None:
                raise ValueError("解药与毒药同一夜不得同时使用")
            if use_save:
                if not can_save:
                    raise ValueError("此窗口不能使用解药（药量、刀口或自救限制）")
                self.save_available, saved = False, True
                self.events.append(ev.potion_used(self.day, witch.seat, "解药", kill_target))
            if poison is not None:
                if not self.poison_available:
                    raise ValueError("毒药已用完")
                self._validate_target(poison, initial_alive, "毒药")
                self.poison_available, poisoned = False, poison
                self.events.append(ev.potion_used(self.day, witch.seat, "毒药", poison))
            elif not use_save:
                self.events.append(ev.potion_used(self.day, witch.seat, "未用药", None))
        causes: dict[int, set[DeathCause]] = {}
        protected = kill_target is not None and guarded == kill_target
        if kill_target is not None and ((not saved and not protected) or
                (saved and protected and self.board.guard_double_save_kills)):
            causes.setdefault(kill_target, set()).add(DeathCause.KNIFE)
        if poisoned is not None:
            causes.setdefault(poisoned, set()).add(DeathCause.POISON)
        self.pending_deaths = causes
        self._settling = True
        if self.day == 1 and self.board.sheriff:
            self._elect_sheriff()
        self.phase, self.stage = ev.Phase.DAY, "dawn"
        self.events.append(ev.dawn(self.day))
        if not causes:
            self.events.append(ev.peaceful_night(self.day))
            self.events.extend(self._election_records)
            self._election_records.clear()
        self._resolve_deaths(causes, first_night=self.day == 1)
        self.pending_deaths = {}
        self._settling = False
        self._check_game_over()

    def _resolve_deaths(self, causes: dict[int, set[DeathCause]], *, first_night=False, exile=False):
        """公布整批、完成所有反击，再遗言和警徽；死因仅引擎私有。"""
        self._settling = True
        original = [s for s in causes if self.player(s).alive]
        if self.board.id != "six-classic":
            original.sort()
        queue = list(original)
        for s in original:
            self.player(s).alive = False
            self.death_causes[s] = frozenset(causes[s])
        if original:
            self.last_dead_seat = original[0]
            if not exile:
                self.events.append(ev.death(self.day, original))
        self.events.extend(self._election_records)
        self._election_records.clear()
        while queue:
            seat = queue.pop(0)
            if self.role_of(seat) is not Role.HUNTER or DeathCause.POISON in self.death_causes[seat]:
                continue
            self.phase, self.stage = ev.Phase.HUNTER, "hunter_shot"
            self.events.append(ev.Event(ev.EventKind.HUNTER_READY, self.day, self.phase,
                "你有一次开枪或弃枪窗口。", public=False, viewers=frozenset({seat}), seat=seat))
            target = self._action("hunter_shot", seat, self.alive_seats(), empty=True)
            if target is not None:
                self.player(seat).revealed = True
                self._public(ev.EventKind.HUNTER_SHOT, f"{seat + 1}号猎人开枪带走{target + 1}号。",
                             seat=seat, role=Role.HUNTER.value, target=target)
                self.player(target).alive = False
                self.death_causes[target] = frozenset({DeathCause.SHOT})
                self.events.append(ev.death(self.day, [target]))
                self.last_dead_seat = target
                queue.append(target)
        if self.board.last_words and (first_night or exile):
            self._speeches(original, stage="last_words", phase=ev.Phase.LAST_WORDS)
        if self.sheriff is not None and not self.player(self.sheriff).alive:
            self.phase, self.stage = ev.Phase.BADGE, "badge"
            actor = self.sheriff
            target = self._action("badge", actor, [s for s in self.alive_seats() if self.player(s).can_vote], empty=True)
            self.sheriff = target
            self._public(ev.EventKind.BADGE, f"{actor + 1}号移交警徽给{target + 1}号。" if target is not None else f"{actor + 1}号撕毁警徽。",
                         seat=actor, target=target)

    def _speeches(self, speakers, *, stage="speech", phase=ev.Phase.DAY, rounds=(1,)):
        order = tuple(speakers)
        for round_number in rounds:
            actual_stage = "reply" if stage == "speech" and round_number == 2 else stage
            self._stage(actual_stage, phase, order=list(order), round_number=round_number)
            for i, seat in enumerate(order):
                future = order if stage == "speech" and round_number == 1 else order[i + 1:]
                context = SpeechContext(self.day, seat, order, i + 1, order[:i], order[i + 1:],
                    tuple((p.seat, p.name) for p in self.players), round_number,
                    actual_stage, phase, future, not self._settling)
                text = self.hooks.speech(seat, self.view(seat), self.day, context)
                self.speech_counter += 1
                self.events.append(ev.speech(self.day, seat, self.player(seat).name, self.speech_counter,
                    text, round_number=round_number, phase=phase, stage=actual_stage))

    def _election_vote(self, voters, candidates):
        votes = [(s, self._action("sheriff_vote", s, candidates, empty=True)) for s in voters]
        effective = [(s, t) for s, t in votes if s not in self.pending_deaths]
        tally = {}
        for _, target in effective:
            if target is not None:
                tally[target] = tally.get(target, 0) + 1
        # 延迟明细：实际资格也不进入下一位选民的上下文。
        self._election_records.append(ev.Event(ev.EventKind.SHERIFF_VOTES, self.day, ev.Phase.SHERIFF,
            "警长选举票型：" + "、".join(f"{s+1}号→{t+1}号" if t is not None else f"{s+1}号弃票" for s,t in effective),
            extra={"votes": effective, "invalid_voters": [s for s, _ in votes if s in self.pending_deaths], "tally": tally}))
        return self._leaders(tally)

    def _elect_sheriff(self):
        self._stage("sheriff_run", ev.Phase.SHERIFF)
        public_alive = self.alive_seats()
        candidates = [s for s in public_alive if self._action("sheriff_run", s)]
        self._public(ev.EventKind.SHERIFF_CANDIDATES, "警长报名：" + "、".join(f"{s+1}号" for s in candidates), candidates=candidates)
        original_candidates = set(candidates)
        self._speeches(candidates, stage="sheriff_speech", phase=ev.Phase.SHERIFF)
        self._stage("sheriff_withdraw", ev.Phase.SHERIFF)
        withdrawn = [s for s in candidates if self._action("sheriff_withdraw", s)]
        self._public(ev.EventKind.SHERIFF_WITHDRAW, "退水：" + "、".join(f"{s+1}号" for s in withdrawn), seats=withdrawn)
        candidates = [s for s in candidates if s not in withdrawn]
        leaders = candidates if len(candidates) <= 1 else []
        if len(candidates) > 1:
            voters = [s for s in public_alive if s not in original_candidates]
            self._stage("sheriff_vote", ev.Phase.SHERIFF, candidates=candidates)
            leaders = self._election_vote(voters, candidates)
            if len(leaders) > 1:
                self._speeches(sorted(leaders), stage="sheriff_pk", phase=ev.Phase.PK)
                self._stage("sheriff_revote", ev.Phase.SHERIFF, candidates=leaders)
                leaders = self._election_vote(voters, leaders)
        self.sheriff = leaders[0] if len(leaders) == 1 else None
        self._public(ev.EventKind.SHERIFF_ELECTED,
            f"{self.sheriff + 1}号当选警长。" if self.sheriff is not None else "本局无警长。", seat=self.sheriff)

    def _play_day(self):
        alive = self.alive_seats()
        n = len(self.players)
        if self.sheriff is not None:
            self._stage("sheriff_direction", ev.Phase.DAY)
            direction = self._action("sheriff_direction", self.sheriff, options=("clockwise", "counterclockwise"))
            step = 1 if direction == "clockwise" else -1
            order = [(self.sheriff + step * i) % n for i in range(1, n + 1)]
            self._public(ev.EventKind.SHERIFF_DIRECTION, f"警长选择{'顺' if step == 1 else '逆'}时针发言。", seat=self.sheriff, direction=direction)
        else:
            start = 0 if self.last_dead_seat is None else (self.last_dead_seat + 1) % n
            order = [(start + i) % n for i in range(n)]
        self._speeches([s for s in order if s in alive], rounds=(1, 2))
        candidates = [s for s in alive if not (self.role_of(s) is Role.IDIOT and self.player(s).revealed)]
        voters = [s for s in alive if self.player(s).can_vote]
        leaders = self._exile_vote(voters, candidates)
        if len(leaders) > 1 and self.board.exile_pk:
            self._speeches(sorted(leaders), stage="exile_pk", phase=ev.Phase.PK)
            leaders = self._exile_vote([s for s in voters if s not in leaders], leaders, stage="exile_revote")
        exiled = leaders[0] if len(leaders) == 1 else None
        self.phase, self.stage = ev.Phase.DAY, "exile"
        if exiled is not None and self.role_of(exiled) is Role.IDIOT and not self.player(exiled).revealed:
            self.player(exiled).revealed, self.player(exiled).can_vote = True, False
            self._public(ev.EventKind.IDIOT_REVEAL, f"{exiled+1}号白痴翻牌免死，永久失去投票权。", seat=exiled, role=Role.IDIOT.value)
            if self.sheriff == exiled:
                self.sheriff = None
                self._public(ev.EventKind.BADGE, "白痴翻牌失票，警徽自动撕毁。", seat=exiled, target=None)
        else:
            self.events.append(ev.exile(self.day, exiled))
            if exiled is not None:
                self._resolve_deaths({exiled: {DeathCause.EXILE}}, exile=True)
        self._settling = False
        self._check_game_over()

    def _exile_vote(self, voters, candidates, *, stage="vote"):
        weights = {s: 3 if s == self.sheriff else 2 for s in voters}
        self._stage(stage, ev.Phase.DAY, candidates=list(candidates), voters=list(voters), weights=weights)
        votes = []
        for seat in voters:
            target = self.hooks.vote(seat, candidates, self.view(seat), self.day)
            if target is not None:
                self._validate_target(target, candidates, "投票")
            votes.append((seat, target))
        units = {}
        for s, t in votes:
            if t is not None:
                units[t] = units.get(t, 0) + weights[s]
        self.events.append(ev.vote_result(self.day, {s: u / 2 for s, u in units.items()}, votes))
        self.events[-1].extra.update(weights=weights, tally_units=units, stage=stage)
        return self._leaders(units)

    @staticmethod
    def _leaders(tally):
        return sorted(s for s, c in tally.items() if c == max(tally.values())) if tally else []

    def _check_game_over(self):
        if self.winner is not None or self._settling or self.pending_deaths:
            return
        wolves = len(self.wolves_alive())
        good = sum(p.alive and team_of(p.role) is Team.GOOD for p in self.players)
        if wolves == 0:
            self.winner, self.win_reason = Team.GOOD.value, "狼人全部出局"
        elif self.board.victory == "parity" and wolves >= good:
            self.winner, self.win_reason = Team.WOLF.value, f"存活狼人（{wolves}）不少于存活好人（{good}）"
        elif self.board.victory == "edge":
            for label, predicate in (("平民", lambda r: r is Role.VILLAGER), ("神职", is_god)):
                if not any(p.alive and predicate(p.role) for p in self.players):
                    self.winner, self.win_reason = Team.WOLF.value, f"屠边：{label}全部出局"
                    break

    def _validate_target(self, target, legal: Sequence[int], action: str) -> int:
        if type(target) is not int or target not in legal:
            raise ValueError(f"{action} 非法目标：{target}，合法集合 {list(legal)}")
        return target
