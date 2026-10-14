"""标准板子的规则矩阵与公开阶段/私有视角不变量。"""
from argparse import Namespace
from collections import Counter
from dataclasses import FrozenInstanceError, replace
import json
import random

import pytest

from engine import events as ev
from engine.boards import BOARDS, SIX_CLASSIC, NINE_STANDARD, TWELVE_STANDARD, TWELVE_GUARD, get_board
from engine.game import DeathCause, DecisionHooks, Game
from engine.roles import Role, Team, deal_roles, is_god
from main import build_game, save_record


def make(board=NINE_STANDARD, **overrides):
    hooks = dict(wolf_kill=lambda w, k, v, d: k[-1],
                 seer_check=lambda s, k, v, d: k[0],
                 witch_act=lambda *a: (False, None),
                 speech=lambda s, v, d, c: "保留判断。",
                 vote=lambda *a: None,
                 action=lambda w, v: False if w.action in {"sheriff_run", "sheriff_withdraw"}
                 else "clockwise" if w.action == "sheriff_direction" else None)
    hooks.update(overrides)
    game = Game([f"玩家{s+1}" for s in range(board.player_count)], DecisionHooks(**hooks), random.Random(42), board)
    # 场景测试固定身份，仍按真实角色配置；离线桩不是实际玩家推理。
    game.roles = list(board.setup)
    for p, r in zip(game.players, game.roles):
        p.role = r
    game.events = [e for e in game.events if e.kind is ev.EventKind.BOARD]
    game.events += [ev.role_assigned(0, p.seat, p.role.value) for p in game.players]
    wolves = game.wolves_alive()
    game.events.append(ev.wolf_reveal(0, wolves, "、".join(str(s+1) for s in wolves)))
    return game


def seat(game, role):
    return next(p.seat for p in game.players if p.role is role)


def night(game):
    game.day += 1
    game._play_night()


@pytest.mark.parametrize("board", BOARDS.values(), ids=BOARDS.keys())
def test_board_snapshot_full_shuffle_reproducibility_and_immutable_definition(board):
    roles = deal_roles(board=board, rng=random.Random(321))
    assert roles == deal_roles(board=board, rng=random.Random(321))
    assert Counter(roles) == dict(board.role_counts)
    assert len(roles) == board.player_count
    assert sum(is_god(r) for r in roles) == (2 if board is SIX_CLASSIC else 3 if board is NINE_STANDARD else 4)
    projection = board.public_projection()
    projection["role_counts"].clear()
    assert board.public_projection()["role_counts"]
    assert not {"seats", "viewers", "knife", "resources", "pending_deaths"} & projection.keys()
    with pytest.raises(FrozenInstanceError):
        board.version = "2"
    with pytest.raises(ValueError):
        deal_roles(board=board, seats=board.player_count - 1)


@pytest.mark.parametrize("board", BOARDS.values(), ids=BOARDS.keys())
def test_dynamic_controller_and_personas(board):
    first = build_game(Namespace(mock=True, reconfig=False, seed=21, board=board.id))
    second = build_game(Namespace(mock=True, reconfig=False, seed=21, board=board.id))
    game, controller, _, _ = first
    assert len(game.players) == board.player_count
    assert len(controller.agents) == board.player_count - 1
    assert len({a.persona.name for a in controller.agents.values()}) == board.player_count - 1
    assert game.roles == second[0].roles and game.names == second[0].names
    assert controller.human == second[1].human


def test_invalid_board_and_missing_skill_handler_fail_before_configuration(monkeypatch):
    monkeypatch.setattr("main.ensure_config", lambda **kw: pytest.fail("不得访问模型配置"))
    with pytest.raises(ValueError, match="未知板子"):
        build_game(Namespace(mock=False, reconfig=False, seed=1, board="unknown"))
    with pytest.raises(ValueError, match="处理器"):
        Game([str(i) for i in range(9)], DecisionHooks(*([lambda *a: None] * 5)), board=NINE_STANDARD)
    hooks = make().hooks
    hooks.seer_check = None
    with pytest.raises(ValueError, match="seer_check"):
        Game([str(i) for i in range(9)], hooks, board=NINE_STANDARD)
    with pytest.raises(ValueError):
        replace(NINE_STANDARD, role_counts=((Role.WEREWOLF, 3), (Role.VILLAGER, 5)))
    with pytest.raises(ValueError):
        replace(NINE_STANDARD, guard_double_save_kills=True)
    with pytest.raises(ValueError):
        replace(NINE_STANDARD, role_counts=list(NINE_STANDARD.role_counts))


@pytest.mark.parametrize("board", [NINE_STANDARD, TWELVE_STANDARD, TWELVE_GUARD])
def test_edge_victory_and_wolf_extinction_priority(board):
    game = make(board)
    keep = {0, 1, seat(game, Role.SEER), seat(game, Role.VILLAGER)}
    for p in game.players:
        p.alive = p.seat in keep
    game._check_game_over()
    assert game.winner is None  # 两狼两好但神民都在。
    game.player(seat(game, Role.VILLAGER)).alive = False
    game._check_game_over()
    assert game.winner == Team.WOLF.value and "平民" in game.win_reason
    game = make(board)
    for p in game.players:
        p.alive = p.role is Role.VILLAGER
    game._check_game_over()
    assert game.winner == Team.GOOD.value


@pytest.mark.parametrize("board", [NINE_STANDARD, TWELVE_STANDARD, TWELVE_GUARD])
@pytest.mark.parametrize("target", [0, 1, None])
def test_friendly_fire_and_empty_knife(board, target):
    game = make(board, wolf_kill=lambda *a: target)
    night(game)
    assert not game.player(target).alive if target is not None else all(p.alive for p in game.players)
    if target is not None:
        assert game.death_causes[target] == {DeathCause.KNIFE}


def test_empty_knife_still_offers_poison_and_night_start_roles_act():
    game = make(wolf_kill=lambda *a: None, witch_act=lambda *a: (False, 0))
    night(game)
    assert game.death_causes[0] == {DeathCause.POISON}
    game = make()
    seer, witch = seat(game, Role.SEER), seat(game, Role.WITCH)
    game.hooks.wolf_kill = lambda *a: seer
    game.hooks.witch_act = lambda *a: (False, witch)
    night(game)
    assert any(e.kind is ev.EventKind.SEER_CHECK for e in game.view(seer))
    assert any(e.kind is ev.EventKind.POTION_USED for e in game.view(witch))
    assert not game.player(seer).alive and not game.player(witch).alive


def test_self_save_first_night_only_and_exhausted_knife_hidden():
    game = make()
    witch = seat(game, Role.WITCH)
    game.hooks.wolf_kill = lambda *a: witch
    game.hooks.witch_act = lambda s, k, cs, cp, v, d: (cs, None)
    night(game)
    assert game.player(witch).alive and not game.save_available
    captured = []
    game.hooks.wolf_kill = lambda *a: 0
    game.hooks.witch_act = lambda *a: captured.append(a) or (False, None)
    night(game)
    assert captured[0][1] is None and captured[0][2] is False
    assert not any(e.kind is ev.EventKind.WITCH_KNEW and e.day == 2 for e in captured[0][-2])
    game = make()
    witch = seat(game, Role.WITCH)
    game.day = 1
    game.hooks.wolf_kill = lambda *a: witch
    game.hooks.witch_act = lambda *a: (True, None)
    with pytest.raises(ValueError, match="自救限制"):
        night(game)


@pytest.mark.parametrize("guarded,saved,poisoned,dies", [
    (True, False, False, False), (False, True, False, False),
    (True, True, False, True), (False, False, False, True),
    (True, False, True, True), (False, False, True, True),
])
def test_guard_and_potions_rule_matrix(guarded, saved, poisoned, dies):
    game = make(TWELVE_GUARD)
    target = seat(game, Role.VILLAGER)
    original_action = game.hooks.action
    game.hooks.action = lambda w, v: target if w.action == "guard" and guarded else original_action(w, v)
    game.hooks.wolf_kill = lambda *a: target
    game.hooks.witch_act = lambda *a: (saved, target if poisoned else None)
    night(game)
    assert game.player(target).alive is not dies
    assert game.save_available is not saved
    guard = seat(game, Role.GUARD)
    assert not any(e.kind is ev.EventKind.GUARD for e in game.view(0))
    assert any(e.kind is ev.EventKind.GUARD for e in game.view(guard))


def test_guard_self_empty_and_repeat_restriction():
    game = make(TWELVE_GUARD, wolf_kill=lambda *a: None)
    guard = seat(game, Role.GUARD)
    choices = iter([guard, None, guard])
    base = game.hooks.action
    game.hooks.action = lambda w, v: next(choices) if w.action == "guard" else base(w, v)
    for _ in range(3):
        night(game)
    assert game.guard_previous == guard
    game.hooks.action = lambda w, v: guard if w.action == "guard" else base(w, v)
    with pytest.raises(ValueError, match="非法目标"):
        night(game)


@pytest.mark.parametrize("causes,shot", [
    ({DeathCause.KNIFE}, True), ({DeathCause.EXILE}, True), ({DeathCause.SHOT}, True),
    ({DeathCause.POISON}, False), ({DeathCause.KNIFE, DeathCause.POISON}, False),
])
def test_hunter_actual_death_causes_and_death_deduplication(causes, shot):
    game = make()
    hunter = seat(game, Role.HUNTER)
    called = []
    game.hooks.action = lambda w, v: called.append(w) or 0
    game.day = 1
    game._resolve_deaths({hunter: causes}, first_night=True)
    assert len(called) == int(shot)
    assert all(w.actor == hunter and hunter not in w.alive and not w.settled for w in called)
    assert sum(e.kind is ev.EventKind.DEATH and hunter in e.extra['seats'] for e in game.events) == 1
    events = [e for e in game.events if e.kind in {ev.EventKind.HUNTER_SHOT, ev.EventKind.SPEECH}]
    assert events[-1].extra["stage"] == "last_words"
    if shot:
        assert events[0].kind is ev.EventKind.HUNTER_SHOT
        assert events[0].extra["role"] == "猎人"
    else:
        assert not any(e.kind is ev.EventKind.HUNTER_READY for e in game.events)


def test_hunter_can_reverse_apparent_edge_loss_before_judgment_and_abstain_privacy():
    game = make()
    hunter, villager = seat(game, Role.HUNTER), seat(game, Role.VILLAGER)
    for p in game.players:
        p.alive = p.seat in {0, hunter, villager}
    game.hooks.action = lambda w, v: 0
    game._resolve_deaths({hunter: {DeathCause.EXILE}}, exile=True)
    game._settling = False
    game._check_game_over()
    assert game.winner == Team.GOOD.value
    game = make()
    hunter = seat(game, Role.HUNTER)
    game._resolve_deaths({hunter: {DeathCause.KNIFE}})
    assert not any(e.public and e.kind in {ev.EventKind.HUNTER_SHOT, ev.EventKind.HUNTER_READY} for e in game.events)


def test_idiot_exile_immunity_actual_vote_rights_and_badge_destroyed():
    game = make(TWELVE_STANDARD)
    idiot = seat(game, Role.IDIOT)
    game.day, game.sheriff = 1, idiot
    game.hooks.vote = lambda *a: idiot
    game._play_day()
    assert game.player(idiot).alive and not game.player(idiot).can_vote and game.player(idiot).revealed
    assert game.sheriff is None
    assert not any(e.kind is ev.EventKind.EXILE and e.seat == idiot for e in game.events)
    assert not any(e.kind is ev.EventKind.SPEECH and e.extra["stage"] == "last_words" for e in game.events)
    captured = []
    game.hooks.vote = lambda s, candidates, v, d: captured.append((s, candidates)) or None
    game.day = 2
    game._play_day()
    assert all(s != idiot and idiot not in candidates for s, candidates in captured)
    assert sum(e.kind is ev.EventKind.SPEECH and e.seat == idiot for e in game.events) == 4
    for p in game.players:
        if is_god(p.role) and p.seat != idiot:
            p.alive = False
    game._check_game_over()
    assert game.winner is None
    game._resolve_deaths({idiot: {DeathCause.POISON}})
    game._settling = False
    game._check_game_over()
    assert game.winner == Team.WOLF.value and "神职" in game.win_reason


@pytest.mark.parametrize("first_night,exile,expected", [(True, False, True), (False, True, True), (False, False, False)])
def test_last_words_sources_and_fake_role_never_grants_skill(first_night, exile, expected):
    contexts = []
    game = make(speech=lambda s, v, d, c: contexts.append(c) or "我是猎人。")
    villager = seat(game, Role.VILLAGER)
    game._resolve_deaths({villager: {DeathCause.EXILE if exile else DeathCause.KNIFE}}, first_night=first_night, exile=exile)
    assert len(contexts) == int(expected)
    assert not any(e.kind is ev.EventKind.HUNTER_READY for e in game.events)


def election_game(*, registrations=(0, 1), withdrawals=(), choices=None, dead=(2,)):
    captures = []
    def action(w, view):
        captures.append((w, list(view)))
        if w.action == "sheriff_run": return w.actor in registrations
        if w.action == "sheriff_withdraw": return w.actor in withdrawals
        if w.action == "sheriff_vote":
            return choices(w) if choices else w.targets[0]
        if w.action == "sheriff_direction": return "clockwise"
        return None
    game = make(TWELVE_STANDARD, action=action)
    game.day, game.pending_deaths, game._settling = 1, {s: {DeathCause.KNIFE} for s in dead}, True
    game._elect_sheriff()
    return game, captures


def test_first_night_election_hides_deaths_votes_and_pending_candidate_can_win():
    game, captures = election_game(registrations=(0,), dead=(0, 2))
    assert game.sheriff == 0
    assert all(0 in w.alive and 2 in w.alive for w, _ in captures)
    assert all(not any(e.kind in {ev.EventKind.DEATH, ev.EventKind.SHERIFF_VOTES} for e in v) for _, v in captures)
    contexts = []
    game = make(TWELVE_STANDARD, speech=lambda s,v,d,c: contexts.append(c) or "竞选。")
    game.hooks.action = lambda w,v: w.actor in {0, 1} if w.action == "sheriff_run" else False if w.action == "sheriff_withdraw" else w.targets[0]
    game.day, game.pending_deaths, game._settling = 1, {0: {DeathCause.KNIFE}}, True
    game._elect_sheriff()
    assert contexts[0].order == (0, 1) and not contexts[0].settled
    assert all(c.stage == "sheriff_speech" for c in contexts)


def test_election_barrier_actual_voters_and_withdrawn_dont_gain_votes():
    game, captures = election_game(registrations=(0, 1, 3), withdrawals=(3,), dead=(2,))
    votes = [(w, v) for w, v in captures if w.action == "sheriff_vote"]
    assert [w.actor for w, _ in votes] == [2, 4, 5, 6, 7, 8, 9, 10, 11]
    assert all(2 in w.alive and not any(e.kind is ev.EventKind.SHERIFF_VOTES for e in v) for w,v in votes)
    assert not any(e.kind is ev.EventKind.SHERIFF_VOTES for e in game.events)
    game._resolve_deaths(game.pending_deaths, first_night=True)
    result = next(e for e in game.events if e.kind is ev.EventKind.SHERIFF_VOTES)
    assert 2 not in dict(result.extra['votes']) and result.extra['invalid_voters'] == [2]
    assert game.events.index(result) > next(i for i,e in enumerate(game.events) if e.kind is ev.EventKind.DEATH)


@pytest.mark.parametrize("registrations,withdrawals", [((), ()), ((0,), (0,)), (tuple(range(12)), ())])
def test_no_sheriff_without_candidate_or_valid_electorate(registrations, withdrawals):
    game, _ = election_game(registrations=registrations, withdrawals=withdrawals, dead=())
    assert game.sheriff is None


def test_sheriff_tie_exactly_one_pk_and_all_abstention_ends_election():
    game, captures = election_game(dead=(), choices=lambda w: w.targets[w.actor % 2])
    speeches = [e for e in game.events if e.kind is ev.EventKind.SPEECH]
    assert len(speeches) == 4 and sum(e.extra['stage'] == 'sheriff_pk' for e in speeches) == 2
    assert game.sheriff is None
    assert sum(w.action == 'sheriff_vote' for w,_ in captures) == 20
    game, _ = election_game(choices=lambda w: None)
    assert game.sheriff is None


def test_weighted_exile_and_sheriff_speaks_last_in_both_rounds():
    game = make(TWELVE_STANDARD)
    game.day, game.sheriff = 1, 0
    game.hooks.action = lambda w,v: "counterclockwise" if w.action == "sheriff_direction" else None
    game.hooks.vote = lambda s,*a: 1 if s == 0 else 2 if s == 3 else None
    game._play_day()
    result = next(e for e in game.events if e.kind is ev.EventKind.VOTE_RESULT)
    assert result.extra['tally_units'] == {1: 3, 2: 2}
    assert result.extra['tally'] == {1: 1.5, 2: 1}
    assert not game.player(1).alive and game.player(2).alive
    speeches = [e.seat for e in game.events if e.kind is ev.EventKind.SPEECH and e.extra['stage'] in {'speech','reply'}]
    assert speeches == list(range(11, -1, -1)) * 2


@pytest.mark.parametrize("board", [NINE_STANDARD, TWELVE_STANDARD, TWELVE_GUARD])
def test_exile_pk_excludes_candidates_and_uses_barrier(board):
    captures = []
    def vote(s, candidates, view, day):
        captures.append((s, tuple(candidates), len([e for e in view if e.kind is ev.EventKind.VOTE_RESULT])))
        if len(candidates) == board.player_count:
            return 0 if s == 2 else 1 if s == 3 else None
        return None
    game = make(board, vote=vote)
    game.day = 1
    game._play_day()
    second = [row for row in captures if len(row[1]) == 2]
    assert all(s not in {0, 1} and count == 1 for s, _, count in second)
    assert len(second) == board.player_count - 2
    assert all(p.alive for p in game.players)
    assert sum(e.kind is ev.EventKind.SPEECH and e.extra['stage'] == 'exile_pk' for e in game.events) == 2


def test_dead_sheriff_badge_after_skills_and_no_re_election():
    game = make(TWELVE_STANDARD)
    hunter = seat(game, Role.HUNTER)
    game.sheriff, game.day = hunter, 1
    seen = []
    def action(w, view):
        seen.append(w)
        return 0 if w.action == 'hunter_shot' else 1 if w.action == 'badge' else None
    game.hooks.action = action
    game._resolve_deaths({hunter: {DeathCause.KNIFE}}, first_night=True)
    assert game.sheriff == 1
    badge = seen[-1]
    assert badge.action == 'badge' and 0 not in badge.targets and hunter not in badge.alive
    events = [e.kind for e in game.events]
    assert events.index(ev.EventKind.HUNTER_SHOT) < events.index(ev.EventKind.SPEECH) < events.index(ev.EventKind.BADGE)
    game.hooks.action = lambda *a: None
    game._resolve_deaths({1: {DeathCause.KNIFE}})
    assert game.sheriff is None


@pytest.mark.parametrize("board", BOARDS.values(), ids=BOARDS.keys())
def test_complete_deterministic_games_and_public_records(board, tmp_path):
    game = make(board)
    # 有界、确定性、全知测试代理验证引擎流程，不模拟胜率。
    game.hooks.wolf_kill = lambda wolves, targets, v, d: next(s for s in targets if game.role_of(s) is not Role.WEREWOLF)
    game.hooks.vote = lambda s, targets, v, d: next((t for t in targets if game.role_of(t) is Role.WEREWOLF), None)
    assert game.run() in {t.value for t in Team}
    assert game.day <= board.player_count
    _, _, _, client = build_game(Namespace(mock=True, reconfig=False, seed=4, board=board.id))
    path = tmp_path / f'{board.id}.json'
    save_record(path, game, client, 42)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['board'] == board.public_projection() and data['version'] == 2
    assert len(data['revealed_roles']) == board.player_count
    assert not {ev.EventKind.GUARD.value, ev.EventKind.HUNTER_READY.value, ev.EventKind.WOLF_KILL.value} & {e['kind'] for e in data['events']}
    with pytest.raises(FileExistsError):
        save_record(path, game, client, 42)
