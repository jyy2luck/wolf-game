"""engine/game 单元测试：夜晚结算各分支、计票与平票、胜负条件、视角隔离不变量。"""

import random

import pytest

from engine import events as ev
from engine.game import DecisionHooks, Game
from engine.events import EventKind
from engine.roles import Role, Team


# ---------------------------------------------------------------- 测试工具


def default_hooks(**overrides) -> DecisionHooks:
    """默认行为桩：狼刀/查验取第一个合法目标、女巫不用药、固定发言、投第一个他人。"""
    base = dict(
        wolf_kill=lambda wolves, killable, view, day: killable[0],
        seer_check=lambda seer, checkable, view, day: checkable[0],
        witch_act=lambda witch, kill, can_save, can_poison, view, day: (False, None),
        speech=lambda seat, view, day, context: f"座位{seat}的发言",
        vote=lambda seat, alive, view, day: next((s for s in alive if s != seat), None),
    )
    base.update(overrides)
    return DecisionHooks(**base)


def make_game(seed: int = 42, **hook_overrides) -> Game:
    """构造 6 人局；hook 可用闭包捕获 game 做角色感知决策。"""
    game = Game(
        names=["A", "B", "C", "D", "E", "F"],
        hooks=default_hooks(),
        rng=random.Random(seed),
    )
    if hook_overrides:
        game.hooks = default_hooks(**hook_overrides)
    return game


def wolves_of(game: Game) -> list[int]:
    return [p.seat for p in game.players if p.role is Role.WEREWOLF]


def goods_of(game: Game) -> list[int]:
    return [p.seat for p in game.players if p.role not in (Role.WEREWOLF,)]


def run_night(game: Game) -> None:
    game.day += 1
    game._play_night()


# ---------------------------------------------------------------- 角色分配


def test_initial_deal_and_private_events():
    game = make_game()
    assert len(game.players) == 6
    assert len(wolves_of(game)) == 2

    for p in game.players:
        # 每人可见自己的角色事件，且不可见他人角色事件
        mine = [
            e for e in game.view(p.seat)
            if e.kind is EventKind.ROLE_ASSIGNED
        ]
        assert len(mine) == 1
        assert p.role.value in mine[0].text


def test_wolf_reveal_shared_only_between_wolves():
    game = make_game()
    wolves = wolves_of(game)
    for seat in range(6):
        reveals = [
            e for e in game.view(seat)
            if e.kind is EventKind.WOLF_REVEAL
        ]
        if seat in wolves:
            assert len(reveals) == 1
        else:
            assert reveals == []


# ---------------------------------------------------------------- 夜晚结算


def test_night_kill_no_potion():
    game = make_game()
    wolves = wolves_of(game)
    target = next(s for s in range(6) if s not in wolves)  # 第一个好人
    game.hooks = default_hooks(wolf_kill=lambda w, k, v, d: target)
    run_night(game)

    assert not game.players[target].alive
    deaths = [e for e in game.events if e.kind is EventKind.DEATH]
    assert len(deaths) == 1 and deaths[0].seat is None  # 死讯不点验主体座位字段
    assert f"{target + 1} 号" in deaths[0].text
    # 女巫没被唤醒用药时也有"未用药"记录吗？——女巫活着就必然被告知刀口并决策
    assert not any(
        e.kind is EventKind.PEACEFUL_NIGHT for e in game.events
    )


def test_night_witch_saves():
    game = make_game()
    wolves = wolves_of(game)
    target = next(s for s in range(6) if s not in wolves)
    game.hooks = default_hooks(
        wolf_kill=lambda w, k, v, d: target,
        witch_act=lambda witch, kill, cs, cp, v, d: (True, None),
    )
    run_night(game)

    assert game.players[target].alive  # 被救
    assert not game.save_available
    potions = [e.extra["potion"] for e in game.events if e.kind is EventKind.POTION_USED]
    assert potions == ["解药"]
    assert any(
        e.kind is EventKind.PEACEFUL_NIGHT for e in game.events
    )


def test_night_witch_poisons():
    game = make_game()
    wolves = wolves_of(game)
    target = next(s for s in range(6) if s not in wolves)
    victim = next(s for s in range(6) if s not in wolves and s != target)
    game.hooks = default_hooks(
        wolf_kill=lambda w, k, v, d: target,
        witch_act=lambda witch, kill, cs, cp, v, d: (False, victim),
    )
    run_night(game)

    assert not game.players[target].alive
    assert not game.players[victim].alive
    assert not game.poison_available
    death_events = [e for e in game.events if e.kind is EventKind.DEATH]
    assert len(death_events) == 1
    text = death_events[0].text
    assert f"{target + 1} 号" in text and f"{victim + 1} 号" in text


def test_night_empty_kill_when_witch_saves_is_peaceful():
    # 救刀即平安夜（空刀场景：无死亡）
    game = make_game()
    wolves = wolves_of(game)
    target = next(s for s in range(6) if s not in wolves)
    game.hooks = default_hooks(
        wolf_kill=lambda w, k, v, d: target,
        witch_act=lambda witch, kill, cs, cp, v, d: (True, None),
    )
    run_night(game)
    assert any(
        e.kind is EventKind.PEACEFUL_NIGHT for e in game.events
    )


def test_witch_both_potions_same_night_raises():
    game = make_game()
    game.hooks = default_hooks(
        witch_act=lambda witch, kill, cs, cp, v, d: (True, 0),
    )
    with pytest.raises(ValueError):
        run_night(game)


def test_witch_potions_limited_once():
    game = make_game()
    wolves = wolves_of(game)

    def witch_save_self_limit(witch, kill, cs, cp, v, d):
        assert cs is game.save_available  # 状态可见
        return (True, None)

    game.hooks = default_hooks(witch_act=witch_save_self_limit)
    run_night(game)
    assert game.save_available is False

    # 第二夜：解药已用完，hook 违规再用应报错
    game.day += 1
    game.hooks = default_hooks(witch_act=lambda w, k, cs, cp, v, d: (True, None))
    with pytest.raises(ValueError):
        game._play_night()


def test_seer_check_writes_private_event():
    game = make_game()
    seer_seat = next(p.seat for p in game.players if p.role is Role.SEER)
    wolves = set(wolves_of(game))
    target = next(s for s in range(6) if s != seer_seat and s in wolves)
    game.hooks = default_hooks(seer_check=lambda seer, ck, v, d: target)
    run_night(game)

    checks = [
        e for e in game.view(seer_seat)
        if e.kind is EventKind.SEER_CHECK
    ]
    assert len(checks) == 1
    assert checks[0].extra["is_wolf"] is True
    assert "狼人" in checks[0].text
    # 其他人看不到
    other = next(s for s in range(6) if s != seer_seat)
    assert not [
        e for e in game.view(other) if e.kind is EventKind.SEER_CHECK
    ]


def test_witch_told_kill_target_privately():
    game = make_game()
    witch_seat = next(p.seat for p in game.players if p.role is Role.WITCH)
    run_night(game)
    knew = [
        e for e in game.view(witch_seat)
        if e.kind is EventKind.WITCH_KNEW
    ]
    assert len(knew) == 1
    # 其他座位看不到
    other = next(s for s in range(6) if s != witch_seat)
    assert not [e for e in game.view(other) if e.kind is EventKind.WITCH_KNEW]


def test_illegal_wolf_target_raises():
    game = make_game()
    wolves = wolves_of(game)
    game.hooks = default_hooks(
        wolf_kill=lambda w, k, v, d: wolves[0]  # 不能刀队友
    )
    with pytest.raises(ValueError):
        run_night(game)


# ---------------------------------------------------------------- 白天流程


def test_speech_context_follows_official_events_after_seat_five_dies():
    captured = []

    def speak(seat, view, day, context):
        captured.append(context)
        assert context.speaker == seat and context.day == day
        official = [e.seat for e in view if e.kind is EventKind.SPEECH and e.day == day and e.extra.get("round_number", 1) == context.round_number]
        assert list(context.spoken) == official
        assert all(e.public or seat in e.viewers for e in view)
        return f"正式发言 {seat + 1}"

    game = make_game(speech=speak, vote=lambda *args: None)
    game.day = 1
    game.players[4].alive = False
    game.last_dead_seat = 4
    game.events.append(ev.death(1, [4]))
    game._play_day()
    order = (5, 0, 1, 2, 3)
    assert tuple(c.speaker for c in captured) == order * 2
    for index, context in enumerate(captured):
        i = index % len(order)
        assert context.round_number == index // len(order) + 1
        assert context.order == order
        assert context.position == i + 1
        assert context.spoken == order[:i]
        assert context.remaining == order[i + 1:]
        assert context.names == tuple(enumerate(game.names))
        assert set(vars(context)) == {"day", "speaker", "order", "position", "spoken", "remaining", "names", "round_number", "stage", "phase", "future_responders", "settled"}


@pytest.mark.parametrize("scenario,expected", [
    ("peaceful", (0, 1, 2, 3, 4, 5)),
    ("prior_exile", (5, 0, 1, 2, 3)),
    ("multiple_deaths", (5, 0, 2, 3)),
])
def test_speech_context_uses_actual_night_and_exile_scheduling(scenario, expected):
    captured = []
    game = make_game()
    roles = [Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.WITCH, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(game.players, roles):
        player.role = role
    if scenario == "prior_exile":
        game.players[4].alive = False
        game.last_dead_seat = 4
        game.events.append(ev.exile(1, 4))
        game.day = 1
    game.hooks = default_hooks(
        wolf_kill=lambda w, k, v, d: k[-1] if scenario == "prior_exile" else 4,
        witch_act=lambda *args: (False, 1) if scenario == "multiple_deaths" else (True, None),
        speech=lambda s, v, d, c: captured.append(c) or "保留判断。",
        vote=lambda *args: None,
    )
    run_night(game)
    game._play_day()
    assert tuple(c.speaker for c in captured) == expected * 2
    assert all(c.order == expected for c in captured)


def test_speech_progress_resets_each_day_and_keeps_all_history():
    captured = []
    game = make_game(
        speech=lambda s, v, d, c: captured.append((c, v)) or f"第 {d} 天观点",
        vote=lambda *args: None,
    )
    game.day = 1
    game._play_day()
    day_one = [e for e in game.events if e.kind is EventKind.SPEECH]
    game.day = 2
    game.events.append(ev.night_falls(2))
    game.events.append(ev.peaceful_night(2))
    game._play_day()
    context, view = captured[12]
    assert context.day == 2 and context.spoken == () and context.position == 1
    assert [e for e in view if e.kind is EventKind.SPEECH] == day_one
    assert [e.speech_index for e in game.events if e.kind is EventKind.SPEECH] == list(range(1, 25))


def test_speech_order_starts_after_last_dead():
    game = make_game()
    wolves = wolves_of(game)
    target = next(s for s in range(6) if s not in wolves)
    game.hooks = default_hooks(wolf_kill=lambda w, k, v, d: target)

    game.day += 1
    game._play_night()  # 死者 = target
    alive_after_night = game.alive_seats()  # 白天放逐前的存活快照
    game._play_day()

    speeches = [e.seat for e in game.events if e.kind is EventKind.SPEECH]
    expected_order = []
    for i in range(6):
        s = (target + 1 + i) % 6
        if s in alive_after_night:
            expected_order.append(s)
    assert speeches == expected_order * 2
    # 发言编号全局递增
    indexes = [
        e.speech_index for e in game.events if e.kind is EventKind.SPEECH
    ]
    assert indexes == list(range(1, len(indexes) + 1))


def test_votes_are_published_only_after_everyone_finishes():
    cast = []

    def vote(seat, alive, view, day):
        assert not any(e.kind is EventKind.VOTE_RESULT and e.day == day for e in view)
        assert len([e for e in view if e.kind is EventKind.SPEECH and e.day == day]) == 12
        target = None if seat == 0 else 1
        cast.append((seat, target))
        return target

    game = make_game(vote=vote)
    game.day = 1
    game._play_day()
    result = next(e for e in game.events if e.kind is EventKind.VOTE_RESULT)
    assert result.extra["votes"] == cast and result.extra["tally"] == {1: 5}
    assert "1号弃票" in result.text and "6号→2号" in result.text


def test_vote_majority_exile():
    game = make_game()
    wolves = wolves_of(game)
    victim = next(s for s in range(6) if s not in wolves)

    # 所有人投 victim
    game.hooks = default_hooks(vote=lambda seat, alive, v, d: victim)
    game._play_day()

    exiles = [e for e in game.events if e.kind is EventKind.EXILE]
    assert len(exiles) == 1
    assert exiles[0].seat == victim
    assert not game.players[victim].alive
    assert game.last_dead_seat == victim


def test_vote_tie_no_exile():
    game = make_game()
    # 0、1 投 2；2、3 投 3 之外的人 → 制造平票：0→2, 1→2, 3→4, 4→4
    votes = {0: 2, 1: 2, 2: 4, 3: 4, 4: 5, 5: 0}
    game.hooks = default_hooks(vote=lambda seat, alive, v, d: votes[seat])
    game._play_day()

    exiles = [e for e in game.events if e.kind is EventKind.EXILE]
    assert len(exiles) == 1 and exiles[0].seat is None
    assert all(p.alive for p in game.players)
    tally_event = next(e for e in game.events if e.kind is EventKind.VOTE_RESULT)
    assert tally_event.extra["tally"] == {2: 2, 4: 2, 5: 1, 0: 1}


def test_vote_all_abstain_no_exile():
    game = make_game()
    game.hooks = default_hooks(vote=lambda seat, alive, v, d: None)
    game._play_day()
    exiles = [e for e in game.events if e.kind is EventKind.EXILE]
    assert exiles[0].seat is None
    assert all(p.alive for p in game.players)


def test_vote_illegal_target_raises():
    game = make_game()
    dead_seat = None
    game.hooks = default_hooks(vote=lambda seat, alive, v, d: 99)
    with pytest.raises(ValueError):
        game._play_day()
    assert dead_seat is None


# ---------------------------------------------------------------- 胜负判定


def test_good_win_all_wolves_exiled():
    game = make_game()
    wolves = sorted(wolves_of(game))

    def vote_wolf(seat, alive, v, d):
        return next(s for s in alive if s in wolves and game.players[s].alive)

    game.hooks = default_hooks(vote=vote_wolf)
    winner = game.run()
    assert winner == Team.GOOD.value
    assert all(
        not game.players[s].alive for s in wolves
    )
    over = next(e for e in game.events if e.kind is EventKind.GAME_OVER)
    assert "好人" in over.text


def test_wolf_win_night_kills():
    # 狼刀全砍好人 + 好人互相投好人 → 狼数 ≥ 好数即狼胜
    game = make_game(seed=7)
    goods = sorted(goods_of(game))

    game.hooks = default_hooks(
        wolf_kill=lambda w, k, v, d: k[0],
        vote=lambda seat, alive, v, d: next(
            s for s in alive if s in goods and s != seat
        ) if any(s in goods and s != seat for s in alive) else None,
    )
    winner = game.run()
    assert winner == Team.WOLF.value
    over = next(e for e in game.events if e.kind is EventKind.GAME_OVER)
    assert "狼人" in over.text


def test_wolf_win_after_exile_good():
    # 1 狼 vs 2 好人时放逐好人 → 狼胜
    game = make_game(seed=7)
    wolves = sorted(wolves_of(game))
    goods = sorted(goods_of(game))

    # 手动推进：直到只剩 1 狼 2 好人，然后白天放逐好人
    game.hooks = default_hooks(
        wolf_kill=lambda w, k, v, d: k[0],
        vote=lambda seat, alive, v, d: next(
            s for s in alive if s in goods and game.players[s].alive
        ),
    )
    winner = game.run()
    assert winner == Team.WOLF.value


def test_identity_reveal_available_at_end():
    game = make_game()
    game.run()
    # 终局翻牌：引擎状态保留全部真实身份
    assert all(p.role is not None for p in game.players)


# ---------------------------------------------------------------- 视角隔离不变量


def test_view_isolation_invariant_full_game():
    """任何角色视角不含越权私有事件（全流程不变量）。"""
    game = make_game()
    game.run()

    wolf_kinds = {EventKind.WOLF_REVEAL, EventKind.WOLF_KILL}
    witch_kinds = {EventKind.WITCH_KNEW, EventKind.POTION_USED}

    for p in game.players:
        for e in game.view(p.seat):
            assert e.public or p.seat in e.viewers  # 结构不变量
            if e.public:
                continue
            if e.kind in wolf_kinds:
                assert p.role is Role.WEREWOLF, (p, e)
            if e.kind is EventKind.SEER_CHECK:
                assert p.role is Role.SEER, (p, e)
            if e.kind in witch_kinds:
                assert p.role is Role.WITCH, (p, e)
            if e.kind is EventKind.ROLE_ASSIGNED:
                assert e.viewers == {p.seat}, (p, e)


def test_human_wolf_sees_teammate():
    # 人类玩家（任意座位）为狼时，能看到队友座位（与 AI 同机制）
    game = make_game(seed=1)
    wolves = wolves_of(game)
    seat = wolves[0]
    reveals = [e for e in game.view(seat) if e.kind is EventKind.WOLF_REVEAL]
    assert len(reveals) == 1
    for w in wolves:
        assert f"{w + 1} 号" in reveals[0].text
