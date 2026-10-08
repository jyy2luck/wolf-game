"""engine/events 单元测试：视角过滤纯函数。"""

from engine import events as ev
from engine.events import EventKind, Phase, visible_events


def _sample_events() -> list[ev.Event]:
    return [
        ev.role_assigned(1, 0, "狼人"),
        ev.role_assigned(1, 1, "狼人"),
        ev.role_assigned(1, 2, "平民"),
        ev.role_assigned(1, 3, "预言家"),
        ev.role_assigned(1, 4, "女巫"),
        ev.role_assigned(1, 5, "平民"),
        ev.wolf_reveal(1, [0, 1], "1 号 A、2 号 B"),
        ev.night_falls(1),
        ev.wolf_kill(1, [0, 1], 4),
        ev.seer_check(1, 3, 2, False),
        ev.witch_knew(1, 5, 4),
        ev.dawn(1),
        ev.death(1, [4]),
        ev.speech(1, 2, "C", 1, "我觉得……"),
        ev.vote_result(1, {0: 2, 2: 1}),
        ev.exile(1, 0),
    ]


def test_public_events_visible_to_all():
    stream = _sample_events()
    public = [e for e in stream if e.public]
    for seat in range(6):
        view = visible_events(stream, seat)
        assert all(e in view for e in public)


def test_private_events_isolated():
    stream = _sample_events()
    # 平民（座位 4）看不到狼刀、查验、女巫信息
    view4 = visible_events(stream, 4)
    kinds4 = {e.kind for e in view4 if not e.public}
    assert kinds4 == {EventKind.ROLE_ASSIGNED}
    assert all(e.viewers == {4} for e in view4 if not e.public)

    # 预言家（座位 3）能看到自己的查验，看不到狼刀
    view3 = visible_events(stream, 3)
    kinds3 = {e.kind for e in view3 if not e.public}
    assert EventKind.SEER_CHECK in kinds3
    assert EventKind.WOLF_KILL not in kinds3
    assert EventKind.WITCH_KNEW not in kinds3

    # 狼人（座位 0）能看到队友揭示与刀口，看不到查验
    view0 = visible_events(stream, 0)
    kinds0 = {e.kind for e in view0 if not e.public}
    assert EventKind.WOLF_REVEAL in kinds0
    assert EventKind.WOLF_KILL in kinds0
    assert EventKind.SEER_CHECK not in kinds0


def test_visible_events_is_pure():
    stream = _sample_events()
    snapshot = [e for e in stream]
    visible_events(stream, 0)
    visible_events(stream, 3)
    assert stream == snapshot  # 不修改原事件流


def test_speech_event_has_numbering():
    e = ev.speech(2, 3, "阿明", 7, "我怀疑 5 号")
    assert e.speech_index == 7
    assert "[#7]" in e.text
    assert "阿明" in e.text and "4 号" in e.text
    assert e.phase is Phase.DAY


def test_death_requires_seats():
    try:
        ev.death(1, [])
        raised = False
    except ValueError:
        raised = True
    assert raised