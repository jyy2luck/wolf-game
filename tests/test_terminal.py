"""Controller → AI → 引擎正式事件 → 终端的确定性集成测试。"""

from copy import deepcopy
from argparse import Namespace
from io import BytesIO, StringIO
import json
import random
import re

import pytest
from rich.console import Console

from config import LLMConfig
from engine import events as ev
from engine.game import Game
from llm.agents import AIPlayer, Persona, SAFE_SPEECH
from llm.client import LLMClient
from main import Controller, build_game
from tests.test_game import default_hooks
from ui import terminal

BAD = "2 号一上来就急着说话，太可疑了。"
FINAL = "等 2 号发言后再判断。"
HUMAN = "我先保留判断。"
NAMES = ["人类", "阿飞", "老猫", "大树", "铁叔", "学姐"]


def pack(speech, suspect=0, **extra):
    return json.dumps({"speech": speech, "suspect": suspect, "mood": "平静", "references": [], **extra}, ensure_ascii=False)


class ScriptClient(LLMClient):
    def __init__(self, replies=()):
        super().__init__(LLMConfig("mock://local", "mock", "mock"), mock=True)
        self.replies = list(replies)
        self.requests = []
        self.public_callbacks = []

    def chat(self, messages, **kwargs):
        self.requests.append((deepcopy(messages), kwargs["stream"]))
        self.usage.calls += 1
        if not kwargs["stream"]:
            return '{"vote": 0}'
        reply = self.replies.pop(0) if self.replies else pack("我还想听听大家的观点。")
        if isinstance(reply, Exception):
            raise reply
        # 故意模拟流式 delta；白天生成必须没有公开回调。
        callback = kwargs.get("on_delta")
        if callback:
            self.public_callbacks.append(reply)
            callback(reply)
        return reply


@pytest.fixture
def captured_terminal(monkeypatch):
    output = StringIO()
    console = Console(file=output, width=180, force_terminal=True, color_system=None)
    monkeypatch.setattr(terminal, "console", console)
    stopped = []
    actual_live = terminal.Live

    class TrackedLive(actual_live):
        def stop(self):
            super().stop()
            stopped.append(self)

    monkeypatch.setattr(terminal, "Live", TrackedLive)
    return output, stopped


def text_output(output):
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.getvalue())


def setup_game(replies, monkeypatch):
    game = Game(names=NAMES, hooks=default_hooks(), rng=random.Random(42))
    game.day = 1
    game.players[4].alive = False
    game.last_dead_seat = 4
    game.events.append(ev.death(1, [4]))
    ui = terminal.TerminalUI(game, 0)
    monkeypatch.setattr(ui, "prompt_speech", lambda: HUMAN)
    monkeypatch.setattr(ui, "prompt_vote", lambda candidates: None)
    clients = {s: ScriptClient(replies if s == 5 else ()) for s in (1, 2, 3, 5)}
    agents = {
        s: AIPlayer(c, seat=s, name=NAMES[s], role=game.role_of(s),
                    persona=Persona(NAMES[s], "冷静", "保留意见"))
        for s, c in clients.items()
    }
    controller = Controller(game, ui, agents, 0)
    game.hooks = controller.hooks
    return game, ui, agents, clients


@pytest.mark.parametrize("replies,expected,attempts", [
    ([pack(BAD, 2), pack(FINAL)], FINAL, 2),
    ([pack(BAD, 2)] * 3, SAFE_SPEECH, 3),
    ([RuntimeError("服务不可用")], SAFE_SPEECH, 1),
])
def test_only_final_or_fallback_enters_history_and_terminal(
    replies, expected, attempts, monkeypatch, captured_terminal,
):
    output, stopped = captured_terminal
    game, ui, agents, clients = setup_game(replies, monkeypatch)
    agents[5].last_suspect = 3
    game._play_day()
    ui.render_new()
    ui.render_new()
    speeches = [e for e in game.events if e.kind is ev.EventKind.SPEECH]
    assert len(speeches) == 5
    assert [e.speech_index for e in speeches] == [1, 2, 3, 4, 5]
    assert speeches[0].extra["content"] == expected
    assert all(BAD not in e.text for e in game.events)
    assert agents[5].last_suspect is None
    for client in clients.values():
        assert client.public_callbacks == []
        assert any(not stream for _, stream in client.requests)  # 兜底也继续投票。
    assert sum(stream for _, stream in clients[5].requests) == attempts
    next_view = clients[1].requests[0][0][0]["content"]
    assert expected in next_view and BAD not in next_view
    assert "第 1 天 · 白天 · 玩家声明：[#1] 6 号 学姐" in next_view
    text = text_output(output)
    assert "生成与校验中" in text
    assert text.count(expected) == 1
    assert "[#1] 学姐（6 号）" in text
    assert BAD not in text and '"speech"' not in text and '"references"' not in text
    assert len(stopped) == 4


def test_user_scene_then_next_day_valid_human_history_reference(monkeypatch, captured_terminal):
    citation = "第 1 天 1 号在 [#2] 说过‘我先保留判断’，我想再听他的看法。"
    refs = [{"speech_index": 2, "seat": 1, "day": 1, "quote": "我先保留判断"}]
    game, ui, agents, clients = setup_game([pack(BAD, 2), pack(FINAL), pack(citation, references=refs)], monkeypatch)
    game._play_day()
    day_one = [e for e in game.events if e.kind is ev.EventKind.SPEECH]
    game.day = 2
    game.events.append(ev.night_falls(2))
    game.events.append(ev.peaceful_night(2))
    game._play_day()
    ui.render_new()
    speeches = [e for e in game.events if e.kind is ev.EventKind.SPEECH]
    assert len(speeches) == 10
    assert speeches[:5] == day_one
    assert speeches[5].speech_index == 6 and speeches[5].extra["content"] == citation
    assert speeches[5].day == 2
    first_requests = [messages for messages, stream in clients[5].requests if stream]
    assert len(first_requests) == 3
    assert "当日已完成正式发言：（无）" in first_requests[2][0]["content"]
    assert "第 1 天 · 白天 · 玩家声明：[#2] 1 号 人类" in first_requests[2][0]["content"]
    assert all(BAD not in messages[0]["content"] for client in clients.values() for messages, _ in client.requests)
    output, stopped = captured_terminal
    text = text_output(output)
    assert text.count(citation) == 1
    assert text.count(FINAL) == 1 and BAD not in text
    assert len(stopped) == 8


def test_wait_state_is_cleaned_on_unexpected_exception(captured_terminal):
    output, stopped = captured_terminal
    game = Game(names=NAMES, hooks=default_hooks(), rng=random.Random(42))
    ui = terminal.TerminalUI(game, 0)
    with pytest.raises(RuntimeError, match="测试异常"):
        with ui.wait_ai_speech("学姐", 5):
            raise RuntimeError("测试异常")
    assert len(stopped) == 1
    terminal.console.print("清理后可以继续输出")
    assert "清理后可以继续输出" in text_output(output)


def test_new_game_has_separate_events_and_ai_state():
    args = Namespace(mock=True, reconfig=False, seed=42)
    first, first_controller, _, _ = build_game(args)
    ai_seat = next(iter(first_controller.agents))
    first.events.append(ev.speech(1, ai_seat, first.names[ai_seat], 1, "只属于上一局的发言。"))
    first_controller.agents[ai_seat].my_votes.append((1, 2))
    first_controller.agents[ai_seat].last_suspect = 2
    second, second_controller, _, _ = build_game(args)
    assert first.events is not second.events
    assert not any(e.kind is ev.EventKind.SPEECH for e in second.events)
    assert second_controller.agents[ai_seat].my_votes == []
    assert second_controller.agents[ai_seat].last_suspect is None


def scripted_byte_input(monkeypatch, content, errors="strict"):
    """以实际损坏字节模拟终端逐行解码；覆盖严格和 surrogateescape 策略。"""
    source = BytesIO(content)

    def read_line():
        data = source.readline()
        if not data:
            raise EOFError
        return data.rstrip(b"\r\n").decode("utf-8", errors=errors)

    monkeypatch.setattr("builtins.input", read_line)


@pytest.mark.parametrize("errors", ["strict", "surrogateescape"])
def test_speech_retries_invalid_utf8_and_keeps_successful_lines(monkeypatch, captured_terminal, errors):
    game = Game(names=NAMES, hooks=default_hooks(), rng=random.Random(42))
    ui = terminal.TerminalUI(game, 0)
    first = "我才是预言家。"
    second = "6 号是狼，大家不要被带偏了。"
    # 狼字的 UTF-8 尾字节被删掉，随后中文标点使该序列无法解码。
    damaged = "6 号是".encode("utf-8") + "狼".encode("utf-8")[:-1] + "，大家不要被带偏了。\n".encode("utf-8")
    scripted_byte_input(monkeypatch, first.encode("utf-8") + b"\n" + damaged + second.encode("utf-8") + b"\n\n", errors)
    assert ui.prompt_speech() == first + "\n" + second
    output, _ = captured_terminal
    assert text_output(output).count("请重新输入当前行") == 1


@pytest.mark.parametrize("action,expected", [
    (lambda ui: ui.prompt_vote([1]), 1),
    (lambda ui: ui.prompt_seer_check([1]), 1),
    (lambda ui: ui.prompt_wolf_kill([1], []), 1),
])
def test_invalid_utf8_in_seat_inputs_does_not_interrupt_choice_validation(
    action, expected, monkeypatch, captured_terminal,
):
    game = Game(names=NAMES, hooks=default_hooks(), rng=random.Random(42))
    ui = terminal.TerminalUI(game, 0)
    scripted_byte_input(monkeypatch, b"\xff\n9\n2\n")
    assert action(ui) == expected
    text = text_output(captured_terminal[0])
    assert "请重新输入当前行" in text and "非法输入" in text


def test_witch_action_and_target_recover_from_invalid_utf8(monkeypatch, captured_terminal):
    game = Game(names=NAMES, hooks=default_hooks(), rng=random.Random(42))
    ui = terminal.TerminalUI(game, 0)
    scripted_byte_input(monkeypatch, b"\xff\np\n\xff\n2\n")
    assert ui.prompt_witch_act(1, True, True, [1]) == (False, 1)


def test_damaged_human_input_is_retried_before_entering_official_history(monkeypatch, captured_terminal):
    game, ui, _, clients = setup_game([pack(FINAL)], monkeypatch)
    monkeypatch.setattr(ui, "prompt_speech", terminal.TerminalUI.prompt_speech.__get__(ui))
    scripted_byte_input(monkeypatch, b"\xff\n" + HUMAN.encode("utf-8") + b"\n\n")
    game._play_day()
    speeches = [e for e in game.events if e.kind is ev.EventKind.SPEECH]
    assert len(speeches) == 5 and game.speech_counter == 5
    assert speeches[1].seat == 0 and speeches[1].extra["content"] == HUMAN
    assert "第 1 天 · 白天 · 玩家声明：[#2] 1 号 人类" in clients[1].requests[0][0][0]["content"]
    assert any(e.kind is ev.EventKind.VOTE_RESULT for e in game.events)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, EOFError])
def test_input_recovery_does_not_swallow_exit_signals(monkeypatch, captured_terminal, exception):
    def stop():
        raise exception

    monkeypatch.setattr("builtins.input", stop)
    with pytest.raises(exception):
        terminal.TerminalUI._read_input("> ")
