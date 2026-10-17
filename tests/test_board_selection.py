"""启动菜单：玩法发现、输入恢复与入口接线。"""

from argparse import Namespace
from collections import Counter
from io import StringIO
import json
import sys

import pytest
from rich.console import Console

import main
from engine import events as ev
from engine.boards import BOARDS
from ui import terminal


@pytest.fixture
def output(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(terminal, "console", Console(file=stream, width=80, color_system=None))
    return stream


@pytest.mark.parametrize("choice,expected", [
    ("", "six-classic"), ("1", "six-classic"), ("2", "nine-standard"),
    ("3", "twelve-standard"), ("4", "twelve-guard"),
])
def test_menu_choices_and_public_setup(monkeypatch, output, choice, expected):
    monkeypatch.setattr("builtins.input", lambda: choice)
    assert terminal.TerminalUI.prompt_board() == expected
    text = output.getvalue()
    for board in BOARDS.values():
        assert board.name in text and str(board.player_count) in text
    assert "白痴×1" in text and "守卫×1" in text and "警长" in text
    assert "回车默认 1" in text


def test_menu_retries_invalid_and_damaged_input(monkeypatch, output):
    inputs = iter(["0", "5", "-1", "九人", "²", "9" * 5000, "\udcff", " 2 "])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    assert terminal.TerminalUI.prompt_board() == "nine-standard"
    assert output.getvalue().count("非法输入") == 6
    assert "输入含无法识别的字符" in output.getvalue()


@pytest.mark.parametrize("failure", [KeyboardInterrupt, EOFError])
def test_menu_propagates_exit_signals(monkeypatch, output, failure):
    def stop():
        raise failure
    monkeypatch.setattr("builtins.input", stop)
    with pytest.raises(failure):
        terminal.TerminalUI.prompt_board()


@pytest.mark.parametrize("direct", [False, True], ids=["menu", "flag"])
@pytest.mark.parametrize("number,board_id", list(enumerate(BOARDS, 1)), ids=BOARDS)
def test_startup_builds_selected_board_and_exports_record(monkeypatch, output, tmp_path, capsys,
                                                         number, board_id, direct):
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path / "user")
    record = tmp_path / "game.json"
    argv = ["main.py", "--mock", "--seed", "42", "--record", str(record)]
    if direct:
        argv += ["--board", board_id]
    monkeypatch.setattr(sys, "argv", argv)
    def reply():
        assert not direct, "显式 --board 不应读取菜单输入"
        return str(number)
    monkeypatch.setattr("builtins.input", reply)
    monkeypatch.setattr(main, "ensure_config", lambda **_: pytest.fail("离线模式不应加载配置"))
    original_build = main.build_game
    def build(args):
        assert args.board == board_id and args.seed == 42 and args.record == record
        game, controller, ui, client = original_build(args)
        board = BOARDS[board_id]
        assert len(game.players) == board.player_count
        assert Counter(p.role for p in game.players) == dict(board.role_counts)
        assert len(controller.agents) == board.player_count - 1
        assert all(agent.board is board for agent in controller.agents.values())
        def finish():
            game.winner = "好人阵营"
            game.events.append(ev.game_over(1, game.winner, "入口接线验证"))
        monkeypatch.setattr(game, "run", finish)
        return game, controller, ui, client
    monkeypatch.setattr(main, "build_game", build)
    main.main()
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["board"] == BOARDS[board_id].public_projection()
    assert data["seed"] == 42 and len(data["names"]) == BOARDS[board_id].player_count
    assert ("选择游戏" in output.getvalue()) is not direct
    assert "LLM 调用" in capsys.readouterr().out


def test_build_game_keeps_six_person_default_with_unselected_args():
    games = [main.build_game(Namespace(mock=True, reconfig=False, seed=42, **values))
             for values in ({}, {"board": None}, {"board": "six-classic"})]
    try:
        assert all(game.board.id == "six-classic" for game, _, _, _ in games)
        assert all(game.names == games[0][0].names and game.roles == games[0][0].roles
                   for game, _, _, _ in games)
    finally:
        for _, _, _, client in games:
            client.close()


@pytest.mark.parametrize("reconfig", [False, True])
def test_board_selection_precedes_configuration(monkeypatch, output, tmp_path, reconfig):
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path / "user")
    monkeypatch.setattr(sys, "argv", ["main.py"] + (["--reconfig"] if reconfig else []))
    inputs = iter(["invalid", "2"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    calls = []
    def configure(**kwargs):
        assert list(inputs) == [] and "非法输入" in output.getvalue()
        calls.append(kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(main, "ensure_config", configure)
    with pytest.raises(SystemExit, match="130"):
        main.main()
    assert calls == [{"reconfig": reconfig}]


@pytest.mark.parametrize("failure", [KeyboardInterrupt, EOFError])
def test_startup_cancel_preserves_files_and_never_builds(monkeypatch, output, tmp_path, capsys, failure):
    config_dir = tmp_path / "user"
    config_dir.mkdir()
    config = config_dir / "config.json"
    config.write_text("已有配置", encoding="utf-8")
    record = tmp_path / "existing.json"
    record.write_text("已有记录", encoding="utf-8")
    monkeypatch.setattr(main, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(sys, "argv", ["main.py", "--reconfig", "--record", str(record)])
    def stop():
        raise failure
    monkeypatch.setattr("builtins.input", stop)
    monkeypatch.setattr(main, "build_game", lambda _: pytest.fail("取消选择不应建局或加载配置"))
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert caught.value.code == 130 and "已退出游戏" in capsys.readouterr().out
    assert config.read_text(encoding="utf-8") == "已有配置"
    assert record.read_text(encoding="utf-8") == "已有记录"
    assert "status=interrupted" in next((config_dir / "logs").glob("*.log")).read_text()


def test_unknown_board_is_rejected_before_menu(monkeypatch, output, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--board", "unknown"])
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert caught.value.code == 2 and not output.getvalue()
    error = capsys.readouterr().err
    assert "unknown" in error and all(board_id in error for board_id in BOARDS)
