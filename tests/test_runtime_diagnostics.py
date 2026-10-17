"""入口日志：增量留档、敏感内容、退出和磁盘故障。"""

from argparse import Namespace
from io import StringIO
import json
import logging
import os

import pytest
from rich.console import Console

import main
from engine import events as ev
from llm.agents import AIPlayer, Persona, SAFE_SPEECH, VotePack, call_structured
from llm.client import LLMClient
from tests.test_client import httpx, make_client, response
from ui import terminal


@pytest.fixture
def log_home(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path / "user")
    return main.CONFIG_DIR / "logs"


def test_incremental_unique_logs_and_logger_restoration(log_home, capsys):
    loggers = [main.logger, logging.getLogger(), logging.getLogger("openai")]
    states = [(log.handlers, log.level, log.propagate, log.disabled) for log in loggers]
    paths = []
    for _ in range(2):
        with main._game_logging() as run_log:
            paths.append(run_log.path)
            main.logger.info("即时中文记录")
            assert "即时中文记录" in run_log.path.read_text(encoding="utf-8")
            logging.warning("根日志秘密")
            logging.getLogger("openai").warning("SDK 秘密")
            child = logging.getLogger("httpx.request")
            child.warning("HTTP 秘密")
        assert run_log.stream is None
        assert states == [(log.handlers, log.level, log.propagate, log.disabled) for log in loggers]
    assert paths[0] != paths[1]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.count("即时中文记录") == 1 and "秘密" not in text
        if os.name == "posix":
            assert path.stat().st_mode & 0o777 == 0o600
            assert log_home.stat().st_mode & 0o777 == 0o700
    captured = capsys.readouterr()
    assert "秘密" not in captured.out + captured.err
    assert captured.out.count("诊断日志已保存") == 2


def test_retries_metrics_and_fallback_do_not_log_model_content(log_home, make_client, capsys):
    secret = "私有策略与失败候选哨兵"
    bad = json.dumps({"vote": -1, secret: secret})
    client, requests = make_client(
        response(secret, finish="length"), response(bad), response('{"vote": 2}'),
        httpx.Response(401, json={"error": {"message": f"test-key {secret}"}}),
    )
    with main._game_logging() as run_log:
        pack, err = call_structured(client, [], VotePack)
        assert pack.vote == 2 and err is None and len(requests) == 3
        assert call_structured(client, [], VotePack)[0] is None
        def unavailable(_):
            raise RuntimeError(secret)
        local = LLMClient(client.config, mock=True, mock_fn=unavailable)
        player = AIPlayer(local, seat=0, name="学姐", role=main.Role.VILLAGER,
                          persona=Persona("学姐", "冷静", "保留意见"))
        assert player.day_speech([], 1, list(range(6)))[0] == SAFE_SPEECH
    text = run_log.path.read_text(encoding="utf-8")
    assert all(value not in text for value in (secret, "test-key", "private reasoning", '"vote": -1'))
    assert "finish_reason=length" in text and "扩大输出预算重试" in text
    assert "校验" in text and "accepted" in text and "http_401" in text
    assert "降级" in text and "RuntimeError" in text
    assert "estimated_input_tokens" in text and "elapsed_seconds" in text
    captured = capsys.readouterr()
    assert "LLM 请求" not in captured.out + captured.err and secret not in captured.out + captured.err


@pytest.fixture
def entry(monkeypatch, log_home):
    args = Namespace(mock=True, reconfig=False, seed=42, record=None)
    game, controller, ui, client = main.build_game(args)
    game.winner = "好人阵营"
    game.events.append(ev.game_over(1, game.winner, "离线验证"))
    closed = []
    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main, "build_game", lambda _: (game, controller, ui, client))
    monkeypatch.setattr(game, "run", lambda: None)
    monkeypatch.setattr(client, "close", lambda: closed.append(True))
    monkeypatch.setattr(terminal, "console", Console(file=StringIO(), width=180, color_system=None))
    return args, game, client, closed


def test_entry_preserves_stats_and_public_export(entry, log_home, tmp_path, capsys):
    args, game, client, closed = entry
    args.record = tmp_path / "public.json"
    client.config.api_key = "密钥哨兵"
    client.usage.calls, client.usage.prompt_tokens, client.usage.completion_tokens = 3, 10, 20
    client.usage.reasoning_tokens = 4
    client.usage.structured_retries, client.usage.fallback_actions = 2, 1
    main.main()
    output = capsys.readouterr().out
    assert "LLM 调用 3 次" in output and "tokens 30（入 10 / 出 20，其中思考 4）" in output
    assert "结构化重试 2 次，降级 1 次" in output and "公开对局记录已保存" in output
    assert json.loads(args.record.read_text())["version"] == 2
    text = next(log_home.glob("*.log")).read_text(encoding="utf-8")
    assert "seed=42 mode=mock" in text and "board=six-classic model=mock" in text
    assert "status=completed" in text and "密钥哨兵" not in text and "LLM 请求" not in text
    assert closed == [True]


@pytest.mark.parametrize("failure,code,status", [
    (KeyboardInterrupt, 130, "interrupted"), (ValueError, 1, "error"),
])
def test_entry_exit_keeps_logs_without_exception_body(entry, log_home, monkeypatch, capsys, failure, code, status):
    _, game, _, closed = entry
    def stop():
        main.logger.info("退出前记录")
        raise failure("异常正文秘密")
    monkeypatch.setattr(game, "run", stop)
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert caught.value.code == code and closed == [True]
    text = next(log_home.glob("*.log")).read_text(encoding="utf-8")
    assert "退出前记录" in text and f"status={status}" in text
    if status == "error":
        assert "type=ValueError" in text and "in stop" in text
    output = capsys.readouterr()
    assert "诊断日志已保存" in output.out
    assert "异常正文秘密" not in text + output.out + output.err
    assert "Traceback" not in output.err


def test_connection_close_failure_is_reported_safely(entry, log_home, monkeypatch, capsys):
    _, _, client, _ = entry
    def fail():
        raise RuntimeError("连接错误正文秘密")
    monkeypatch.setattr(client, "close", fail)
    with pytest.raises(SystemExit, match="1"):
        main.main()
    text = next(log_home.glob("*.log")).read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "type=RuntimeError" in text and "游戏无法继续" in captured.err
    assert "连接错误正文秘密" not in text + captured.err


def test_log_creation_failure_allows_game_to_finish(entry, monkeypatch, capsys):
    _, game, _, _ = entry
    def fail(**_):
        raise PermissionError("权限错误正文")
    notices = []
    def continue_game():
        notices.append(capsys.readouterr().err)
    monkeypatch.setattr(game, "run", continue_game)
    monkeypatch.setattr(main.tempfile, "NamedTemporaryFile", fail)
    main.main()
    captured = capsys.readouterr()
    assert "".join(notices).count("本局诊断日志无法保存") == 1
    assert "本局诊断日志无法保存" not in captured.err
    assert "LLM 调用" in captured.out and "诊断日志已保存" not in captured.out
    assert "权限错误正文" not in captured.err


@pytest.mark.parametrize("operation", ["write", "flush", "close"])
def test_log_io_failure_warns_once_and_preserves_partial_file(log_home, monkeypatch, capsys, operation):
    with main._game_logging() as run_log:
        main.logger.info("故障前已写入")
        actual_close = run_log.stream.close
        def fail(*_):
            if operation == "close":
                actual_close()
            raise OSError("磁盘错误正文")
        monkeypatch.setattr(run_log.stream, operation, fail)
        main.logger.warning("此条可能未保存")
        main.logger.warning("不应反复提示")
        print("游戏继续")
    captured = capsys.readouterr()
    assert captured.err.count("本局诊断日志无法保存") == 1
    assert "游戏继续" in captured.out and "诊断日志不完整" in captured.out
    assert "故障前已写入" in run_log.path.read_text(encoding="utf-8")
    assert "磁盘错误正文" not in captured.err and "Traceback" not in captured.err
