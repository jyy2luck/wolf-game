"""入口：加载/引导配置 → 建局 → 主循环 → 终局。

用法：
    python main.py              # 正常启动（无配置则进入引导）
    python main.py --reconfig   # 重新配置 LLM 连接
    python main.py --mock       # 离线演示模式（不调用真实 LLM）
    python main.py --seed 42    # 固定随机种子（复现对局）
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import sys
import tempfile
import traceback

from config import CONFIG_DIR, LLMConfig, ensure_config
from engine.game import ActionWindow, DecisionHooks, Game, SpeechContext
from engine.boards import BOARDS, get_board
from engine import events as ev
from engine.roles import Role
from llm.agents import AIPlayer, assign_personas
from llm.client import LLMClient
from ui.terminal import TerminalUI

logger = logging.getLogger("llm_murder")


class _RunLog(logging.StreamHandler):
    """本次运行的文件日志；磁盘故障只提示一次，不打断游戏。"""

    def __init__(self) -> None:
        super().__init__()
        self.stream = None
        self.path = None
        self.failed = False
        self._warned = False
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        try:
            CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory = CONFIG_DIR / "logs"
            directory.mkdir(mode=0o700, exist_ok=True)
            self.stream = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), suffix=".log", delete=False,
            )
            self.path = Path(self.stream.name)
        except OSError:
            self.failed = True

    def _fail(self) -> None:
        self.failed = True
        if not self._warned:
            self._warned = True
            print("本局诊断日志无法保存，游戏仍可继续。", file=sys.stderr)

    def emit(self, record) -> None:
        if not self.failed:
            super().emit(record)

    def handleError(self, record) -> None:
        self._fail()

    def close(self) -> None:
        try:
            if self.stream is not None:
                self.stream.close()
        except OSError:
            self._fail()
        finally:
            self.stream = None
            super().close()


@contextmanager
def _game_logging():
    run_log = _RunLog()
    loggers = [logger, logging.getLogger(), *(logging.getLogger(name) for name in
               ("openai", "httpx", "httpx2", "httpcore", "httpcore2"))]
    previous = [(log, log.handlers, log.level, log.propagate, log.disabled) for log in loggers]
    try:
        for log in loggers:
            log.handlers = [run_log] if log is logger else [logging.NullHandler()]
            log.setLevel(logging.INFO if log is logger else logging.CRITICAL + 1)
            log.propagate = False
            log.disabled = False
        yield run_log
    finally:
        run_log.close()
        for log, handlers, level, propagate, disabled in previous:
            log.handlers = handlers
            log.setLevel(level)
            log.propagate, log.disabled = propagate, disabled
        if run_log.failed:
            run_log._fail()
        if run_log.path is not None:
            label = "诊断日志不完整" if run_log.failed else "诊断日志已保存"
            print(f"{label}：{run_log.path}")


def _unwired(*_args, **_kwargs):  # noqa: ANN002, ANN003
    raise RuntimeError("决策钩子未接线（Controller 未初始化）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="🐺 LLM 狼人杀（单机标准板子）")
    parser.add_argument("--board", choices=tuple(BOARDS), default="six-classic", help="选择具名板子")
    parser.add_argument("--reconfig", action="store_true", help="重新配置 LLM 连接")
    parser.add_argument("--mock", action="store_true", help="离线演示模式，不调用真实 LLM")
    parser.add_argument("--seed", type=int, default=None, help="固定随机种子")
    parser.add_argument("--record", type=Path, help="终局保存公开对局记录到新的 JSON 文件")
    return parser.parse_args()


class Controller:
    """决策路由：引擎 hooks → 人类玩家走终端 UI，其余路由到 AI 玩家。"""

    def __init__(
        self,
        game: Game,
        ui: TerminalUI,
        agents: dict[int, AIPlayer],
        human_seat: int,
    ) -> None:
        self.game = game
        self.ui = ui
        self.agents = agents
        self.human = human_seat
        self.hooks = DecisionHooks(
            wolf_kill=self.on_wolf_kill,
            seer_check=self.on_seer_check,
            witch_act=self.on_witch_act,
            speech=self.on_speech,
            vote=self.on_vote,
            action=self.on_action,
        )

    # ------------------------------------------------------------ 夜晚技能

    def on_wolf_kill(
        self, wolf_seats: list[int], killable: list[int], view, day: int
    ) -> int:
        self.ui.render_new()  # 夜幕与此前事件
        if self.human in wolf_seats:
            mates = [s for s in wolf_seats if s != self.human]
            choice = self.ui.prompt_wolf_kill(killable, mates)
            if choice is not None:
                return None if choice == "empty" else choice
            # 委托 AI 队友（此时队友必然存活）
        ai_wolf = next(s for s in wolf_seats if s != self.human)
        agent = self.agents[ai_wolf]
        target = agent.wolf_den_decision(
            self.game.view(ai_wolf), day, killable
        )
        if agent.team_plan is not None:
            self.game.events.append(ev.wolf_plan(day, wolf_seats, agent.team_plan))
        return target

    def on_action(self, window: ActionWindow, view):
        self.ui.render_new()
        if window.actor == self.human:
            return self.ui.prompt_action(window)
        return self.agents[window.actor].window_decision(window, view)

    def on_seer_check(
        self, seer_seat: int, checkable: list[int], view, day: int
    ) -> int:
        self.ui.render_new()
        if seer_seat == self.human:
            return self.ui.prompt_seer_check(checkable)
        return self.agents[seer_seat].seer_decision(view, day, checkable)

    def on_witch_act(
        self, witch_seat: int, kill_target: int | None, can_save: bool,
        can_poison: bool, view, day: int,
    ) -> tuple[bool, int | None]:
        self.ui.render_new()
        if witch_seat == self.human:
            return self.ui.prompt_witch_act(
                kill_target, can_save, can_poison, self.game.alive_seats()
            )
        return self.agents[witch_seat].witch_decision(
            view, day, kill_target, can_save, can_poison, self.game.alive_seats()
        )

    # ------------------------------------------------------------ 白天

    def on_speech(self, seat: int, view, day: int, context: SpeechContext) -> str:
        self.ui.render_new()
        if seat == self.human:
            self.ui.show_speech_context(context)
            if context.round_number == 2:
                self.ui.show_reply_prompt()
            return self.ui.prompt_speech()
        agent = self.agents[seat]
        with self.ui.wait_ai_speech(agent.name, seat):
            speech, _suspect, _mood = agent.day_speech(
                view, day, self.game.alive_seats(), context=context,
            )
        return speech

    def on_vote(self, seat: int, candidates: list[int], view, day: int) -> int | None:
        self.ui.render_new()
        if seat == self.human:
            return self.ui.prompt_vote(candidates)
        window = ActionWindow(self.game.stage, seat, day, self.game.phase,
            tuple(self.game.alive_seats()), tuple(candidates), True,
            vote_weight=3 if seat == self.game.sheriff else 2)
        return self.agents[seat].day_vote(view, day, candidates, window=window)


def build_game(args: argparse.Namespace) -> tuple[Game, Controller, TerminalUI, LLMClient]:
    board = get_board(getattr(args, "board", "six-classic"))
    if args.mock:
        config = LLMConfig(base_url="mock://local", api_key="mock", model="mock")
    else:
        config = ensure_config(reconfig=args.reconfig)
    client = LLMClient(config, mock=args.mock)
    rng = random.Random(args.seed)

    # 座位与名字：人类随机座位，AI 用人设名
    human_seat = rng.randrange(board.player_count)
    ai_seats = [s for s in range(board.player_count) if s != human_seat]
    personas = assign_personas(rng, board.player_count - 1)
    names = [""] * board.player_count
    names[human_seat] = "你"
    for seat, persona in zip(ai_seats, personas):
        names[seat] = persona.name

    game = Game(names=names, hooks=DecisionHooks(*([_unwired] * 5), action=_unwired), rng=rng, board=board)
    ui = TerminalUI(game, human_seat)

    wolves = [p.seat for p in game.players if p.role is Role.WEREWOLF]
    agents: dict[int, AIPlayer] = {}
    for seat, persona in zip(ai_seats, personas):
        role = game.role_of(seat)
        mates = [w for w in wolves if w != seat] if role is Role.WEREWOLF else None
        agents[seat] = AIPlayer(
            client, seat=seat, name=names[seat], role=role,
            persona=persona, teammate_seats=mates,
            board=board,
        )

    controller = Controller(game, ui, agents, human_seat)
    game.hooks = controller.hooks
    return game, controller, ui, client


def save_record(path: Path, game: Game, client: LLMClient, seed: int | None) -> None:
    payload = {
        "version": 2, "seed": seed, "model": client.config.model,
        "board": game.board.public_projection(),
        "names": game.names, "winner": game.winner,
        "revealed_roles": [p.role.value for p in game.players] if game.winner is not None else None,
        "usage": asdict(client.usage),
        "requests": client.requests,
        "events": [{"kind": e.kind.value, "day": e.day, "phase": e.phase.value,
                    "seat": e.seat, "speech_index": e.speech_index, "text": e.text, "extra": e.extra}
                   for e in game.events if e.public],
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(content)


def main() -> None:
    args = parse_args()
    with _game_logging() as run_log:
        logger.info("启动 seed=%s mode=%s", args.seed, "mock" if getattr(args, "mock", False) else "online")
        try:
            with ExitStack() as resources:
                game, _controller, ui, client = build_game(args)
                resources.callback(client.close)
                logger.info("建局 board=%s model=%s", game.board.id, client.config.model)
                ui.show_opening()
                if run_log.failed:
                    run_log._fail()  # 开局清屏之后提示，避免日志故障信息被擦掉。
                game.run()
                ui.render_new()   # 终局事件
                ui.show_game_over()
                usage = client.usage
                print(
                    f"\n📊 本次对局 LLM 调用 {usage.calls} 次，"
                    f"tokens {usage.total_tokens}（入 {usage.prompt_tokens} / 出 {usage.completion_tokens}，"
                    f"其中思考 {usage.reasoning_tokens}）；"
                    f"结构化重试 {usage.structured_retries} 次，降级 {usage.fallback_actions} 次"
                )
                if args.record is not None:
                    try:
                        save_record(args.record, game, client, args.seed)
                        print(f"公开对局记录已保存：{args.record}")
                    except OSError as exc:
                        print(f"对局记录保存失败：{exc}", file=sys.stderr)
            logger.info("退出 status=completed")
        except KeyboardInterrupt:
            logger.info("退出 status=interrupted")
            print("\n已退出游戏。")
            raise SystemExit(130) from None
        except Exception as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            locations = "\n".join(f"{f.filename}:{f.lineno} in {f.name}" for f in frames)
            logger.error("退出 status=error type=%s\n%s", type(exc).__name__, locations)
            print(f"游戏无法继续（{type(exc).__name__}）。", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
