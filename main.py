"""入口：加载/引导配置 → 建局 → 主循环 → 终局。

用法：
    python main.py              # 正常启动（无配置则进入引导）
    python main.py --reconfig   # 重新配置 LLM 连接
    python main.py --mock       # 离线演示模式（不调用真实 LLM）
    python main.py --seed 42    # 固定随机种子（复现对局）
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

from config import LLMConfig, ensure_config
from engine.game import DecisionHooks, Game
from engine.roles import Role
from llm.agents import AIPlayer, assign_personas
from llm.client import LLMClient
from ui.terminal import TerminalUI

logger = logging.getLogger("llm_murder")


def _unwired(*_args, **_kwargs):  # noqa: ANN002, ANN003
    raise RuntimeError("决策钩子未接线（Controller 未初始化）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="🐺 LLM 狼人杀（6 人局单机）")
    parser.add_argument("--reconfig", action="store_true", help="重新配置 LLM 连接")
    parser.add_argument("--mock", action="store_true", help="离线演示模式，不调用真实 LLM")
    parser.add_argument("--seed", type=int, default=None, help="固定随机种子")
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
                return choice
            # 委托 AI 队友（此时队友必然存活）
        ai_wolf = next(s for s in wolf_seats if s != self.human)
        return self.agents[ai_wolf].wolf_den_decision(
            self.game.view(ai_wolf), day, killable
        )

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

    def on_speech(self, seat: int, view, day: int) -> str:
        self.ui.render_new()
        self.ui.show_alive_update()
        if seat == self.human:
            return self.ui.prompt_speech()
        agent = self.agents[seat]
        with self.ui.stream_ai_speech(agent.name, seat) as handle:
            speech, _suspect, _mood = agent.day_speech(
                view, day, self.game.alive_seats(), on_delta=handle
            )
        return speech

    def on_vote(self, seat: int, candidates: list[int], view, day: int) -> int | None:
        self.ui.render_new()
        if seat == self.human:
            return self.ui.prompt_vote(candidates)
        return self.agents[seat].day_vote(view, day, candidates)


def build_game(args: argparse.Namespace) -> tuple[Game, Controller, TerminalUI, LLMClient]:
    if args.mock:
        config = LLMConfig(base_url="mock://local", api_key="mock", model="mock")
    else:
        config = ensure_config(reconfig=args.reconfig)
    client = LLMClient(config, mock=args.mock)
    rng = random.Random(args.seed)

    # 座位与名字：人类随机座位，AI 用人设名
    human_seat = rng.randrange(6)
    ai_seats = [s for s in range(6) if s != human_seat]
    personas = assign_personas(rng, 5)
    names = ["你"] + [""] * 5
    for seat, persona in zip(ai_seats, personas):
        names[seat] = persona.name

    game = Game(names=names, hooks=DecisionHooks(*([_unwired] * 5)), rng=rng)
    ui = TerminalUI(game, human_seat)

    wolves = [p.seat for p in game.players if p.role is Role.WEREWOLF]
    agents: dict[int, AIPlayer] = {}
    for seat, persona in zip(ai_seats, personas):
        role = game.role_of(seat)
        mates = [w for w in wolves if w != seat] if role is Role.WEREWOLF else None
        agents[seat] = AIPlayer(
            client, seat=seat, name=names[seat], role=role,
            persona=persona, teammate_seats=mates,
        )

    controller = Controller(game, ui, agents, human_seat)
    game.hooks = controller.hooks
    return game, controller, ui, client


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    try:
        game, _controller, ui, client = build_game(args)
        ui.show_opening()
        game.run()
        ui.render_new()   # 终局事件
        ui.show_game_over()
        usage = client.usage
        print(
            f"\n📊 本次对局 LLM 调用 {usage.calls} 次，"
            f"tokens {usage.total_tokens}（入 {usage.prompt_tokens} / 出 {usage.completion_tokens}）"
        )
    except KeyboardInterrupt:
        print("\n已退出游戏。")
        sys.exit(130)


if __name__ == "__main__":
    main()