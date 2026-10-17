"""全 AI 对局：沿用正式控制器，仅将人类座位置为空；输出公开记录及安全指标。

uv run python scripts/benchmark_games.py --output /tmp/wolf-benchmark
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LLMConfig, load_config
from engine.boards import BOARDS, get_board
from engine.game import DecisionHooks, Game
from engine.roles import Role
from llm.agents import AIPlayer, Persona, assign_personas
from llm.client import LLMClient
from main import Controller, _unwired, save_record


class GameLimit(RuntimeError):
    pass


class HeadlessUI:
    def __init__(self, game, max_days):
        self.game, self.max_days = game, max_days

    def render_new(self):
        if self.game.day > self.max_days:
            raise GameLimit("max_days")

    def show_alive_update(self):
        pass

    def wait_ai_speech(self, *_):
        return nullcontext()


def run_game(board_id, seed, output: Path, *, mock=False, max_days=10):
    config = LLMConfig("mock://local", "mock", "mock") if mock else load_config()
    if config is None:
        raise ValueError("缺少本机 API 配置")
    client = LLMClient(config, mock=mock)
    board, rng = get_board(board_id), random.Random(seed)
    personas = assign_personas(rng, board.player_count-1) + [Persona("阿衡", "沉稳", "先核对依据")]
    game = Game([p.name for p in personas], DecisionHooks(*([_unwired]*5), action=_unwired), rng=rng, board=board)
    wolves = [p.seat for p in game.players if p.role is Role.WEREWOLF]
    agents = {s: AIPlayer(client, seat=s, name=p.name, role=game.role_of(s), persona=p, board=board,
                         teammate_seats=[w for w in wolves if w != s] if s in wolves else None)
              for s, p in enumerate(personas)}
    controller = Controller(game, HeadlessUI(game, max_days), agents, -1)
    game.hooks = controller.hooks
    started, status = time.monotonic(), "complete"
    try:
        game.run()
    except GameLimit:
        status = "max_days"
    finally:
        client.close()
    output.mkdir(parents=True, exist_ok=True)
    save_record(output/f"{board_id}-{seed}.json", game, client, seed)
    durations = sorted(r['elapsed_seconds'] for r in client.requests)
    def percentile(q):
        return durations[max(0, math.ceil(len(durations)*q)-1)] if durations else 0
    summary = dict(board=board_id, seed=seed, model=config.model, status=status, days=game.day,
                   winner=game.winner, elapsed_seconds=round(time.monotonic()-started, 2),
                   request_p50_seconds=percentile(.5), request_p95_seconds=percentile(.95),
                   request_max_seconds=max(durations, default=0), usage=asdict(client.usage))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boards', nargs='+', choices=tuple(BOARDS), default=['six-classic','nine-standard','twelve-standard'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpx2').setLevel(logging.WARNING)
    summaries = []
    for board in args.boards:
        for seed in args.seeds:
            summaries.append(run_game(board, seed, args.output, mock=args.mock))
            (args.output/'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
