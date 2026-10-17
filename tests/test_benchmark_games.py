"""真实控制器的全 AI 路由，不向 AI 暴露引擎隐藏状态。"""
import json

import pytest

from engine.boards import BOARDS
from scripts.benchmark_games import run_game


@pytest.mark.parametrize('board', BOARDS)
@pytest.mark.parametrize('seed', [42,43])
def test_headless_games_complete_and_save_safe_records(tmp_path, board, seed):
    result=run_game(board,seed,tmp_path,mock=True)
    assert result['status']=='complete' and result['usage']['calls']>0
    assert result['usage']['fallback_actions']==0
    record=json.loads((tmp_path/f'{board}-{seed}.json').read_text())
    assert record['winner'] and record['requests']==[]
    assert not {'seer_check','wolf_kill','wolf_plan','potion_used'} & {e['kind'] for e in record['events']}
    assert 'api_key' not in record


def test_headless_game_limit_reports_incomplete(tmp_path):
    result=run_game('six-classic',42,tmp_path,mock=True,max_days=0)
    assert result['status']=='max_days' and result['winner'] is None
    record=json.loads((tmp_path/'six-classic-42.json').read_text())
    assert record['revealed_roles'] is None
