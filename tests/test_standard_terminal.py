"""实际终端输入路径、首夜面板及四板子的 Controller/离线模型完整流程。"""
from argparse import Namespace
from io import StringIO
import json
import re

import pytest
from rich.console import Console

from engine import events as ev
from engine.boards import BOARDS, NINE_STANDARD, TWELVE_STANDARD, TWELVE_GUARD
from engine.game import ActionWindow
from engine.roles import Role
from llm.agents import PrivateStrategy
from main import build_game, save_record
from tests.test_standard_boards import make, seat
from ui import terminal


@pytest.fixture
def output(monkeypatch):
    stream=StringIO()
    monkeypatch.setattr(terminal,'console',Console(file=stream,width=180,color_system=None))
    return stream


def feed(monkeypatch, ui, values):
    iterator=iter(values)
    def reply(question):
        terminal.console.print(question)
        return next(iterator)
    monkeypatch.setattr(ui,'_read_input',reply)


@pytest.mark.parametrize('action,empty_label', [('guard','空守'),('hunter_shot','弃枪'),('badge','撕徽'),('sheriff_vote','弃选举票')])
def test_human_new_target_windows_multidigit_invalid_retry_and_explicit_empty(monkeypatch,output,action,empty_label):
    game=make(TWELVE_GUARD)
    ui=terminal.TerminalUI(game,0)
    window=ActionWindow(action,0,1,ev.Phase.NIGHT,tuple(range(12)),(9,11),True)
    feed(monkeypatch,ui,['99','1','12'])
    assert ui.prompt_action(window)==11
    assert '非法输入' in output.getvalue() and empty_label in output.getvalue()
    feed(monkeypatch,ui,['0'])
    assert ui.prompt_action(window) is None


def test_human_boolean_direction_wolf_empty_delegation_and_witch_hidden_knife(monkeypatch,output):
    game=make(TWELVE_GUARD)
    ui=terminal.TerminalUI(game,0)
    feed(monkeypatch,ui,['x','y'])
    assert ui.prompt_action(ActionWindow('sheriff_run',0,1,ev.Phase.SHERIFF,tuple(range(12)))) is True
    feed(monkeypatch,ui,['n'])
    assert ui.prompt_action(ActionWindow('sheriff_withdraw',0,1,ev.Phase.SHERIFF,tuple(range(12)))) is False
    feed(monkeypatch,ui,['0','2'])
    assert ui.prompt_action(ActionWindow('sheriff_direction',0,1,ev.Phase.DAY,tuple(range(12))))=='counterclockwise'
    feed(monkeypatch,ui,['0'])
    assert ui.prompt_wolf_kill(list(range(12)),[1])=='empty'
    feed(monkeypatch,ui,['d'])
    assert ui.prompt_wolf_kill(list(range(12)),[1]) is None
    feed(monkeypatch,ui,['d','0'])
    assert ui.prompt_wolf_kill(list(range(12)),[])=='empty'
    feed(monkeypatch,ui,['s','p','12'])
    assert ui.prompt_witch_act(None,False,True,list(range(12)))==(False,11)
    assert '无可见刀口' in output.getvalue() and '自救限制' in output.getvalue()


def test_opening_and_sheriff_phase_panel_do_not_reveal_pending_deaths(output):
    game=make(TWELVE_STANDARD)
    game.pending_deaths={2:{'knife'}}
    game.stage='sheriff_speech'
    ui=terminal.TerminalUI(game,0)
    ui.show_opening()
    ui.show_alive_update()
    text=output.getvalue()
    assert '十二人预女猎白' in text and '12' in text and '屠边' in text
    assert '出局' not in text.split('座位表')[1].split('游戏即将开始')[0]
    assert '玩家3' in text


@pytest.mark.parametrize('board',BOARDS.values(),ids=BOARDS.keys())
def test_complete_controller_mock_game_actual_human_input_and_record(board,monkeypatch,output,tmp_path):
    game,controller,ui,client=build_game(Namespace(mock=True,reconfig=False,seed=42,board=board.id))
    speech_line=False
    def reply(question):
        nonlocal speech_line
        if question in {'> ','… '}:
            speech_line=not speech_line
            return '离线人类发言。' if speech_line else ''
        if '是否报名' in question: return 'y'
        if '是否退水' in question: return 'n'
        if '发言方向' in question: return '1'
        if '刀口座位号' in question:
            if board.empty_kill:return '0'
            return str(next(s+1 for s in game.alive_seats() if game.role_of(s) is not Role.WEREWOLF))
        if '用解药 /' in question:return 'n'
        match=re.search(r'（([\d、]+)）',question)
        if match:return match[1].split('、')[0]
        raise AssertionError(question)
    monkeypatch.setattr(ui,'_read_input',reply)
    for a in controller.agents.values():
        a.private_strategy=PrivateStrategy(assessment='私有哨兵',goal='合法演示',claim='私有说法',prediction='私有预判哨兵')
    game.run()
    ui.render_new()
    ui.show_game_over()
    assert game.day<=board.player_count
    assert client.usage.structured_retries==client.usage.fallback_actions==0
    indices=[e.speech_index for e in game.events if e.kind is ev.EventKind.SPEECH]
    assert indices==list(range(1,len(indices)+1))
    path=tmp_path/'public.json'
    save_record(path,game,client,42)
    text=path.read_text(encoding='utf-8')
    assert '私有预判哨兵' not in text and '私有哨兵' not in output.getvalue()
    assert json.loads(text)['board']['id']==board.id


@pytest.mark.parametrize('role',[Role.WEREWOLF,Role.SEER,Role.WITCH,Role.HUNTER,Role.IDIOT,Role.GUARD])
def test_human_role_controller_routes_skill_window_and_dead_last_words(role,monkeypatch,output):
    # 寻找可复现的人类角色；整个输入仍走真实终端受限选项。
    board=TWELVE_GUARD if role is Role.GUARD else TWELVE_STANDARD
    for seed_value in range(100):
        game,controller,ui,client=build_game(Namespace(mock=True,reconfig=False,seed=seed_value,board=board.id))
        if game.role_of(controller.human) is role:break
    else:pytest.fail('种子未覆盖角色')
    human=controller.human
    if role is Role.GUARD:
        feed(monkeypatch,ui,[str(human+1)])
        assert controller.on_action(ActionWindow('guard',human,1,ev.Phase.NIGHT,tuple(range(12)),(human,),True),game.view(human))==human
    elif role is Role.HUNTER:
        game.player(human).alive=False
        target=game.alive_seats()[-1]
        feed(monkeypatch,ui,[str(target+1)])
        assert controller.on_action(ActionWindow('hunter_shot',human,1,ev.Phase.HUNTER,tuple(game.alive_seats()),(target,),True,False),game.view(human))==target
    elif role is Role.SEER:
        target=next(s for s in game.alive_seats() if s!=human)
        feed(monkeypatch,ui,[str(target+1)])
        assert controller.on_seer_check(human,[target],game.view(human),1)==target
    elif role is Role.WITCH:
        feed(monkeypatch,ui,['s'])
        assert controller.on_witch_act(human,human,True,True,game.view(human),1)==(True,None)
    elif role is Role.WEREWOLF:
        feed(monkeypatch,ui,['0'])
        assert controller.on_wolf_kill(game.wolves_alive(),game.alive_seats(),game.view(human),1) is None
    else:
        game.player(human).revealed=True
        game.player(human).can_vote=False
        ui.show_seat_table()
        assert '无投票权' in output.getvalue()
    assert client.usage.calls==0
