"""长局上下文预算、角色隔离、原文回查和完整事实校验。"""
from copy import deepcopy
import json

import pytest

from config import LLMConfig
from engine import events as ev
from engine.boards import BOARDS
from engine.game import ActionWindow
from engine.roles import Role
from llm.agents import AIPlayer, Persona, PrivateStrategy, SAFE_SPEECH
from llm.client import LLMClient
from llm.context import InputBudgetError, estimate_tokens, fit_context, recall_message, render_memory


def history(count=120):
    return [ev.role_assigned(0, 0, '预言家'), ev.seer_check(1, 0, 1, True),
            *[ev.speech((i-1)//12+1, i%6, f'玩家{i%6}', i,
                         f'第{i}次说法：我可能是预言家，但只是玩家声明。'+'还要核对依据。'*25) for i in range(1,count+1)],
            ev.vote_result(10, {1:3}, [(0,1)]), ev.death(10, [5])]


def test_compaction_keeps_sources_private_facts_and_originals():
    view = history()
    original = deepcopy(view)
    messages = [{'role':'system', 'content':'你是预言家。\n'+render_memory(view)+'\n私有策略：隐藏身份。'},
                {'role':'user', 'content':'按当前窗口投票 JSON。'}]
    fitted = fit_context(messages, view, 4500)
    text = fitted[0]['content']
    assert estimate_tokens(fitted) <= 4500
    assert '不完整节选' in text and '[#120]' in text and 'recall' in text
    assert '查验 2 号：[狼人]' in text and '角色私有事实' in text
    assert '私有策略：隐藏身份。' in text and view[-1].text in text
    assert '玩家声明' in text and '省略不表示从未发生' in text
    assert view[-3].extra['content'] in text
    assert fitted[1] == messages[1] and view == original
    assert view[2].extra['content'] in recall_message([1], view)


def test_budget_cannot_silently_drop_fixed_identity_or_private_facts():
    with pytest.raises(InputBudgetError, match='固定规则'):
        fit_context([{'role':'system','content':'真实身份。'*1000}], [], 100)


@pytest.mark.parametrize('selector', [None, [], [True], [0], [1,1], [1,2,3,4], [999], ['1']])
def test_recall_rejects_invalid_or_invisible_selectors(selector):
    with pytest.raises(ValueError):
        recall_message(selector, history(3))


@pytest.mark.parametrize('board', ['six-classic','nine-standard','twelve-standard','twelve-guard'])
@pytest.mark.parametrize('action', ['speech','vote','wolf','seer','witch','window'])
def test_all_action_paths_fit_large_histories_without_hidden_facts(board, action):
    board = BOARDS[board]
    view = history()
    captured = []
    def reply(messages):
        captured.append(messages)
        return json.dumps({'speech':'我暂时保留判断。','vote':2,'target':2,'save':False,'poison':0})
    client = LLMClient(LLMConfig('mock://local','mock','mock'), mock=True, mock_fn=reply)
    player = AIPlayer(client, seat=0, name='测试者', role=Role.SEER,
                      persona=Persona('测试者','冷静','核对事实'), board=board)
    player.private_strategy = PrivateStrategy(assessment='真实查验与说辞分开。',goal='保护身份。',claim='保留意见。')
    hidden = ev.seer_check(1, 3, 4, False)
    visible = ev.visible_events(view+[hidden], 0)
    alive = list(range(board.player_count))
    if action == 'speech': player.day_speech(visible, 11, alive)
    elif action == 'vote': player.day_vote(visible, 11, alive)
    elif action == 'wolf': player.wolf_den_decision(visible, 11, alive[1:])
    elif action == 'seer': player.seer_decision(visible, 11, alive[1:])
    elif action == 'witch': player.witch_decision(visible,11,1,True,True,alive)
    else: player.window_decision(ActionWindow('guard',0,11,ev.Phase.NIGHT,tuple(alive),tuple(alive),True),visible)
    assert captured and client.usage.fallback_actions == 0
    for messages in captured:
        assert estimate_tokens(messages) <= client.config.max_input_tokens
        text='\n'.join(m['content'] for m in messages)
        assert hidden.text not in text and '真实查验与说辞分开。' in text
    assert len(view) == 124


@pytest.mark.parametrize('fabricated', [False, True])
def test_recall_old_original_then_validate_against_full_journal(fabricated):
    view = history(120)
    quote = '从未说过的伪造原话' if fabricated else '我可能是预言家'
    final = json.dumps({'speech':f'第1天2号说“{quote}”[#1]。', 'references':[
        {'speech_index':1,'day':1,'seat':2,'quote':quote}]}, ensure_ascii=False)
    replies = iter(['{"recall":[1]}', final, final])
    captured=[]
    def answer(messages):
        captured.append(messages)
        return next(replies)
    client=LLMClient(LLMConfig('mock://local','mock','mock'), mock=True,mock_fn=answer)
    player=AIPlayer(client,seat=0,name='测试',role=Role.SEER,persona=Persona('测试','冷静','核实'))
    prior=PrivateStrategy(assessment='已有事实',goal='原计划',claim='原说辞')
    player.private_strategy=prior
    result=player.day_speech(view,11,list(range(6)))
    assert client.usage.calls <= 3 and client.usage.recalls == 1
    assert view[2].extra['content'] in captured[1][-1]['content']
    if fabricated:
        assert result[0] == SAFE_SPEECH and player.private_strategy == prior
    else:
        assert result[0] != SAFE_SPEECH and client.usage.fallback_actions == 0


def test_recall_never_returns_private_speech_like_event():
    hidden=ev.speech(1,3,'隐藏来源',77,'不应泄露的私有文本')
    hidden.public=False
    hidden.viewers=frozenset({3})
    with pytest.raises(ValueError,match='当前可见公开发言'):
        recall_message([77],history(3)+[hidden])


def test_repeated_recall_cannot_extend_attempts_or_commit_strategy():
    view=history(3)
    client=LLMClient(LLMConfig('mock://local','mock','mock'),mock=True,
        mock_fn=lambda _:json.dumps({'recall':[1],'private_strategy':{
            'assessment':'未通过候选','goal':'不得提交','claim':'不得提交'}}))
    player=AIPlayer(client,seat=0,name='测试',role=Role.SEER,persona=Persona('测试','冷静','核实'))
    prior=PrivateStrategy(assessment='原状态',goal='原计划',claim='原说辞')
    player.private_strategy=prior
    assert player.day_speech(view,11,list(range(6)))[0]==SAFE_SPEECH
    assert client.usage.calls==3 and client.usage.recalls==1 and client.usage.fallback_actions==1
    assert player.private_strategy==prior
