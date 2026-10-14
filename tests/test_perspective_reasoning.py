"""成对固定局面检验合法参考、推断完备性和成功提交边界，不评估胜率。"""
from dataclasses import replace
import json

import pytest

from engine import events as ev
from engine.boards import BOARDS, SIX_CLASSIC, NINE_STANDARD, TWELVE_GUARD
from engine.game import ActionWindow, SpeechContext
from engine.roles import Role
from llm import strategies
from llm.agents import AIPlayer, Persona, PrivateStrategy, build_speech_prompt, build_vote_prompt, build_witch_prompt, build_wolf_den_prompt
from llm.grounding import SpeechReference, speech_index, validate_speech_facts
from llm.strategies import CURRENT_RULES, MAX_CARDS, MAX_REFERENCE_CHARS, StrategyLibrary, get_strategy_advice, load_strategy_library, public_claims
from tests.test_agents import FakeClient

PERSONA = Persona("晨晨", "谨慎", "核对票型")


def advice(board=NINE_STANDARD, role=Role.VILLAGER, action='vote', seat=0, view=(), alive=None, **kw):
    return get_strategy_advice(board=board, role=role, action=action, seat=seat, view=list(view),
                               day=kw.pop('day', 2), alive_seats=list(range(board.player_count)) if alive is None else alive, **kw)


def player(role=Role.VILLAGER, board=NINE_STANDARD, responses=()):
    client = FakeClient(list(responses))
    return AIPlayer(client, seat=0, name=PERSONA.name, role=role, persona=PERSONA, board=board), client


def test_other_hidden_events_never_change_strategy_input_or_inferred_facts():
    common = [ev.speech(1, 1, '甲', 1, '我是预言家，我验了3号是好人。')]
    hidden_a = [ev.seer_check(1, 1, 2, True), ev.potion_used(1, 3, '毒药', 4)]
    hidden_b = [ev.seer_check(1, 1, 8, False), ev.potion_used(1, 3, '解药', 4)]
    first, second = advice(view=common + hidden_a), advice(view=common + hidden_b)
    assert first == second
    assert not first.situation.known_wolves
    assert first.situation.known_good == {0}


def test_wolf_knife_cross_window_reference_covers_poison_shot_guard_with_unknown_resources():
    result = advice(TWELVE_GUARD, Role.WEREWOLF, 'wolf_kill', teammate_seats=[1,2,3], can_save=False, can_poison=False)
    assert {c.role for c in result.other_cards} == {'女巫','猎人','守卫'}
    assert {'W03', 'H03'} <= {c.id for c in result.other_cards}
    assert '条件假设' in result.text and '资源可用' in result.text
    assert len(result.cards) + len(result.other_cards) <= MAX_CARDS
    assert len(result.text) <= MAX_REFERENCE_CHARS
    nine = advice(NINE_STANDARD, Role.WEREWOLF, 'wolf_kill', teammate_seats=[1,2])
    assert '守卫' not in {c.role for c in nine.other_cards}


@pytest.mark.parametrize('board', BOARDS.values(), ids=BOARDS.keys())
def test_knife_good_and_peaceful_night_use_only_board_specific_rules(board):
    result = advice(board, Role.WITCH, 'witch_act', view=[ev.witch_knew(1,0,1),ev.peaceful_night(1)], kill_target=1)
    assert (1 in result.situation.known_good) == (board is SIX_CLASSIC)
    if board is not SIX_CLASSIC:
        assert 'critical_certain' not in result.situation.flags
        assert 'W04' not in {c.id for c in result.cards}
    assert 'poison_available' in result.situation.flags


def test_same_counts_parity_and_edge_produce_different_valid_ranges_and_methods():
    parity = advice(SIX_CLASSIC, alive=[0,1,2])
    edge = advice(NINE_STANDARD, alive=[0,1,2])
    assert parity.situation.wolf_counts == (1,)
    assert edge.situation.wolf_counts == (1,)
    larger_edge = advice(NINE_STANDARD, alive=[0,1,2,3])
    assert larger_edge.situation.wolf_counts == (1,2)
    assert edge.situation.villager_counts == (1,) and edge.situation.god_counts == (1,)
    assert 'critical_certain' in parity.situation.flags and 'critical_certain' not in edge.situation.flags
    assert 'C10' in {c.id for c in edge.cards}


def test_public_reveals_are_facts_but_claims_and_matching_wolf_methods_arent_certification():
    reveal = ev.Event(ev.EventKind.IDIOT_REVEAL, 1, ev.Phase.DAY, '白痴翻牌。', seat=1, extra={'role':'白痴'})
    from engine.boards import TWELVE_STANDARD
    result = advice(TWELVE_STANDARD, view=[reveal, ev.speech(1,2,'乙',1,'我是预言家，我验了4号是狼。')])
    assert 1 in result.situation.known_good and 3 not in result.situation.known_wolves
    assert 'idiot_revealed' in result.situation.flags
    normal = advice(view=[ev.speech(1,1,'乙',1,'我是狼人。')])
    assert 1 not in normal.situation.known_wolves


def test_self_role_capacity_and_seer_results_prove_only_complete_constraints():
    result = advice(TWELVE_GUARD, Role.SEER, alive=[0,1,2,3], view=[ev.seer_check(1,0,1,False)])
    assert result.situation.inference_complete
    assert result.situation.wolf_counts == (1,2)
    assert not result.situation.known_wolves
    contradiction = advice(NINE_STANDARD, Role.SEER, view=[ev.seer_check(1,0,1,True),ev.seer_check(2,0,1,False)])
    assert not contradiction.situation.wolf_counts and not contradiction.situation.known_wolves
    assert 'contradictory_facts' in contradiction.situation.flags


def test_truncation_never_certifies_intersection_or_resource(monkeypatch):
    monkeypatch.setattr(strategies,'MAX_INFERENCE_BRANCHES',1)
    result = advice(TWELVE_GUARD, Role.SEER, view=[ev.seer_check(1,0,1,False)])
    assert not result.situation.inference_complete
    assert result.situation.known_good == {0,1} and not result.situation.known_wolves
    assert not result.situation.wolf_counts and '非完备' in result.text


def test_unknown_rules_disable_certification_and_pending_shot_disables_continuation_bound():
    unknown = replace(CURRENT_RULES, capabilities=frozenset())
    result = get_strategy_advice(role=Role.SEER, seat=0, action='vote',view=[],day=2,alive_seats=[0,1,2],rules=unknown)
    assert not result.situation.wolf_counts and not result.situation.known_good
    pending = advice(NINE_STANDARD, Role.HUNTER, 'hunter_shot', alive=[1,2], settled=False)
    assert pending.situation.wolf_counts == (0,1,2)
    assert 0 in pending.situation.known_good


@pytest.mark.parametrize('text,expected', [
    ('我是预言家，我验了12号是狼。', 11),
    ('我不是预言家，我没有验12号。', None),
    ('如果我是预言家，我验了12号是狼。', None),
    ('10号说，我是预言家，我验了12号是狼。', None),
    ('10号在[#1]说过“我是预言家，我验了12号是狼。”。', None),
])
def test_multidigit_claims_negation_reports_and_hypotheses(text, expected):
    claims = public_claims([ev.speech(1,9,'甲',1,text)], set(range(12)))
    checks = [c.target for c in claims if c.target is not None]
    assert checks == ([] if expected is None else [expected])
    if expected is not None:
        assert not [c for c in public_claims([ev.speech(1,8,'甲',1,text)],set(range(9))) if c.target is not None]


def test_twelve_seat_reference_vote_skill_fact_validation_and_out_of_board_rejection():
    history = [ev.Event(ev.EventKind.BOARD,0,ev.Phase.NIGHT,'规则',extra=TWELVE_GUARD.public_projection()),
               ev.speech(1,11,'晨晨',1,'保留判断。'),ev.vote_result(1,{9:1},[(11,9)])]
    reference = SpeechReference(speech_index=1,seat=12,day=1,quote='保留判断')
    assert validate_speech_facts('12号在[#1]说过“保留判断”。',[reference],records=speech_index(history),day=1,view=history) is None
    assert validate_speech_facts('12号今天投给10号。',[],records={},day=1,view=history) is None
    assert validate_speech_facts('12号今天投给11号。',[],records={},day=1,view=history)
    assert validate_speech_facts('12号开枪了。',[],records={},day=1,view=history)
    shot = ev.Event(ev.EventKind.HUNTER_SHOT,1,ev.Phase.HUNTER,'真实开枪',seat=11,extra={'role':'猎人','target':9})
    assert validate_speech_facts('12号开枪了。',[],records={},day=1,view=history+[shot]) is None
    history[0].extra = NINE_STANDARD.public_projection()
    assert validate_speech_facts('12号值得关注。',[],records={},day=1,view=history)


def test_speech_and_skill_dynamic_targets_retry_without_failed_summary_commits():
    bad = {'vote':12,'private_strategy':{'assessment':'失败哨兵','goal':'失败','claim':'失败','prediction':'失败预判'}}
    actor,client=player(responses=[json.dumps(bad),'{"vote":9}'])
    assert actor.day_vote([],1,list(range(9))) == 8
    assert len(client.calls)==2 and actor.private_strategy is None
    actor,client=player(board=TWELVE_GUARD,responses=['{"speech":"保留判断。","suspect":12}'])
    assert actor.day_speech([],1,list(range(12)))[1] == 11
    assert len(client.calls)==1
    actor,client=player(Role.SEER,TWELVE_GUARD,['{"target":12}'])
    assert actor.seer_decision([],1,list(range(1,12)))==11


@pytest.mark.parametrize('action,role,good,expected', [
    ('guard',Role.GUARD,{'target':12},11),('hunter_shot',Role.HUNTER,{'target':12},11),
    ('badge',Role.SEER,{'target':12},11),('sheriff_vote',Role.VILLAGER,{'target':12},11),
    ('sheriff_run',Role.VILLAGER,{'choice':True},True),
    ('sheriff_withdraw',Role.VILLAGER,{'choice':False},False),
    ('sheriff_direction',Role.SEER,{'direction':'counterclockwise'},'counterclockwise')])
def test_new_window_valid_requests_use_one_call_and_successful_summary(action,role,good,expected):
    summary={'assessment':'合法窗口','goal':'本轮行动','claim':'公开说法','prediction':'12号可能改变支持','switch_condition':'公开票型反驳时改策'}
    actor,client=player(role,TWELVE_GUARD,[json.dumps(dict(good,private_strategy=summary),ensure_ascii=False)])
    window=ActionWindow(action,0,1,ev.Phase.HUNTER,(1,11),(11,),True,False,3,('clockwise','counterclockwise'))
    assert actor.window_decision(window,[]) == expected
    assert actor.private_strategy.prediction == summary['prediction'] and len(client.calls)==1
    text=client.calls[0][0]['content']
    assert 'twelve-guard' in text and '最多三名' in text and '1.5' in text
    assert '死亡调用者' in text


@pytest.mark.parametrize('action,fallback', [('guard',None),('hunter_shot',None),('sheriff_vote',None),('badge',None),('sheriff_run',False),('sheriff_withdraw',False),('sheriff_direction','clockwise')])
def test_new_window_bad_output_three_attempts_safe_fallback_keeps_memory(action,fallback):
    actor,client=player(Role.HUNTER,TWELVE_GUARD,['{"target":99,"direction":"bad"}']*3)
    previous=PrivateStrategy(assessment='之前事实',goal='之前目标',claim='之前说法',prediction='之前预判')
    actor.private_strategy=previous
    window=ActionWindow(action,0,1,ev.Phase.HUNTER,(1,11),(11,),True,False,2,('clockwise','counterclockwise'))
    assert actor.window_decision(window,[]) == fallback
    assert len(client.calls)==3 and actor.private_strategy is previous


def test_optional_predictions_preserved_reviewed_and_isolated_from_other_player_new_game():
    actor,client=player(responses=['{"vote":0,"private_strategy":{"assessment":"新判断","goal":"新目标","claim":"新说法"}}'])
    actor.private_strategy=PrivateStrategy(assessment='旧判断',goal='旧目标',claim='旧说法',prediction='2号可能投3号',switch_condition='2号改票则重审')
    view=[ev.vote_result(1,{3:1},[(1,3)])]
    actor.day_vote(view,2,list(range(9)))
    assert actor.private_strategy.prediction=='2号可能投3号'
    text=client.calls[0][0]['content']
    assert '预判核对与修正' in text and '2号→4号' in text and '支持者改票' in text
    other,_=player()
    assert other.private_strategy is None
    other_text=build_vote_prompt(seat=1,name='其他',role=Role.VILLAGER,persona=PERSONA,view=view,day=2,alive_seats=list(range(9)),my_votes=[],board=NINE_STANDARD)[0]['content']
    assert '2号可能投3号' not in other_text


def test_four_wolf_shared_decision_is_one_request_and_keeps_personal_summaries_separate():
    actor, client = player(Role.WEREWOLF, TWELVE_GUARD, ['{"target":0,"team_plan":"甲争票，乙切割，丙保留备选，丁观察公开改策信号。"}'])
    actor.teammate_seats = [1, 2, 3]
    actor.private_strategy = PrivateStrategy(assessment='调用者个人摘要',goal='选刀',claim='自己的说法')
    peer, _ = player(Role.WEREWOLF, TWELVE_GUARD)
    peer.private_strategy = PrivateStrategy(assessment='队友秘密哨兵',goal='别的目标',claim='别的说法')
    view = [ev.wolf_reveal(0,[0,1,2,3],'四狼')]
    assert actor.wolf_den_decision(view,1,list(range(12))) is None
    assert len(client.calls)==1 and actor.team_plan is not None
    text = client.calls[0][0]['content']
    assert '存活狼队成员：1 号、2 号、3 号、4 号' in text
    assert '调用者个人摘要' in text and '队友秘密哨兵' not in text
    assert '白天狼队无即时私聊' in text


def test_pk_prompt_keeps_public_alive_count_separate_from_legal_candidates_and_weight():
    actor,client=player(Role.VILLAGER,TWELVE_GUARD,['{"vote":12}'])
    window=ActionWindow('exile_revote',0,2,ev.Phase.DAY,tuple(range(12)),(10,11),True,True,3)
    assert actor.day_vote([],2,[10,11],window=window)==11
    text=client.calls[0][0]['content']
    assert '当前存活 12 人' in text and '本票权重：1.5票' in text
    assert '合法投票目标：11 号、12 号' in text


def test_public_weighted_tally_decimal_speech_is_checked_without_splitting_digits():
    history = [ev.vote_result(1,{11:1.5},[(0,11)])]
    assert validate_speech_facts('12号得了1.5票。',[],records={},day=1,view=history) is None
    assert validate_speech_facts('12号得了2.5票。',[],records={},day=1,view=history)


def test_same_visible_clash_changes_methods_with_actual_private_check_and_real_last_window():
    view=[ev.speech(1,1,'甲',1,'我是预言家，我验了3号是好人。'),ev.speech(1,2,'乙',2,'我是预言家，我验了4号是狼。')]
    civilian=advice(view=view,action='speech')
    true_seer=advice(role=Role.SEER,view=view+[ev.seer_check(1,0,2,True)],action='speech')
    assert 2 not in civilian.situation.known_wolves and 2 in true_seer.situation.known_wolves
    assert 'S03' in {c.id for c in true_seer.cards}
    context=SpeechContext(1,0,(1,2,0),3,(1,2),(),(),2,'reply',ev.Phase.DAY,(),True)
    messages=build_speech_prompt(seat=0,name='晨晨',role=Role.SEER,persona=PERSONA,teammate_seats=None,
        view=view+[ev.seer_check(1,0,2,True)],day=1,alive_seats=list(range(9)),my_votes=[],last_suspect=None,context=context,board=NINE_STANDARD)
    assert '其他好人已经相信' in messages[0]['content']
    assert '无法再回答' in messages[0]['content'] and '最后一次发言' in messages[0]['content']


def test_invalid_links_and_version_one_compatibility_fallback(monkeypatch):
    library=load_strategy_library()
    data=library.model_dump(mode='json')
    legacy=dict(data,version=1)
    for c in legacy['cards']:
        for name in ['uses','required_skills','related_windows','clues','responses','failure_cost','counters','exit_conditions','example','counterexample']:
            c.pop(name,None)
    assert StrategyLibrary.model_validate(legacy).version==1
    data['cards'][0]['counters']=['NO99']
    class Resource:
        def joinpath(self,*a):return self
        def read_text(self,**kw):return json.dumps(data)
    monkeypatch.setattr(strategies,'files',lambda *a:Resource())
    load_strategy_library.cache_clear()
    try:
        result=advice(TWELVE_GUARD)
        assert not result.cards and '策略库暂不可用' in result.text
        prompt=build_witch_prompt(seat=0,persona=PERSONA,view=[],day=2,kill_target=None,can_save=False,can_poison=True,alive_seats=list(range(12)),board=TWELVE_GUARD)
        assert 'twelve-guard' in prompt[0]['content'] and '无可见刀口' in prompt[0]['content']
        assert '6 人局' not in prompt[0]['content']
    finally:
        load_strategy_library.cache_clear()
