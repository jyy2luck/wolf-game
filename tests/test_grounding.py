"""公开原始记录的引用合同与发言行为断言的正反例。"""

import json
from types import SimpleNamespace

import pytest

from engine import events as ev
from engine.game import SpeechContext
from engine.roles import Role
from llm.agents import MAX_SPEECH_CHARS, SAFE_SPEECH, SpeechPack, parse_pack
from llm.client import OutputTruncatedError
from llm.grounding import SpeechReference, speech_index, validate_speech_facts
from tests.test_agents import ALIVE, make_player, speech_json

NAMES = ("人类", "阿飞", "老猫", "大树", "铁叔", "学姐")
FIRST = SpeechContext(1, 5, (5, 0, 1, 2, 3), 1, (), (0, 1, 2, 3), tuple(enumerate(NAMES)))
SECOND_DAY = SpeechContext(2, 5, (5, 0, 1, 2, 3), 1, (), (0, 1, 2, 3), tuple(enumerate(NAMES)))
HISTORY = [ev.speech(1, 1, "阿飞", 3, "我怀疑 4 号，但还想多听听。"),
           ev.speech(1, 4, "铁叔", 4, "我保留判断。"), ev.death(2, [4])]


def reference(index=3, seat=2, day=1, quote="我怀疑 4 号"):
    return {"speech_index": index, "seat": seat, "day": day, "quote": quote}


def check(text, refs=(), history=None, day=2, context=SECOND_DAY):
    view = HISTORY if history is None else history
    return validate_speech_facts(
        text, [SpeechReference(**r) for r in refs],
        records=speech_index(view), day=day, context=context, view=view,
    )


@pytest.mark.parametrize("speech,refs", [
    ("第 1 天 2 号在 [#3] 说过‘我怀疑 4 号’，我觉得还要观察。", [reference()]),
    ("阿飞昨天在 [#3] 说过‘我怀疑 4 号’，我暂时保留判断。", [reference()]),
    ("历史发言 [#3] 让我有些怀疑 2 号，但这是我的猜测。", [reference()]),
    ("第 1 天 5 号在 [#4] 说过‘我保留判断’，出局也不妨碍回顾。",
     [reference(4, 5, 1, "我保留判断")]),
    ("目前没有行为依据，我只是暂时怀疑 2 号。", []),
])
def test_valid_cross_day_dead_player_and_subjective_references(speech, refs):
    assert check(speech, refs) is None


@pytest.mark.parametrize("speech,refs,error", [
    ("2 号在 [#999] 说过这些。", [reference(999)], "不存在"),
    ("第 1 天 2 号 [#3]。", [reference(seat=6)], "发言人"),
    ("第 1 天 2 号 [#3]。", [reference(day=2)], "属于第 1 天"),
    ("第 1 天 2 号 [#3]。", [reference(quote="我认定 6 号是狼")], "原文片段"),
    ("第 1 天 2 号 [#3]。", [reference(quote="   ")], "原文片段"),
    ("第 1 天 6 号在 [#3] 说过这些。", [reference()], "正文"),
    ("第 1 天学姐在 [#3] 说过这些。", [reference()], "正文"),
    ("第 2 天 2 号在 [#3] 说过这些。", [reference()], "正文时间"),
    ("2 号今天刚才在 [#3] 说过这些。", [reference()], "正文时间"),
    ("2 号在 [#3] 今天刚才说过这些。", [reference()], "正文时间"),
    ("昨天 2 号在 [#3] 说过‘我绝对不是狼’。", [reference()], "引号内"),
    ("我暂时保留判断。", [reference()], "未在正文使用"),
    ("第 1 天 2 号在 [#3] 说过这些。", [], "缺少 references"),
])
def test_reference_mismatch_reports_specific_conflict(speech, refs, error):
    assert error in check(speech, refs)


def test_references_default_to_empty_but_legacy_citations_fail():
    pack, err = parse_pack(speech_json("我只是暂时怀疑 2 号。"), SpeechPack)
    assert err is None and pack.references == []
    player, client = make_player(replies=[speech_json("[#1]有问题")] * 3)
    assert player.day_speech([], 1, ALIVE) == (SAFE_SPEECH, None, "平静")
    assert "缺少 references" in client.calls[1][-1]["content"]


@pytest.mark.parametrize("speech", [
    "2 号一上来就急着说话，我觉得很可疑。",
    "我觉得 2 号一上来就急着说话。",
    "阿飞刚才说了那么多，我暂时怀疑他。",
    "2号发言太积极了。",
    "2号已经改口了。",
    "3号故意不说话，4号躲发言。",
    "大树故意沉默。",
    "2号长篇大论，一直在给自己辩解。",
    "我觉得阿飞刚才长篇大论。",
    "2号一直辩解。",
])
def test_first_speaker_cannot_invent_speech_or_intentional_silence(speech):
    assert check(speech, history=[], day=1, context=FIRST) is not None


@pytest.mark.parametrize("speech", [
    "等 2 号发言后再判断。",
    "等阿飞说话后，我再考虑。",
    "如果 2 号急着说话，我可能会怀疑他。",
    "假如阿飞刚才说了这种话，我会怀疑他。",
    "目前没有行为依据，我只是暂时怀疑 2 号。",
    "还没听到 3、4 号今天的观点，暂时无法判断。",
    "2 号并没有说过这句话。",
    "我是好人，我暂时怀疑 2 号。",
    "我是预言家，昨晚我验了2号是狼。",
    "我是女巫，昨晚我救了2号。",
    "如果2号眼神躲闪，我可能会怀疑他。",
    "2号并没有眼神躲闪。",
    "2号没有拍桌子。",
])
def test_waiting_hypothetical_negative_and_role_play_are_allowed(speech):
    assert check(speech, history=[], day=1, context=FIRST) is None


@pytest.mark.parametrize("speech", ["2号今天刚才说了这些。", "阿飞今日表示了怀疑。"])
def test_old_record_does_not_prove_today_speech(speech):
    assert "第 2 天尚无发言记录" in check(speech)


def test_current_record_still_requires_explicit_reference_for_behavior():
    assert "缺少真实编号" in check("2号昨天说过这些。")


@pytest.mark.parametrize("speech", [
    "6号在 [#1] 声称‘2号抢话’，但不能证明2号抢话。",
    "6号在 [#1] 声称2号抢话，但2号尚无发言记录。",
    "6号说过‘2号抢话’ [#1]，这只是他的指控。",
])
def test_rebutting_a_real_but_unsupported_accusation_is_allowed(speech):
    history = [ev.speech(1, 5, "学姐", 1, "2号抢话")]
    refs = [reference(1, 6, 1, "2号抢话")]
    assert check(speech, refs, history=history, day=1, context=FIRST) is None
    # 只有转述者的记录，不会生成被指控者的记录。
    assert "整局尚无发言记录" in check("2号抢话。", history=history, day=1, context=FIRST)


@pytest.mark.parametrize("claim", ["2号抢话，太积极了", "2号抢话。3号改口了", "2号抢话，3号也抢话！"])
def test_quoted_punctuation_does_not_end_supported_report_scope(claim):
    history = [ev.speech(1, 5, "学姐", 1, claim)]
    text = f"6号在 [#1] 声称‘{claim}’，但这是无依据的指控。"
    assert check(text, [reference(1, 6, 1, claim)], history=history, day=1, context=FIRST) is None


def test_fact_retry_uses_specific_error_and_never_exposes_deltas():
    player, client = make_player(seat=5, replies=[
        speech_json("2 号一上来就急着说话。", 2, "激动"),
        speech_json("等 2 号发言后再判断。"),
    ])
    deltas = []
    assert player.day_speech([], 1, ALIVE, deltas.append, context=FIRST) == ("等 2 号发言后再判断。", None, "平静")
    assert len(client.calls) == 2 and deltas == []
    assert "2 号整局尚无发言记录" in client.calls[1][-1]["content"]


def test_fallback_clears_previous_and_failed_suspect():
    player, client = make_player(seat=5, replies=[speech_json("2号抢话。", 2, "愤怒")] * 3)
    player.last_suspect = 3
    assert player.day_speech([], 1, ALIVE, context=FIRST) == (SAFE_SPEECH, None, "平静")
    assert player.last_suspect is None and len(client.calls) == 3


@pytest.mark.parametrize("replies,expected,expected_calls", [
    (["bad JSON", speech_json("2号抢话。"), "bad JSON"], (SAFE_SPEECH, None, "平静"), 3),
    (["bad JSON", speech_json("2号抢话。"), speech_json("我是狼", 2)], ("我是狼", 1, "平静"), 3),
    ([speech_json("我是狼", 2)], ("我是狼", 1, "平静"), 1),
])
def test_mixed_errors_share_three_attempts_without_rejecting_role_play(replies, expected, expected_calls):
    player, client = make_player(role=Role.WEREWOLF, seat=5, replies=replies)
    assert player.day_speech([], 1, ALIVE, context=FIRST) == expected
    assert len(client.calls) == expected_calls
    assert player.last_suspect == expected[1]


def test_truncation_fact_and_format_failures_share_budget_and_ceiling():
    player, client = make_player(seat=5)
    client.config = SimpleNamespace(max_tokens=500, max_tokens_limit=750)
    budgets = []
    replies = [OutputTruncatedError(500, "half-json", None), speech_json("2号抢话。"), "bad"]

    def chat(messages, **kwargs):
        client.calls.append(messages)
        budgets.append(kwargs["max_tokens"])
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    client.chat = chat
    assert player.day_speech([], 1, ALIVE, context=FIRST) == (SAFE_SPEECH, None, "平静")
    assert budgets == [500, 750, 750]
    assert "half-json" not in json.dumps(client.calls)


@pytest.mark.parametrize("long_speech", [
    "字" * MAX_SPEECH_CHARS + "昨天2号在[#3]说过‘我怀疑 4 号’。",
    "字" * (MAX_SPEECH_CHARS - 2) + "[#3]",
], ids=["after_limit", "split_marker"])
def test_oversized_speech_with_late_reference_reports_length_not_fact(long_speech):
    final = "昨天2号在[#3]说过‘我怀疑 4 号’，我还想听听。"
    player, client = make_player(replies=[
        speech_json(long_speech, references=[reference()]),
        speech_json(final, references=[reference()]),
    ])
    assert player.day_speech(HISTORY, 2, ALIVE)[0] == final
    assert len(client.calls) == 2
    feedback = client.calls[1][-1]["content"]
    assert "发言超过" in feedback and "未在正文使用" not in feedback


def test_reference_at_exact_speech_length_limit_is_kept_and_validated():
    opening = "昨天2号说过‘我怀疑 4 号’"
    final = opening + "字" * (MAX_SPEECH_CHARS - len(opening) - len("[#3]")) + "[#3]"
    player, client = make_player(replies=[speech_json(final, references=[reference()])])
    assert player.day_speech(HISTORY, 2, ALIVE)[0] == final
    assert len(client.calls) == 1


def test_shortened_speech_still_checks_facts_within_shared_retry_budget():
    player, client = make_player(replies=[
        speech_json("字" * MAX_SPEECH_CHARS + "2号抢话。"),
        speech_json("2号抢话。"),
        speech_json("等2号发言后再判断。"),
    ])
    assert player.day_speech([], 1, ALIVE)[0] == "等2号发言后再判断。"
    assert len(client.calls) == 3
    assert "发言超过" in client.calls[1][-1]["content"]
    assert "整局尚无发言记录" in client.calls[2][-1]["content"]


@pytest.mark.parametrize("text", ["2号眼神躲闪。", "阿飞拍桌子抢话。", "2号表情明显慌张。"])
def test_text_only_history_does_not_prove_visual_observations(text):
    assert "观察依据" in check(text)


def test_real_quote_can_be_selected_and_motives_can_be_distorted_but_words_cannot_be_forged():
    history = [ev.speech(1, 1, "阿飞", 1, "我暂时怀疑3号，但证据还不够。")]
    refs = [reference(1, 2, 1, "我暂时怀疑3号")]
    allowed = "第1天2号在[#1]说过‘我暂时怀疑3号’，2号嘴上说证据不足，其实就是想把3号推出去。"
    assert check(allowed, refs, history=history) is None
    assert "引号内" in check("第1天2号在[#1]说过‘3号一定是狼’。", refs, history=history)


@pytest.mark.parametrize("claim", ["2号眼神躲闪", "2号长篇大论", "2号死了", "2号有3票"])
def test_rebutting_supported_quotes_does_not_adopt_their_false_records(claim):
    history = [ev.speech(1, 5, "学姐", 1, claim)]
    refs = [reference(1, 6, 1, claim)]
    assert check(f"6号在[#1]声称‘{claim}’，但这不是系统确认的事实。", refs,
                 history=history, day=1, context=FIRST) is None


@pytest.mark.parametrize("text", ["2号说过自己有3票。", "2号说过自己死了。"])
def test_public_results_do_not_substitute_for_a_claimed_speech(text):
    history = [ev.vote_result(1, {1: 3}), ev.exile(1, 1)]
    assert "整局尚无发言记录" in check(text, history=history)


@pytest.mark.parametrize("text,error", [
    ("2号今天死了。", "公开记录没有"),
    ("5号昨天死了。", "公开记录没有"),
    ("5号今天被放逐了。", "公开记录没有"),
    ("2号今天得了3票。", "尚无公开票型"),
    ("2号昨天得了2票。", "是 3 票"),
    ("3号昨天有1票。", "是 0 票"),
])
def test_explicit_death_and_vote_conflicts_use_public_events(text, error):
    history = [ev.vote_result(1, {1: 3}), ev.death(2, [4])]
    assert error in check(text, history=history)


@pytest.mark.parametrize("text", [
    "5号今天死了。", "5号昨晚死了。", "2号昨天被放逐了。", "2号昨天有3票。",
    "3号昨天得了0票。", "如果2号死了，我会怀疑3号。", "2号会有3票。",
    "我估计2号会得到3票。", "2号并没有死了。", "2号昨天并没有得到2票。",
])
def test_true_public_results_negations_and_predictions_remain_allowed(text):
    history = [ev.vote_result(1, {1: 3}), ev.exile(1, 1), ev.death(2, [4])]
    assert check(text, history=history) is None
