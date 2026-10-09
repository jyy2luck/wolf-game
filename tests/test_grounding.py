"""公开原始记录的引用合同与发言行为断言的正反例。"""

import json
from types import SimpleNamespace

import pytest

from engine import events as ev
from engine.game import SpeechContext
from engine.roles import Role
from llm.agents import SAFE_SPEECH, SpeechPack, parse_pack
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
    return validate_speech_facts(
        text, [SpeechReference(**r) for r in refs],
        records=speech_index(HISTORY if history is None else history), day=day, context=context,
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


@pytest.mark.parametrize("replies,expected_calls", [
    (["bad JSON", speech_json("2号抢话。"), speech_json("我是狼", 2)], 3),
    ([speech_json("我是狼", 2), "bad JSON", speech_json("我是狼", 2)], 3),
    ([speech_json("我是狼", 2), speech_json("我是狼", 2), speech_json("安全的话")], 2),
])
def test_mixed_errors_share_three_attempts_and_only_one_leak_correction(replies, expected_calls):
    player, client = make_player(role=Role.WEREWOLF, seat=5, replies=replies)
    assert player.day_speech([], 1, ALIVE, context=FIRST) == (SAFE_SPEECH, None, "平静")
    assert len(client.calls) == expected_calls
    assert player.last_suspect is None


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


def test_reference_cut_by_final_length_limit_is_revalidated():
    player, client = make_player(replies=[
        speech_json("字" * 150 + "昨天2号[#3]说过这些。", references=[reference()]),
        speech_json("我暂时保留判断。"),
    ])
    assert player.day_speech(HISTORY, 2, ALIVE)[0] == "我暂时保留判断。"
    assert len(client.calls) == 2
    assert "未在正文使用" in client.calls[1][-1]["content"]
