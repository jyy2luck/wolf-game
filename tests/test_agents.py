"""AI 解析、策略性表达、私有记忆与有界降级。"""

import json

import pytest

from engine.roles import Role
from engine import events as ev
from llm.agents import (
    PERSONA_POOL,
    SAFE_SPEECH,
    MAX_SPEECH_CHARS,
    MAX_REPLY_CHARS,
    AIPlayer,
    Persona,
    PrivateStrategy,
    VotePack,
    WitchPack,
    SpeechPack,
    TargetPack,
    assign_personas,
    build_speech_prompt,
    build_vote_prompt,
    build_wolf_den_prompt,
    build_seer_prompt,
    build_witch_prompt,
    extract_json_block,
    parse_pack,
)
from llm.client import LLMClient
from config import LLMConfig


# ---------------------------------------------------------------- 工具


class FakeClient:
    """按脚本顺序返回预设文本的假客户端。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self.options = []

    def chat(self, messages, **kwargs):  # noqa: ANN002, ANN003
        self.calls.append(messages)
        self.options.append(kwargs)
        if self.replies:
            return self.replies.pop(0)
        return "{不是json"


def make_player(
    role: Role = Role.VILLAGER,
    replies: list[str] | None = None,
    seat: int = 2,
    teammate_seats: list[int] | None = None,
) -> tuple[AIPlayer, FakeClient]:
    client = FakeClient(replies or [])
    player = AIPlayer(
        client,
        seat=seat,
        name="测试侠",
        role=role,
        persona=Persona("测试侠", "冷静理性", "口癖是「按理说」"),
        teammate_seats=teammate_seats,
    )
    return player, client


ALIVE = [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("builder,options,schema", [
    (build_speech_prompt, {"seat": 1, "name": "学姐", "role": Role.VILLAGER,
                          "teammate_seats": None, "alive_seats": ALIVE,
                          "my_votes": [], "last_suspect": None}, SpeechPack),
    (build_vote_prompt, {"seat": 1, "name": "学姐", "role": Role.VILLAGER,
                        "alive_seats": ALIVE, "my_votes": []}, VotePack),
    (build_wolf_den_prompt, {"wolf_seats": [1, 2], "killable": [0, 3, 4, 5]}, TargetPack),
    (build_seer_prompt, {"seat": 1, "checkable": [0, 2, 3, 4, 5]}, TargetPack),
    (build_witch_prompt, {"seat": 1, "kill_target": 0, "can_save": True,
                         "can_poison": True, "alive_seats": ALIVE}, WitchPack),
])
def test_action_prompt_examples_are_valid_json(builder, options, schema):
    messages = builder(persona=Persona("学姐", "理性", "从逻辑上讲"),
                       view=[], day=1, **options)
    pack, err = parse_pack(messages[-1]["content"], schema)
    assert pack is not None and err is None
    assert "2 名狼人、1 名预言家、1 名女巫、2 名平民" in messages[0]["content"]
    assert "没有守卫" in messages[0]["content"]


def speech_json(speech: str, suspect: int = 0, mood: str = "平静", references=None, **extra) -> str:
    data = {"speech": speech, "suspect": suspect, "mood": mood, **extra}
    if references is not None:
        data["references"] = references
    return json.dumps(
        data, ensure_ascii=False
    )


# ---------------------------------------------------------------- 基础组件


def test_assign_personas_distinct_and_within_pool():
    personas = assign_personas(__import__("random").Random(3), 5)
    assert len(personas) == 5
    names = {p.name for p in personas}
    assert len(names) == 5
    assert all(p in PERSONA_POOL for p in personas)


def test_extract_json_block_variants():
    obj, err = extract_json_block('{"vote": 3}')
    assert err is None and obj == {"vote": 3}
    obj, err = extract_json_block('好的，我的选择：```json\n{"vote": 2}\n``` 以上')
    assert err is None and obj == {"vote": 2}
    obj, err = extract_json_block("我认为应该投票。没有json")
    assert obj is None and err is not None


def test_parse_pack_schema_validation():
    pack, err = parse_pack('{"vote": 3}', VotePack)
    assert pack.vote == 3 and err is None
    pack, err = parse_pack('{"vote": 99}', VotePack)
    assert pack is None and "校验" in err
    pack, err = parse_pack('{"save": true, "poison": 0}', WitchPack)
    assert pack is not None and err is None


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("speech", [
    "我是预言家，我验过3号是狼。",
    "昨晚我查验了3号，是好人。",
    "我是女巫，我救了4号，我的毒药还在。",
    "我昨晚毒了4号。",
    "我是狼，2号是我的队友。",
    "我们狼人要赢了。",
    "今晚刀口是他。",
])
def test_all_roles_can_disguise_claim_private_actions_or_reveal_identity(role, speech):
    player, client = make_player(role=role, replies=[speech_json(speech, 2)])
    view = [ev.role_assigned(0, player.seat, role.value)]
    original = list(view)
    assert player.day_speech(view, 1, ALIVE) == (speech, 1, "平静")
    assert len(client.calls) == 1
    assert player.role is role and view == original


# ---------------------------------------------------------------- 发言


def test_day_speech_success():
    player, client = make_player(
        replies=[speech_json("我觉得 [#1] 有点问题，按理说首发发言最激进。", 3, "冷静",
                             [{"speech_index": 1, "seat": 1, "day": 1, "quote": "我怀疑2号"}])]
    )
    speech, suspect, mood = player.day_speech([ev.speech(1, 0, "人类", 1, "我怀疑2号")], 1, ALIVE)
    assert speech.startswith("我觉得 [#1]")
    assert suspect == 2  # 1-based 3 → 0-based 2
    assert mood == "冷静"
    assert player.last_suspect == 2
    assert len(client.calls) == 1


def test_day_speech_retry_then_success():
    player, client = make_player(
        replies=["这不是json", '{"speach": "字段名打错"}', speech_json("好的我分析一下", 5)]
    )
    speech, suspect, _ = player.day_speech([], 1, ALIVE)
    assert speech == "好的我分析一下"
    assert suspect == 4
    assert len(client.calls) == 3
    # 重试 prompt 携带错误提示
    assert "无法使用" in client.calls[1][-1]["content"]


def test_day_speech_degrade_after_3_failures():
    player, client = make_player(replies=["坏输出", "还是坏", "依然坏"])
    speech, suspect, mood = player.day_speech([], 1, ALIVE)
    assert speech == SAFE_SPEECH
    assert suspect is None
    assert len(client.calls) == 3  # 首次 + 最多 2 次重试


def test_day_speech_illegal_suspect_triggers_retry():
    # suspect 指向已死玩家（4 号死了）→ 校验失败重试
    player, client = make_player(
        replies=[speech_json("怀疑5号", 5), speech_json("那怀疑2号", 2)]
    )
    speech, suspect, _ = player.day_speech([], 1, [0, 1, 2, 3, 5])
    assert suspect == 1
    assert "不在存活玩家中" in client.calls[1][-1]["content"]


STRATEGY = {"assessment": "真实身份是狼人，3号是好人。", "goal": "推动放逐好人。", "claim": "假跳预言家报3号查杀。"}


@pytest.mark.parametrize("method,args,reply,role", [
    ("day_speech", (2, ALIVE), speech_json("我继续保留判断。"), Role.WEREWOLF),
    ("day_vote", (2, ALIVE), '{"vote": 2}', Role.WEREWOLF),
    ("wolf_den_decision", (2, [0, 2, 3, 4]), '{"target": 3}', Role.WEREWOLF),
    ("seer_decision", (2, [0, 2, 3, 4]), '{"target": 3}', Role.SEER),
    ("witch_decision", (2, 0, True, True, ALIVE), '{"save": true, "poison": 0}', Role.WITCH),
])
def test_private_strategy_reaches_later_speech_votes_and_night_actions(method, args, reply, role):
    strategy = {**STRATEGY, "assessment": f"真实身份是{role.value}，假查验只是说辞。"}
    public = "我是预言家，昨晚验了3号是狼。"
    player, client = make_player(role=role, seat=1, teammate_seats=[5], replies=[
        speech_json(public, 3, private_strategy=strategy), reply,
    ])
    view = [ev.role_assigned(0, 1, role.value)]
    result = player.day_speech(view, 1, ALIVE)
    assert result == (public, 2, "平静")
    assert player.private_strategy == PrivateStrategy(**strategy)
    view.append(ev.speech(1, 1, player.name, 1, result[0]))
    getattr(player, method)(view, *args)
    system = client.calls[-1][0]["content"]
    assert player.private_strategy.model_dump_json() in system
    assert "玩家声明：[#1]" in system and public in system
    assert "角色私有事实：查验" not in system
    if method == "day_speech":
        assert "上轮公开怀疑对象（不等于真实判断）：3 号" in system
    if method == "day_vote":
        assert player.my_votes == [(2, 1)]  # 真实投票可以不同于公开攻击对象。


@pytest.mark.parametrize("method,args,reply,role", [
    ("day_vote", (2, ALIVE), {"vote": 2}, Role.VILLAGER),
    ("wolf_den_decision", (2, [0, 2, 3, 4]), {"target": 3}, Role.WEREWOLF),
    ("seer_decision", (2, [0, 2, 3, 4]), {"target": 3}, Role.SEER),
    ("witch_decision", (2, 0, True, True, ALIVE), {"save": True, "poison": 0}, Role.WITCH),
])
def test_each_action_commits_only_accepted_new_summary(method, args, reply, role):
    updated = {"assessment": "听完全部发言后修正判断。", "goal": "根据新证据改变行动。", "claim": "说明公开改口。"}
    player, client = make_player(role=role, seat=1, teammate_seats=[5], replies=[
        json.dumps({**reply, "private_strategy": updated}, ensure_ascii=False),
        json.dumps(reply),
        *[json.dumps({**reply, "private_strategy": {**updated, "goal": "x" * 61}})] * 3,
    ])
    getattr(player, method)([], *args)
    accepted = player.private_strategy
    assert accepted == PrivateStrategy(**updated)
    getattr(player, method)([], *args)
    assert player.private_strategy is accepted
    getattr(player, method)([], *args)
    assert player.private_strategy is accepted
    assert all(options["temperature"] == 0.3 for options in client.options)
    assert accepted.model_dump_json() in client.calls[1][0]["content"]


def test_invalid_target_does_not_commit_candidate_summary():
    updated = {**STRATEGY, "goal": "不能提交非法目标的摘要。"}
    player, _ = make_player(replies=[json.dumps({"vote": 6, "private_strategy": updated})] * 3)
    player.private_strategy = PrivateStrategy(**STRATEGY)
    assert player.day_vote([], 2, [0, 1, 2]) is None
    assert player.private_strategy.goal == STRATEGY["goal"]


def test_reply_has_shorter_limit_and_all_first_round_history():
    from engine.game import SpeechContext

    context = SpeechContext(1, 2, tuple(ALIVE), 3, (0, 1), (3, 4, 5), (), round_number=2)
    player, client = make_player(replies=[speech_json("字" * (MAX_REPLY_CHARS + 1)), speech_json("我根据新证据调整投票方向。")])
    view = [ev.speech(1, s, str(s), s + 1, f"首轮观点{s}") for s in ALIVE]
    assert player.day_speech(view, 1, ALIVE, context=context)[0] == "我根据新证据调整投票方向。"
    assert str(MAX_REPLY_CHARS) in client.calls[1][-1]["content"]
    prompt = client.calls[0][0]["content"]
    assert "全部首轮发言" in prompt and "投票前最后一次发言" in prompt
    assert all(e.extra["content"] in prompt for e in view)


def test_wolf_rejects_invalid_team_plan_without_leaking_or_retaining_it():
    player, _ = make_player(role=Role.WEREWOLF, seat=1, teammate_seats=[5], replies=[
        json.dumps({"target": 3, "team_plan": "x" * 241}), json.dumps({"target": 3, "team_plan": "2号主导悍跳，6号争取归票。"}),
        *[json.dumps({"target": 6, "team_plan": "失败候选计划"})] * 3,
    ])
    assert player.wolf_den_decision([], 1, [0, 2, 3, 4]) == 2
    assert player.team_plan == "2号主导悍跳，6号争取归票。"
    player.wolf_den_decision([], 2, [0, 2, 3, 4])
    assert player.team_plan is None


def test_failed_candidates_missing_and_null_summaries_preserve_accepted_strategy():
    rejected = {"assessment": "失败候选秘密", "goal": "失败候选目的", "claim": "失败候选说辞"}
    player, client = make_player(replies=[
        speech_json("保留判断。", 2, private_strategy=STRATEGY),
        speech_json("2号刚才长篇大论。", 2, private_strategy=rejected),
        speech_json("继续听听。"),
        speech_json("还是保留意见。", private_strategy=None),
        *[speech_json("2号抢话。", 2, private_strategy=rejected)] * 3,
        speech_json("调整看法。", private_strategy={**STRATEGY, "goal": "改为收集信息。"}),
    ])
    player.day_speech([], 1, ALIVE)
    accepted = player.private_strategy
    original_chat = client.chat

    def chat(messages, **kwargs):
        assert player.private_strategy is accepted
        return original_chat(messages, **kwargs)

    client.chat = chat
    player.day_speech([], 2, ALIVE)
    player.day_speech([], 2, ALIVE)
    assert player.private_strategy is accepted
    assert player.day_speech([], 3, ALIVE) == (SAFE_SPEECH, None, "平静")
    assert player.last_suspect is None and player.private_strategy is accepted
    assert player.day_speech([], 4, ALIVE)[0] == "调整看法。"
    assert player.private_strategy.goal == "改为收集信息。"


@pytest.mark.parametrize("field,limit", [("assessment", 120), ("goal", 60), ("claim", 120)])
def test_private_strategy_limits_retry_without_logging_private_content(field, limit, caplog):
    secret = "隐私" * (limit + 1)
    player, client = make_player(replies=[
        speech_json("保留判断。", private_strategy={**STRATEGY, field: secret}),
        speech_json("继续听听。", private_strategy=STRATEGY),
    ])
    assert player.day_speech([], 1, ALIVE)[0] == "继续听听。"
    assert len(client.calls) == 2 and player.private_strategy == PrivateStrategy(**STRATEGY)
    assert secret not in caplog.text


def test_legacy_pack_defaults_to_no_private_strategy():
    pack, err = parse_pack(speech_json("保留意见。"), SpeechPack)
    assert err is None and pack.private_strategy is None


def test_day_speech_oversized_speech_requests_complete_shorter_reply(caplog):
    long_speech = "字" * 300
    final = "我暂时保留判断。"
    player, client = make_player(replies=[speech_json(long_speech), speech_json(final)])
    assert player.day_speech([], 1, ALIVE) == (final, None, "平静")
    assert len(client.calls) == 2
    feedback = client.calls[1][-1]["content"]
    assert "300" in feedback and str(MAX_SPEECH_CHARS) in feedback
    assert "缩写" in feedback and "references" in feedback
    assert "校验/length失败" in caplog.text


@pytest.mark.parametrize("length", [MAX_SPEECH_CHARS - 1, MAX_SPEECH_CHARS])
def test_day_speech_within_limit_preserves_complete_body_after_trimming(length):
    final = "字" * (length - 1) + "。"
    player, client = make_player(replies=[speech_json(" \n" + final + "\n ")])
    assert player.day_speech([], 1, ALIVE)[0] == final
    assert len(client.calls) == 1


def test_oversized_speech_exhaustion_preserves_accepted_strategy_and_clears_suspect():
    rejected = {**STRATEGY, "goal": "不得接受的超长候选计划。"}
    player, client = make_player(replies=[
        speech_json("字" * (MAX_SPEECH_CHARS + 1), 2, "愤怒", private_strategy=rejected),
    ] * 3)
    accepted = PrivateStrategy(**STRATEGY)
    player.private_strategy = accepted
    player.last_suspect = 3
    assert player.day_speech([], 1, ALIVE) == (SAFE_SPEECH, None, "平静")
    assert player.private_strategy is accepted and player.last_suspect is None
    assert len(client.calls) == 3


# ---------------------------------------------------------------- 投票


def test_day_vote_success_and_history():
    player, client = make_player(replies=['{"vote": 4}'])
    target = player.day_vote([], 1, ALIVE)
    assert target == 3
    assert player.my_votes == [(1, 3)]
    assert "vote" in client.calls[0][-1]["content"]


def test_day_vote_abstain_and_degrade():
    player, _ = make_player(replies=['{"vote": 0}'])
    assert player.day_vote([], 1, ALIVE) is None
    assert player.my_votes == [(1, None)]

    player2, client2 = make_player(replies=["bad", "bad2", "bad3"])
    assert player2.day_vote([], 2, ALIVE) is None  # 降级弃票
    assert player2.my_votes == [(2, None)]
    assert len(client2.calls) == 3


def test_day_vote_dead_target_retry():
    player, client = make_player(replies=['{"vote": 6}', '{"vote": 1}'])
    target = player.day_vote([], 1, [0, 1, 2, 3])  # 5 号（座位6）已死
    assert target == 0
    assert "不在存活玩家中" in client.calls[1][-1]["content"]


def test_wolf_can_vote_for_known_teammate():
    player, _ = make_player(role=Role.WEREWOLF, teammate_seats=[1], replies=['{"vote": 2}'])
    assert player.day_vote([], 1, ALIVE) == 1


# ---------------------------------------------------------------- 技能决策


def test_wolf_den_decision():
    player, client = make_player(
        role=Role.WEREWOLF, replies=['{"target": 5}'], seat=2, teammate_seats=[1]
    )
    target = player.wolf_den_decision([], 1, [0, 3, 4, 5])
    assert target == 4
    # 狼窝 prompt 包含两狼座位
    content = client.calls[0][0]["content"]
    assert "狼队" in content


def test_wolf_den_degrade_to_first_killable():
    player, _ = make_player(
        role=Role.WEREWOLF, replies=["bad", "bad", "bad"], seat=2, teammate_seats=[1]
    )
    assert player.wolf_den_decision([], 1, [0, 3, 4]) == 0


def test_seer_decision_and_degrade():
    player, _ = make_player(role=Role.SEER, replies=['{"target": 3}'])
    assert player.seer_decision([], 1, [0, 1, 2, 3]) == 2

    player2, client2 = make_player(role=Role.SEER, replies=["x", "y", "z"])
    assert player2.seer_decision([], 1, [0, 1, 2, 3]) == 0  # 降级查首位
    assert len(client2.calls) == 3


def test_witch_decision_save():
    player, _ = make_player(role=Role.WITCH, replies=['{"save": true, "poison": 0}'])
    assert player.witch_decision([], 1, 4, True, True, ALIVE) == (True, None)


def test_witch_decision_poison():
    player, _ = make_player(role=Role.WITCH, replies=['{"save": false, "poison": 3}'])
    assert player.witch_decision([], 1, 4, True, True, ALIVE) == (False, 2)


def test_witch_decision_both_potions_invalid_retry():
    player, client = make_player(
        role=Role.WITCH,
        replies=['{"save": true, "poison": 3}', '{"save": true, "poison": 0}'],
    )
    use_save, poison = player.witch_decision([], 1, 4, True, True, ALIVE)
    assert (use_save, poison) == (True, None)
    assert "同一夜不得同时使用" in client.calls[1][-1]["content"]


def test_witch_decision_no_potion_left_degrades():
    player, _ = make_player(role=Role.WITCH, replies=['{"save": true, "poison": 3}'] * 3)
    assert player.witch_decision([], 1, 4, False, False, ALIVE) == (False, None)


def test_witch_poison_dead_target_retry():
    player, client = make_player(
        role=Role.WITCH, replies=['{"save": false, "poison": 5}', '{"save": false, "poison": 1}']
    )
    _, poison = player.witch_decision([], 1, 4, False, True, [0, 1, 2, 3])
    assert poison == 0
    assert "不在存活玩家中" in client.calls[1][-1]["content"]


# ---------------------------------------------------------------- mock 模式联测


def test_llmclient_mock_reply_priority():
    cfg = LLMConfig(base_url="https://x/v1", api_key="k", model="m")
    # mock_fn 优先于 mock_reply
    client = LLMClient(cfg, mock=True, mock_fn=lambda m: '{"vote": 1}')
    assert (
        client.chat([], mock_reply=lambda: '{"vote": 2}') == '{"vote": 1}'
    )
    # 无 mock_fn 时使用本次注入的 mock_reply
    client2 = LLMClient(cfg, mock=True)
    assert (
        client2.chat([], mock_reply=lambda: '{"vote": 2}') == '{"vote": 2}'
    )
    # 都没有时回退默认发言包
    assert '"speech"' in client2.chat([])


def test_mock_mode_produces_valid_votes():
    cfg = LLMConfig(base_url="mock://local", api_key="m", model="m")
    client = LLMClient(cfg, mock=True)
    player = AIPlayer(
        client, seat=0, name="甲", role=Role.VILLAGER,
        persona=Persona("甲", "冷静", "口癖"),
    )
    target = player.day_vote([], 1, [0, 1, 2])
    assert target == 1  # mock 合法投票：第一个非自己的存活座位
    assert client.usage.calls == 1  # 一次调用即成功，未走降级重试


def test_mock_mode_produces_valid_skill_decisions():
    cfg = LLMConfig(base_url="mock://local", api_key="m", model="m")
    client = LLMClient(cfg, mock=True)
    player = AIPlayer(
        client, seat=2, name="乙", role=Role.WEREWOLF,
        persona=Persona("乙", "冷静", "口癖"), teammate_seats=[4],
    )
    assert player.wolf_den_decision([], 1, [0, 1, 3]) == 0
    assert player.seer_decision([], 1, [0, 1, 3]) == 0
    save, poison = player.witch_decision([], 1, 0, True, True, [0, 1, 2])
    assert (save, poison) == (False, None)
    assert client.usage.calls == 3  # 各一次调用即成功


def test_llmclient_mock_with_scripted_fn():
    cfg = LLMConfig(base_url="https://x/v1", api_key="k", model="m")
    replies = ['{"vote": 2}', '{"vote": 3}']

    def mock_fn(messages):
        return replies.pop(0)

    client = LLMClient(cfg, mock=True, mock_fn=mock_fn)
    assert client.chat([{"role": "user", "content": "?"}]) == '{"vote": 2}'
    assert client.usage.calls == 1
    assert client.chat([{"role": "user", "content": "?"}]) == '{"vote": 3}'
