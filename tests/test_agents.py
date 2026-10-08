"""llm/agents 单元测试：mock LLM 返回，覆盖解析成功/重试/降级、穿帮检出/兜底。"""

import json

import pytest

from engine.roles import Role
from llm.agents import (
    PERSONA_POOL,
    SAFE_SPEECH,
    AIPlayer,
    Persona,
    VotePack,
    WitchPack,
    assign_personas,
    detect_leak,
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

    def chat(self, messages, **kwargs):  # noqa: ANN002, ANN003
        self.calls.append(messages)
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


def speech_json(speech: str, suspect: int = 0, mood: str = "平静") -> str:
    return json.dumps(
        {"speech": speech, "suspect": suspect, "mood": mood}, ensure_ascii=False
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


def test_detect_leak_matrix():
    assert detect_leak("我觉得我的队友有问题", Role.VILLAGER) is not None
    assert detect_leak("我们狼人要赢了", Role.SEER) is not None
    assert detect_leak("今晚刀口是他", Role.WITCH) is not None  # 女巫也不能提刀口
    assert detect_leak("我验过3号是好人", Role.SEER) is None
    assert detect_leak("我验过3号是好人", Role.VILLAGER) is not None
    assert detect_leak("我救了4号", Role.WITCH) is None
    assert detect_leak("我救了4号", Role.SEER) is not None
    assert detect_leak("大家听听我的分析", Role.WEREWOLF) is None


# ---------------------------------------------------------------- 发言


def test_day_speech_success():
    player, client = make_player(
        replies=[speech_json("我觉得 [#1] 有点问题，按理说首发发言最激进。", 3, "冷静")]
    )
    speech, suspect, mood = player.day_speech([], 1, ALIVE)
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


def test_day_speech_leak_retry_then_clean():
    player, client = make_player(
        role=Role.WEREWOLF,
        replies=[
            speech_json("我和我的队友昨晚聊过，我觉得3号有问题", 3),
            speech_json("我觉得3号的发言太急了", 3),
        ],
    )
    speech, _, _ = player.day_speech([], 1, ALIVE)
    assert speech == "我觉得3号的发言太急了"
    assert player.leak_count == 0
    assert len(client.calls) == 2
    assert "警告" in client.calls[1][-1]["content"]  # 重试携带穿帮警告


def test_day_speech_double_leak_fallback():
    player, client = make_player(
        role=Role.WEREWOLF,
        replies=[
            speech_json("我作为狼人觉得2号有问题", 2),
            speech_json("我们狼人先刀谁？不对，我是说2号有问题", 2),
        ],
    )
    speech, _, _ = player.day_speech([], 1, ALIVE)
    assert speech == SAFE_SPEECH
    assert player.leak_count == 1


def test_day_speech_oversized_speech_truncated():
    long_speech = "字" * 300
    player, _ = make_player(replies=[speech_json(long_speech, 0)])
    speech, _, _ = player.day_speech([], 1, ALIVE)
    assert len(speech) == 150


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