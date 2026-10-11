"""所有行动共享完整历史，今日进度与往日声明独立呈现。"""

import pytest

from engine import events as ev
from engine.game import SpeechContext
from engine.roles import Role
from llm.agents import (
    Persona, PrivateStrategy, build_speech_prompt, build_vote_prompt, build_wolf_den_prompt,
    build_seer_prompt, build_witch_prompt,
)

PERSONA = Persona("学姐", "理性", "从逻辑上讲")
ALIVE = [0, 1, 2, 3, 5]


def speech_prompt(view, day=2, context=None):
    return build_speech_prompt(
        seat=5, name="学姐", role=Role.VILLAGER, persona=PERSONA,
        teammate_seats=None, view=view, day=day, alive_seats=ALIVE,
        my_votes=[(1, 1)], last_suspect=None, context=context,
    )


@pytest.mark.parametrize("builder,options", [
    (build_speech_prompt, {"seat": 5, "name": "学姐", "role": Role.SEER,
                          "teammate_seats": None, "alive_seats": ALIVE,
                          "my_votes": [], "last_suspect": None}),
    (build_vote_prompt, {"seat": 5, "name": "学姐", "role": Role.SEER,
                        "alive_seats": ALIVE, "my_votes": []}),
    (build_wolf_den_prompt, {"wolf_seats": [0, 5], "killable": [1, 2, 3]}),
    (build_seer_prompt, {"seat": 5, "checkable": [0, 1, 2, 3]}),
    (build_witch_prompt, {"seat": 5, "kill_target": 0, "can_save": True,
                         "can_poison": True, "alive_seats": ALIVE}),
])
def test_every_action_keeps_complete_visible_history(builder, options):
    history = [
        ev.role_assigned(0, 5, "预言家"),
        ev.speech(1, 1, "人类", 1, "我先保留判断。"),
        ev.speech(1, 4, "已出局AI", 2, "我今天怀疑 3 号。"),
        *[ev.speech(day, 5, "学姐", day + 2, f"我的第{day}次观点。") for day in range(1, 5)],
        ev.death(5, [4]),
        ev.vote_result(5, {1: 3, 2: 2}),
        ev.seer_check(5, 5, 0, True),
        ev.seer_check(5, 2, 3, False),
        ev.wolf_kill(5, [0, 1], 3),
        ev.potion_used(5, 3, "毒药", 2),
    ]
    view = ev.visible_events(history, 5)
    strategy = PrivateStrategy(assessment="只有真实查验才可信。", goal="隐藏真实身份保护自己。", claim="维持之前的假身份。")
    messages = builder(persona=PERSONA, view=view, day=6, private_strategy=strategy, **options)
    text = messages[0]["content"]
    for event in view:
        if event.kind is ev.EventKind.SPEECH:
            assert f"第 {event.day} 天 · 白天 · 玩家声明：[#{event.speech_index}] {event.seat + 1} 号" in text
            assert event.extra["content"] in text
    assert "角色私有事实：查验 1 号：[狼人]。" in text
    for hidden in history[-3:]:
        assert hidden.text not in text
    assert "系统公开事实：票型：2 号 3 票，3 号 2 票" in text
    assert "记录只证明该玩家说过这句话" in text
    assert "不能推断其他玩家的个人投票目标" in text
    assert strategy.model_dump_json() in text and "不是事实或本轮指令" in text
    assert "所有角色均可伪装身份或阵营" in text
    assert "不必每轮撒谎" in text and "编造的查验不能当作真实查验" in text
    # 开始另一局，新的视角没有旧局内容。
    fresh = builder(persona=PERSONA, view=[ev.role_assigned(0, 5, "预言家")], day=1, **options)
    assert "我的第1次观点" not in fresh[0]["content"]
    assert "已出局AI" not in fresh[0]["content"]
    assert strategy.claim not in fresh[0]["content"]


def test_first_speaker_has_actual_order_and_no_history():
    context = SpeechContext(1, 5, (5, 0, 1, 2, 3), 1, (), (0, 1, 2, 3),
                            tuple(enumerate(["人类", "阿飞", "老猫", "大树", "小蝶", "学姐"])))
    text = speech_prompt([ev.death(1, [4])], 1, context)[0]["content"]
    assert "引擎当日实际顺序：6 号、1 号、2 号、3 号、4 号" in text
    assert "当日已完成正式发言：（无）" in text
    assert "当前发言人之后尚未轮到：1 号、2 号、3 号、4 号" in text
    assert "整局尚无公开发言记录" in text
    assert "正常轮候不是故意沉默的证据" in text


def test_day_two_progress_does_not_erase_yesterday_or_personal_votes():
    context = SpeechContext(2, 5, (5, 0, 1, 2, 3), 1, (), (0, 1, 2, 3), ())
    view = [ev.speech(1, 1, "人类", 3, "我暂时怀疑 4 号。"), ev.vote_result(1, {3: 3})]
    text = speech_prompt(view, 2, context)[0]["content"]
    assert "第 1 天 · 白天 · 玩家声明：[#3] 2 号 人类：我暂时怀疑 4 号。" in text
    assert "当日已完成正式发言：（无）" in text
    assert "今天尚未发言不等于整局从未发言" in text
    assert "第1天投给2 号" in text


def test_direct_call_does_not_invent_unknown_speaking_order():
    view = [ev.speech(1, 1, "阿飞", 1, "保留判断。"), ev.speech(2, 0, "人类", 2, "再听听。")]
    text = speech_prompt(view)[0]["content"]
    assert "当日完整顺序与位置未知" in text
    assert "当日已完成正式发言：1 号" in text
    assert "引擎当日实际顺序" not in text
