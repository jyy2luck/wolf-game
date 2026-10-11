"""策略检索的规则适配、实时资源变化、未知信息与五个请求入口。"""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from engine import events as ev
from engine.roles import Role
from llm import strategies
from llm.agents import (
    AIPlayer, Persona, build_seer_prompt, build_speech_prompt, build_vote_prompt,
    build_witch_prompt, build_wolf_den_prompt,
)
from llm.strategies import (
    CURRENT_RULES, MAX_CARDS, MAX_REFERENCE_CHARS, RuleProfile, StrategyLibrary,
    get_strategy_advice, load_strategy_library,
)

PERSONA = Persona("测试", "谨慎", "慢慢说")
DESIGN = Path(__file__).resolve().parents[1] / "docs/ai-strategy-library-design.md"


def advice(role=Role.VILLAGER, action="vote", alive=None, view=None, **kwargs):
    return get_strategy_advice(
        role=role, seat=0, action=action, alive_seats=alive or [0, 1, 2, 3, 4],
        view=view or [], day=kwargs.pop("day", 2), **kwargs,
    )


def ids(result):
    return {card.id for card in result.cards}


def test_library_covers_reviewed_cards_and_ten_complete_population_guides():
    import re

    library = load_strategy_library()
    assert library is not None
    documented_ids = set(re.findall(r"^\| ([A-Z]{1,2}\d{2}) \|", DESIGN.read_text(encoding="utf-8"), re.M))
    assert {card.id for card in library.cards} == documented_ids
    assert len(library.cards) == 67
    assert len(library.roles) == 10
    for guides in library.roles.values():
        assert len(guides) == 7
        for count in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 20, 100]:
            assert sum(row.matches(count) for row in guides) == 1


@pytest.mark.parametrize("mutation", ["duplicate", "missing_base", "overlap", "bad_role"])
def test_malformed_catalog_is_rejected(mutation):
    data = load_strategy_library().model_dump(mode="json")
    if mutation == "duplicate":
        data["cards"].append(data["cards"][0])
    elif mutation == "missing_base":
        data["cards"] = [card for card in data["cards"] if card["id"] != "C01"]
    elif mutation == "overlap":
        data["roles"]["平民"][0]["min_alive"] = 0
    else:
        data["cards"][0]["role"] = "未定义身份"
    with pytest.raises(ValidationError):
        StrategyLibrary.model_validate(data)


def test_unknown_wolf_count_stays_a_branch_and_claims_do_not_confirm_identity():
    result = advice(view=[ev.speech(2, 1, "假预言家", 1, "我验过3号是狼，4号是好人。")])
    assert result.situation.wolf_counts == (1, 2)
    assert result.situation.known_wolves == frozenset()
    assert result.situation.known_good == frozenset({0})
    assert "1狼4好人 / 2狼3好人" in result.text
    assert "部分候选" in result.text


def test_first_single_night_death_and_continuing_exile_allow_rule_deduction():
    view = [ev.death(1, [5])]
    before = advice(alive=[0, 1, 2, 3, 4], view=view)
    assert before.situation.wolf_counts == (2,)
    assert "全部候选" in before.text
    after = advice(alive=[0, 1, 2, 3], view=view + [ev.exile(1, 4)])
    assert after.situation.wolf_counts == (1,)
    assert "1狼3好人" in after.text


def test_double_death_does_not_assign_death_causes_by_list_order():
    result = advice(alive=[0, 1, 2, 3], view=[ev.death(1, [4, 5])])
    assert result.situation.known_good == frozenset({0})
    assert result.situation.wolf_counts == (1,)


def test_private_results_only_reach_their_legal_owner_and_true_role():
    events = [ev.seer_check(1, 0, 1, True), ev.seer_check(1, 3, 2, False),
              ev.witch_knew(1, 4, 2), ev.potion_used(1, 4, "毒药", 3)]
    result = advice(role=Role.SEER, view=events)
    assert result.situation.known_wolves == frozenset({1})
    assert 2 not in result.situation.known_good
    assert "S03" in ids(result)
    ordinary = advice(role=Role.VILLAGER, view=events)
    assert ordinary.situation.known_wolves == frozenset()
    assert ordinary.situation.known_good == frozenset({0})


def test_real_seer_information_changes_cards_and_candidate_wolf_counts():
    initial = advice(role=Role.SEER, action="speech")
    checked = advice(role=Role.SEER, action="speech", view=[ev.seer_check(1, 0, 1, True)])
    assert "S03" not in ids(initial)
    assert "S03" in ids(checked)
    assert checked.cards[2].id == "S03"
    assert checked.situation.known_wolves == frozenset({1})


def test_witch_uses_actual_resources_and_knife_not_claimed_potions():
    result = advice(role=Role.WITCH, action="witch_act", alive=[0, 1, 2],
                    kill_target=1, can_save=False, can_poison=True,
                    view=[ev.speech(1, 2, "玩家", 1, "我是女巫，解药还在。")])
    assert result.situation.known_good == frozenset({0, 1})
    assert result.situation.wolf_counts == (1,)
    assert "W04" in ids(result)
    assert "W01" not in ids(result) and "W02" not in ids(result)
    assert result.cards[2].id == "W04"


def test_witch_can_compare_saving_and_poisoning_in_three_player_night():
    result = advice(role=Role.WITCH, action="witch_act", alive=[0, 1, 2],
                    kill_target=0, can_save=True, can_poison=True)
    assert "W02" in ids(result)
    assert "W04" not in ids(result)
    assert "W07" in ids(result)


def test_first_night_save_card_and_exhausted_poison_filter():
    first = advice(role=Role.WITCH, action="witch_act", alive=list(range(6)), day=1,
                   kill_target=1, can_save=True, can_poison=True)
    assert "W01" in ids(first)
    assert "W04" not in ids(first)
    exhausted = advice(role=Role.WITCH, action="witch_act", alive=[0, 1, 2],
                       kill_target=1, can_save=False, can_poison=False)
    assert not {"W01", "W02", "W03", "W04", "W07"} & ids(exhausted)


def test_witch_speech_resources_can_be_derived_from_own_potion_history():
    result = advice(role=Role.WITCH, action="speech",
                    view=[ev.potion_used(1, 0, "解药", 1), ev.witch_knew(1, 0, 1)])
    assert "antidote_unavailable" in result.situation.flags
    assert "poison_available" in result.situation.flags
    assert 1 in result.situation.known_good


def test_allowing_self_kill_prevents_knife_good_certification():
    rules = replace(CURRENT_RULES, capabilities=CURRENT_RULES.capabilities - {"wolves_kill_only_good", "single_night_death_good"})
    result = advice(role=Role.WITCH, action="witch_act", rules=rules, kill_target=1,
                    can_save=False, can_poison=True, view=[ev.witch_knew(2, 0, 1)])
    assert 1 not in result.situation.known_good
    assert "W04" not in ids(result)


def test_custom_victory_rules_and_pending_shot_do_not_use_parity_bounds():
    rules = replace(CURRENT_RULES, capabilities=CURRENT_RULES.capabilities - {"stable_two_team_parity"})
    result = advice(alive=[0, 1, 2], rules=rules)
    assert not result.situation.wolf_counts
    assert "1狼2好人" not in result.text
    assert "普通人数残局表" in result.text
    pending = advice(role=Role.WITCH, action="witch_act", alive=[0, 1, 2],
                     can_save=False, can_poison=True, kill_target=1, settled=False)
    assert "W04" not in ids(pending)
    assert "普通人数残局表" in pending.text


def test_missing_initial_wolf_count_can_still_use_public_parity_constraints():
    result = advice(alive=[0, 1, 2], rules=replace(CURRENT_RULES, initial_wolves=None))
    assert result.situation.wolf_counts == (1,)


def test_contradictory_private_records_remain_unknown_instead_of_inventing_board():
    result = advice(role=Role.SEER, view=[ev.seer_check(1, 0, 1, True), ev.seer_check(2, 0, 1, False)])
    assert not result.situation.wolf_counts
    assert 1 not in result.situation.known_good
    assert 1 not in result.situation.known_wolves


@pytest.mark.parametrize("role,action", [
    (Role.VILLAGER, "speech"), (Role.VILLAGER, "vote"),
    (Role.WEREWOLF, "speech"), (Role.WEREWOLF, "wolf_kill"),
    (Role.SEER, "speech"), (Role.SEER, "seer_check"),
    (Role.WITCH, "speech"), (Role.WITCH, "witch_act"),
])
def test_references_are_bounded_and_match_true_role_rules_and_action(role, action):
    result = advice(role=role, action=action)
    assert len(result.cards) <= MAX_CARDS
    assert len(result.text) <= MAX_REFERENCE_CHARS
    assert {"C01", "C05"} <= ids(result)
    for card in result.cards:
        assert card.role in {"*", role.value}
        assert action in card.actions
        assert card.required_rules <= CURRENT_RULES.capabilities
        assert card.required_flags <= result.situation.flags
    assert "V06" not in ids(result)
    assert "S07" not in ids(result)
    assert "L10" not in ids(result)


def test_future_hunter_profile_gets_hunter_cards_only_with_explicit_capabilities():
    profile = RuleProfile("twelve-hunter", 12, 4, frozenset({"平民", "狼人", "猎人", "预言家", "女巫"}),
                          frozenset({"stable_two_team_parity", "hunter_shot", "hunter_optional_shot"}))
    result = advice(role="猎人", action="hunter_shot", rules=profile, alive=list(range(8)))
    assert {"H03", "H04"} <= ids(result)
    assert result.situation.wolf_counts == (1, 2, 3)
    assert "GU01" not in ids(result)
    disabled = advice(role="猎人", action="hunter_shot")
    assert all(card.role == "*" for card in disabled.cards)


def test_future_rules_with_death_counteraction_disable_forced_poison():
    profile = replace(CURRENT_RULES, roles=CURRENT_RULES.roles | {"猎人"},
                      capabilities=CURRENT_RULES.capabilities - {"no_death_counteraction", "single_night_death_good"})
    result = advice(role=Role.WITCH, action="witch_act", rules=profile, alive=[0, 1, 2],
                    can_save=False, can_poison=True, kill_target=1)
    assert "W04" not in ids(result)


@pytest.mark.parametrize("payload", [None, "not json", '{"version": 999}'])
def test_missing_or_invalid_package_data_falls_back_without_breaking_request(monkeypatch, payload):
    class Resource:
        def joinpath(self, name):
            return self

        def read_text(self, **kwargs):
            if payload is None:
                raise FileNotFoundError("missing")
            return payload

    load_strategy_library.cache_clear()
    monkeypatch.setattr(strategies, "files", lambda package: Resource())
    try:
        result = advice()
        assert result.cards == ()
        assert "策略库暂不可用" in result.text
        prompt = build_vote_prompt(seat=0, name="测试", role=Role.VILLAGER, persona=PERSONA,
                                   view=[], day=2, alive_seats=[0, 1, 2], my_votes=[])
        assert "策略库暂不可用" in prompt[0]["content"]
        assert '"vote": 0' in prompt[1]["content"]
    finally:
        load_strategy_library.cache_clear()


@pytest.mark.parametrize("builder,kwargs", [
    (build_speech_prompt, dict(seat=0, name="测试", role=Role.VILLAGER, teammate_seats=None,
                              alive_seats=[0, 1, 2, 3], my_votes=[], last_suspect=None)),
    (build_vote_prompt, dict(seat=0, name="测试", role=Role.VILLAGER, alive_seats=[0, 1, 2, 3], my_votes=[])),
    (build_seer_prompt, dict(seat=0, checkable=[1, 2, 3])),
    (build_witch_prompt, dict(seat=0, kill_target=1, can_save=False, can_poison=True, alive_seats=[0, 1, 2])),
    (build_wolf_den_prompt, dict(wolf_seats=[0, 5], seat=0, killable=[1, 2, 3])),
])
def test_every_prompt_includes_current_strategy_reference(builder, kwargs):
    prompt = builder(persona=PERSONA, view=[ev.exile(1, 5)], day=2, **kwargs)
    text = prompt[0]["content"]
    assert text.count("# 本轮策略库参考（不公开）") == 1
    assert "[C01]" in text and "[C05]" in text
    assert "自己的骗局和他人声明不变成真实信息" in text


def test_wolf_night_does_not_count_dead_teammate_as_alive():
    text = build_wolf_den_prompt(wolf_seats=[0, 5], seat=0, persona=PERSONA,
                                 view=[ev.exile(1, 5)], day=2, killable=[1, 2, 3])[0]["content"]
    assert "当前存活 4 人" in text
    assert "1狼3好人" in text
    assert "存活狼队成员：1 号。" in text
    assert "确认的存活狼人：1号。" in text


def test_real_player_requests_refresh_strategy_after_resource_and_population_change():
    class Client:
        def __init__(self):
            self.messages = []

        def chat(self, messages, **kwargs):
            self.messages.append(messages)
            return '{"save": false, "poison": 0}'

    client = Client()
    player = AIPlayer(client, seat=0, name="测试", role=Role.WITCH, persona=PERSONA)
    player.witch_decision([], 1, 1, True, True, list(range(6)))
    player.witch_decision([ev.potion_used(1, 0, "解药", 1)], 2, 1, False, True, [0, 1, 2])
    assert len(client.messages) == 2  # 每个行动仍只有原有的一次调用。
    first, second = (messages[0]["content"] for messages in client.messages)
    assert "[W01]" in first and "[W04]" not in first
    assert "[W04]" in second and "[W01]" not in second
    assert "当前存活 3 人" in second
