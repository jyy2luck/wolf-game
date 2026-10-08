"""engine/roles 单元测试：角色配置与随机分配。"""

from collections import Counter

from engine.roles import (
    NIGHT_ORDER,
    SIX_PLAYER_SETUP,
    Role,
    Team,
    deal_roles,
    team_of,
)


def test_fixed_setup_composition():
    counts = Counter(SIX_PLAYER_SETUP)
    assert counts[Role.WEREWOLF] == 2
    assert counts[Role.SEER] == 1
    assert counts[Role.WITCH] == 1
    assert counts[Role.VILLAGER] == 2
    assert sum(counts.values()) == 6


def test_deal_roles_always_valid_composition():
    for seed in range(50):
        roles = deal_roles(rng=__import__("random").Random(seed))
        assert Counter(roles) == Counter(SIX_PLAYER_SETUP)


def test_deal_roles_is_shuffled():
    results = set()
    for seed in range(100):
        results.add(tuple(deal_roles(rng=__import__("random").Random(seed))))
    assert len(results) > 1  # 洗牌生效


def test_teams():
    assert team_of(Role.WEREWOLF) is Team.WOLF
    for role in (Role.SEER, Role.WITCH, Role.VILLAGER):
        assert team_of(role) is Team.GOOD


def test_night_order():
    # 夜晚按 狼刀 → 预言家 → 女巫 的顺序唤醒
    assert NIGHT_ORDER == (Role.WEREWOLF, Role.SEER, Role.WITCH)