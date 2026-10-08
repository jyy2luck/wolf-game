"""角色定义与 6 人局配置：角色枚举、阵营、技能说明、固定配置与技能时序。

6 人局固定配置（无警长）：狼人 ×2、预言家 ×1、女巫 ×1、平民 ×2。
"""

from __future__ import annotations

import enum
import random


class Role(str, enum.Enum):
    """游戏角色。"""

    WEREWOLF = "狼人"
    SEER = "预言家"
    WITCH = "女巫"
    VILLAGER = "平民"


class Team(str, enum.Enum):
    """阵营。"""

    WOLF = "狼人阵营"
    GOOD = "好人阵营"


ROLE_TEAM: dict[Role, Team] = {
    Role.WEREWOLF: Team.WOLF,
    Role.SEER: Team.GOOD,
    Role.WITCH: Team.GOOD,
    Role.VILLAGER: Team.GOOD,
}

# 角色技能说明（角色卡展示用）
ROLE_DESCRIPTION: dict[Role, str] = {
    Role.WEREWOLF: (
        "每晚与另一名狼人共同猎杀一名玩家。伪装好人、误导投票，"
        "狼人数量不少于好人时获胜。切勿暴露队友与刀口。"
    ),
    Role.SEER: (
        "每晚查验一名玩家的阵营（狼人/好人）。合理跳身份带领好人放逐狼人，"
        "但过早暴露可能被狼人夜刀。"
    ),
    Role.WITCH: (
        "拥有一瓶解药与一瓶毒药，各限用一次且同一夜不得同时使用。"
        "每晚得知刀口，可救刀口、毒一名玩家或不用药。"
    ),
    Role.VILLAGER: (
        "没有夜间技能，靠白天听发言、找逻辑矛盾，投票放逐狼人。"
    ),
}

# 夜晚技能唤醒时序（引擎按此顺序结算）
NIGHT_ORDER: tuple[Role, ...] = (Role.WEREWOLF, Role.SEER, Role.WITCH)

# 6 人局固定角色配置
SIX_PLAYER_SETUP: tuple[Role, ...] = (
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.SEER,
    Role.WITCH,
    Role.VILLAGER,
    Role.VILLAGER,
)


def team_of(role: Role) -> Team:
    return ROLE_TEAM[role]


def deal_roles(seats: int = 6, rng: random.Random | None = None) -> list[Role]:
    """把 6 人局角色随机分配到座位。返回 index=座位号(0-based) 的角色列表。"""
    rng = rng or random.Random()
    roles = list(SIX_PLAYER_SETUP)
    rng.shuffle(roles)
    return roles[:seats] if seats <= len(roles) else roles