"""执行角色、阵营/神民分类；保留原六人牌堆与时序兼容入口。"""

from __future__ import annotations

import enum
import random


class Role(str, enum.Enum):
    """游戏角色。"""

    WEREWOLF = "狼人"
    SEER = "预言家"
    WITCH = "女巫"
    VILLAGER = "平民"
    HUNTER = "猎人"
    IDIOT = "白痴"
    GUARD = "守卫"


class Team(str, enum.Enum):
    """阵营。"""

    WOLF = "狼人阵营"
    GOOD = "好人阵营"


ROLE_TEAM: dict[Role, Team] = {
    Role.WEREWOLF: Team.WOLF,
    Role.SEER: Team.GOOD,
    Role.WITCH: Team.GOOD,
    Role.VILLAGER: Team.GOOD,
    Role.HUNTER: Team.GOOD,
    Role.IDIOT: Team.GOOD,
    Role.GUARD: Team.GOOD,
}

# 角色技能说明（角色卡展示用）
ROLE_DESCRIPTION: dict[Role, str] = {
    Role.WEREWOLF: (
        "每晚与存活狼队共同决定一个刀口。伪装好人、影响投票，按当局规则获胜。"
    ),
    Role.SEER: (
        "每晚查验一名玩家的阵营（狼人/好人）。合理跳身份带领好人放逐狼人，"
        "但过早暴露可能被狼人夜刀。"
    ),
    Role.WITCH: (
        "拥有一瓶解药与一瓶毒药，各限用一次且同一夜不得同时使用。"
        "按板子规定获知刀口和自救资格，可救刀口、毒一名玩家或不用药。"
    ),
    Role.VILLAGER: (
        "没有夜间技能，靠白天听发言、找逻辑矛盾，投票放逐狼人。"
    ),
    Role.HUNTER: "合法死亡后可开枪带走一名存活玩家或弃枪；中毒（包括混合死因）不能开枪。",
    Role.IDIOT: "首次被放逐翻牌免死并永久失票，此后不能再被放逐；仍可发言、计为神职，可被夜刀或毒杀。",
    Role.GUARD: "每夜可自守或空守，不能连续两夜守同一非空目标；守护只挡狼刀，同守同救致死，毒药不受保护。",
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


def is_god(role: Role) -> bool:
    return team_of(role) is Team.GOOD and role is not Role.VILLAGER


def deal_roles(seats: int | None = None, rng: random.Random | None = None, *, board=None) -> list[Role]:
    """完整洗牌；不允许截断牌堆。默认兼容六人局。"""
    rng = rng or random.Random()
    roles = list(board.setup if board is not None else SIX_PLAYER_SETUP)
    if seats is not None and seats != len(roles):
        raise ValueError(f"玩家数 {seats} 与板子角色数 {len(roles)} 不一致")
    rng.shuffle(roles)
    return roles
