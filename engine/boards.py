"""不可变、带版本的公开规则；不包含任何对局私有状态。"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from engine.roles import Role, ROLE_DESCRIPTION, is_god


@dataclass(frozen=True)
class Board:
    id: str
    name: str
    role_counts: tuple[tuple[Role, int], ...]
    version: str = "1"
    victory: str = "parity"
    sheriff: bool = False
    wolf_friendly_fire: bool = False
    empty_kill: bool = False
    witch_self_save: str = "always"
    knife_after_save: bool = True
    guard_double_save_kills: bool = False
    last_words: bool = False
    exile_pk: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if type(self.role_counts) is not tuple or any(type(row) is not tuple for row in self.role_counts):
            raise ValueError("角色数量必须使用不可变元组")
        if not self.id or not self.version or self.victory not in {"parity", "edge"}:
            raise ValueError("板子编号、版本或胜利规则无效")
        counts = dict(self.role_counts)
        if len(counts) != len(self.role_counts) or any(
            not isinstance(r, Role) or type(n) is not int or n < 1 for r, n in self.role_counts
        ):
            raise ValueError("角色配置重复、未支持或数量无效")
        if counts.get(Role.WEREWOLF, 0) < 1 or counts.get(Role.VILLAGER, 0) < 1:
            raise ValueError("板子需要狼人和平民")
        if any(counts.get(r, 0) > 1 for r in Role if is_god(r)) or not any(is_god(r) for r in counts):
            raise ValueError("本版本神职必须互异且至少一名")
        if self.player_count > 12 or self.witch_self_save not in {"always", "first_night"}:
            raise ValueError("未支持的人数或女巫规则")
        if self.guard_double_save_kills and Role.GUARD not in counts:
            raise ValueError("同守同救需要守卫")

    @property
    def setup(self) -> tuple[Role, ...]:
        return tuple(r for r, n in self.role_counts for _ in range(n))

    @property
    def player_count(self) -> int:
        return sum(n for _, n in self.role_counts)

    @property
    def capabilities(self) -> frozenset[str]:
        caps = {"stable_alignment_checks", "witch_save", "witch_one_of_each", "witch_acts_before_dawn",
                "repeat_daily_speech", "individual_votes"}
        caps.add("stable_two_team_parity" if self.victory == "parity" else "edge_victory")
        if not self.wolf_friendly_fire:
            caps.add("wolves_kill_only_good")
        if not self.empty_kill and not self.wolf_friendly_fire and Role.GUARD not in self.setup:
            caps.add("single_night_death_good")
        if Role.GUARD not in self.setup:
            caps.add("no_other_night_protection")
        else:
            caps.update({"guard", "guard_no_repeat", "double_protection_kills"})
        if Role.HUNTER not in self.setup:
            caps.add("no_death_counteraction")
        else:
            caps.update({"hunter_shot", "hunter_optional_shot", "poison_blocks_shot"})
        if Role.IDIOT in self.setup:
            caps.add("idiot_reveal")
        if self.witch_self_save == "always":
            caps.add("witch_self_save")
        else:
            caps.add("witch_first_night_self_save")
        if self.empty_kill:
            caps.add("empty_kill")
        if self.wolf_friendly_fire:
            caps.add("wolf_friendly_fire")
        if self.sheriff:
            caps.add("sheriff")
        if self.last_words:
            caps.add("last_words")
        return frozenset(caps)

    def public_projection(self) -> dict:
        return {"id": self.id, "name": self.name, "version": self.version,
                "player_count": self.player_count,
                "role_counts": {r.value: n for r, n in self.role_counts},
                "gods": [r.value for r, _ in self.role_counts if is_god(r)],
                "victory": self.victory, "sheriff": self.sheriff,
                "capabilities": sorted(self.capabilities), "rules": self.rules_text(),
                "wolf_rules": {"friendly_fire": self.wolf_friendly_fire, "empty_kill": self.empty_kill},
                "witch_rules": {"self_save": self.witch_self_save, "knife_after_save": self.knife_after_save, "one_potion_per_night": True},
                "guard_rules": {"enabled": Role.GUARD in self.setup, "no_repeat": True, "double_save_kills": self.guard_double_save_kills},
                "hunter_rules": {"enabled": Role.HUNTER in self.setup, "poison_blocks_shot": True, "settle_before_victory": True},
                "idiot_rules": {"enabled": Role.IDIOT in self.setup, "first_exile_immune": True, "loses_vote": True},
                "flow": {"night_order": [r.value for r in (Role.GUARD, Role.WEREWOLF, Role.SEER, Role.WITCH) if r in self.setup],
                         "last_words": self.last_words, "exile_pk": self.exile_pk, "speech_rounds": 2,
                         "sheriff_vote_units": 3, "normal_vote_units": 2}}

    def goal(self, role: Role) -> str:
        if role is not Role.WEREWOLF:
            return "狼人全部出局则好人获胜；按真实技能与信息争取同阵营支持。"
        return ("仍有狼人时，平民或神职全部出局则狼人获胜；狼全灭优先判好人胜。"
                if self.victory == "edge" else "存活狼数不少于好人数时狼人获胜。")

    def rules_text(self) -> str:
        lines = [f"# 本局固定规则：{self.id} · {self.name} · v{self.version}",
                 f"{self.player_count} 人局：" + "、".join(f"{r.value}{n}名" for r, n in self.role_counts) + "。",
                 ("屠边胜利：狼全灭优先判好人胜；否则平民全部出局或神职全部出局则狼人胜，狼好人数相等本身不终局。" if self.victory == "edge" else "狼全灭优先判好人胜；存活狼数不少于好人数则狼人胜。"),
                 "狼人每晚统一刀一人；" + ("允许自刀、刀队友和空刀。" if self.empty_kill else "只能刀好人，不能空刀。"),
                 "预言家每晚私下查验阵营；女巫解药/毒药各一次，同夜不能双药。",
                 ("女巫仅首夜可自救，解药耗尽后不再获得新刀口，空刀夜仍可用毒。" if not self.knife_after_save else "女巫可自救，每晚得知刀口，包括解药耗尽后。")]
        for r in (Role.HUNTER, Role.IDIOT, Role.GUARD):
            if r in self.setup:
                lines.append(ROLE_DESCRIPTION[r])
        lines.extend(["夜初存活角色完成当夜行动后，统一公布死亡；所有枪、遗言、警徽处置结束后才判胜。",
                      "死亡和放逐不翻牌、不公开死因；真实开枪、白痴翻牌才揭示身份；夜死名单按座位排序。",
                      "首夜死亡与放逐死者各一次遗言，猎人先枪后遗言；后续夜死和枪击目标无遗言。" if self.last_words else "本局无遗言。",
                      "白天首轮发言及一轮回应后统一投票，全部选票收齐才公布个人票型。",
                      "正票最高者并列进行一次PK，候选不投票；再平或全弃票无人出局。" if self.exile_pk else "最高票平票或全弃票无人出局。"])
        if self.sheriff:
            lines.append("首夜死讯前竞选警长，待死亡者可参选；报名统一公示、发言后退水。仅最初警下且实际存活者的票有效，明细死讯后披露；单候选直接当选，一次PK仍平则无警长。警长选择顺/逆时针、两轮末置发言，放逐1.5票，其他有票者1票；死亡后可移交给存活有票者或撕徽，翻牌白痴自动撕徽；不重新竞选。")
        return "\n".join(lines)


SIX_CLASSIC = Board("six-classic", "原六人局", ((Role.WEREWOLF, 2), (Role.SEER, 1), (Role.WITCH, 1), (Role.VILLAGER, 2)))
_NEW = dict(victory="edge", wolf_friendly_fire=True, empty_kill=True,
            witch_self_save="first_night", knife_after_save=False, last_words=True, exile_pk=True)
NINE_STANDARD = Board("nine-standard", "九人预女猎", ((Role.WEREWOLF, 3), (Role.SEER, 1), (Role.WITCH, 1), (Role.HUNTER, 1), (Role.VILLAGER, 3)), **_NEW)
TWELVE_STANDARD = Board("twelve-standard", "十二人预女猎白", ((Role.WEREWOLF, 4), (Role.SEER, 1), (Role.WITCH, 1), (Role.HUNTER, 1), (Role.IDIOT, 1), (Role.VILLAGER, 4)), sheriff=True, **_NEW)
TWELVE_GUARD = Board("twelve-guard", "十二人预女猎守", ((Role.WEREWOLF, 4), (Role.SEER, 1), (Role.WITCH, 1), (Role.HUNTER, 1), (Role.GUARD, 1), (Role.VILLAGER, 4)), sheriff=True, guard_double_save_kills=True, **_NEW)
BOARDS = MappingProxyType({b.id: b for b in (SIX_CLASSIC, NINE_STANDARD, TWELVE_STANDARD, TWELVE_GUARD)})


def get_board(name: str = "six-classic") -> Board:
    try:
        return BOARDS[name]
    except KeyError:
        raise ValueError(f"未知板子 {name}，可用：{', '.join(BOARDS)}") from None
