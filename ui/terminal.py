"""终端 UI：rich 渲染（角色卡/对局面板/私有信息/流式发言/终局）与玩家输入。

- 渲染一律基于事件流：`render_new` 增量渲染人类玩家视角内的新事件
  （公开事件 + 属于自己的私有事件），与 AI 共用引擎的视角过滤机制
- 玩家输入做受限选项校验，非法输入拒绝重输
- AI 发言通过 Live 逐字流式渲染，并显示「打字中」状态
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from engine import events as ev
from engine.events import Event, EventKind
from engine.game import Game
from engine.roles import ROLE_DESCRIPTION, Role, Team, team_of

console = Console()


class TerminalUI:
    """人类玩家的终端界面。"""

    def __init__(self, game: Game, my_seat: int) -> None:
        self.game = game
        self.my_seat = my_seat
        self._rendered = 0  # 已渲染的人类视角事件数（增量渲染）

    # ------------------------------------------------------------ 开局（4.1）

    def show_opening(self) -> None:
        game = self.game
        me = game.player(self.my_seat)
        console.clear()
        console.print(
            Panel(
                f"[bold]{me.name}[/bold] · 座位 {me.seat + 1}\n\n"
                f"角色：[bold red]{me.role.value}[/bold red]（{'狼人阵营' if team_of(me.role) is Team.WOLF else '好人阵营'}）\n"
                f"{ROLE_DESCRIPTION[me.role]}\n\n"
                "白天按座位顺序发言，全员投票放逐；夜晚狼人猎杀。\n"
                "狼人全部出局 → 好人胜；狼人数量不少于好人 → 狼人胜。",
                title="🐺 你的角色卡",
                border_style="red" if team_of(me.role) is Team.WOLF else "cyan",
            )
        )
        self.show_seat_table()
        # 开局私有信息（狼人队友）
        for e in game.view(self.my_seat):
            if not e.public:
                self._print_private_event(e)
        console.print("\n[dim]游戏即将开始……[/dim]")

    def show_seat_table(self) -> None:
        game = self.game
        table = Table(title="座位表", expand=False)
        table.add_column("座位", justify="center")
        table.add_column("名字")
        table.add_column("状态", justify="center")
        for p in game.players:
            status = "[green]存活[/green]" if p.alive else "[red]出局[/red]"
            mark = " ← 你" if p.seat == self.my_seat else ""
            table.add_row(str(p.seat + 1), p.name + mark, status)
        console.print(table)

    # ------------------------------------------------------------ 事件渲染（4.2 / 4.3）

    def render_new(self) -> None:
        """增量渲染人类视角内尚未展示的事件（公开 + 自己的私有）。"""
        view = self.game.view(self.my_seat)
        for e in view[self._rendered:]:
            self._print_event(e)
        self._rendered = len(view)

    def _print_event(self, e: Event) -> None:
        if not e.public:
            self._print_private_event(e)
            return
        style = {
            EventKind.NIGHT_FALLS: "dim magenta",
            EventKind.DAWN: "yellow",
            EventKind.DEATH: "bold red",
            EventKind.PEACEFUL_NIGHT: "green",
            EventKind.SPEECH: "white",
            EventKind.VOTE_RESULT: "cyan",
            EventKind.EXILE: "bold red",
            EventKind.GAME_OVER: "bold yellow",
        }.get(e.kind, "white")
        console.print(e.text, style=style)

    def _print_private_event(self, e: Event) -> None:
        console.print(f"🔒（私有）{e.text}", style="magenta")

    def phase_banner(self, day: int, phase: str) -> None:
        label = f"第 {day} 夜" if phase == "night" else f"第 {day} 天 · 白天"
        console.rule(f"[bold]{label}")

    def show_alive_update(self) -> None:
        alive = "、".join(
            f"{p.seat + 1}号 {p.name}"
            for p in self.game.players if p.alive
        )
        console.print(f"[dim]存活：{alive}[/dim]")

    # ------------------------------------------------------------ 玩家输入（4.4）

    def prompt_speech(self) -> str:
        """白天多行自由发言；空行结束，内容不能为空。"""
        console.print("[bold]轮到你发言[/bold]（可多行输入，单独回车结束）：")
        lines: list[str] = []
        while True:
            prompt = "> " if not lines else "… "
            line = console.input(prompt).strip()
            if line:
                lines.append(line)
            elif lines:
                break
            else:
                console.print("[red]发言不能为空，请重新输入。[/red]")
        return "\n".join(lines)

    def _ask_seat_choice(
        self, candidates: list[int], question: str, allow_abstain: bool = False
    ) -> int | None:
        """受限选项输入：仅接受合法座位号（0-based 输入显示），非法拒绝重输。"""
        opts = "、".join(f"{s + 1}" for s in candidates)
        abstain = "（0 = 弃票）" if allow_abstain else ""
        while True:
            raw = console.input(f"{question}[bold]（{opts}）{abstain}：[/bold]").strip()
            if raw.isdigit():
                val = int(raw)
                if allow_abstain and val == 0:
                    return None
                if (val - 1) in candidates:
                    return val - 1
            console.print("[red]非法输入，请重输。[/red]")

    def prompt_wolf_kill(
        self, killable: list[int], teammate_seats: list[int]
    ) -> int | None:
        """狼人刀口：由玩家确认，或委托 AI 队友决策（返回 None 表示委托）。

        teammate_seats 应只含存活队友；为空时不提供委托选项。
        """
        if teammate_seats:
            mates = "、".join(
                f"{s + 1} 号 {self.game.player(s).name}" for s in teammate_seats
            )
            console.print(f"🔒 你的狼人队友：{mates}", style="magenta")
        console.print("可刀目标：" + "、".join(f"{s + 1} 号" for s in killable))
        delegate_hint = "，或输入 [bold]d[/bold] 委托队友决策" if teammate_seats else ""
        while True:
            raw = console.input(f"输入刀口座位号{delegate_hint}：").strip()
            if raw.lower() == "d" and teammate_seats:
                return None
            if raw.isdigit() and (int(raw) - 1) in killable:
                return int(raw) - 1
            console.print("[red]非法输入，请重输。[/red]")

    def prompt_seer_check(self, checkable: list[int]) -> int:
        return self._ask_seat_choice(checkable, "查验哪一位？")

    def prompt_witch_act(
        self, kill_target: int, can_save: bool, can_poison: bool, alive_seats: list[int]
    ) -> tuple[bool, int | None]:
        """女巫用药：展示刀口与药品状态，返回 (用解药, 毒杀目标或 None)。"""
        note = "—— 就是你自己！" if kill_target == self.my_seat else ""
        console.print(
            f"🔒 今晚刀口：{kill_target + 1} 号 {note}", style="magenta"
        )
        console.print(
            f"解药：{'✅ 还在' if can_save else '❌ 已用完'}；"
            f"毒药：{'✅ 还在' if can_poison else '❌ 已用完'}（同一夜只能用一瓶）"
        )
        while True:
            action = console.input("输入 [bold]s[/bold] 用解药 / [bold]p[/bold] 用毒药 / [bold]n[/bold] 不用药：").strip().lower()
            if action == "s" and can_save:
                return True, None
            if action == "s":
                console.print("[red]解药已用完。[/red]")
                continue
            if action == "p" and can_poison:
                target = self._ask_seat_choice(alive_seats, "毒杀哪一位？")
                return False, target
            if action == "p":
                console.print("[red]毒药已用完。[/red]")
                continue
            if action == "n":
                return False, None
            console.print("[red]非法输入，请重输。[/red]")

    def prompt_vote(self, alive_seats: list[int]) -> int | None:
        return self._ask_seat_choice(alive_seats, "投票放逐谁？", allow_abstain=True)

    # ------------------------------------------------------------ AI 流式渲染（4.5）

    @contextmanager
    def stream_ai_speech(self, name: str, seat: int) -> Iterator[Callable[[str], None]]:
        """AI 发言流式渲染：先显示发言人标识与「打字中」，逐字追加。

        Live 结束后内容被清除，最终以事件流中的规范记录（[#n] 编号行）展示。
        """
        console.print(f"💬 [bold]{name}[/bold]（{seat + 1} 号）", style="cyan")
        buf = ""
        live = Live(console=console, refresh_per_second=20, transient=True)
        live.start()
        live.update(Text("（打字中…）", style="dim italic"))

        def handle(delta: str) -> None:
            nonlocal buf
            buf += delta
            live.update(Text(buf))

        try:
            yield handle
        finally:
            live.stop()
            console.print()

    # ------------------------------------------------------------ 终局（4.6）

    def show_game_over(self) -> None:
        game = self.game
        console.rule("[bold]终局")
        over = next(e for e in game.events if e.kind is EventKind.GAME_OVER)
        console.print(Panel(over.text, border_style="yellow"))

        # 身份揭示表
        table = Table(title="身份揭示")
        table.add_column("座位", justify="center")
        table.add_column("名字")
        table.add_column("角色")
        table.add_column("阵营", justify="center")
        table.add_column("结局", justify="center")
        for p in game.players:
            wolf = team_of(p.role) is Team.WOLF
            table.add_row(
                str(p.seat + 1),
                p.name + (" ← 你" if p.seat == self.my_seat else ""),
                f"[bold red]{p.role.value}[/bold red]" if wolf else p.role.value,
                "狼人阵营" if wolf else "好人",
                "存活" if p.alive else "出局",
            )
        console.print(table)

        # 关键事件回顾（刀口/查验/用药/放逐顺序）
        recap = Table(title="关键事件回顾")
        recap.add_column("阶段")
        recap.add_column("事件")
        for e in game.events:
            if e.kind in (
                EventKind.WOLF_KILL,
                EventKind.SEER_CHECK,
                EventKind.POTION_USED,
                EventKind.DEATH,
                EventKind.PEACEFUL_NIGHT,
                EventKind.EXILE,
            ):
                phase = f"第{e.day}夜" if e.phase.value == "night" else f"第{e.day}天"
                recap.add_row(phase, e.text)
        console.print(recap)