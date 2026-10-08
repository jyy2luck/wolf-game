"""LLM 连接配置管理：首次启动引导、连通性验证、600 权限持久化与自动加载。

规格：
- 不存在 `~/.llm_murder/config.json` 时进入交互式引导
  （base_url 有默认值可改、api_key 不回显、model 有默认值可改）
- 写盘前发起一次最小连通性请求，验证通过才写入
- 配置目录 700、文件 600，绝不写入项目目录
- `--reconfig` 强制重新引导（由 main.py 解析参数后传入）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

CONFIG_DIR = Path.home() / ".llm_murder"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class LLMConfig:
    """OpenAI 兼容服务的连接三元组。"""

    base_url: str
    api_key: str
    model: str


# ---------------------------------------------------------------- 加载与保存


def load_config() -> LLMConfig | None:
    """静默加载已有配置；不存在返回 None。加载时顺手把权限修正为 600。"""
    if not CONFIG_PATH.exists():
        return None
    os.chmod(CONFIG_PATH, 0o600)
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return LLMConfig(
        base_url=str(data["base_url"]).rstrip("/"),
        api_key=str(data["api_key"]),
        model=str(data["model"]),
    )


def save_config(config: LLMConfig) -> None:
    """写入配置并强制 600 权限（目录 700）。"""
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        os.chmod(CONFIG_PATH, 0o600)  # 覆盖旧文件前先收紧权限
    CONFIG_PATH.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(CONFIG_PATH, 0o600)


# ---------------------------------------------------------------- 连通性验证


def verify_config(config: LLMConfig, timeout: float = 15.0) -> str | None:
    """用所填配置发起一次最小请求。返回错误描述；None 表示验证通过。"""
    try:
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=timeout,
            max_retries=0,
        )
        client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return None
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # 网络 / 鉴权 / 模型不存在等均视为验证失败
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 交互式引导


def prompt_config() -> LLMConfig:
    """交互式引导收集配置，验证通过后返回。

    api_key 输入不回显；输入为空（api_key 无默认值）或 Ctrl-C 时直接退出，
    不写入任何配置文件。
    """
    console.print(
        Panel(
            "首次启动需要配置 OpenAI 兼容的 LLM 服务。\n"
            "配置将保存到 [cyan]~/.llm_murder/config.json[/cyan]（权限 600）。",
            title="🐺 LLM 狼人杀 · 配置引导",
            border_style="cyan",
        )
    )
    while True:
        base_url = Prompt.ask(
            "[bold]base_url[/bold]", default=DEFAULT_BASE_URL
        ).strip().rstrip("/")
        api_key = Prompt.ask("[bold]api_key[/bold]", password=True).strip()
        if not api_key:
            console.print("[red]api_key 为空，已退出，未写入任何配置。[/red]")
            raise SystemExit(1)
        model = Prompt.ask("[bold]model[/bold]", default=DEFAULT_MODEL).strip()

        config = LLMConfig(base_url=base_url, api_key=api_key, model=model)

        console.print("正在验证连通性（发起一次最小请求）…", style="dim")
        error = verify_config(config)
        if error is None:
            console.print("[green]✓ 验证通过[/green]")
            return config
        console.print(f"[red]✗ 验证失败：{error}[/red]")
        console.print("请检查后重新填写（Ctrl-C 退出）。\n")


def ensure_config(reconfig: bool = False) -> LLMConfig:
    """主入口：已有配置则静默加载；否则（或 --reconfig）进入引导并保存。"""
    if not reconfig:
        existing = load_config()
        if existing is not None:
            return existing
    config = prompt_config()
    save_config(config)
    console.print(f"配置已保存到 [cyan]{CONFIG_PATH}[/cyan]（权限 600）")
    return config