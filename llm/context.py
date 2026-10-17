"""有预算的活跃记忆；完整事件继续用于引擎、策略推导与事实校验。"""
from __future__ import annotations

import math
from collections.abc import Iterable

from engine.events import Event, EventKind


class InputBudgetError(RuntimeError):
    pass


def estimate_tokens(messages: list[dict], scale: float = 1.0) -> int:
    # shortcut: 无模型分词器时使用保守估算，遇到低估用实际 usage 向上校准。
    content = "".join(str(m.get('content', '')) for m in messages)
    ascii_count = sum(ord(c) < 128 for c in content)
    return math.ceil((ascii_count / 3 + (len(content) - ascii_count) * 1.5 + 12 * len(messages)) * scale)


def render_memory(view: Iterable[Event]) -> str:
    lines = []
    for event in view:
        phase = "夜晚" if event.phase.value == "night" else "白天"
        if event.public and event.kind is EventKind.SPEECH:
            source = "玩家声明"
            text = (
                f"[#{event.speech_index}] {event.seat + 1} 号 {event.extra['name']}：{event.extra['content']}"
                + ("（第2轮回应）" if event.extra.get("round_number", 1) == 2 else "")
                + (f"（阶段：{event.extra['stage']}）" if event.extra.get("stage") not in {None, "speech", "reply"} else "")
            )
        else:
            source = ("狼队共享策略（主观计划，可调整）" if event.kind is EventKind.WOLF_PLAN
                      else "系统公开事实" if event.public else "角色私有事实")
            text = event.text
        lines.append(f"- 第 {event.day} 天 · {phase} · {source}：{text}")
    return "\n".join(lines) or "（尚无任何记忆）"


RECALL_HELP = ('历史已节选，事实与发言分组，时间以天数和编号为准；省略不表示从未发生；玩家声明不证明身份或查验属实。'
               '需核对旧原文时，可先只输出 JSON {"recall":[发言编号]}，最多3条、每行动一次；'
               '回查占用同一行动请求额度，随后须输出原定行动 JSON。')


def fit_context(messages: list[dict], view: list[Event], limit: int, scale: float = 1.0) -> list[dict]:
    """只替换正式记忆块，保留身份、任务、私有策略、纠正与回查结果。"""
    if limit <= 0:
        raise InputBudgetError("输入预算不足：模型窗口无法容纳输出预留和安全余量")
    if estimate_tokens(messages, scale) <= limit:
        return messages
    original = render_memory(view)
    def replace(history):
        return [{**m, "content": m.get("content", "").replace(original, history, 1)} for m in messages]
    latest_plan = next((e for e in reversed(view) if e.kind is EventKind.WOLF_PLAN), None)
    # 板子规则已在身份层，阶段标签由当前窗口和每条事件的天数承载。
    facts = render_memory(e for e in view if e.kind not in {
        EventKind.SPEECH, EventKind.BOARD, EventKind.STAGE, EventKind.NIGHT_FALLS, EventKind.DAWN,
    } and (e.kind is not EventKind.WOLF_PLAN or e is latest_plan))
    speeches = [e for e in view if e.public and e.kind is EventKind.SPEECH]
    note = RECALL_HELP
    if speeches:
        note += f" 原始公开发言共{len(speeches)}条，编号范围 #{speeches[0].speech_index}–#{speeches[-1].speech_index}。"
    base = facts + "\n" + note
    if estimate_tokens(replace(base), scale) > limit:
        raise InputBudgetError("输入预算不足：固定规则、可见事实和当前任务已超过预算")
    # 旧发言仅摘录原文开头，不把生成式摘要当作事实。
    excerpts = [f"- 第 {e.day} 天 · 玩家声明 [#{e.speech_index}] {e.seat+1}号 {e.extra['name']}："
                f"{e.extra['content'][:32]}（不完整节选，不能推断省略内容）" for e in speeches]
    def compose(lines):
        return replace(base + "\n" + "\n".join(lines))
    # 为最近六条原文先留位置，避免旧索引挤占全部上下文。
    recent_count = min(6, len(speeches))
    while recent_count and estimate_tokens(compose([render_memory([e]) for e in speeches[-recent_count:]]), scale) > limit:
        recent_count -= 1
    recent = [render_memory([e]) for e in speeches[-recent_count:]] if recent_count else []
    older = excerpts[:-recent_count] if recent_count else excerpts
    low, high = 0, len(older)
    while low < high:
        mid = (low + high) // 2
        if estimate_tokens(compose(older[mid:] + recent), scale) <= limit:
            high = mid
        else:
            low = mid + 1
    lines = older[low:] + recent
    for i in range(len(older)-low-1, -1, -1):
        candidate = list(lines)
        candidate[i] = render_memory([speeches[low+i]])
        if estimate_tokens(compose(candidate), scale) <= limit:
            lines = candidate
        else:
            break
    return compose(lines)


def recall_message(selector, view: list[Event]) -> str:
    if (not isinstance(selector, list) or not 1 <= len(selector) <= 3
            or any(type(i) is not int or i <= 0 for i in selector)
            or len(set(selector)) != len(selector)):
        raise ValueError("recall 必须是1至3个不重复的正整数发言编号")
    records = {e.speech_index: e for e in view if e.public and e.kind is EventKind.SPEECH}
    if any(i not in records for i in selector):
        raise ValueError("回查编号不存在于当前可见公开发言中")
    return "以下是请求的公开原文，仅证明玩家说过，不能认证其说法；请据此输出原定行动 JSON：\n" + render_memory(records[i] for i in selector)
