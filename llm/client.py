"""OpenAI 兼容客户端封装：流式输出、网络级重试、token 用量统计、mock 模式。

- 同步串行调用（对局流程本身串行，见设计 D6）
- 流式模式可通过 on_delta 收集增量，由调用方控制展示时机
- mock 模式：离线测试/演示。调用方可注入 mock_fn(messages) -> str 生成
  受控回复；未注入时返回通用安全 JSON（配合 agents 层的降级路径可跑通流程）
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import OpenAI

from config import LLMConfig


@dataclass
class UsageStats:
    """累计调用量与 token 用量（费用估算的基础）。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


DEFAULT_MOCK_REPLY = json.dumps(
    {"speech": "（离线模式）我先听听大家的看法，保留意见。", "suspect": 2, "mood": "平静", "references": []},
    ensure_ascii=False,
)


class OutputTruncatedError(RuntimeError):
    """预算耗尽，需由结构化调用层扩大预算；不属于网络故障。"""

    def __init__(self, max_tokens: int, content: str, usage) -> None:
        self.max_tokens = max_tokens
        self.finish_reason = "length"
        self.content_chars = len(content)
        self.completion_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens = getattr(details, "reasoning_tokens", None)
        super().__init__(
            f"输出被截断：finish_reason=length，预算={max_tokens}，"
            f"输出 tokens={self.completion_tokens}，思考 tokens={self.reasoning_tokens}，"
            f"正文长度={self.content_chars}"
        )


class LLMClient:
    """OpenAI 兼容客户端。网络失败自动重试（指数退避）；mock=True 时不发请求。"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        mock: bool = False,
        mock_fn: Callable[[list[dict]], str] | None = None,
        request_timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.config = config
        self.mock = mock
        self.mock_fn = mock_fn
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.usage = UsageStats()
        self._client: OpenAI | None = None
        if not mock:
            self._client = OpenAI(
                base_url=config.base_url, api_key=config.api_key,
                timeout=request_timeout, max_retries=0,
            )

    # ------------------------------------------------------------- 内部工具

    def _record_usage(self, usage) -> None:  # noqa: ANN001 - openai usage 对象
        self.usage.calls += 1
        if usage is None:
            return
        self.usage.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        self.usage.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0

    # ------------------------------------------------------------- 主入口

    def chat(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        mock_reply: Callable[[], str] | None = None,
    ) -> str:
        """发起一次 chat 调用并返回完整文本。

        stream=True 时把增量文本回调给 on_delta，并统计 usage。
        mock 模式回复优先级：mock_fn（构造时注入）> mock_reply（本次调用注入，
        由调用方按请求 schema 生成合法 JSON）> DEFAULT_MOCK_REPLY。
        max_tokens 未指定时使用配置预算；length 完成原因抛出 OutputTruncatedError。
        """
        if self.mock:
            self.usage.calls += 1
            if self.mock_fn:
                return self.mock_fn(messages)
            if mock_reply is not None:
                return mock_reply()
            return DEFAULT_MOCK_REPLY

        assert self._client is not None
        budget = self.config.max_tokens if max_tokens is None else max_tokens
        options: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": budget,
        }
        is_deepseek = urlparse(self.config.base_url).hostname == "api.deepseek.com"
        json_mode = self.config.json_mode
        if json_mode is None:
            json_mode = is_deepseek
        if json_mode:
            options["response_format"] = {"type": "json_object"}
        thinking = self.config.thinking
        if thinking is None and is_deepseek:
            thinking = "disabled"
        if thinking is not None:
            options["extra_body"] = {"thinking": {"type": thinking}}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if stream:
                    chunks = self._client.chat.completions.create(
                        **options,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    parts: list[str] = []
                    usage = None
                    finish_reason = None
                    for chunk in chunks:
                        if getattr(chunk, "usage", None) is not None:
                            usage = chunk.usage
                        if chunk.choices:
                            if chunk.choices[0].finish_reason is not None:
                                finish_reason = chunk.choices[0].finish_reason
                            delta = chunk.choices[0].delta.content
                            if delta:
                                parts.append(delta)
                                if on_delta is not None:
                                    on_delta(delta)
                    self._record_usage(usage)
                    content = "".join(parts)
                    if finish_reason == "length":
                        raise OutputTruncatedError(budget, content, usage)
                    return content

                resp = self._client.chat.completions.create(**options)
                self._record_usage(resp.usage)
                content = resp.choices[0].message.content or ""
                if resp.choices[0].finish_reason == "length":
                    raise OutputTruncatedError(budget, content, resp.usage)
                return content

            except (KeyboardInterrupt, OutputTruncatedError):
                raise
            except Exception as exc:  # 网络抖动 / 限流：退避后重试
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"LLM 调用连续失败 {self.max_retries} 次: {last_error}"
        ) from last_error
