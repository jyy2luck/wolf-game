"""OpenAI 兼容客户端封装：流式输出、网络级重试、token 用量统计、mock 模式。

- 同步串行调用（对局流程本身串行，见设计 D6）
- 流式模式通过 on_delta 回调逐段输出给终端渲染层
- mock 模式：离线测试/演示。调用方可注入 mock_fn(messages) -> str 生成
  受控回复；未注入时返回通用安全 JSON（配合 agents 层的降级路径可跑通流程）
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from openai import OpenAI

from config import LLMConfig


@dataclass
class UsageStats:
    """累计调用量与 token 用量（费用估算的基础）。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


DEFAULT_MOCK_REPLY = json.dumps(
    {"speech": "（离线模式）我先听听大家的看法，保留意见。", "suspect": 2, "mood": "平静"},
    ensure_ascii=False,
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
                base_url=config.base_url, api_key=config.api_key, timeout=request_timeout
            )

    # ------------------------------------------------------------- 内部工具

    def _record_usage(self, usage) -> None:  # noqa: ANN001 - openai usage 对象
        if usage is None:
            return
        self.usage.calls += 1
        self.usage.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    # ------------------------------------------------------------- 主入口

    def chat(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 500,
        mock_reply: Callable[[], str] | None = None,
    ) -> str:
        """发起一次 chat 调用并返回完整文本。

        stream=True 时把增量文本回调给 on_delta（逐字渲染），并统计 usage。
        mock 模式回复优先级：mock_fn（构造时注入）> mock_reply（本次调用注入，
        由调用方按请求 schema 生成合法 JSON）> DEFAULT_MOCK_REPLY。
        """
        if self.mock:
            self.usage.calls += 1
            if self.mock_fn:
                return self.mock_fn(messages)
            if mock_reply is not None:
                return mock_reply()
            return DEFAULT_MOCK_REPLY

        assert self._client is not None
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if stream:
                    chunks = self._client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    parts: list[str] = []
                    usage = None
                    for chunk in chunks:
                        if getattr(chunk, "usage", None) is not None:
                            usage = chunk.usage
                        if chunk.choices:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                parts.append(delta)
                                if on_delta is not None:
                                    on_delta(delta)
                    if usage is not None:
                        self._record_usage(usage)
                    else:
                        self.usage.calls += 1
                    return "".join(parts)

                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._record_usage(resp.usage)
                return resp.choices[0].message.content or ""

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # 网络抖动 / 限流：退避后重试
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"LLM 调用连续失败 {self.max_retries} 次: {last_error}"
        ) from last_error