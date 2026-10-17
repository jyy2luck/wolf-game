"""同步游戏接口，内部复用异步连接以取消超时的完整请求（包括流式读取）。"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import AsyncOpenAI, APIConnectionError, APIStatusError

from config import LLMConfig
from llm.context import estimate_tokens

logger = logging.getLogger("llm_murder")


@dataclass
class UsageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    structured_retries: int = 0
    fallback_actions: int = 0
    cached_tokens: int = 0
    output_truncations: int = 0
    input_context_errors: int = 0
    timeouts: int = 0
    validation_errors: int = 0
    context_compactions: int = 0
    recalls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


DEFAULT_MOCK_REPLY = json.dumps(
    {"speech": "（离线模式）我先听听大家的看法，保留意见。", "suspect": 2, "mood": "平静", "references": []},
    ensure_ascii=False,
)


class OutputTruncatedError(RuntimeError):
    def __init__(self, max_tokens: int, content: str, usage) -> None:
        self.max_tokens = max_tokens
        self.finish_reason = "length"
        self.content_chars = len(content)
        self.completion_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens = getattr(details, "reasoning_tokens", None)
        super().__init__(f"输出被截断：finish_reason=length，预算={max_tokens}，"
                         f"输出 tokens={self.completion_tokens}，思考 tokens={self.reasoning_tokens}，"
                         f"正文长度={self.content_chars}")


class RequestError(RuntimeError):
    def __init__(self, category: str, *, retryable: bool = False) -> None:
        self.category = category
        self.retryable = retryable
        super().__init__(f"LLM 请求失败：{category}")


def classify_error(exc: Exception) -> RequestError:
    if isinstance(exc, APIStatusError):
        # 仅用于分类，服务端错误正文不进入日志或公开记录。
        body = str(exc).lower()
        if exc.status_code in (400, 413, 422) and any(s in body for s in (
            "context_length", "context window", "context length", "maximum context", "too many tokens",
            "input tokens", "prompt is too long", "上下文",
        )):
            return RequestError("input_context")
        return RequestError(f"http_{exc.status_code}", retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500)
    return RequestError("network" if isinstance(exc, APIConnectionError) else "client_error",
                        retryable=isinstance(exc, APIConnectionError))


class LLMClient:
    def __init__(self, config: LLMConfig, *, mock: bool = False,
                 mock_fn: Callable[[list[dict]], str] | None = None,
                 request_timeout: float = 60.0, max_retries: int = 3) -> None:
        self.config = config
        self.mock = mock
        self.mock_fn = mock_fn
        self.request_timeout = request_timeout
        self.max_retries = max(1, min(max_retries, 3))
        self.usage = UsageStats()
        self.output_budget = config.max_tokens
        self.token_scale = 1.0
        self.requests: list[dict] = []
        self._runner = asyncio.Runner() if not mock else None
        self._client = None if mock else AsyncOpenAI(
            base_url=config.base_url, api_key=config.api_key, timeout=request_timeout, max_retries=0)

    def close(self) -> None:
        if self._runner is not None:
            try:
                self._runner.run(self._client.close())
            finally:
                self._runner.close()
                self._runner = None

    def mark_response(self, outcome: str) -> None:
        if self.requests and not self.mock:
            self.requests[-1]["validation"] = outcome
            logger.info("LLM 响应校验 request=%d result=%s", len(self.requests), outcome)

    def pause(self, seconds: float, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= seconds:
            self.usage.timeouts += 1
            raise RequestError("action_timeout")
        if self._runner is not None:
            self._runner.run(asyncio.sleep(seconds))

    async def _request(self, options, stream, on_delta, remaining):
        async with asyncio.timeout(remaining):
            if not stream:
                response = await self._client.chat.completions.create(**options)
                return response.choices[0].message.content or "", response.choices[0].finish_reason, response.usage
            chunks = await self._client.chat.completions.create(
                **options, stream=True, stream_options={"include_usage": True})
            parts, usage, finish = [], None, None
            try:
                async for chunk in chunks:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.choices:
                        finish = chunk.choices[0].finish_reason or finish
                        delta = chunk.choices[0].delta.content
                        if delta:
                            parts.append(delta)
                            if on_delta is not None:
                                on_delta(delta)
            finally:
                await chunks.close()
            return "".join(parts), finish, usage

    def _record(self, started, estimate, budget, category, usage=None, input_limit=None) -> None:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        if cached is None:
            cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        self.usage.prompt_tokens += prompt
        self.usage.completion_tokens += completion
        self.usage.reasoning_tokens += reasoning
        self.usage.cached_tokens += cached
        if category == "input_context":
            self.usage.input_context_errors += 1
        if category == "output_truncated":
            self.usage.output_truncations += 1
        if category == "action_timeout":
            self.usage.timeouts += 1
        if prompt > estimate > 0:
            self.token_scale *= prompt / estimate * 1.1
        record = dict(estimated_input_tokens=estimate, max_tokens=budget, prompt_tokens=prompt,
                      completion_tokens=completion, reasoning_tokens=reasoning, cached_tokens=cached,
                      elapsed_seconds=round(time.monotonic()-started, 3), result=category,
                      input_limit=input_limit)
        self.requests.append(record)
        logger.info("LLM 请求 %s", json.dumps(record, ensure_ascii=False))

    def chat(self, messages: list[dict], *, stream: bool = False,
             on_delta: Callable[[str], None] | None = None, temperature: float = 0.8,
             max_tokens: int | None = None, mock_reply: Callable[[], str] | None = None,
             deadline: float | None = None, input_limit: int | None = None) -> str:
        if self.mock:
            self.usage.calls += 1
            return self.mock_fn(messages) if self.mock_fn else mock_reply() if mock_reply else DEFAULT_MOCK_REPLY
        if self._runner is None:
            raise RequestError("client_closed")
        shared_deadline = deadline is not None
        deadline = deadline if shared_deadline else time.monotonic() + self.config.action_timeout_seconds
        budget = self.output_budget if max_tokens is None else max_tokens
        options = dict(model=self.config.model, messages=messages, temperature=temperature, max_tokens=budget)
        is_deepseek = urlparse(self.config.base_url).hostname == "api.deepseek.com"
        if self.config.json_mode is True or (self.config.json_mode is None and is_deepseek):
            options["response_format"] = {"type": "json_object"}
        if is_deepseek or self.config.thinking is not None:
            options["extra_body"] = {"thinking": {"type": "disabled"}}
        for attempt in range(1 if shared_deadline else self.max_retries):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise RequestError("action_timeout")
            started, estimate = time.monotonic(), estimate_tokens(messages, self.token_scale)
            self.usage.calls += 1
            usage, category = None, "ok"
            try:
                content, finish, usage = self._runner.run(self._request(options, stream, on_delta, remaining))
                if finish == "length":
                    category = "output_truncated"
                    raise OutputTruncatedError(budget, content, usage)
                return content
            except OutputTruncatedError:
                raise
            except TimeoutError:
                category = "action_timeout"
                raise RequestError(category) from None
            except Exception as exc:
                error = classify_error(exc)
                category = error.category
                if shared_deadline or not error.retryable or attempt == self.max_retries-1:
                    raise error from None
            finally:
                self._record(started, estimate, budget, category, usage, input_limit)
            self.pause(2**attempt, deadline)
        raise RequestError("network")
