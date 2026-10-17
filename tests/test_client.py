"""通过真实 OpenAI SDK 与模拟 HTTP/SSE 响应验证输出截断管道。"""

import json

import pytest
from openai import AsyncOpenAI

try:
    from openai._base_client import httpx2 as httpx
except ImportError:  # OpenAI SDK 1.x/2.x 使用 httpx，3.x 使用 httpx2。
    import httpx

from config import LLMConfig
from engine import events as ev
from engine.roles import Role
from llm.agents import AIPlayer, Persona, SAFE_SPEECH, VotePack, call_structured
from llm.client import LLMClient, OutputTruncatedError


def response(content, *, stream=False, finish="stop", tokens=20, reasoning=0, usage=True):
    stats = {
        "prompt_tokens": 10, "completion_tokens": tokens, "total_tokens": tokens + 10,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    } if usage else None
    base = {"id": "chat-test", "created": 1, "model": "test-model"}
    if not stream:
        return httpx.Response(200, json={
            **base, "object": "chat.completion", "usage": stats,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        })
    # usage 由 choices 为空的尾包返回，完成原因在前一个包。
    chunks = [
        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"reasoning_content": "private reasoning"}, "finish_reason": None}]},
        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
    ]
    if usage:
        chunks.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": stats})
    data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})


@pytest.fixture
def make_client(monkeypatch):
    clients = []

    def make(*replies, config=None, max_retries=3):
        requests = []
        pending = iter(replies)

        def handler(request):
            requests.append(json.loads(request.content))
            return next(pending)

        cfg = config or LLMConfig("https://api.deepseek.com/v1", "test-key", "deepseek-flash")
        sdk = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                     http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), max_retries=0)
        monkeypatch.setattr("llm.client.AsyncOpenAI", lambda **kwargs: sdk)
        client = LLMClient(cfg, max_retries=max_retries)
        clients.append(client)
        return client, requests

    yield make
    for client in clients:
        client.close()


@pytest.mark.parametrize("stream", [False, True])
def test_defaults_produce_complete_reply_and_count_usage(make_client, stream):
    client, requests = make_client(response('{"vote": 2}', stream=stream))
    deltas = []
    assert client.chat([{"role": "user", "content": "Return JSON"}], stream=stream, on_delta=deltas.append) == '{"vote": 2}'
    assert requests[0]["max_tokens"] == 4096
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert client.usage.calls == 1 and client.usage.total_tokens == 30
    assert deltas == (['{"vote": 2}'] if stream else [])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content", ["", '{"speech": "half', '{"vote": 2}'])
def test_length_is_not_retried_as_network_failure(make_client, stream, content):
    client, requests = make_client(response(content, stream=stream, finish="length", tokens=500, reasoning=483))
    with pytest.raises(OutputTruncatedError) as caught:
        client.chat([], stream=stream, max_tokens=500)
    assert caught.value.finish_reason == "length"
    assert caught.value.reasoning_tokens == 483
    assert caught.value.content_chars == len(content)
    assert requests[0]["max_tokens"] == 500
    assert len(requests) == 1
    assert client.usage.calls == 1 and client.usage.completion_tokens == 500
    assert client.usage.reasoning_tokens == 483
    assert client.usage.total_tokens == 510  # 思考已包含在输出中，不重复计费。


@pytest.mark.parametrize("stream", [False, True])
def test_structured_retry_expands_only_current_action_without_adding_truncated_text(make_client, stream, caplog):
    cfg = LLMConfig("https://api.deepseek.com", "test-key", "deepseek-flash", thinking="enabled")
    client, requests = make_client(
        response("", stream=stream, finish="length", tokens=4096, reasoning=0),
        response('{"vote": 3}', stream=stream, tokens=200, reasoning=194),
        # 下一行动恢复初始预算。
        response('{"vote": 2}', stream=not stream), config=cfg,
    )
    messages = [{"role": "user", "content": 'Return JSON {"vote": 0}'}]
    pack, err = call_structured(client, messages, VotePack, stream=stream)
    assert err is None and pack.vote == 3
    assert [r["max_tokens"] for r in requests] == [4096, 8192]
    assert all(r["messages"] == messages for r in requests)
    assert all(r["thinking"] == {"type": "disabled"} for r in requests)
    assert client.usage.calls == 2 and client.usage.completion_tokens == 4296
    assert client.usage.reasoning_tokens == 194
    assert "finish_reason=length" in caplog.text and "思考 tokens=0" in caplog.text
    assert "结构化输出解析失败" not in caplog.text
    assert "test-key" not in caplog.text and "private reasoning" not in caplog.text
    next_pack, err = call_structured(client, [{"role": "user", "content": "Next action JSON"}], VotePack, stream=not stream)
    assert err is None and next_pack.vote == 2
    assert [r["max_tokens"] for r in requests] == [4096, 8192, 4096]
    assert client.usage.structured_retries == 1 and client.usage.fallback_actions == 0
    assert cfg.max_tokens == 4096 and cfg.max_tokens_limit == 16384
    new_client, _ = make_client(response('{"vote": 2}'), config=cfg)
    assert new_client.output_budget == 4096


@pytest.mark.parametrize("stream", [False, True])
def test_maximum_structured_attempts_degrade_speech_or_vote(make_client, stream):
    cfg = LLMConfig("https://api.deepseek.com", "test-key", "deepseek-flash", max_tokens=500, thinking="disabled")
    client, requests = make_client(
        *(response("", stream=stream, finish="length", tokens=n, reasoning=n) for n in (500, 1000, 2000)), config=cfg,
    )
    player = AIPlayer(client, seat=1, name="学姐", role=Role.VILLAGER, persona=Persona("学姐", "理性", "从逻辑上讲"))
    if stream:
        assert player.day_speech([], 1, list(range(6))) == (SAFE_SPEECH, None, "平静")
    else:
        assert player.day_vote([], 1, list(range(6))) is None
    assert [r["max_tokens"] for r in requests] == [500, 1000, 2000]
    assert client.usage.calls == 3
    assert client.usage.structured_retries == 2 and client.usage.fallback_actions == 1


def test_budget_ceiling_stops_useless_retries(make_client):
    cfg = LLMConfig("https://api.deepseek.com", "test-key", "deepseek-flash", max_tokens=500, max_tokens_limit=750, thinking="disabled")
    client, requests = make_client(response("", finish="length", tokens=500), response("", finish="length", tokens=750), config=cfg)
    pack, err = call_structured(client, [], VotePack)
    assert pack is None and "截断" in err
    assert [r["max_tokens"] for r in requests] == [500, 750]


def test_format_retry_keeps_budget_and_includes_error(make_client):
    client, requests = make_client(response("bad JSON"), response('{"vote": 4}'))
    pack, err = call_structured(client, [], VotePack)
    assert pack.vote == 4 and err is None
    assert [r["max_tokens"] for r in requests] == [4096, 4096]
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": "bad JSON"}
    assert "无法使用" in requests[1]["messages"][-1]["content"]
    assert client.usage.structured_retries == 1 and client.usage.fallback_actions == 0


@pytest.mark.parametrize("stream", [False, True])
def test_missing_usage_still_counts_call(make_client, stream):
    client, _ = make_client(response('{"vote": 2}', stream=stream, usage=False))
    assert client.chat([], stream=stream) == '{"vote": 2}'
    assert client.usage.calls == 1 and client.usage.total_tokens == 0
    assert client.usage.reasoning_tokens == 0


@pytest.mark.parametrize("base_url,json_mode,thinking,expected_json,expected_thinking", [
    ("https://provider.example/v1", None, None, False, None),
    ("https://provider.example/v1", True, "disabled", True, "disabled"),
    ("https://api.deepseek.com", False, "enabled", False, "disabled"),
    ("https://api.deepseek.com/v1", None, "disabled", True, "disabled"),
    ("https://api.deepseek.com/v1", None, None, True, "disabled"),
    ("https://api.deepseek.com.provider.example", None, None, False, None),
])
def test_provider_options_are_compatible_and_overridable(make_client, base_url, json_mode, thinking, expected_json, expected_thinking):
    cfg = LLMConfig(base_url, "test-key", "model", thinking=thinking, json_mode=json_mode)
    client, requests = make_client(response('{"vote": 2}'), config=cfg)
    client.chat([])
    assert ("response_format" in requests[0]) == expected_json
    if expected_thinking:
        assert requests[0]["thinking"] == {"type": expected_thinking}
    else:
        assert "thinking" not in requests[0]
    assert requests[0]["max_tokens"] == 4096


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("base,ceiling,expected", [(4096, 6000, 4096), (10000, 12000, 10000)])
def test_thinking_budget_respects_configured_base_and_ceiling(make_client, stream, base, ceiling, expected):
    cfg = LLMConfig("https://api.deepseek.com", "test-key", "deepseek-flash",
                    max_tokens=base, max_tokens_limit=ceiling)
    client, requests = make_client(response("", stream=stream, finish="length", tokens=expected), config=cfg)
    pack, err = call_structured(client, [], VotePack, stream=stream, max_retries=0)
    assert pack is None and "截断" in err
    assert requests[0]["max_tokens"] == expected
    assert client.output_budget <= ceiling


def test_network_failure_degrades_after_client_retries(make_client, monkeypatch):
    monkeypatch.setattr(LLMClient, "pause", lambda *_: None)
    unavailable = lambda: httpx.Response(503, json={"error": {"message": "service unavailable"}})
    client, requests = make_client(*(unavailable() for _ in range(3)))
    player = AIPlayer(client, seat=1, name="学姐", role=Role.VILLAGER, persona=Persona("学姐", "理性", "从逻辑上讲"))
    assert player.day_vote([], 1, list(range(6))) is None
    assert len(requests) == 3


def test_streamed_candidates_stay_private_until_fact_checks_pass(make_client):
    def pack(speech):
        return json.dumps({"speech": speech, "suspect": 2, "mood": "平静", "references": []})

    client, requests = make_client(
        response(pack("2号一上来就急着说话。"), stream=True),
        response(pack("我是狼，2号可疑。"), stream=True),
    )
    player = AIPlayer(client, seat=5, name="学姐", role=Role.VILLAGER, persona=Persona("学姐", "理性", "从逻辑上讲"))
    deltas = []
    result = player.day_speech([], 1, list(range(6)), on_delta=deltas.append)
    assert result == ("我是狼，2号可疑。", 1, "平静")
    assert deltas == []
    assert len(requests) == 2 and client.usage.calls == 2
    assert "2 号整局尚无发言记录" in requests[1]["messages"][-1]["content"]


def test_streamed_truncation_then_fact_errors_cannot_open_another_retry_loop(make_client):
    invalid = json.dumps({"speech": "2号刚才长篇大论。", "suspect": 2})
    cfg = LLMConfig("https://api.deepseek.com", "key", "model", max_tokens=500, max_tokens_limit=750, thinking="disabled")
    client, requests = make_client(
        response("half-json", stream=True, finish="length"),
        response(invalid, stream=True),
        response(invalid, stream=True), config=cfg,
    )
    player = AIPlayer(client, seat=5, name="学姐", role=Role.WEREWOLF, persona=Persona("学姐", "理性", "从逻辑上讲"))
    player.last_suspect = 3
    assert player.day_speech([], 1, list(range(6))) == (SAFE_SPEECH, None, "平静")
    assert [r["max_tokens"] for r in requests] == [500, 750, 750]
    assert player.last_suspect is None


def test_prompt_reference_example_is_grounded_in_real_history(make_client):
    from llm.agents import SpeechPack, parse_pack

    final = json.dumps({"speech": "我暂时保留判断。"})
    client, requests = make_client(response(final, stream=True))
    player = AIPlayer(client, seat=5, name="学姐", role=Role.VILLAGER, persona=Persona("学姐", "理性", "从逻辑上讲"))
    player.day_speech([ev.speech(1, 1, "人类", 3, "我想听听4号。")], 2, list(range(6)))
    example, err = parse_pack(requests[0]["messages"][-1]["content"], SpeechPack)
    assert err is None and example.references[0].speech_index == 3
    assert example.references[0].seat == 2 and example.references[0].day == 1
    assert example.references[0].quote == "我想听听4号。"


@pytest.mark.parametrize('status', [400,401,403,404,422])
def test_nonretryable_status_stops_without_leaking_body(make_client, status, caplog):
    client, requests=make_client(httpx.Response(status,json={'error':{'message':'secret prompt test-key'}}))
    pack, err=call_structured(client, [], VotePack)
    assert pack is None and f'http_{status}' in err and len(requests)==1
    assert client.requests[-1]['result']==f'http_{status}'
    assert 'test-key' not in caplog.text and 'secret prompt' not in caplog.text


def test_input_context_error_refits_instead_of_repeating_same_input(make_client):
    from llm.context import estimate_tokens, render_memory
    view=[ev.speech(i,1,'玩家',i,'历史观点。'*80) for i in range(1,25)]
    messages=[{'role':'system','content':'规则\n'+render_memory(view)}, {'role':'user','content':'投票 JSON'}]
    client,requests=make_client(
        httpx.Response(400,json={'error':{'code':'context_length_exceeded','message':'maximum context length'}}),
        response('{"vote": 2}'))
    pack,err=call_structured(client,messages,VotePack,view=view)
    assert pack.vote==2 and err is None and len(requests)==2
    assert estimate_tokens(requests[1]['messages']) < estimate_tokens(requests[0]['messages'])
    assert client.usage.input_context_errors==1 and client.usage.context_compactions>=1


def test_network_format_and_truncation_share_three_actual_requests(make_client, monkeypatch):
    monkeypatch.setattr(LLMClient,'pause',lambda *_:None)
    client,requests=make_client(httpx.Response(503,json={'error':{'message':'busy'}}),
                                response('bad JSON'), response('',finish='length'))
    pack,err=call_structured(client,[],VotePack)
    assert pack is None and len(requests)==client.usage.calls==3
    assert client.usage.validation_errors==client.usage.output_truncations==client.usage.fallback_actions==1


def test_known_context_window_reserves_output_and_margin(make_client):
    from llm.context import estimate_tokens, render_memory
    cfg=LLMConfig('https://x','key','model',context_window_tokens=7000)
    client,requests=make_client(response('{"vote": 2}'), config=cfg)
    view=[ev.speech(i,1,'玩家',i,'真实记录。'*80) for i in range(1,30)]
    pack,err=call_structured(client,[{'role':'system','content':'规则\n'+render_memory(view)}],VotePack,view=view)
    assert pack is not None and err is None
    request=requests[0]
    assert estimate_tokens(request['messages'])+request['max_tokens']+512 <= 7000


def test_usage_cache_metrics_and_upward_calibration(make_client):
    reply=response('{"vote":2}')
    body=json.loads(reply.content)
    body['usage'].update(prompt_tokens=500,prompt_tokens_details={'cached_tokens':128})
    client,_=make_client(httpx.Response(200,json=body))
    client.chat([{'role':'user','content':'JSON'}])
    assert client.usage.cached_tokens==128 and client.token_scale>1
    assert client.requests[0]['cached_tokens']==128
    assert 'messages' not in client.requests[0]


def test_action_deadline_cancels_drip_stream_and_closes_response(make_client):
    import asyncio
    import time
    class Drip(httpx.AsyncByteStream):
        closed=False
        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(.01)
                yield b'data: {"id":"test","created":1,"model":"test","choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
        async def aclose(self):
            self.closed=True
    drip=Drip()
    client,requests=make_client(httpx.Response(200,stream=drip,headers={'content-type':'text/event-stream'}),
        config=LLMConfig('https://x','key','model',action_timeout_seconds=.06))
    started=time.monotonic()
    pack,err=call_structured(client,[],VotePack,stream=True)
    assert pack is None and 'action_timeout' in err
    assert time.monotonic()-started < .5 and drip.closed and len(requests)==1
    assert client.usage.timeouts==client.usage.fallback_actions==1


def test_multiple_requests_share_one_action_deadline(make_client):
    import asyncio
    import time
    class DelayedBody(httpx.AsyncByteStream):
        closed=False
        async def __aiter__(self):
            await asyncio.sleep(.04)
            yield response('invalid JSON').content
        async def aclose(self):
            self.closed=True
    bodies=[DelayedBody(),DelayedBody()]
    client,requests=make_client(*(httpx.Response(200,stream=b) for b in bodies),
        config=LLMConfig('https://x','key','model',action_timeout_seconds=.065))
    started=time.monotonic()
    pack,err=call_structured(client,[],VotePack)
    assert pack is None and 'action_timeout' in err
    assert len(requests)==2 and time.monotonic()-started < .2
    assert all(b.closed for b in bodies) and client.usage.timeouts==1


def test_validation_and_recall_diagnostics_do_not_store_output(make_client):
    client,_=make_client(response('{"recall":[1]}'),response('bad JSON'),response('{"vote":2}'))
    pack,err=call_structured(client,[],VotePack,view=[ev.speech(1,1,'玩家',1,'公开原文')])
    assert pack.vote==2 and err is None
    assert [r['validation'] for r in client.requests]==['recall','format_or_target_error','accepted']
    metrics=json.dumps(client.requests,ensure_ascii=False)
    assert '公开原文' not in metrics and 'bad JSON' not in metrics and 'test-key' not in metrics
