from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from app.config import settings
from app.integrations.local_input_budget import LocalEstimateTokenCounter, estimate_tokens
from app.providers.contracts import (
    EmbeddingBatchRequest,
    EmbeddingRequest,
    ProviderError,
    ProviderHTTPStatusError,
    ProviderImagePart,
    ProviderRequest,
    ProviderToolDefinition,
)
from app.providers.gemini import classify_generation_failure
from app.providers.openai_compatible import OpenAICompatibleAdapter
from app.providers import registry

STORED_KEY = "AIzaStoredGeminiKeyThatMustNeverLeave000"


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "openai_compatible")
    monkeypatch.setattr(settings, "OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:11434/v1/")
    monkeypatch.setattr(settings, "OPENAI_COMPAT_API_KEY", None)
    monkeypatch.setattr(settings, "OPENAI_COMPAT_CHAT_MODEL", "qwen3.6:35b")
    monkeypatch.setattr(settings, "OPENAI_COMPAT_MODEL_MAP", "gemini-3.5-flash-lite=gemma4:27b")
    monkeypatch.setattr(settings, "OPENAI_COMPAT_EMBEDDING_MODEL", "embeddinggemma")
    monkeypatch.setattr(settings, "OPENAI_COMPAT_EMBEDDING_TRUNCATE", False)
    monkeypatch.setattr(settings, "OPENAI_COMPAT_JSON_MODE", "json_schema")
    monkeypatch.setattr(settings, "OPENAI_COMPAT_SEND_REASONING_EFFORT", False)
    monkeypatch.setattr(settings, "OPENAI_COMPAT_EXTRA_BODY", "")
    monkeypatch.setattr(settings, "OPENAI_COMPAT_CONTEXT_TOKENS", 32768)


class Recorder:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def _chat(content=None, *, finish="stop", tool_calls=None, usage=None) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    })


def _adapter(recorder: Recorder) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(transport=httpx.MockTransport(recorder),
                                   sync_transport=httpx.MockTransport(recorder))


def _request(**overrides) -> ProviderRequest:
    values = dict(api_key=STORED_KEY, model="gemini-3.1-flash-lite", system_prompt="system",
                  user_prompt="user", max_output_tokens=64, timeout_seconds=5.0)
    values.update(overrides)
    return ProviderRequest(**values)


class Payload(BaseModel):
    answer: str


def test_registry_selects_adapter_by_backend_setting(monkeypatch) -> None:
    assert isinstance(registry.get_provider_adapter("google", "gemini-3.1-flash-lite"), OpenAICompatibleAdapter)
    assert isinstance(registry.get_embedding_adapter("google", "gemini-embedding-2"), OpenAICompatibleAdapter)
    monkeypatch.setattr(settings, "LLM_BACKEND", "gemini")
    assert not isinstance(registry.get_provider_adapter("google", "gemini-3.1-flash-lite"), OpenAICompatibleAdapter)


def test_text_request_maps_profile_and_never_sends_stored_key() -> None:
    recorder = Recorder(_chat("hello"), _chat("hi"))
    adapter = _adapter(recorder)
    first = asyncio.run(adapter.generate_text(_request()))
    asyncio.run(adapter.generate_text(_request(model="gemini-3.5-flash-lite")))

    assert str(recorder.requests[0].url) == "http://127.0.0.1:11434/v1/chat/completions"
    assert recorder.body(0)["model"] == "qwen3.6:35b"
    assert recorder.body(1)["model"] == "gemma4:27b"
    assert recorder.body(0)["messages"] == [{"role": "system", "content": "system"},
                                            {"role": "user", "content": "user"}]
    assert "authorization" not in recorder.requests[0].headers
    assert STORED_KEY not in recorder.requests[0].content.decode()
    assert first.text == "hello" and first.parsed is None and first.finish_reason == "STOP"
    assert (first.usage.input_tokens, first.usage.output_tokens, first.usage.total_tokens) == (11, 7, 18)


def test_configured_server_key_is_sent_as_bearer(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_COMPAT_API_KEY", SecretStr("server-key"))
    recorder = Recorder(_chat("ok"))
    asyncio.run(_adapter(recorder).generate_text(_request()))
    assert recorder.requests[0].headers["authorization"] == "Bearer server-key"


def test_json_schema_request_and_parsed_object() -> None:
    recorder = Recorder(_chat('```json\n{"answer": "yes"}\n```'))
    response = asyncio.run(_adapter(recorder).generate_json(
        _request(response_schema=Payload, response_mime_type="application/json")))

    response_format = recorder.body()["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"]["properties"]["answer"]["type"] == "string"
    assert response.parsed == {"answer": "yes"}


def test_json_object_mode_and_reasoning_block_are_handled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_COMPAT_JSON_MODE", "json_object")
    recorder = Recorder(_chat('<think>planning</think>{"answer": "no"}'))
    response = asyncio.run(_adapter(recorder).generate_json(
        _request(response_schema={"type": "object"}, response_mime_type="application/json")))
    assert recorder.body()["response_format"] == {"type": "json_object"}
    assert response.text == '{"answer": "no"}' and response.parsed == {"answer": "no"}


def test_extra_body_and_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_COMPAT_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    monkeypatch.setattr(settings, "OPENAI_COMPAT_SEND_REASONING_EFFORT", True)
    recorder = Recorder(_chat("ok"))
    asyncio.run(_adapter(recorder).generate_text(_request(thinking_level="High")))
    body = recorder.body()
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "high"


def test_invalid_extra_body_fails_before_io(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_COMPAT_EXTRA_BODY", "[1]")
    recorder = Recorder()
    with pytest.raises(ValueError, match="extra_body_invalid"):
        asyncio.run(_adapter(recorder).generate_text(_request()))
    assert recorder.requests == []


def test_images_become_data_urls() -> None:
    recorder = Recorder(_chat("a cat"))
    asyncio.run(_adapter(recorder).generate_text(
        _request(image_parts=(ProviderImagePart(mime_type="image/png", data=b"\x89PNG"),))))
    content = recorder.body()["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "user"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}}


def test_tool_calls_round_trip() -> None:
    tool = ProviderToolDefinition("pick", "Pick one", {"type": "object", "properties": {"id": {"type": "string"}}})
    recorder = Recorder(_chat(None, finish="tool_calls", tool_calls=[
        {"id": "call-1", "type": "function", "function": {"name": "pick", "arguments": '{"id": "a"}'}}]))
    response = asyncio.run(_adapter(recorder).generate_text(_request(tools=(tool,), require_tool_call=True)))

    body = recorder.body()
    assert body["tools"][0]["function"]["name"] == "pick" and body["tool_choice"] == "required"
    assert response.tool_calls[0].name == "pick" and response.tool_calls[0].arguments == {"id": "a"}
    assert response.tool_calls[0].call_id == "call-1" and response.finish_reason == "STOP"


def test_tools_with_json_mode_is_rejected_like_gemini() -> None:
    tool = ProviderToolDefinition("pick", "Pick one", {"type": "object"})
    with pytest.raises(ValueError, match="native_tools_json_mode_conflict"):
        asyncio.run(_adapter(Recorder()).generate_text(
            _request(tools=(tool,), response_mime_type="application/json")))


def test_malformed_tool_arguments_raise_typed_error() -> None:
    recorder = Recorder(_chat(None, tool_calls=[{"function": {"name": "pick", "arguments": "{oops"}}]))
    with pytest.raises(ProviderError) as error:
        asyncio.run(_adapter(recorder).generate_text(_request(tools=(
            ProviderToolDefinition("pick", "Pick", {"type": "object"}),))))
    assert error.value.failure_class == "tool_call_invalid"


@pytest.mark.parametrize(("raw", "expected"), [("length", "MAX_TOKENS"), ("content_filter", "SAFETY"),
                                               ("weird", "OTHER")])
def test_finish_reasons_use_gemini_codes(raw, expected) -> None:
    response = asyncio.run(_adapter(Recorder(_chat("x", finish=raw))).generate_text(_request()))
    assert response.finish_reason == expected


@pytest.mark.parametrize(("code", "failure", "retryable"), [
    (429, "rate_limited", True), (503, "provider_unavailable", True), (404, "model_unavailable", False),
    (401, "auth_failed", False)])
def test_http_errors_classify_like_gemini(code, failure, retryable) -> None:
    with pytest.raises(ProviderHTTPStatusError) as error:
        asyncio.run(_adapter(Recorder(httpx.Response(code, text="detail " + STORED_KEY))).generate_text(_request()))
    classified = classify_generation_failure(error.value)
    assert (classified.failure_class, classified.retryable) == (failure, retryable)
    normalized = OpenAICompatibleAdapter().normalize_error(error.value, api_key=STORED_KEY)
    assert normalized.provider_code == code and normalized.retryable is retryable
    assert STORED_KEY not in str(normalized)


def test_missing_chat_model_fails_before_io(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_COMPAT_CHAT_MODEL", "")
    recorder = Recorder()
    with pytest.raises(ValueError, match="chat_model_unset"):
        asyncio.run(_adapter(recorder).generate_text(_request()))
    assert recorder.requests == []


def _vector(size: int, value: float = 0.5) -> list[float]:
    return [value] * size


def test_embedding_single_and_batch() -> None:
    recorder = Recorder(
        httpx.Response(200, json={"data": [{"index": 0, "embedding": _vector(768)}]}),
        httpx.Response(200, json={"data": [{"index": 1, "embedding": _vector(768, 0.2)},
                                           {"index": 0, "embedding": _vector(768, 0.1)}]}),
    )
    adapter = _adapter(recorder)
    single = asyncio.run(adapter.embed_measured(
        EmbeddingRequest(STORED_KEY, "gemini-embedding-2", "query", 768), timeout_seconds=5))
    batch = asyncio.run(adapter.embed_batch_measured(
        EmbeddingBatchRequest(STORED_KEY, "gemini-embedding-2", ("a", "b"), 768), timeout_seconds=5))

    assert recorder.body(0) == {"model": "embeddinggemma", "input": "query", "dimensions": 768}
    assert recorder.body(1)["input"] == ["a", "b"]
    assert len(single.vector) == 768 and single.physical_attempts == 1
    assert batch.vectors[0][0] == 0.1 and batch.vectors[1][0] == 0.2


def test_embedding_sync_path() -> None:
    recorder = Recorder(httpx.Response(200, json={"data": [{"index": 0, "embedding": _vector(768)}]}))
    vector = _adapter(recorder).embed_sync(EmbeddingRequest(STORED_KEY, "gemini-embedding-2", "lore", 768))
    assert len(vector) == 768


def test_embedding_dimension_mismatch_and_truncation(monkeypatch) -> None:
    long_vector = httpx.Response(200, json={"data": [{"index": 0, "embedding": _vector(1024)}]})
    request = EmbeddingRequest(STORED_KEY, "gemini-embedding-2", "query", 768)
    with pytest.raises(ProviderError) as error:
        asyncio.run(_adapter(Recorder(long_vector)).embed_measured(request, timeout_seconds=5))
    assert error.value.failure_class == "embedding_dimension_mismatch"

    monkeypatch.setattr(settings, "OPENAI_COMPAT_EMBEDDING_TRUNCATE", True)
    long_vector = httpx.Response(200, json={"data": [{"index": 0, "embedding": _vector(1024)}]})
    result = asyncio.run(_adapter(Recorder(long_vector)).embed_measured(request, timeout_seconds=5))
    assert len(result.vector) == 768
    assert math.isclose(math.sqrt(sum(v * v for v in result.vector)), 1.0)


def test_embedding_transport_failure_is_classified() -> None:
    def fail(request):
        raise httpx.ConnectError("refused", request=request)

    adapter = OpenAICompatibleAdapter(transport=httpx.MockTransport(fail))
    with pytest.raises(ProviderError) as error:
        asyncio.run(adapter.embed_measured(
            EmbeddingRequest(STORED_KEY, "gemini-embedding-2", "query", 768), timeout_seconds=5))
    assert error.value.failure_class == "transport_failed" and error.value.retryable is True


def test_local_token_estimate_over_counts_and_matches_fingerprint() -> None:
    counter = LocalEstimateTokenCounter(context_tokens=4096)
    request = _request(user_prompt="안녕하세요 world " * 10)
    profile = asyncio.run(counter.profile(request))
    tokens, fingerprint = asyncio.run(counter.count(request, profile))

    assert profile.input_limit == profile.context_limit == 4096
    assert tokens >= 50  # every Hangul syllable counts as a whole token
    assert len(fingerprint) == 64
    assert estimate_tokens({"inlineData": {"data": "A" * 10000}, "text": "abc"}) == 1
