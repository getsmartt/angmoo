"""OpenAI-compatible text, JSON, tool and embedding adapter.

Serves Ollama, Lemonade, LM Studio and llama.cpp's server through their
``/v1/chat/completions`` and ``/v1/embeddings`` routes. Requests keep the
Gemini profile names stored on credentials; ``OPENAI_COMPAT_*`` settings map
them to the server's own model names. One physical HTTP attempt per call: the
caller owns retries, as with the Gemini adapter's explicit ``sdk_attempts``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx
from pydantic import BaseModel

from app.config import settings
from app.core.redaction import redact_exact_secret_text
from app.providers.contracts import (
    EmbeddingBatchRequest,
    EmbeddingRequest,
    MeasuredEmbeddingBatchResponse,
    MeasuredEmbeddingResponse,
    ProviderCapabilities,
    ProviderError,
    ProviderHTTPStatusError,
    ProviderRequest,
    ProviderResponse,
    ProviderToolCall,
    ProviderUsage,
)
from app.providers.diagnostics import schema_evidence

# Gemini wire codes, because callers and diagnostics only accept those.
_FINISH_REASONS = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "function_call": "STOP",
    "content_filter": "SAFETY",
}
# Status names shared with the Gemini SDK so rate-limit/overload checks match.
_HTTP_STATUS_NAMES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    502: "BAD_GATEWAY",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str | None
    chat_model: str
    model_map: dict[str, str]
    embedding_model: str
    embedding_truncate: bool
    context_tokens: int
    json_mode: str
    send_reasoning_effort: bool
    extra_body: dict[str, Any]

    @classmethod
    def from_settings(cls) -> OpenAICompatibleConfig:
        model_map: dict[str, str] = {}
        for entry in settings.OPENAI_COMPAT_MODEL_MAP.split(","):
            if not entry.strip():
                continue
            source, separator, target = entry.partition("=")
            if not separator or not source.strip() or not target.strip():
                raise ValueError("openai_compatible_model_map_invalid")
            model_map[source.strip()] = target.strip()
        extra_body: Any = {}
        if settings.OPENAI_COMPAT_EXTRA_BODY.strip():
            try:
                extra_body = json.loads(settings.OPENAI_COMPAT_EXTRA_BODY)
            except ValueError:
                raise ValueError("openai_compatible_extra_body_invalid") from None
            if not isinstance(extra_body, dict):
                raise ValueError("openai_compatible_extra_body_invalid")
        api_key = settings.OPENAI_COMPAT_API_KEY
        return cls(
            base_url=settings.OPENAI_COMPAT_BASE_URL.rstrip("/"),
            api_key=api_key.get_secret_value() if api_key is not None else None,
            chat_model=settings.OPENAI_COMPAT_CHAT_MODEL.strip(),
            model_map=model_map,
            embedding_model=settings.OPENAI_COMPAT_EMBEDDING_MODEL.strip(),
            embedding_truncate=settings.OPENAI_COMPAT_EMBEDDING_TRUNCATE,
            context_tokens=max(1, settings.OPENAI_COMPAT_CONTEXT_TOKENS),
            json_mode=settings.OPENAI_COMPAT_JSON_MODE,
            send_reasoning_effort=settings.OPENAI_COMPAT_SEND_REASONING_EFFORT,
            extra_body=extra_body,
        )

    def chat_model_for(self, profile: str) -> str:
        model = self.model_map.get(profile) or self.chat_model
        if not model:
            raise ValueError("openai_compatible_chat_model_unset")
        return model

    def embedding_model_for(self, profile: str) -> str:
        model = self.model_map.get(profile) or self.embedding_model
        if not model:
            raise ValueError("openai_compatible_embedding_model_unset")
        return model

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def json_schema_for(schema: Any) -> dict[str, Any] | None:
    """The plain JSON Schema for a caller's dict, Pydantic class or SDK Schema."""
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        from app.providers.gemini import build_gemini_developer_response_schema

        return build_gemini_developer_response_schema(schema)
    if isinstance(schema, BaseModel):  # google.genai types.Schema (OpenAPI form)
        def lower_types(node: Any) -> Any:
            if isinstance(node, list):
                return [lower_types(item) for item in node]
            if not isinstance(node, dict):
                return node
            return {key: (value.lower() if key == "type" and isinstance(value, str)
                          else lower_types(value)) for key, value in node.items()}

        return lower_types(schema.model_dump(mode="json", exclude_none=True))
    raise ValueError("openai_compatible_response_schema_unsupported")


def build_chat_body(request: ProviderRequest, config: OpenAICompatibleConfig) -> dict[str, Any]:
    if request.require_tool_call and not request.tools:
        raise ValueError("required_tool_declarations_missing")
    if request.tools and (request.response_schema is not None or request.response_mime_type is not None):
        raise ValueError("native_tools_json_mode_conflict")
    user_content: Any = request.user_prompt
    if request.image_parts:
        user_content = [{"type": "text", "text": request.user_prompt}]
        for part in request.image_parts:
            if part.data is not None:
                url = f"data:{part.mime_type};base64,{base64.b64encode(part.data).decode()}"
            elif part.url:
                url = part.url
            else:
                continue
            user_content.append({"type": "image_url", "image_url": {"url": url}})
    body: dict[str, Any] = {
        **config.extra_body,
        "model": config.chat_model_for(request.model),
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": request.max_output_tokens,
        "stream": False,
    }
    schema = json_schema_for(request.response_schema)
    if schema is not None and config.json_mode == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }
    elif schema is not None or request.response_mime_type == "application/json":
        body["response_format"] = {"type": "json_object"}
    if request.tools:
        body["tools"] = [
            {"type": "function", "function": {
                "name": tool.name, "description": tool.description, "parameters": tool.parameters}}
            for tool in request.tools
        ]
        body["tool_choice"] = "required" if request.require_tool_call else "auto"
    if config.send_reasoning_effort and request.thinking_level:
        body["reasoning_effort"] = request.thinking_level.strip().lower()
    return body


def _wants_json(request: ProviderRequest) -> bool:
    return request.response_schema is not None or request.response_mime_type == "application/json"


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content
                          if isinstance(part, dict) and isinstance(part.get("text"), str))
    if not isinstance(content, str):
        return ""
    # Servers that inline reasoning put it first; Gemini never returns it as text.
    return _THINK_BLOCK.sub("", content, count=1).strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    fenced = _JSON_FENCE.match(text)
    try:
        value = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _usage(payload: dict[str, Any], *, call_type: str, duration_ms: int) -> ProviderUsage:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}

    def count(container: Any, key: str) -> int | None:
        value = container.get(key) if isinstance(container, dict) else None
        return value if type(value) is int and value >= 0 else None

    return ProviderUsage(
        input_tokens=count(usage, "prompt_tokens"),
        output_tokens=count(usage, "completion_tokens"),
        thought_tokens=count(usage.get("completion_tokens_details"), "reasoning_tokens"),
        total_tokens=count(usage, "total_tokens"),
        cached_input_tokens=count(usage.get("prompt_tokens_details"), "cached_tokens"),
        call_type=call_type,
        duration_ms=duration_ms,
    )


def _tool_calls(message: dict[str, Any]) -> tuple[ProviderToolCall, ...]:
    calls = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ProviderError("Tool call shape invalid", failure_class="tool_call_invalid")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                raise ProviderError("Tool call arguments invalid", failure_class="tool_call_invalid") from None
        if not isinstance(arguments, dict):
            raise ProviderError("Tool call arguments invalid", failure_class="tool_call_invalid")
        call_id = item.get("id")
        calls.append(ProviderToolCall(name=function["name"], arguments=arguments,
                                      call_id=call_id if isinstance(call_id, str) else None))
    return tuple(calls)


def _diagnose(request: ProviderRequest, evidence: dict[str, Any]) -> None:
    if request.diagnostic_callback is None:
        return
    try:
        request.diagnostic_callback(evidence)
    except Exception:
        pass  # Observation never changes provider behavior.


async def _post_json(
    path: str,
    body: dict[str, Any],
    *,
    config: OpenAICompatibleConfig,
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    async with asyncio.timeout(timeout_seconds), httpx.AsyncClient(
        timeout=timeout_seconds, follow_redirects=False, transport=transport,
    ) as client:
        async with client.stream("POST", config.base_url + path, json=body, headers=config.headers()) as response:
            if response.status_code != 200:
                raise ProviderHTTPStatusError(
                    f"OpenAI-compatible request failed (HTTP {response.status_code})",
                    code=response.status_code,
                    status=_HTTP_STATUS_NAMES.get(response.status_code),
                )
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > _MAX_RESPONSE_BYTES:
                    raise ProviderError("Provider response too large", failure_class="response_too_large")
    try:
        payload = json.loads(data)
    except ValueError:
        raise ProviderError("Provider response was not JSON", failure_class="response_invalid") from None
    if not isinstance(payload, dict):
        raise ProviderError("Provider response was not an object", failure_class="response_invalid")
    return payload


def _post_json_sync(path: str, body: dict[str, Any], *, config: OpenAICompatibleConfig,
                    timeout_seconds: float, transport: httpx.BaseTransport | None) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_seconds, follow_redirects=False, transport=transport) as client:
        response = client.post(config.base_url + path, json=body, headers=config.headers())
    if response.status_code != 200:
        raise ProviderHTTPStatusError(
            f"OpenAI-compatible request failed (HTTP {response.status_code})",
            code=response.status_code, status=_HTTP_STATUS_NAMES.get(response.status_code))
    if len(response.content) > _MAX_RESPONSE_BYTES:
        raise ProviderError("Provider response too large", failure_class="response_too_large")
    try:
        payload = response.json()
    except ValueError:
        raise ProviderError("Provider response was not JSON", failure_class="response_invalid") from None
    if not isinstance(payload, dict):
        raise ProviderError("Provider response was not an object", failure_class="response_invalid")
    return payload


def _embedding_vectors(payload: dict[str, Any], *, count: int, dimension: int,
                       truncate: bool) -> tuple[tuple[float, ...], ...]:
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != count:
        raise ProviderError("Embedding response shape invalid", failure_class="embedding_cardinality_mismatch")
    ordered = sorted(rows, key=lambda row: row.get("index", 0) if isinstance(row, dict) else 0)
    vectors = []
    for row in ordered:
        values = row.get("embedding") if isinstance(row, dict) else None
        if (not isinstance(values, list) or not values
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
            raise ProviderError("Embedding vector invalid", failure_class="empty_embedding")
        vector = [float(v) for v in values]
        if len(vector) > dimension and truncate:
            vector = vector[:dimension]
            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0:
                raise ProviderError("Embedding vector invalid", failure_class="empty_embedding")
            vector = [v / norm for v in vector]
        if len(vector) != dimension:
            raise ProviderError("Embedding dimension did not match the request",
                                failure_class="embedding_dimension_mismatch")
        vectors.append(tuple(vector))
    return tuple(vectors)


class OpenAICompatibleAdapter:
    capabilities = ProviderCapabilities(
        text=True,
        structured_json=True,
        image_input=True,
        embedding=True,
        embedding_batch=True,
        tool_calls=True,
    )

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None,
                 sync_transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport
        self._sync_transport = sync_transport

    async def generate_text(self, request: ProviderRequest) -> ProviderResponse:
        return await self._generate(request)

    async def generate_json(self, request: ProviderRequest) -> ProviderResponse:
        return await self._generate(request)

    async def _generate(self, request: ProviderRequest) -> ProviderResponse:
        config = OpenAICompatibleConfig.from_settings()
        body = build_chat_body(request, config)
        _diagnose(request, {
            "capture_boundary": "sdk_input", "model": request.model,
            "config_keys": sorted(key for key in body if key not in {"messages", "model"}),
            "schema_field": "response_format" if "response_format" in body else None,
            "max_output_tokens": request.max_output_tokens,
            "response_mime_type": request.response_mime_type,
            "thinking_config": body.get("reasoning_effort"),
            "timeout_ms": max(1, int(request.timeout_seconds * 1000)),
            "sdk_retry_attempts": 1,
            "sdk_retry_status": "explicit",
            "system_sha256": hashlib.sha256(request.system_prompt.encode()).hexdigest(),
            "system_chars": len(request.system_prompt),
            "user_sha256": hashlib.sha256(request.user_prompt.encode()).hexdigest(),
            "user_chars": len(request.user_prompt),
            **schema_evidence(json_schema_for(request.response_schema)),
        })
        started = monotonic()
        payload = await _post_json("/chat/completions", body, config=config,
                                   timeout_seconds=request.timeout_seconds, transport=self._transport)
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        raw_finish = choice.get("finish_reason")
        finish_reason = _FINISH_REASONS.get(raw_finish, "OTHER") if isinstance(raw_finish, str) else None
        text = _message_text(message)
        tool_calls = _tool_calls(message)
        parsed = _parse_json_object(text) if _wants_json(request) and text else None
        _diagnose(request, {
            "capture_boundary": "sdk_response", "response_evidence_version": 1, "model": request.model,
            "candidate_count": len(choices) if isinstance(choices, list) else 0,
            "prompt_feedback_present": False, "prompt_block_reason": None,
            "candidate_finish_reasons": [finish_reason],
            "candidate_part_counts": [int(bool(text)) + len(tool_calls)],
            "candidate_text_part_counts": [int(bool(text))],
            "candidate_metadata_truncated": False,
            "parsed_present": parsed is not None,
        })
        return ProviderResponse(
            text=text,
            parsed=parsed,
            usage=_usage(payload, call_type="generate_content",
                         duration_ms=int((monotonic() - started) * 1000)),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def _embedding_body(self, config: OpenAICompatibleConfig, model: str, texts: Any, dimension: int) -> dict:
        return {"model": config.embedding_model_for(model), "input": texts, "dimensions": dimension}

    async def embed(self, request: EmbeddingRequest) -> list[float]:
        response = await self.embed_measured(request, timeout_seconds=60)
        return list(response.vector)

    def embed_sync(self, request: EmbeddingRequest) -> list[float]:
        config = OpenAICompatibleConfig.from_settings()
        payload = _post_json_sync("/embeddings",
                                  self._embedding_body(config, request.model, request.text, request.output_dimension),
                                  config=config, timeout_seconds=60, transport=self._sync_transport)
        return list(_embedding_vectors(payload, count=1, dimension=request.output_dimension,
                                       truncate=config.embedding_truncate)[0])

    async def embed_measured(self, request: EmbeddingRequest, *, timeout_seconds: float) -> MeasuredEmbeddingResponse:
        if not 0 < timeout_seconds <= 60:
            raise ValueError("embedding_timeout_invalid")
        result = await self._embed_texts(request.model, request.text, 1, request.output_dimension,
                                         timeout_seconds=timeout_seconds, call_type="embed_content")
        return MeasuredEmbeddingResponse(result.vectors[0], result.usage, physical_attempts=1)

    async def embed_batch_measured(self, request: EmbeddingBatchRequest, *,
                                   timeout_seconds: float) -> MeasuredEmbeddingBatchResponse:
        from app.providers.gemini_embedding_batch import batch_fits

        if not batch_fits(request.texts) or not 0 < timeout_seconds <= 60:
            raise ValueError("embedding_batch_limit_invalid")
        return await self._embed_texts(request.model, list(request.texts), len(request.texts),
                                       request.output_dimension, timeout_seconds=timeout_seconds,
                                       call_type="batch_embed_contents")

    async def _embed_texts(self, model: str, texts: Any, count: int, dimension: int, *,
                           timeout_seconds: float, call_type: str) -> MeasuredEmbeddingBatchResponse:
        started = monotonic()
        try:
            config = OpenAICompatibleConfig.from_settings()
            payload = await _post_json("/embeddings", self._embedding_body(config, model, texts, dimension),
                                       config=config, timeout_seconds=timeout_seconds, transport=self._transport)
            vectors = _embedding_vectors(payload, count=count, dimension=dimension,
                                         truncate=config.embedding_truncate)
            duration_ms = int((monotonic() - started) * 1000)
            return MeasuredEmbeddingBatchResponse(vectors, _usage(payload, call_type=call_type,
                                                                  duration_ms=duration_ms), physical_attempts=1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            from app.providers.gemini import classify_generation_failure

            error = exc if isinstance(exc, ProviderError) else classify_generation_failure(exc)
            error.physical_attempts = 1
            error.duration_ms = int((monotonic() - started) * 1000)
            raise error from None

    def normalize_error(self, exc: BaseException, *, api_key: str | None = None) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        configured = settings.OPENAI_COMPAT_API_KEY
        message = redact_exact_secret_text(str(exc), api_key)
        if configured is not None:
            message = redact_exact_secret_text(message, configured.get_secret_value())
        code = getattr(exc, "code", None)
        status = getattr(exc, "status", None)
        retryable = (code in {429, 500, 502, 503, 504}
                     or isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.NetworkError)))
        return ProviderError(
            message[:1000],
            failure_class=type(exc).__name__,
            provider_status=str(status) if status is not None else None,
            provider_code=code if isinstance(code, int) else None,
            retryable=retryable,
        )
