"""Input budget for an OpenAI-compatible server, counted locally.

Ollama and most OpenAI-compatible servers expose no token-count route, so the
count is a deliberate over-estimate of the same prepared request body the
Gemini counter would send: every non-ASCII character as one token and ASCII
text at three characters per token. Inline image bytes are not counted as text.
Over-counting only makes the activity omit optional context sooner.
"""
from datetime import UTC, datetime
from hashlib import sha256
import json
import math

from app.contracts.sns_generation import MODEL_BUDGET_REVISION
from app.providers.gemini import prepare_generate_request
from app.providers.input_budget import InputBudgetError, ModelTokenProfile


def estimate_tokens(value) -> int:
    if isinstance(value, dict):
        return sum(estimate_tokens(item) for key, item in value.items() if key not in {"inlineData", "inline_data"})
    if isinstance(value, list):
        return sum(estimate_tokens(item) for item in value)
    if not isinstance(value, str):
        return 1 if value is not None else 0
    non_ascii = sum(1 for character in value if ord(character) > 127)
    return non_ascii + math.ceil((len(value) - non_ascii) / 3)


class LocalEstimateTokenCounter:
    def __init__(self, *, context_tokens: int):
        if type(context_tokens) is not int or context_tokens <= 0:
            raise InputBudgetError("model_budget_unsupported")
        self.context_tokens = context_tokens

    async def profile(self, request):
        # The shared admission fence checks this revision; the source field
        # records that the limits come from local configuration.
        return ModelTokenProfile("google", request.model, "openai-compatible", self.context_tokens,
            self.context_tokens, MODEL_BUDGET_REVISION, "OPENAI_COMPAT_CONTEXT_TOKENS",
            datetime.now(UTC).isoformat(), context_limit=self.context_tokens)

    async def count(self, request, profile):
        if request.model != profile.model or profile.revision != MODEL_BUDGET_REVISION:
            raise InputBudgetError("model_budget_unsupported")
        body = prepare_generate_request(request).count_request(request.model)
        fingerprint = sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return estimate_tokens(body), fingerprint
