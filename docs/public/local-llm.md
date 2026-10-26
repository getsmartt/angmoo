# Local text models (OpenAI-compatible server)

Angmoo can send all of its text, image-recognition and embedding calls to one
OpenAI-compatible server instead of the Google Gemini API: Ollama, Lemonade,
LM Studio or llama.cpp's `llama-server`. This is a **server-wide switch** for a
self-hosted install. It is experimental and not part of the official support
matrix.

## How it works

Credentials and settings keep their Gemini profile names (`gemini-3.1-flash-lite`,
`gemini-3.5-flash-lite`, `gemini-embedding-2`). When `LLM_BACKEND=openai_compatible`,
the provider registry hands every call to
[`openai_compatible.py`](../../backend/app/providers/openai_compatible.py), which
maps those names to your server's models and talks to `/v1/chat/completions` and
`/v1/embeddings`. Nothing in the database or the frontend changes, and switching
back to `gemini` restores the old behaviour.

- **API keys:** the settings screens still ask for a key. Any non-empty text works.
  Stored keys are **never** sent to the local server; set
  `OPENAI_COMPAT_API_KEY` if your server needs one (Lemonade with an API key does).
- **Token budget:** Gemini's countTokens API is replaced by a local over-estimate
  limited by `OPENAI_COMPAT_CONTEXT_TOKENS`. Set it to the context length your
  server actually loads the model with.
- **Retries:** one HTTP attempt per call; Angmoo's own retry policy applies.

## Settings (`backend/.env`)

```env
LLM_BACKEND=openai_compatible
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:11434/v1
# OPENAI_COMPAT_API_KEY=
OPENAI_COMPAT_CHAT_MODEL=qwen3.6:35b-a3b
# Optional: a different model per Gemini profile
# OPENAI_COMPAT_MODEL_MAP=gemini-3.5-flash-lite=gemma4:27b
OPENAI_COMPAT_EMBEDDING_MODEL=embeddinggemma
OPENAI_COMPAT_CONTEXT_TOKENS=32768
# json_schema (default) or json_object for servers without schema support
# OPENAI_COMPAT_JSON_MODE=json_schema
# Pass Angmoo's thinking level as reasoning_effort
# OPENAI_COMPAT_SEND_REASONING_EFFORT=false
# Merged into every chat request, e.g. to turn off thinking on llama.cpp/Lemonade
# OPENAI_COMPAT_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
# Local servers have no free-tier quota; raise the per-credential pacing
DIRECT_LLM_DEFAULT_RPM_LIMIT=600
```

| Server | Base URL |
| --- | --- |
| Ollama | `http://127.0.0.1:11434/v1` |
| Lemonade | `http://127.0.0.1:13305/api/v1` |
| LM Studio | `http://127.0.0.1:1234/v1` |
| llama.cpp `llama-server` | `http://127.0.0.1:8080/v1` |

## Choosing models

- **Chat model:** Angmoo relies heavily on structured JSON output and tool calls.
  Use a capable instruction model with tool-calling support (for example Qwen3.6
  or Gemma 4 class models). Small models fail validation more often.
- **Image recognition:** the chat model, or the model mapped to the recognition
  profile, must accept images.
- **Embeddings must be 768-dimensional.** Memory and lore vectors are stored as
  `vector(768)`. `embeddinggemma` and `nomic-embed-text` produce 768 values. A
  larger Matryoshka-trained model (such as Qwen3-Embedding) can be cut to 768 with
  `OPENAI_COMPAT_EMBEDDING_TRUNCATE=true`; do not use that with other models.
- **Switching embedding models makes existing vectors incomparable.** Memories and
  lore embedded by Gemini (or another local model) must be embedded again.

## Checks

```sh
cd backend
.venv/bin/python -m pytest tests/test_openai_compatible_provider.py
```
