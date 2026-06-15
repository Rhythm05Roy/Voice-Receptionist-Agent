<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a0033,50:4a0e4e,100:81162e&height=180&section=header&text=Voice%20Receptionist%20Agent&fontSize=44&fontColor=ffffff&fontAlignY=38&desc=LLM-Driven%20Telephony%20Agent%20%7C%20Twilio%20%C2%B7%20ElevenLabs%20%C2%B7%20OpenAI&descAlignY=58&descSize=18&descColor=f4a8c9" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Twilio](https://img.shields.io/badge/Twilio-Voice-F22F46?style=flat-square&logo=twilio&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)
![ElevenLabs](https://img.shields.io/badge/ElevenLabs-TTS-6E56CF?style=flat-square)
![AssemblyAI](https://img.shields.io/badge/AssemblyAI-STT-1A1A2E?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

**A FastAPI microservice that powers real-time AI voice calls for service businesses — an LLM-driven conversational agent that answers calls, handles bookings, tracks orders, and transfers to humans, fully integrated with Twilio telephony.**

</div>

---

## ⚠️ Repository Status

> **This repository has moved.** It is preserved here for reference, but active development continues in a successor repository. The architecture and code described below reflect the last state of this codebase.

---

## Overview

Voice Receptionist Agent is a stateless FastAPI microservice that acts as the voice front-end for a service business's phone line. Incoming calls are received via **Twilio Voice webhooks**, transcribed via **AssemblyAI** (with a local stub for development), reasoned about by an **LLM-driven `ConversationEngine`** (GPT-4o, with Anthropic available as an alternative), spoken back to the caller via **ElevenLabs TTS** (Bahraini Arabic voice support), and routed back to the caller through **Twilio TwiML** responses.

Rather than a rigid state-machine intake flow, the conversation engine gives the LLM full conversation history and a set of callable tools — `submit_booking`, `track_booking`, and `transfer_to_human` — so it can handle corrections, side-questions, multi-intent turns, and natural conversational flow. All agent configuration (services offered, policies, FAQs) and persisted bookings live in a separate **backend API**; this service itself holds no durable business data.

Legacy **Vonage** telephony routes are retained for backward compatibility, though Twilio is the primary and recommended provider.

---

## Architecture

```
                                Caller
                                  │
                                  ▼
                    Twilio Voice (TwiML webhooks)
                                  │
                    /api/v1/twilio/webhook/incoming
                                  ▼
            ┌──────────────────────────────────────────┐
            │              FastAPI Service               │
            │                                            │
            │   RequestIDMiddleware · APIKeyAuth ·       │
            │   Rate Limiting (slowapi)                  │
            │                                            │
            │            ConversationEngine              │
            │     ┌──────────────┬──────────────────┐   │
            │     │              │                  │   │
            │     ▼              ▼                  ▼   │
            │ SessionManager  BackendClient    Tool Handlers │
            │ (in-memory or   (agent config,   (booking,      │
            │  Redis)          bookings)        tracking,     │
            │                                    transfer)     │
            │     │              │                  │   │
            │     ▼              │                  │   │
            │ OpenAI / Anthropic │                  │   │
            │   (GPT-4o reply +  │                  │   │
            │    tool calls)     │                  │   │
            │     │              │                  │   │
            │     ▼              │                  │   │
            │ ElevenLabs TTS ────┘                  │   │
            │ (Bahraini Arabic voice)                │   │
            │     │                                  │   │
            │     ▼                                  │   │
            │ AssemblyAI STT (real-time stub hook)   │   │
            └──────────────────┬─────────────────────┘   │
                                │
                                ▼
                  Twilio TwiML response
              (<Gather> / <Dial> / <Hangup>)
```

### Example call flow

```text
1) Twilio hits the inbound webhook (no speech yet)
2) Service returns TwiML: <Gather> + <Say>(greeting)
3) Caller speaks; Twilio sends SpeechResult -> same webhook
4) ConversationEngine processes the turn via GPT-4o,
   optionally calling submit_booking / track_booking / transfer_to_human
5) TwiML returned: <Gather>(reply) or <Dial>(transfer)
6) When done: <Say>(summary) -> <Hangup>
```

---

## Project Structure

```text
Voice-Receptionist-Agent/
├── src/
│   ├── main.py                       # FastAPI app, lifespan, /health endpoint
│   ├── config/
│   │   └── settings.py               # Pydantic Settings — env-driven configuration
│   ├── middleware/
│   │   ├── request_id.py             # RequestIDMiddleware (request_id for logging)
│   │   └── auth.py                   # APIKeyAuthMiddleware for management endpoints
│   ├── api/
│   │   ├── deps.py                   # Dependency injection — clients, rate limiter
│   │   ├── exceptions.py             # Custom exception hierarchy + handlers
│   │   └── routers/
│   │       ├── twilio_webhooks.py    # Twilio TwiML webhooks (primary telephony)
│   │       ├── telephony.py          # Legacy Vonage webhook routes
│   │       ├── agent.py              # Agent config, previews, phone number management
│   │       ├── voice.py              # /voice/tts, cached audio retrieval
│   │       ├── onboarding.py         # Voice-driven agent onboarding flow
│   │       └── analytics.py          # Dashboard metrics & call activity feed
│   ├── core/
│   │   ├── conversation/
│   │   │   ├── engine.py             # ConversationEngine + CallSessionManager
│   │   │   ├── handlers.py           # Tool-call execution (booking/tracking/transfer)
│   │   │   ├── onboarding_engine.py  # Onboarding conversation logic
│   │   │   ├── prompts.py            # System prompt construction
│   │   │   └── redis_session.py      # Redis-backed session manager
│   │   ├── services/
│   │   │   ├── backend_client.py     # Backend API client (agent config + bookings)
│   │   │   ├── openai.py             # OpenAI/Anthropic LLM client wrapper
│   │   │   ├── elevenlabs.py         # ElevenLabs TTS client + audio caching
│   │   │   ├── assemblyai.py         # AssemblyAI STT (real-time stub)
│   │   │   ├── twilio_client.py      # Twilio TwiML + signature verification
│   │   │   ├── vonage.py             # Legacy Vonage client
│   │   │   ├── call_store.py         # In-memory call record store (analytics)
│   │   │   └── audio_utils.py        # Audio format helpers
│   │   └── types/
│   │       └── schemas.py            # Core domain types (AgentConfig, etc.)
│   └── schemas/
│       ├── agent.py                  # Agent config / preview / phone number schemas
│       ├── call.py                   # Call-related request/response models
│       ├── voice.py                  # TTS request/response models
│       └── error.py                  # Error response models
├── tests/
│   ├── api/                          # API/router tests
│   ├── core/                         # Conversation engine & service tests
│   ├── conftest.py                   # Pytest fixtures
│   ├── mock_backend.py               # Mock backend API for local testing
│   ├── local_call_simulator.py       # Simulates a full call without telephony
│   ├── local_voice_conversation.py   # Local interactive voice conversation harness
│   └── dummy_business_db.json        # Sample agent/business configuration data
├── Dockerfile                        # Single-stage Python 3.11-slim build
├── docker-compose.yml                # Single-service compose definition
├── run-dev.sh                        # Local dev entrypoint (Poetry + reload)
├── pyproject.toml                    # Poetry project & dependency definitions
└── requirements.txt                  # Pip-installable dependency list
```

---

## Component Details

### `src/main.py` — Application Entrypoint

Configures structured logging via **loguru** (request-ID-tagged), and defines a **lifespan handler** that:
- Creates a shared `httpx.AsyncClient`
- Initializes the session manager — **Redis-backed** if `REDIS_URL` is set and reachable, otherwise an **in-memory** `CallSessionManager`
- Constructs singleton service clients (`BackendClient`, `OpenAIClient`, `ElevenLabsClient`) and the `ConversationEngine`, stored on `app.state`
- Exposes `GET /health`, which reports the status of session storage and the backend API dependency

### `src/core/conversation/engine.py` — `ConversationEngine`

The heart of the service. Rather than a scripted question flow, it:

1. Builds a system prompt from the business's `AgentConfig` (services, policies, FAQs) via `prompts.py`
2. Sends the full conversation history plus the latest user message to GPT-4o
3. Lets the LLM decide what to say **and** when to invoke tools
4. Executes tool calls via `handlers.py` and feeds results back to the LLM for a natural follow-up

`CallSession` tracks conversation history, turn count, session state (booking references, flags), and a cached compiled system prompt per call.

### `src/core/conversation/handlers.py` — Tool Execution

Implements the three tools the LLM can call:

| Tool | Purpose |
|---|---|
| `submit_booking` | Sends booking details to the backend via `BackendClient.book_service()` |
| `track_booking` | Queries booking/order status from the backend |
| `transfer_to_human` | Returns a transfer instruction, used to generate a `<Dial>` TwiML response |

### `src/core/services/` — External Integrations

| Service | Client | Role |
|---|---|---|
| Twilio | `twilio_client.py` | TwiML generation, inbound webhook signature verification |
| Vonage *(legacy)* | `vonage.py` | Backward-compatible inbound webhook handling |
| OpenAI / Anthropic | `openai.py` | LLM replies and function-calling for the conversation engine |
| ElevenLabs | `elevenlabs.py` | Text-to-speech synthesis (Bahraini Arabic voice), with retry-on-429/5xx and audio caching |
| AssemblyAI | `assemblyai.py` | Speech-to-text — currently a local stub simulating the real-time websocket API |
| Backend API | `backend_client.py` | Source of truth for agent configuration and booking persistence |
| Call Store | `call_store.py` | In-memory call records powering the analytics dashboard |

### `src/api/routers/` — API Surface

| Router | Prefix | Purpose |
|---|---|---|
| `twilio_webhooks.py` | `/twilio` | Primary inbound/status webhooks, TwiML responses |
| `telephony.py` | `/telephony` | Legacy Vonage inbound webhook + event handling |
| `agent.py` | `/agent` | Agent config preview, phone number search/provision/assign/release/rebind, call reports |
| `voice.py` | `/voice` | Direct TTS generation (`/voice/tts`) and cached audio retrieval |
| `onboarding.py` | `/onboarding` | Voice-driven agent setup conversation (Figma SOW flow) |
| `analytics.py` | `/analytics` | Dashboard summary metrics and call activity feed |

### `src/middleware/`

- **`request_id.py`** — Attaches a unique request ID to every request for log correlation
- **`auth.py`** — `APIKeyAuthMiddleware` protects management/agent endpoints with an API key

### `src/config/settings.py` — Configuration

A single Pydantic `Settings` class (env-driven, `.env`-aware) covering backend connectivity, LLM provider keys, ElevenLabs voice config, Twilio credentials and signature validation, legacy Vonage credentials, Redis session storage, CORS origins, and management API authentication.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) (or `pip` with `requirements.txt`)
- A running backend API that serves agent configuration and booking endpoints
- API credentials for whichever providers you intend to use (Twilio, OpenAI/Anthropic, ElevenLabs, AssemblyAI)

### Installation

```bash
git clone https://github.com/Rhythm05Roy/Voice-Receptionist-Agent.git
cd Voice-Receptionist-Agent

poetry install
cp .env.example .env   # fill in your secrets
poetry run uvicorn src.main:app --reload
```

Or with pip:

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload
```

### Docker

```bash
docker build -t ai-voice-service .
docker run -p 8000:8000 --env-file .env ai-voice-service
```

Or via Compose:

```bash
docker-compose up -d
```

---

## Local Webhook Testing with ngrok

```bash
ngrok http 8000
# Set the Twilio Voice webhook to:
# https://<ngrok-id>.ngrok.io/api/v1/twilio/webhook/incoming
```

---

## Telephony Configuration

### Twilio (primary)

1. Buy/select a Twilio phone number.
2. Under **Active Numbers → Voice Configuration**, set:
   - **A Call Comes In**: `Webhook`, `HTTP POST`
   - URL: `https://<your-domain>/api/v1/twilio/webhook/incoming`
3. Optional status callback URL: `https://<your-domain>/api/v1/twilio/webhook/status`
4. Set `PUBLIC_BASE_URL=https://<your-domain>` in `.env`.
5. For production, set `TWILIO_VALIDATE_SIGNATURE=true`.

### Vonage (legacy route)

Continue using `/api/v1/telephony/webhook/inbound` only if you still operate Vonage numbers.

---

## API Examples

**Health check**
```bash
curl http://localhost:8000/health
```

**Text-to-speech**
```bash
curl -X POST http://localhost:8000/api/v1/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"مرحبا من المنامة"}'
```

**Agent greeting preview**
```bash
curl -X POST http://localhost:8000/api/v1/agent/preview \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"default"}'
```

**Simulated inbound webhook (legacy Vonage route)**
```bash
curl -X POST http://localhost:8000/api/v1/telephony/webhook/inbound \
  -H "Content-Type: application/json" \
  -d '{"from":"+973111","to":"+973222","uuid":"call-123"}'
```

---

## Configuration

All configuration is environment-driven via `src/config/settings.py` (`.env` file supported).

```bash
# ── Environment ────────────────────────────────────────────────────
ENVIRONMENT=development
LOG_LEVEL=INFO
LOCAL_TEST_MODE=false

# ── Backend API ────────────────────────────────────────────────────
BACKEND_BASE_URL=http://localhost:9000
BACKEND_API_KEY=dev-backend-key

# ── LLM Providers ──────────────────────────────────────────────────
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...           # optional

# ── ElevenLabs (TTS) ───────────────────────────────────────────────
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...

# ── Twilio (primary telephony) ─────────────────────────────────────
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=...
TWILIO_WEBSOCKET_URL=...
PUBLIC_BASE_URL=https://your-domain.com
TWILIO_VALIDATE_SIGNATURE=true   # recommended for production

# ── Vonage (legacy) ─────────────────────────────────────────────────
VONAGE_API_KEY=...
VONAGE_API_SECRET=...
VONAGE_APPLICATION_ID=...
VONAGE_PRIVATE_KEY=...
VONAGE_WEBHOOK_SECRET=...

# ── Sessions ────────────────────────────────────────────────────────
REDIS_URL=                       # optional; falls back to in-memory
CONTEXT_REFRESH_TTL_SECONDS=60

# ── Misc ────────────────────────────────────────────────────────────
REQUEST_TIMEOUT=15
CORS_ORIGINS=*
API_AUTH_TOKEN=...                # protects management endpoints
```

> If `REDIS_URL` is unset, unreachable, or the `redis` package is missing, the service automatically falls back to the in-memory `CallSessionManager` — no configuration changes needed for local development.

---

## Testing

```bash
poetry run pytest
```

The `tests/` directory includes:

| Tool | Purpose |
|---|---|
| `tests/api/` | Router-level API tests |
| `tests/core/` | `ConversationEngine`, tool handlers, and service client tests |
| `tests/mock_backend.py` | Mock backend API server for local/CI testing |
| `tests/local_call_simulator.py` | Simulates a full inbound call lifecycle without real telephony |
| `tests/local_voice_conversation.py` | Interactive local harness for testing conversation flow |
| `tests/dummy_business_db.json` | Sample agent/business config used by the mock backend |

---

## Monitoring & Observability

- **Structured logs** via `loguru`, tagged with `request_id` and `call_id` for cross-request correlation
- **Rate limiting** via `slowapi` on telephony webhooks (10 requests/min per IP)
- **Planned**: Prometheus metrics (e.g. per-call latency) and centralized log shipping (ELK)

---

## Deployment Notes

- Runs as a single container — suitable for Railway, Render, Fly.io, or any container host. Expose port `8000` and mount `.env` secrets.
- For production:
  - Configure `REDIS_URL` so sessions survive restarts and scale across instances
  - Enable HTTPS at the edge (required by both Twilio and Vonage webhook URLs)
  - Set `TWILIO_VALIDATE_SIGNATURE=true` to verify inbound webhook authenticity
  - Set `API_AUTH_TOKEN` to protect agent/management endpoints

---

## Key Design Patterns

- **Tool-calling conversation engine** — the LLM drives the conversation and decides when to call `submit_booking`, `track_booking`, or `transfer_to_human`, instead of a rigid intake state machine
- **Stateless service, external source of truth** — agent configuration and booking data live in a separate backend API; this service holds only ephemeral call session state
- **Pluggable session storage** — `CallSessionManager` (in-memory) and `RedisSessionManager` implement the same interface, selected automatically based on `REDIS_URL` availability
- **Provider abstraction** — Twilio is primary, with Vonage retained behind the same webhook-response contract for backward compatibility
- **Stub-friendly integrations** — AssemblyAI real-time STT ships as a local stub with a clear production swap-in point, and ElevenLabs responses are cached to reduce redundant TTS calls
- **Local-first testability** — `local_call_simulator.py`, `local_voice_conversation.py`, and `mock_backend.py` allow full end-to-end conversation testing without any telephony or external API credentials

---

## License

MIT — see `LICENSE`.

<div align="center">

**Built by [Ridam Roy](https://github.com/Rhythm05Roy)**

[![GitHub](https://img.shields.io/badge/GitHub-Rhythm05Roy-181717?style=flat-square&logo=github)](https://github.com/Rhythm05Roy)
[![Email](https://img.shields.io/badge/Email-ridam15--4260%40diu.edu.bd-D44638?style=flat-square&logo=gmail&logoColor=white)](mailto:ridam15-4260@diu.edu.bd)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a0033,50:4a0e4e,100:81162e&height=80&section=footer" width="100%"/>

</div>
