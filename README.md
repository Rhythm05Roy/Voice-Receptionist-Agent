# AI Voice Service

FastAPI microservice powering real-time Bahraini home-services voice calls. It integrates Vonage (telephony), ElevenLabs (Bahraini Arabic TTS), AssemblyAI (STT), and OpenAI/Anthropic (LLM). The service is stateless; configs and bookings live in the main backend API.

## Architecture
```
Caller
  │
  ▼
Vonage Voice (NCCO webhooks)
  │  /api/v1/telephony/webhook/inbound
  ▼
FastAPI + ConversationEngine
  │   ├─ SessionManager (in-memory TTL; swap to Redis later)
  │   ├─ BackendClient  (agent config + booking)
  │   ├─ OpenAI/Anthropic (LLM replies)
  │   ├─ ElevenLabs (TTS data URLs)
  │   └─ AssemblyAI stub (real-time STT hook)
  ▼
Vonage NCCO response (talk → listen → connect/hangup)
```

Example call flow (ASCII):
```
1) Vonage hits inbound webhook (no speech yet)
2) Service returns NCCO: talk(greeting) -> listen(eventUrl=self)
3) Caller speaks; Vonage sends speech result -> webhook
4) ConversationEngine processes intake Qs, may call backend booking
5) NCCO returned: talk(reply) -> listen or connect(transfer)
6) If booking done: talk(summary) -> hangup
```

## Setup
```bash
poetry install
cp .env.example .env  # fill secrets
poetry run uvicorn src.main:app --reload
```
For Docker:
```bash
docker build -t ai-voice-service .
docker run -p 8000:8000 --env-file .env ai-voice-service
```

## Local webhook testing with ngrok
```bash
ngrok http 8000
# set Vonage Answer URL to: https://<ngrok-id>.ngrok.io/api/v1/telephony/webhook/inbound
```

## Vonage configuration
1. Create Voice application in Vonage dashboard.
2. Answer URL (POST JSON): `https://<your-domain>/api/v1/telephony/webhook/inbound`
3. Event URL: optional (call status) – not handled yet.
4. Link your purchased number to the application.
5. Put application private key into `.env` (single-line PEM).

## Monitoring / observability
- Structured logs via loguru (request_id + call_id). View with `poetry run uvicorn ... --reload`.
- Rate limiting (slowapi) on webhook: 10/min per IP.
- Future: add Prometheus metrics (e.g., per-call latency) and ship logs to ELK.

## Deployment tips
- Works as a single container on Railway / Render / Fly.io. Expose port 8000, mount `.env` secrets.
- For production, swap SessionManager to Redis and enable HTTPS at the edge (Vonage requires https URLs).

## Example Requests
Health check:
```bash
curl http://localhost:8000/health
```
TTS:
```bash
curl -X POST http://localhost:8000/api/v1/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"مرحبا من المنامة"}'
```
Agent greeting preview:
```bash
curl -X POST http://localhost:8000/api/v1/agent/preview \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"default"}'
```
Simulated inbound webhook:
```bash
curl -X POST http://localhost:8000/api/v1/telephony/webhook/inbound \
  -H "Content-Type: application/json" \
  -d '{"from":"+973111","to":"+973222","uuid":"call-123"}'
```

## Environment Variables (see .env.example)
- BACKEND_BASE_URL / BACKEND_API_KEY
- OPENAI_API_KEY (ANTHROPIC_API_KEY optional)
- ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID
- ASSEMBLYAI_API_KEY
- VONAGE_API_KEY / VONAGE_API_SECRET / VONAGE_APPLICATION_ID / VONAGE_PRIVATE_KEY
- REQUEST_TIMEOUT, LOG_LEVEL, ENVIRONMENT
- REDIS_URL (optional future for session store)

## Testing
```bash
poetry run pytest
```
