# AI Voice Service

FastAPI microservice powering real-time Bahraini home-services voice calls. It integrates Twilio (primary telephony), ElevenLabs (Bahraini Arabic TTS), AssemblyAI (STT), and OpenAI/Anthropic (LLM). Vonage routes are still kept for backward compatibility. The service is stateless; configs and bookings live in the main backend API.

## Architecture
```
Caller
  │
  ▼
Twilio Voice (TwiML webhooks)
  │  /api/v1/twilio/webhook/incoming
  ▼
FastAPI + ConversationEngine
  │   ├─ SessionManager (in-memory TTL; swap to Redis later)
  │   ├─ BackendClient  (agent config + booking)
  │   ├─ OpenAI/Anthropic (LLM replies)
  │   ├─ ElevenLabs (TTS data URLs)
  │   └─ AssemblyAI stub (real-time STT hook)
  ▼
Twilio TwiML response (<Gather> / <Dial> / <Hangup>)
```

Example call flow (ASCII):
```
1) Twilio hits inbound webhook (no speech yet)
2) Service returns TwiML: Gather + Say(greeting)
3) Caller speaks; Twilio sends SpeechResult -> same webhook
4) ConversationEngine processes intake Qs, may call backend booking
5) TwiML returned: Gather(reply) or Dial(transfer)
6) If done: Say(summary) -> Hangup
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
# set Twilio Voice webhook to: https://<ngrok-id>.ngrok.io/api/v1/twilio/webhook/incoming
```

## Twilio configuration (primary)
1. Buy/select your Twilio phone number.
2. In **Active Numbers -> Voice Configuration**, set:
   - **A Call Comes In**: `Webhook`, `HTTP POST`
   - URL: `https://<your-domain>/api/v1/twilio/webhook/incoming`
3. Optional status callback URL: `https://<your-domain>/api/v1/twilio/webhook/status`
4. Set `PUBLIC_BASE_URL=https://<your-domain>` in `.env`.
5. For production, set `TWILIO_VALIDATE_SIGNATURE=true`.

## Vonage configuration (legacy route)
- Keep using `/api/v1/telephony/webhook/inbound` only if you still run Vonage.

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
- TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER
- PUBLIC_BASE_URL / TWILIO_VALIDATE_SIGNATURE (recommended for production)
- ASSEMBLYAI_API_KEY
- VONAGE_API_KEY / VONAGE_API_SECRET / VONAGE_APPLICATION_ID / VONAGE_PRIVATE_KEY (optional legacy)
- REQUEST_TIMEOUT, LOG_LEVEL, ENVIRONMENT
- REDIS_URL (optional future for session store)

## Testing
```bash
poetry run pytest
```
# This repo has moved to a new repo 
