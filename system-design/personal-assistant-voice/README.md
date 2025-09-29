# GenAI Voice Personal Assistant (STT → LLM → TTS)

## 1. Problem Statement
Provide a hands-free personal assistant supporting conversational queries, reminders, calendar actions, and knowledge lookups with end-to-end round-trip latency < 1.5s (speech start to speech reply) for short queries.

## 2. Latency Budget (Target p95)
| Stage | Target (ms) |
|-------|------------|
| Audio capture buffering | 80 |
| Streaming STT partials | 250 (first partial) |
| Intent + context assembly | 80 |
| LLM first token (speculative) | 350 |
| LLM streaming remainder | 400 |
| Streaming TTS start | 200 |
| Total (overlapped) | <1500 |

Overlapping: STT streaming partials allow early LLM start; TTS commences after partial answer prefix.

## 3. High-Level Architecture
```
Microphone → Audio Chunker → Streaming STT → Intent Router + Context Builder → LLM Inference (Speculative + KV cache)
   → Function Calls (Calendar/Weather/Notes) → Response Stream → Streaming TTS Synthesizer → Speaker
```
Support Systems: Wake word detector, Local command fallback, Edge cache (recent queries), Vector store (personal notes & preferences), Privacy vault.

## 4. Component Detail
### 4.1 Wake Word & VAD
- Lightweight on-device model (e.g., Porcupine) triggers capture.
- Voice Activity Detection trims silence; energy + spectral gating.

### 4.2 Streaming STT
- Low-latency Conformer / Whisper small distilled variant with chunk-wise decoding.
- Partial hypotheses emitted every ~200ms; endpointing via silence or stable posterior.

### 4.3 Intent & Context
- Classifier decides: (Knowledge Query, Personal Action, Chit-Chat, Control Command).
- Retrieve relevant personal context (calendar events, todo list) from local encrypted store + vector similarity for notes (k=5).
- Compose compact prompt: system (persona + privacy), recent 2 turns summarized, dynamic context snippets.

### 4.4 LLM Inference
- Speculative decoding: draft 3–5 tokens with a small model; accept ratio tracked.
- KV cache reuse for session; summarization of history > 2k tokens.
- Route simple queries to a distilled 3B model; complex reasoning → larger model.

### 4.5 Tool / Function Calls
- Calendar (add/list), Weather, Reminders, Notes CRUD, Smart Home actions.
- JSON schema outputs; synchronous for quick results (<200ms) else asynchronous callback with follow-up speech.

### 4.6 Streaming TTS
- Neural codec or fast vocoder (HiFi-GAN) with chunked synthesis.
- Begin playback after first sentence tokens; cross-fade updates if late function result modifies answer.

### 4.7 Privacy & Security
- On-device STT + wake detection to minimize cloud audio; only text sent to cloud LLM (if remote) after privacy filter.
- PII masking for notes retrieval; encrypted local storage (AES-GCM) with device key.

### 4.8 Caching & Optimization
- Cache embeddings of frequent queries (timer, weather) + pre-generated TTS for canonical responses.
- Prefix KV caching for system prompt and persona.
- Pre-fetch weather/calendar data every few minutes.

### 4.9 Observability
- Metrics: end-to-end latency, STT WER, first-token latency, acceptance rate (speculative), TTS start time, tool call latency.
- User feedback tag (“helpful”, “incorrect”, “privacy concern”).

## 5. Data Flow (Answer Query)
1. Wake word triggers audio capture.
2. Audio chunks → streaming STT; partial text emitted.
3. Once stable prefix recognized, intent router fetches context & starts LLM.
4. LLM streams tokens; speculative model accelerates first tokens.
5. If tool needed, JSON call executed; response injected mid-stream.
6. TTS begins after first sentence; continues streaming.
7. Final answer + optional follow-up suggestion.

## 6. Latency Strategies
- Parallel STT + early LLM start.
- Speculative decoding + quantized models.
- Streaming TTS initiation on partial answer.
- Local edge inference for small model; fallback cloud large model.

## 7. Reliability
- Offline fallback: small on-device model answers simple queries when network down.
- Circuit breaker: if tool latency > threshold, answer with partial + "I'll update you shortly" continuation.

## 8. Safety & Abuse Prevention
- Block unsafe commands (e.g., unlocking doors) without secondary auth (voice PIN / biometrics).
- Toxicity / self-harm classifier gating responses.

## 9. Cost Optimization
- Use small model for ≥60% of queries (heuristic classifier).
- Quantize TTS & STT models; batch background tasks.
- Precompute common TTS phrases.

## 10. Tech Stack (Example)
| Layer | Choice |
|-------|--------|
| STT | Distilled Whisper small / Conformer streaming |
| LLM | Local 3B quant + Cloud 13B fallback |
| Vector Store | Lite local (SQLite + embeddings) |
| TTS | FastPitch + HiFi-GAN |
| Wake | Porcupine / Vosk |
| Tool Runtime | Async Python (FastAPI) |

## 11. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Latency jitter from network | Cache & local inference; adaptive bitrate |
| Privacy leakage (raw audio) | On-device STT; transmit text only |
| Speculative misprediction | Limit draft length; monitor accept % |
| Function call race altering early TTS | Sentence boundary buffering; patch response update |
| Wake word false positives | Multi-factor (wake + short user profile voice embedding) |

## 12. Future Enhancements
- Emotion-aware prosody adjustments.
- Personalized voice cloning (opt-in).
- Local summarization of conversation history.

## 13. Interview Summary
"Pipeline overlaps STT, LLM, and TTS with speculative decoding and routing to minimize perceived latency while safeguarding privacy through on-device preprocessing and selective cloud escalation."
