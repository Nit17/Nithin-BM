# Enterprise Chatbot Integrating 25+ REST APIs

## 1. Problem Statement
Provide a unified conversational interface for employees/partners to query and act across >25 heterogeneous REST APIs (HR, CRM, Finance, Inventory, ITSM, Analytics) with secure, auditable, low-latency responses and role-scoped actions.

## 2. Goals & Non-Goals
| Goals | Non-Goals |
|-------|-----------|
| Seamless natural language access to APIs | Replacing underlying transactional systems |
| Fine-grained RBAC & data masking | Building new core domain data models |
| Latency p95 < 1.2s for query-type requests | Supporting offline batch processing |
| Auditable action execution with approvals | Full autonomous multi-step execution without human oversight |
| Multi-tenant (BU / region segregation) | Cross-company federation |

## 3. High-Level Architecture
```
User → Channel (Web/Teams/Slack) → API Gateway → Orchestrator/Router
   → (AuthZ + Policy Engine) → Intent & Slot Extraction (LLM) → Tool Selector
   → API Aggregation Layer (per-domain connectors + caching) → Response Composer (LLM)
   → Safety & Redaction Filters → Output & Audit Log
```
Supporting systems: Vector Store (API schema embeddings, historical Q&A), Feature Store (usage/latency), Secret Manager, Observability Stack.

## 4. Component Detail
### 4.1 Channel Adapters
- Web widget (iframe) + Slack/Teams bots.
- Normalize message model (user_id, tenant_id, channel, text, attachments).

### 4.2 Gateway
- JWT / OAuth2 token validation.
- Rate limiting (token bucket per user + global concurrency guard).
- Request ID & trace propagation.

### 4.3 Policy & AuthZ
- Attribute-based access control (ABAC) combining user role, department, region, data sensitivity label.
- Policy engine (e.g., OPA) returns allow/deny + redaction rules.

### 4.4 Semantic Layer
- Prompt template includes: user intent history (last 3 turns), available tools summary (top 5 relevant), constraints.
- Intent classification (classifier model) → route: (Query, Action, Multi-Step, Escalation).
- Slot filling: structured JSON schema extraction (entity, target_system, date_range, filters).

### 4.5 Tool / API Registry
- Each API registered with: name, description, JSON schema for inputs/outputs, auth scope, rate limits, masking rules.
- Embedding vector for semantic matching between user request and tool descriptions.

### 4.6 API Aggregation Layer
- Connector microservices grouped by domain (e.g., finance-svc, hr-svc) handling retries, pagination, partial failures.
- Circuit breakers & bulkheads to isolate slow dependencies.
- Response normalization to canonical internal schema.

### 4.7 Retrieval Augmentation
- Vector store of historical Q&A pairs, API schema docs, common playbooks.
- Hybrid search (BM25 + dense) to surface relevant examples / schemas in context.

### 4.8 Response Composer
- LLM builds final answer citing API sources (system + tool call logs) and indicates confidence.
- Structured output envelope: `{answer, citations[], actions_executed[], confidence, redactions[]}`.

### 4.9 Safety & Compliance
- PII classifier + redaction (mask digits, emails) before returning.
- Action types needing approval (e.g., finance transfer) routed to approver queue (human-in-loop) → upon approval, tool re-invoked.

### 4.10 Observability & Audit
- Metrics: intent distribution, tool success rate, average tools per conversation, latency breakdown (NLP vs API), refusal / denial counts.
- Full audit event per action: user, pre-prompt hash, tool args (hashed sensitive fields), result, policy decision, timestamp.

## 5. Data Flow (Single Query)
1. User message arrives at channel; forwarded to gateway.
2. Gateway authenticates, enriches with profile, assigns trace ID.
3. Policy engine evaluates + returns allowed tools / fields masks.
4. LLM performs intent + slot extraction.
5. Tool selector ranks candidate API connectors; selects top (or orchestrates multi-call plan) with JSON args.
6. API calls executed (parallel where safe); partial failures summarized.
7. Retrieval adds similar Q&A or schema snippets if clarifying needed.
8. Response composer LLM summarizes results + citations.
9. Safety redaction + final formatting; delivered & audited.

## 6. Scaling & Performance
- Dynamic micro-batching of LLM inference (≤20ms window).
- Cache stable schema/tool descriptions + common read-only query results (TTL by system).
- Prioritize fast path for pure retrieval queries (skip tool plan if high-confidence FAQ match).
- Pre-warm connectors; connection pooling.
- Token optimization: compress history (summarize ≥10 turns); embed tools once.

## 7. Reliability & Resilience
- Circuit breakers per connector; fallback to cached data (with staleness disclaimer).
- Degrade gracefully: if LLM down → direct keyword routed templated queries for high-value endpoints.
- Dead-letter queue for failed actions; operator dashboard.

## 8. Security & Compliance
- Secrets in vault; short-lived access tokens per connector.
- Fine-grained scopes: read vs write split for each API.
- Row/field-level masking rules from policy engine enforced pre-output.
- GDPR: right-to-be-forgotten implemented via event-sourced user data references (indirection layer).

## 9. Human-in-Loop & Feedback
- Thumbs up/down + optional rationale stored with embeddings for future ranking / fine-tune.
- Low-confidence (< threshold) answers trigger clarification question pattern.
- Active learning queue surfaces high-disagreement samples weekly.

## 10. Cost & Optimization
- Track cost per conversation (LLM tokens + API billable events).
- Route simple deterministic FAQs to smaller cheaper model.
- Evaluate speculative decoding & quantized model for lower latency tier.

## 11. Tech Stack (Example)
| Layer | Choice |
|-------|--------|
| LLM Inference | vLLM + quantized 13B + frontier fallback |
| Embeddings | E5-large or GTE-base |
| Vector DB | Qdrant (HNSW + payload filters) |
| Gateway | Kong / Envoy |
| Policy | OPA (Rego) |
| Connectors | FastAPI microservices |
| Queue | Kafka / Redpanda |
| Observability | OpenTelemetry + Prometheus + Grafana |
| Secrets | HashiCorp Vault |

## 12. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Tool explosion (maintenance) | Auto-generate registry from OpenAPI specs + nightly validation |
| Latency spikes from slow APIs | Parallel + timeout + partial result summarization |
| Data leakage via tool response | Schema-level allowlist + redaction filters |
| Prompt injection via API text | Treat API text as untrusted; delimit & sanitize |
| Over-fitting to feedback spam | Weighted sampling + dedupe user IDs |

## 13. Future Enhancements
- Multi-modal (image attachments → OCR tool).
- Proactive suggestions (subscription to events → push notifications).
- Federated retrieval across data lakes.

## 14. Sequence Diagram (Text)
```
User->Gateway: Message
Gateway->Policy: Evaluate(user, resource, action)
Policy-->Gateway: Allow + mask rules
Gateway->LLM: Intent + Slot prompt
LLM-->Gateway: {intent, slots}
Gateway->Selector: Rank tools
Selector-->Gateway: Tool plan
Gateway->Connectors: Parallel API calls
Connectors-->Gateway: Normalized results
Gateway->Retrieval: Similar Q&A/context
Retrieval-->Gateway: Snippets
Gateway->LLM: Compose answer prompt
LLM-->Gateway: Answer + citations
Gateway->Safety: Redact / mask
Safety-->Gateway: Clean answer
Gateway->User: Response + audit ID
```

## 15. Interview Summary (Concise Pitch)
"A policy-aware orchestration layer routes user intents to a registry of toolified enterprise APIs using an LLM for planning and response composition, with retrieval of historical Q&A for context, strong RBAC & redaction, micro-batched inference for latency, and continuous feedback loops for improvement."
