# Secure Enterprise Document Integration (RAG + Compliance)

## 1. Problem Statement
Integrate sensitive enterprise documents (contracts, HR, financial, engineering) into a Retrieval-Augmented Generation pipeline while enforcing strict access control, privacy, retention, and auditability.

## 2. Core Requirements
| Category | Requirement |
|----------|------------|
| Access Control | RBAC + ABAC (attributes: region, clearance, project) |
| Data Privacy | PII detection + redaction policy tiers |
| Encryption | Ingest (TLS), at-rest (AES-256), key rotation (KMS/HSM) |
| Isolation | Index segmentation per sensitivity tier |
| Auditing | Query + document ID + decision rationale logged |
| Compliance | GDPR (RTBF), SOC2, ISO27001 alignment |
| Latency | < 700ms retrieval p95 |

## 3. Architecture Overview
1. Ingestion pipeline pulls sources (SharePoint, Git, S3, GDrive). 
2. PII + classification scan (NER + regex + ML classifier). 
3. Policy engine decides: redact / tokenize / block / encrypt field-level. 
4. Chunking + embeddings (per model per sensitivity tier). 
5. Encrypted storage of chunks + metadata tags: {department, region, sensitivity, retention_date}. 
6. Query flow: user context → authz → filtered vector search (pre-filter) → rerank → dynamic masking → answer synthesis with citations.

## 4. Data Classification
| Sensitivity | Examples | Handling |
|------------|----------|----------|
| PUBLIC | Press releases | Index freely |
| INTERNAL | Process docs | Standard retention |
| CONFIDENTIAL | Financial forecasts | Attribute-gated retrieval |
| RESTRICTED | PII, legal disputes | Encrypted, strict ABAC, no cross-region replication |

## 5. Metadata Strategy
- Mandatory tags: owner, sensitivity, region, retention_date, legal_hold_flag.
- Optional: model_override (force safer model), embargo_until.
- Vector index uses filtered search on sensitivity + region before similarity scoring (guardrail).

## 6. Access Control Model
- RBAC: roles (employee, manager, legal, finance-analyst, admin).
- ABAC: region == doc.region, clearance >= doc.sensitivity_level, project in doc.allowed_projects.
- Policy engine (OPA) evaluates ALLOW/DENY with explanation object.

## 7. Ingestion Pipeline
| Stage | Function |
|-------|----------|
| Source Connector | Pull delta (webhooks / polling) |
| Normalization | Convert to canonical markdown/text |
| PII Detection | Regex + ML + checksum for ID patterns |
| Classification | Zero-shot + rules ensemble |
| Redaction | Replace spans with tokens <PII_TYPE_X> |
| Chunking | Semantic / sliding window |
| Embedding | Sensitivity-specific embedding model |
| Encryption | Field & blob encryption (envelope) |
| Index Write | Upsert with metadata |

## 8. Retrieval Flow
1. User request + auth token.
2. Context builder fetches user attributes (HR DB / IAM claims).
3. Pre-filter query: sensitivity <= clearance && region == user.region.
4. Vector similarity search (k=40) → rerank (cross-encoder) → top_k.
5. Policy second-pass: mask redacted placeholders if user lacks attribute.
6. Answer generation with inline citation markers [doc_id:chunk].
7. Audit log persisted.

## 9. Security Controls
| Control | Description |
|---------|-------------|
| Row-Level Encryption | Per sensitivity key (KMS) |
| Key Rotation | Quarterly; re-encrypt lazily on access |
| Tamper Evidence | Hash chain for chunks & logs |
| Prompt Injection Defense | Strip external URLs, enforce allowlist domains |
| Output Guard | Detect PII re-exfiltration; drop or mask |

## 10. Audit & Forensics
Log schema:
{
  trace_id, user_id_hash, roles, attributes_hash, query_hash,
  retrieved_docs: [{doc_id, sensitivity, used}], policy_decision, timestamp,
  model_version, response_hash, redaction_actions
}
Retention separated from raw documents; supports selective purge.

## 11. Right-To-Be-Forgotten (RTBF)
- Index maintains inverted map user_id → chunk_ids.
- RTBF job: mark chunks tombstoned → remove from index → purge encrypted blob (delay queue for rollback window).
- Adapter retraining / eval sets regenerated excluding purged data.

## 12. Monitoring & Metrics
| Metric | Purpose |
|--------|---------|
| Unauthorized retrieval attempts | Detect probing |
| Redaction coverage % | PII detection efficacy |
| Retrieval latency p95 | SLA tracking |
| Citation groundedness % | Hallucination guard |
| Policy eval latency | Authz scalability |
| RTBF SLA compliance % | Regulatory adherence |

## 13. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| PII missed in scan | Layered detectors + periodic rescans |
| Attribute spoofing | Signed IAM tokens + mTLS |
| Over-broad access grants | Quarterly entitlement review |
| Prompt exfiltration | Output PII detector & domain filter |
| Index linkage attacks | Salted hashing of identifiers |

## 14. Future Enhancements
- Homomorphic encryption for extremely sensitive numeric analytics.
- Differential privacy noise for aggregate answers.
- Real-time adaptive risk scoring on queries.

## 15. Interview Summary
"Secure doc integration layers classification, redaction, encryption, and attribute-based filtered retrieval before similarity scoring; policies enforced twice (pre & post), with comprehensive audit, RTBF workflows, and strict metadata-driven isolation to maintain compliance and prevent sensitive data leakage."