# Full Fine-Tuning vs Adapters (LoRA/QLoRA/PEFT) Decision Framework

## 1. Objective
Provide a rigorous methodology to decide between (a) full model fine-tuning, (b) parameter-efficient adapters (LoRA/QLoRA), (c) prompt engineering + RAG, or (d) fusion strategies—balancing quality, cost, latency, governance, and operational risk.

## 2. Decision Dimensions
| Dimension | Full Fine-Tune | Adapters (LoRA/QLoRA) | Prompt + RAG |
|-----------|----------------|-----------------------|--------------|
| Data Volume (domain examples) | >200k high-quality | 5k–150k | <5k |
| Domain Shift (semantics/style) | Major | Moderate | Light |
| Latency Impact | Potentially higher (bigger base) | Minimal | Minimal |
| Infra Cost | High (multi-GPU epochs) | Low/Moderate | Very Low |
| Update Agility | Slow (retrain) | Fast (swap adapters) | Instant (index refresh) |
| Safety Re-audit Need | High each retrain | Moderate | Low |
| Catastrophic Forgetting Risk | Medium | Low | None |
| IP / Data Leakage Risk | Higher (weights encode) | Moderate | Low |
| Personalization Granularity | Global | Per-segment | Per-user (retrieval) |

## 3. Flowchart (Narrative)
1. Can retrieval of authoritative sources answer ≥80%? → Start with Prompt + RAG.
2. Is residual gap stylistic or minor? → Add lightweight adapter (LoRA).
3. Do you need deep reasoning improvement not solvable via context? → Consider larger base model first.
4. Do you have large proprietary dataset + strong eval signals + budget? → Full fine-tune.
5. Need rapid multi-vertical variants? → Multiple adapters on shared base.

## 4. Quantitative ROI Model
Let:
- C_full = GPU_cost_per_hour * hours_full_train * num_gpus
- C_lora = GPU_cost_per_hour * hours_lora_train * num_gpus + storage_adapters
- Q_x = quality metric (e.g., exact match / grounded correctness)
- ΔQ = Q_after - Q_baseline
- Payback_days = (ΔQ * business_value_per_point * interactions_per_day) / ΔCost
Use to compare scenarios; require Payback_days < threshold (e.g., 90 days) to justify full fine-tune.

## 5. Evaluation Harness
| Layer | Metric |
|-------|--------|
| Retrieval (if RAG) | Recall@k, MRR |
| Generation | Exact Match / ROUGE-L / BLEU (domain) |
| Factuality | Citation support %, hallucination rate |
| Safety | Toxicity %, refusal accuracy |
| Numeric | Relative error %, unit consistency |
| Latency | p95 end-to-end |
| Cost | $ / 1K tokens |

## 6. Risks & Controls
| Risk | Full Fine-Tune | Adapter | RAG |
|------|----------------|---------|-----|
| Overfitting | Strong (needs early stopping) | Moderate | None |
| Data Leakage (PII) | High | Medium | Low (docs sanitized) |
| Drift Response Time | Slow | Fast | Instant (reindex) |
| Governance Overhead | High | Medium | Low |

## 7. Storage & Deployment
- Full fine-tune: replace baseline weights; version with semantic tag (domain-major.minor.build).
- Adapters: store ΔW matrices; chainable; dynamic load (lazy) per request routing header.
- Hybrid: RAG + adapters: first attempt retrieval; fallback to adapter-enhanced reasoning.

## 8. Latency & Throughput Considerations
- Same base model + multiple adapters keeps KV cache reuse (hot path).
- Adapter merge at load-time for inference runtime that lacks native multi-adapter stacking.
- For high QPS multi-tenant: route by adapter ID → batch within adapter queue to avoid cross-adapter divergence.

## 9. Training Pipelines
| Stage | Full Fine-Tune | LoRA/QLoRA |
|-------|----------------|------------|
| Data Curation | Heavy filtering, dedupe, balance | Similar but smaller scale |
| Hardware | Multi-GPU (DP+TP) | Single / few GPUs |
| Precision | FP16/BF16 | 4-bit (QLoRA) + FP16 adapters |
| Checkpoints | Frequent (catastrophic risk) | Lightweight |
| Eval Frequency | Per epoch / mid-epoch | Per epoch or faster |

## 10. Decision Matrix Example
| Scenario | Recommended | Rationale |
|----------|------------|-----------|
| Niche legal assistant (30k docs) | RAG + LoRA | Retrieval for sources; adapter for tone/legal phrasing |
| High-volume support bot (5k transcripts) | RAG only | Not enough data for adapter; focus on retrieval quality |
| Proprietary algorithm explanation (250k labeled pairs) | Full Fine-Tune | Large dataset & deep domain semantics |
| Multi-regional marketing variants | Multiple Adapters | Style localization per region |
| Financial QA with live docs | RAG + Adapter (numeric templates) | Live updates + formatting control |

## 11. Governance & Compliance
- Approval checklist: data provenance → PII scrub → eval scores thresholds → safety panel sign-off.
- Change log per model/adapters includes: dataset hash set, training code commit, hyperparams, evaluation diff, safety exceptions.

## 12. Migration Path
Start: Prompt engineering baseline → Add RAG → Add adapters for stubborn error clusters → Reassess ROI quarterly → Escalate to full fine-tune only if sustained gap persists and dataset ≥ threshold.

## 13. Monitoring Post-Deployment
- Drift: Compare rolling eval (weekly) vs golden baseline; trigger retrain/adaptation if ΔQ < -X%.
- Safety regression: spike in refusal anomalies or toxicity metrics.
- Usage distribution by adapter → consolidate underutilized variants.

## 14. Interview Summary
"Choose full fine-tuning only with large, high-quality domain data and strong ROI; otherwise layer retrieval and parameter-efficient adapters for agility, lower cost, and controlled governance. Maintain evaluation-driven escalation and multi-adapter deployment for rapid verticalization."