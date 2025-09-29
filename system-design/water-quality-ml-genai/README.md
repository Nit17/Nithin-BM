# Water Quality Prediction + GenAI Reporting System

## 1. Problem Statement
Predict near-term water quality metrics (e.g., turbidity, pH, dissolved oxygen) from sensor + weather data and generate natural-language compliance & insights reports with traceable data sources.

## 2. Goals & KPIs
| Goal | KPI |
|------|-----|
| Accurate short-term predictions | MAE / RMSE per metric |
| Explainable & source-linked reports | 100% citation coverage |
| Timely alerts | Detection → alert < 2 min |
| Regulatory compliance | Template adherence rate |

## 3. Architecture Overview
```
Sensors → Ingestion (IoT Hub/MQTT) → Stream Processor (clean, QC, feature enrich) → Feature Store
       → Prediction Service (ML model ensemble) → Time-Series DB
       → Report Generator (Retrieval + LLM) → Compliance Portal / Alerts
       → Feedback (Operator edits) → Label Store → Model Retraining Pipeline
```
Additional: Weather API, Static Site Characteristics DB, Vector Store (historical incidents, remediation actions).

## 4. Data Pipeline
- Streaming ingestion with late data handling (watermarks + windowing).
- Quality checks: sensor heartbeat, range validation, spike detection (MAD-based).
- Feature engineering: rolling averages (1h, 6h, 24h), gradients, weather joins (rainfall, temp), site categorical encodings.
- Store features (point-in-time correctness) in feature store (Feast or custom) keyed by (site_id, timestamp).

## 5. Prediction Layer
- Ensemble: Gradient Boosted Trees (tabular) + Temporal model (LSTM/Temporal Fusion Transformer) + rule overrides (hard thresholds for compliance breaches).
- Model selection per metric based on validation MAE; confidence scoring (ensemble variance).
- Online inference triggered each batch interval (e.g., 5 min windows).

## 6. Retrieval for Reporting
- Retrieve: last 24h anomalies, top features contributing (SHAP summary), similar past incidents (vector similarity on incident summaries), regulatory guideline snippets.
- Compose context ensuring numeric precision (fixed decimal formatting) & units.

## 7. Report Generation Prompt (Simplified)
```
SYSTEM: Generate a concise water quality report. Cite every factual statement with [source_id]. If prediction confidence < threshold, flag as low-confidence.
INPUT DATA: <tabular summary JSON>
ANOMALIES: <list>
PAST_INCIDENTS: <list>
GUIDELINES: <snippets>
FORMAT: {"summary":..., "metrics": [{metric, value, unit, delta, citation}], "alerts": [], "recommendations": []}
```

## 8. Validation & Post-Processing
- Numeric validator ensures reported values match prediction store (tolerance 0).
- Citation checker: each metric has at least one source chunk ID.
- Template linter verifies required JSON fields.

## 9. Alerts & SLA
- Threshold breaches or rapid deltas trigger real-time notification (SMS/email) with plain-text fallback if LLM service degraded.
- SLA: alert pipeline isolated (no dependency on full report generation path).

## 10. Explainability
- Store SHAP at inference; aggregate daily for feature drift monitoring.
- Provide per-report explanation overlay: top 3 contributing features per metric.

## 11. Feedback & Continuous Learning
- Operator edits captured as delta patches; labeler verifies and approves into label store.
- Weekly retraining job uses new labels; champion/challenger evaluation (stat sig tests) before promotion.

## 12. Observability
- Metrics: prediction MAE by site, drift score (PSI) on features, report generation latency, citation coverage %, alert false positive rate.
- Dashboards: feature drift heatmap, anomaly frequency timeline.

## 13. Tech Stack (Example)
| Layer | Choice |
|-------|--------|
| Stream | Kafka / Azure Event Hub |
| Processing | Flink / Spark Structured Streaming |
| Feature Store | Feast |
| Models | XGBoost + TFT + Rule Engine |
| Vector DB | Weaviate / Qdrant |
| Time-Series | InfluxDB / TimescaleDB |
| LLM | Domain-tuned open model (LoRA) |
| Serving | FastAPI + async workers |

## 14. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Sensor drift / calibration issues | Drift detection + recalibration workflow |
| Missing data gaps | Imputation windows + confidence downgrade |
| Hallucinated values | Numeric validator + citation enforcement |
| Latency spikes in retrieval | Cache recent incidents, warm vector index |
| Guideline changes | Version guideline snippets; include version in context |

## 15. Future Enhancements
- Geospatial spillover modeling (graph features).
- Active learning on uncertain predictions.
- Multi-modal (satellite imagery) enrichment.

## 16. Interview Summary
"Streaming feature pipeline feeds an ensemble predictor whose outputs, plus retrieved contextual incidents and regulations, are composed into a citation-enforced JSON report with numeric validation and operator feedback loop for continual improvement."
