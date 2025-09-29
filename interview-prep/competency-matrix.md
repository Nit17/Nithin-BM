# Competency Matrix

| Skill / Category | Mandatory? | Depth Expectation | Evidence Examples | Self-Rating (1-5) | Gap Notes |
|------------------|------------|-------------------|-------------------|-------------------|-----------|
| Python Advanced (iterators, generators, decorators) | Yes | Implement + optimize idiomatic code | Custom iterator, context manager, micro-optimization |  |  |
| Python Functional Constructs (map/filter, functools, itertools) | Yes | Select appropriate paradigm | Refactor loops to lazy pipelines |  |  |
| Data Handling (Pandas, data types, memory) | Yes | Optimize large dataset flows | Chunked ingest, dtype reduction |  |  |
| Feature Engineering | Yes | Design robust pipelines | Target encoding, leakage prevention |  |  |
| ML Algorithms (classic) | Yes | Select & tune | Compare logistic vs tree ensembles |  |  |
| Evaluation & Metrics | Yes | Choose / justify | ROC vs PR rationale |  |  |
| Model Development (PyTorch / TF) | Yes | Build training loop, debug | Custom loss, gradient issue fix |  |  |
| Deployment (Docker + API) | Yes | Ship reliable service | FastAPI + health + logging |  |  |
| Cloud (AWS/Azure/GCP basics) | Yes | Use managed primitives | S3 + Lambda inference POC |  |  |
| Analytical Debugging | Yes | Root cause systematically | Profiling memory spike fix |  |  |
| Communication | Yes | Clear narrative & impact | STAR story with metrics |  |  |
| Agile Collaboration | Yes | Team integration | Sprint planning ownership |  |  |
| Documentation & Best Practices | Yes | Produce reusable docs | ADR + onboarding guide |  |  |
| NLP Fundamentals | No | Working knowledge | Tokenization tradeoff explanation |  |  |
| Computer Vision Basics | No | Conceptual | Augmentation strategy outline |  |  |
| Time Series | No | Conceptual + basic modeling | ARIMA vs LSTM justification |  |  |
| MLOps Tooling (MLflow, Airflow) | No | Instrumentation & tracking | Registered model lifecycle |  |  |
| Big Data (Spark) | No | Optimize distributed job | Partition + predicate pushdown |  |  |
| Visualization (Matplotlib/Seaborn/Plotly) | No | Effective storytelling | Multi-facet anomaly chart |  |  |
| Distributed Computing Concepts | No | Recognize patterns | Data vs model parallel rationale |  |  |
| CI/CD + Version Control | No | Workflow design | GitHub Actions test matrix |  |  |

## Usage
1. Score Self-Rating each week; track movement.
2. Fill Gap Notes with concrete action ("implement X", "read Y").
3. Prioritize Mandatory items with rating ≤3 before touching optional.

## Readiness Thresholds
- Apply only when all Mandatory ≥4 and at least 3 optional ≥3.

## Rubric Guidelines
| Rating | Definition | Evidence Examples |
|--------|-----------|------------------|
| 1 | Aware only | Can define term minimally |
| 2 | Emerging | Can follow existing pattern with guidance |
| 3 | Proficient | Implements independently; explains tradeoffs |
| 4 | Advanced | Optimizes, debugs subtle issues, mentors others |
| 5 | Expert | Designs novel solutions; sets org standards |

Promotion Readiness: Majority Mandatory at 4, at least two at 5 indicating spikes.

## Sample Question Mapping
| Skill | Sample Interview Questions |
|-------|----------------------------|
| Python Advanced | Explain iterator vs generator; implement custom context manager |
| Feature Engineering | Prevent leakage in target encoding? Design real-time vs offline feature sync |
| Evaluation & Metrics | Pick metric for extreme imbalance; calibrate probabilities |
| Deployment | How to blue/green a model API? Add canary shadowing |
| MLOps Tooling | Track lineage and metrics across retrains? |
| Communication | STAR story on reducing inference latency |
| Data Handling | Memory-optimize 10GB CSV ingest |
| Debugging | Diagnose sudden AUC drop after schema change |

## Reading References Per Skill
| Skill | Primary File Section | External Anchor |
|-------|----------------------|----------------|
| Evaluation | ml-algorithms-evaluation (5,14,15) | Calibration paper; fairness metrics |
| Feature Eng | data-feature-engineering (1–7,10) | Feature store blog |
| Deployment | deployment-integration (forthcoming) | FastAPI / K8s docs |
| Python | python-advanced (1–10) | Fluent Python (book) |
| MLOps | mlops-tools-frameworks (planned) | MLflow / Kubeflow docs |

## Improvement Log (Append Below)
| Date | Skill | Action Taken | Impact |
|------|-------|--------------|--------|
