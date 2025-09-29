# Interview Prep Guide

Central hub for structured preparation across mandatory and optional competencies for ML / GenAI engineer roles.

## Structure
| Folder/File | Purpose |
|-------------|---------|
| competency-matrix.md | Snapshot of required vs optional skills |
| python-advanced.md | Deep Python concepts & pitfalls |
| data-feature-engineering.md | Data prep & feature engineering techniques |
| ml-algorithms-evaluation.md | Algorithms & evaluation metrics |
| model-development-frameworks.md | PyTorch / TensorFlow patterns |
| deployment-integration.md | Serving, Docker, APIs, infra |
| analytical-problem-solving.md | Debug frameworks & case studies |
| communication-collaboration.md | STAR stories & translation examples |
| agile-teamwork.md | Agile practices & cross-functional workflow |
| documentation-best-practices.md | Docs, ADRs, style |
| advanced-ai-domains.md | NLP / CV / Time series primers |
| mlops-tools-frameworks.md | MLflow, Kubeflow, Airflow, Jenkins |
| big-data-visualization.md | Spark, Hadoop, visualization libraries |
| distributed-computing.md | Scaling & parallelism concepts |
| version-control-cicd.md | Git workflows & CI/CD patterns |
| question-bank.md | Consolidated practice questions |
| study-plan-30-60-90.md | Sequenced preparation roadmap |
| cheat-sheets.md | Quick reference tables & links |

## How To Use
1. Start with `competency-matrix.md` to identify gaps.
2. Follow `study-plan-30-60-90.md` for pacing.
3. Regularly attempt items in `question-bank.md` (spaced repetition).
4. Use `cheat-sheets.md` day before interviews as rapid refresh.
5. Map your experiences to `communication-collaboration.md` templates.

## Layered Learning Strategy
- Foundation → Application → Optimization → Communication.
- Each file ends with: Checklist / Pitfalls / Practice.

## Cross-References
- RAG & system design scenarios live under `system-design/`.
- Core GenAI concepts in `docs/01-core-genai.md` etc.

## Recommended Reading Paths
| Goal | Sequence |
|------|----------|
| Rapid ML refresh (1 day) | competency-matrix → ml-algorithms-evaluation (sections 1–5) → data-feature-engineering (1–5) → python-advanced (1–6) |
| Deep model craft (week) | model-development-frameworks → ml-algorithms-evaluation (math + tuning) → data-feature-engineering (pipelines + leakage) → deployment-integration |
| Production readiness | deployment-integration → mlops-tools-frameworks → documentation-best-practices → analytical-problem-solving |
| System design narrative | system-design/* READMEs → advanced-ai-domains → cheat-sheets |
| Interview storytelling | communication-collaboration → agile-teamwork → competency-matrix evidence fill |

## External Reading (Non-Exhaustive)
| Topic | Source Type | Reference |
|-------|------------|-----------|
| Gradient Boosting | Paper | Friedman: Greedy Function Approximation |
| Calibration | Paper | Guo et al. (On Calibration of Modern Neural Networks) |
| Drift Detection | Article | Evidently blog: data & concept drift |
| Feature Stores | Blog | Uber Michelangelo / Feast docs |
| ML System Design | Book | Chip Huyen: Designing Machine Learning Systems |
| Testing ML | Article | Google: Rules of ML |
| Serving Optimization | Blog | vLLM / TensorRT-LLM architecture posts |
| Fairness Metrics | Paper | Hardt et al. Equality of Opportunity |
| Ranking Diversity | Paper | Carbonell & Goldstein MMR |
| Data Contracts | Blog | Monte Carlo / Data Contracts concept |

## Reading Cadence Suggestion
- Daily (30m): one focused section + flash review of previous notes.
- Weekly: summarize new insights in personal log to combat forgetting.
- After mock interview: map misses to specific file sections; schedule revisit.

## Progress Tracking
Add a personal log section below to mark readiness (% confidence) per module.

| Module | Confidence % | Notes |
|--------|--------------|-------|
| Python |  |  |
| Data / Features |  |  |
| Algorithms |  |  |
| Deployment |  |  |
| MLOps |  |  |
| Communication |  |  |

---
Personal Notes:
> Maintain examples with quantifiable impact (latency ↓, cost ↓, accuracy ↑).