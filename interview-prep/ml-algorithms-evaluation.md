# ML Algorithms & Evaluation

## 1. Supervised Learning Overview
| Task | Common Models | Notes |
|------|---------------|-------|
| Classification | Logistic Regression, Random Forest, XGBoost, SVM, Neural Nets | Calibrate probabilities |
| Regression | Linear, Ridge/Lasso, Random Forest, Gradient Boosting | Examine residual patterns |
| Ranking | LambdaMART, XGBoost Rank, Neural Rankers | Metrics: NDCG, MRR |
| Sequence | RNN, LSTM, Transformer | Masking & padding awareness |

### When to Prefer Simpler Models
| Situation | Prefer | Rationale |
|-----------|--------|-----------|
| Small dataset (<10k rows) | Linear / Tree | Lower variance |
| High interpretability need | Linear / GAM | Transparent contributions |
| Wide sparse features | Linear (L1) | Feature selection via sparsity |
| Tabular heterogeneous | Gradient Boosting | Handles non-linearity & interactions |
| Extreme class imbalance | Trees / XGBoost + calibration | Robust splits, later probability fix |

## 2. Bias-Variance Intuition
- Underfit: high bias, training & validation poor.
- Overfit: low training error, high validation error.
- Regularization trades variance for bias.

Formal decomposition (regression):  E[(y - f̂(x))^2] = Bias[f̂(x)]^2 + Var[f̂(x)] + σ^2 (irreducible noise).

### Bias Indicators
- High training error, validation not much worse.
### Variance Indicators
- Training near perfect, validation much worse.
### Remediation Mapping
| Symptom | Fix Bias | Fix Variance |
|---------|----------|--------------|
| Underfit | Add features, decrease regularization, increase model capacity | N/A |
| Overfit | N/A | Regularize, more data, simplify model, early stopping |

## 3. Cross-Validation Patterns
| Pattern | Use Case |
|---------|----------|
| k-fold | General small/medium data |
| Stratified k-fold | Imbalanced classification |
| TimeSeriesSplit | Temporal dependency |
| GroupKFold | Group leakage risk |

Edge Cases:
- Nested CV for unbiased performance when also doing model selection.
- Rolling window CV for non-stationary time series.

Leakage Guard: Ensure feature generation does not peek across fold boundaries (e.g., target encoding within fold only).

## 4. Hyperparameter Tuning
| Strategy | Pros | Cons |
|----------|------|------|
| Grid Search | Systematic | Expensive |
| Random Search | Efficient exploration | Non-adaptive |
| Bayesian Optimization | Sample efficiency | Implementation complexity |
| Hyperband / ASHA | Early-stop poor configs | Needs monotonic signal |

Tuning Heuristics:
- Start with logarithmic sampling for learning rates.
- Limit depth / max_leaves early for tree models, then broaden.
- Use early stopping rounds on validation to bound search cost.

Parallelization: Asynchronous schedulers (Ray Tune / Optuna) reduce idle GPU/CPU time.

## 5. Key Metrics
| Metric | Task | Caveat |
|--------|------|--------|
| Accuracy | Classification | Misleading on imbalance |
| Precision/Recall/F1 | Classification | Tune threshold |
| ROC-AUC | Binary | Skew tolerant, not threshold specific |
| PR-AUC | Imbalanced | More informative when positives rare |
| Log Loss | Prob calibration | Sensitive to confident errors |
| MAE / RMSE | Regression | RMSE penalizes large errors |
| MAPE | Regression (%) | Undefined near zero |
| R^2 | Regression | Negative if worse than baseline |
| NDCG | Ranking | Position discount |
| MRR | Ranking | Focus top result |

### Metric Formulas
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (P * R) / (P + R)
- ROC-AUC ≈ Probability(model ranks random positive above random negative)
- Log Loss = - (1/N) Σ [ y log p + (1-y) log(1-p) ]
- RMSE = sqrt( (1/N) Σ (y - ŷ)^2 )
- NDCG@k = DCG@k / IDCG@k ; DCG@k = Σ (2^{rel_i}-1)/log2(i+1)

### Threshold Selection
Use precision-recall curve to locate operating point maximizing business utility U = α*Recall + β*Precision or cost-sensitive loss matrix.

### Probability Calibration
Methods: Platt Scaling, Isotonic Regression, Temperature Scaling (for deep nets).
Calibration Metrics: Brier Score, Expected Calibration Error (ECE).

## 6. Imbalanced Classification Workflow
1. Baseline (no resampling) metrics.
2. Use stratified k-fold.
3. Try class weights or focal loss.
4. Calibrate threshold optimizing F-beta for business tradeoff.
5. Monitor precision-recall drift post-deploy.

Additional Techniques:
- One-class classification for extreme minority novelty detection.
- Ensemble of under-sampled majority subsets (EasyEnsemble).
- Synthetic data caution: ensure no leakage of test distribution or duplication.

## 7. Feature Importance & Explainability
| Method | Notes |
|--------|------|
| Permutation | Model agnostic, slow |
| SHAP | Local + global, costly on large trees |
| Gain (Tree) | Biased toward high-cardinality |
| Coefficients (Linear) | Need standardized inputs |

Global vs Local:
- Global: permutation, gain, aggregated SHAP.
- Local: SHAP, LIME, Integrated Gradients.

Stability Concern: track drift in top-k important features over time; sudden shifts may indicate pipeline or distribution issues.

## 8. Model Comparison Protocol
| Step | Detail |
|------|--------|
| Fix data splits | Reproducibility |
| Define metric hierarchy | Primary vs secondary |
| Statistical test | McNemar / t-test on folds |
| Complexity vs gain | Avoid diminishing returns |

Statistical Testing Examples:
- Classification paired comparison: McNemar's test on disagreement contingency table.
- Regression across folds: paired t-test on fold-wise errors; check normality (Shapiro) or use Wilcoxon signed-rank.

Effect Size: Report Δmetric with confidence interval (bootstrap 1000 samples) for decision transparency.

## 9. Overfitting Controls
| Technique | Example |
|----------|---------|
| Regularization | L1/L2, dropout |
| Early Stopping | Validation patience |
| Data Augmentation | Images: flips; Text: back-translation |
| Ensembling | Bagging/stacking |
| Noise Injection | Input or embedding noise |

Data Augmentation Specifics:
- NLP: synonym replacement, back-translation, insertion/deletion.
- Tabular: SMOTE (caution), Gaussian noise on continuous features.
- Vision: mixup, cutmix for regularization.

## 10. Pipeline Reliability
- Use Pipeline objects to avoid leakage.
- Persist model + preprocessing together.
- Version metrics with model artifact.

Operational Safeguards:
- Hash data schema; alert on change.
- Store feature statistics snapshot (mean/std/min/max) for drift baseline.
- Canary deployment: shadow predictions vs production model.

## 11. Mathematical Snapshot
Logistic: p = σ(w·x). Loss: -[y log p + (1-y) log(1-p)].
Gradient: ∂L/∂w = (p - y) x.

Regularized (L2): Loss + λ/2 ||w||^2 → gradient adds λ w.

SVM (soft margin) primal (simplified): minimize 1/2 ||w||^2 + C Σ ξ_i subject to y_i (w·x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0.

XGBoost Objective: L = Σ l(y_i, ŷ_i^{(t-1)} + f_t(x_i)) + Ω(f_t); with Ω(f)=γT + 1/2 λ Σ w_j^2.

Gradient Boosting Step: Fit base learner to negative gradient of loss.

Calibration (Temperature Scaling): Optimize T minimizing NLL on validation: p_i = softmax(z_i / T).

## 12. Practice Questions
1. Choose metric for fraud detection (1:500) and justify.
2. Diagnose overfitting in gradient boosting logs.
3. Explain when PR-AUC preferred over ROC-AUC.
4. Design tuning plan for XGBoost on 5M rows.
5. Compare SHAP vs permutation importance tradeoffs.
6. Derive gradient for logistic regression and explain each term.
7. Design an experiment to verify calibration improvement.
8. Choose between Random Forest and XGBoost on sparse high-cardinality data—justify.
9. Detect and mitigate feature drift in production fraud model.
10. Explain why PR-AUC can degrade while ROC-AUC stays stable in drift scenario.
11. Provide strategy for ranking system where click labels are noisy (position bias).
12. How to reduce leakage risk in target encoding across CV folds.
13. Compare Isotonic Regression vs Platt Scaling assumptions.
14. Explain difference between ensembling and stacking; when stacking may hurt.
15. Outline fairness evaluation plan for loan approval model.

## 13. Checklist
- [ ] Clear metric hierarchy
- [ ] CV strategy documented
- [ ] Overfitting checks automated
- [ ] Explainability method defined
- [ ] Tuning budget justified
- [ ] Calibration evaluated & adjusted
- [ ] Drift monitoring (data & performance) active
- [ ] Fairness metrics (if applicable) reviewed
- [ ] Confidence intervals on key metrics

## 14. Fairness & Responsible Evaluation
| Concern | Metric | Example |
|---------|--------|---------|
| Demographic Parity | P(Ŷ=1 | A=a) similar | Hiring screening |
| Equal Opportunity | TPR parity across groups | Medical diagnosis |
| Predictive Parity | PPV parity | Credit risk |
| Calibration Within Groups | Reliability curves per group | Lending |

Mitigation: reweighing, group-specific thresholds, adversarial debiasing.

## 15. Drift & Monitoring
| Drift Type | Signal | Tool |
|-----------|--------|------|
| Data (covariate) | PSI, Jensen-Shannon divergence | Evidently / custom |
| Label drift | Δ class balance | Post-label arrival monitor |
| Concept drift | Rising residual error | Shadow challenger model |

Response Runbook:
1. Confirm data ingestion integrity.
2. Quantify impact (metric delta severity).
3. Retrain or recalibrate depending on drift type.

## 16. Model Risk Documentation
Include: purpose, data lineage, feature list, training date, validation metrics, fairness audit, limitations, retrain trigger thresholds.

## 17. Ranking Nuances
- Handle position bias: use inverse propensity weighting.
- Cold start: blend content-based and collaborative features.
- Diversity: apply MMR (λ * relevance - (1-λ) * similarity).

## 18. Regression Diagnostics
| Check | Purpose | Action |
|-------|---------|--------|
| Residual vs fitted scatter | Detect heteroscedasticity | Transform target / weighted loss |
| Q-Q plot | Normality assumption (if needed) | Robust regression |
| Influence (Cook's distance) | Outlier leverage | Investigate / cap |

## 19. Advanced Interview Scenarios
1. You have high ROC-AUC but poor precision at required recall—what do you do?
2. Latency budget cuts allowable ensemble size—prioritize pruning strategies.
3. Business changes positive rate—recalibration vs retrain decision criteria.
4. Ranking model exploited by click spammers—defenses?
5. Explain tradeoff between SHAP global summary and permutation stability.
6. Provide a design for continuous evaluation pipeline (daily rolling window).

## 20. Quick Reference (Cheat)
| Area | Key Reminder |
|------|--------------|
| Calibration | Always check with reliability curve |
| Thresholds | Optimize to cost matrix not raw F1 |
| Drift | Separate covariate vs concept |
| Importance | Prefer permutation for agnostic clarity |
| Experiment | Keep fixed seed & split for comparability |

## 21. Interview Soundbite (Extended)
"Beyond baseline metrics, I enforce calibration, fairness, drift detection, and statistically grounded model comparisons—treating evaluation as an ongoing governance process, not a one-time leaderboard exercise."

<!-- Original soundbite replaced by extended version above -->