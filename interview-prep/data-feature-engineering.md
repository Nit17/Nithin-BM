# Data Handling & Feature Engineering

## 1. Data Ingestion Patterns
| Pattern | Use Case | Notes |
|---------|----------|-------|
| Batch CSV + Chunking | Large historical load | Use iterator + dtype map |
| Streaming (Kafka) | Real-time features | Idempotent consumers |
| API Pull w/ Backoff | External augmentation | Cache ETags / timestamps |
| Data Lake (Parquet) | Columnar analytics | Predicate pushdown |

## 2. Memory Optimization (Pandas)
| Technique | Example |
|-----------|---------|
| Downcast numerics | pd.to_numeric(series, downcast='integer') |
| Category dtype | df['state'] = df['state'].astype('category') |
| Use Parquet | df.to_parquet('file.parquet') |
| Chunked read | pd.read_csv(..., chunksize=100_000) |

## 3. Cleaning & Preprocessing
| Step | Purpose | Tools |
|------|---------|-------|
| Missing value strategy | Preserve signal vs bias | mean/median, model-based |
| Outlier handling | Robust modeling | IQR, winsorizing |
| Text normalization | NLP consistency | lowercasing, unicode normalize |
| Date/time expansion | Cyclical encoding | sine/cosine transforms |

## 4. Feature Engineering Techniques
| Technique | Description | Example |
|-----------|-------------|---------|
| Target Encoding | Encode category via target mean | leak-safe folds |
| Frequency Encoding | Replace with counts | skewed categories |
| Interaction Features | Combine signals | product of price * qty |
| Lag Features | Temporal dependency | sales_t-1, sales_t-7 |
| Rolling Stats | Smoothing / seasonality | 7d mean, 30d std |
| Embeddings | Dense semantics | doc2vec/user2vec |
| Binning | Non-linearity capture | quantile bins |
| Scaling | Normalize variance | StandardScaler / MinMax |

## 5. Leakage Prevention
| Risk | Example | Mitigation |
|------|---------|-----------|
| Future info | Using t+1 sales | time-split validation |
| Target leakage | Pre-aggregated target in features | remove / recompute |
| Duplicated rows | Data duplication inflating metrics | hash + drop duplicates |

## 6. Imbalanced Data Strategies
| Strategy | Description | Notes |
|----------|-------------|-------|
| Resampling | Over/Under sample | Beware overfitting |
| Class Weights | Adjust loss | Works with many algos |
| Focal Loss | Focus hard examples | Useful in CV/NLP |
| Threshold Tuning | Move decision boundary | Optimize F1/Recall |

## 7. Feature Selection
| Method | Type | Notes |
|--------|------|------|
| Variance Threshold | Filter | Remove near-constant |
| Mutual Information | Filter | Non-linear detection |
| Recursive Feature Elimination | Wrapper | Expensive |
| Permutation Importance | Model-based | Post-fit stability |
| SHAP Values | Model-based | Local + global insight |

## 8. Pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
num_pipe = Pipeline([
    ("impute", SimpleImputer()),
    ("scale", StandardScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy='most_frequent')),
    ("encode", OneHotEncoder(handle_unknown='ignore'))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
model = Pipeline([
    ("pre", pre),
    ("clf", GradientBoostingClassifier())
])
```

## 9. Data Quality Checks (Great Expectations Style)
| Check | Example |
|-------|---------|
| Schema | column types stable |
| Range | 0 <= churn_rate <= 1 |
| Uniqueness | id unique |
| Null thresholds | < 5% missing |
| Drift | PSI < 0.2 |

## 10. Feature Store Concepts
| Aspect | Explanation |
|--------|------------|
| Online vs Offline | Low-latency vs batch analytics |
| Point-in-time correctness | Avoid lookahead bias |
| Lineage | Trace source -> feature -> model |

## 11. Practice Questions
1. Optimize memory for a 5GB CSV of transactions.
2. Implement leakage-safe target encoding.
3. Design lag + rolling features for demand forecasting.
4. Choose imbalanced strategy for 1:200 fraud detection.
5. Build data validation suite for daily feed ingest.

## 12. Checklist
- [ ] All features reproducible via code
- [ ] Leakage tests implemented
- [ ] Memory profile documented
- [ ] Drift monitoring configured
- [ ] Feature naming conventions consistent

## 13. Interview Soundbite
"I design reproducible, leakage-safe feature pipelines with memory-optimized ingestion, rigorous validation, and monitoring to sustain model performance over time."