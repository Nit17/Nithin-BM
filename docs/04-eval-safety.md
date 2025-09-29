# 4. Evaluation & Safety

## Core Metrics (BLEU, ROUGE, F1, RAGAS)
- BLEU: n-gram precision + brevity penalty (MT focus); weak on semantics.
- ROUGE: recall n-grams/LCS (summarization coverage); verbosity bias.
- F1: span/token precision & recall (extractive QA, IE).
- RAGAS: faithfulness, context precision/recall, answer relevance, (optionally correctness) via LLM judge.
- Complement: BERTScore/BLEURT (semantic), FactCC/QAFactEval (factual consistency).

## Hallucination & Groundedness
- Hallucination: unsupported or contradictory claims; fabricated citations.
- Reduction: strong retrieval + re-ranking; explicit instructions “answer only from context”; require citations; structured outputs; lower temperature; tool delegation (calculator/API) vs guessing; context hygiene; SFT & preference tuning rewarding abstention.
- Groundedness testing: claim extraction → evidence span check → NLI/entailment → metrics (attribution precision/recall, unsupported rate, citation coverage).

## Constitutional AI
- Constitution: natural-language principles (safety, privacy, honesty, fairness, boundaries).
- Inference-time critique & rewrite: draft → critique vs principles → revise.
- Training: AI preference labels (RLAIF) or DPO/ORPO using preferred (aligned) vs rejected outputs.
- Benefits: scalable alignment signals, auditable; Risks: over/under-refusal, rule conflicts, injection.
- Evaluate: refusal accuracy, harmful-advice rate, red-team bypass rate, principle citation quality.

## Prompt Injection & Data Poisoning
- Prompt injection: untrusted input attempts instruction override, secret exfiltration, unsafe tool use.
- Data poisoning: malicious content seeded in training/index/eval to bias or backdoor.
- Defenses: provenance labeling, immutable system channel, trust tiers, schema-constrained outputs, tool allowlists/timeouts, sanitized retrieval (strip scripts), re-ranking/compression, anomaly & pattern detectors, secrets isolation, data lineage + quality gates, backdoor screening.

## Bias / Toxicity / Fairness Monitoring
- Dimensions: toxicity/hate, stereotyping, refusal disparity, sentiment/politeness gaps, task accuracy parity.
- Datasets: RealToxicityPrompts, ToxiGen, CivilComments, HolisticBias, CrowS-Pairs, StereoSet, BOLD.
- Metrics: toxicity rate per group, refusal rate disparity, accuracy parity, stereotype association scores.
- Methods: counterfactual term swaps, paired prompts, distribution drift tests, ensemble safety classifiers + LLM judge.
- Mitigations: balanced SFT, detox rewrites, decoding constraints, post-filter cascade, retrieval sanitization, debiasing (careful), over-refusal correction list.

## Adversarial vs Jailbreak Prompts
- Adversarial = any exploit attempt (quality degradation, bias, leakage, unsafe output). Jailbreak = subset specifically bypassing safety to elicit disallowed content.
- Techniques: persona override, multi-turn priming, obfuscation, instruction smuggling, context flooding, retrieval poisoning.
- Mitigations: reinforcement of system prompt, normalization (Unicode, zero-width removal), safety cascade (fast filters → classifiers → LLM judge), structured output schemas, adversarial training refresh, provenance tags, streaming safety monitor, escalation after repeated attempts, automated red-team harness.
- KPIs: jailbreak bypass rate, false refusal rate, mean time to patch, tactic coverage.

---
End of Evaluation & Safety.

---
## Quick Reference
- Metrics Trio: Retrieval (Recall@k), Answer (Groundedness/Faithfulness), Safety (Refusal accuracy, Toxicity rate).
- Hallucination Mitigation Ladder: Retrieval quality → Prompt constraints → Tool use → Decode controls → Training alignment.
- Constitutional AI Flow: Draft → Critique → Revise → (Preference tune).
- Safety Cascade: Heuristic filters → Classifiers → LLM Judge → (Rewrite/Refuse).

## Common Pitfalls
| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Single safety classifier reliance | Concept drift blind spots | Ensemble + periodic re-labeling |
| Over-refusal | Utility loss & user frustration | Track false refusal; adjust thresholds |
| Ignoring retrieval recall | Groundedness ceiling | Evaluate recall before answer metrics |
| Unversioned safety prompts | Debug difficulty | Version & log prompt hashes |
| LLM judge temperature >0 | Metric variance | Fix temp=0 & seed |

## Interview Checklist
1. Distinguish hallucination vs lack of grounding.
2. Outline faithfulness evaluation pipeline.
3. Compare RLHF vs Constitutional AI alignment signals.
4. Design dashboard for live safety monitoring.
5. Mitigate prompt injection in RAG context assembly.

## Cross-Links
- Caching interplay with safety updates: see [Caching](08-caching.md#governance).
- Vector DB governance for poisoning: see [Vector Databases](09-vector-databases.md#ingestion--versioning).

## Further Reading
- RAGAS paper & implementation docs
- FactCC, QAFactEval
- Anthropic Constitutional AI blog
- OWASP LLM Top 10

## Practice Exercise
- Create a red-team suite outline covering jailbreak, injection, bias, and hallucination probes with pass/fail criteria.

---
