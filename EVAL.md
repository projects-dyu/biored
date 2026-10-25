# Evaluation protocol

## Metrics
BC8 BioRED Subtask 1 metrics, micro-F1 = 2TP / (2TP + FP + FN), summed over all documents.
A relation is an unordered entity-ID pair; "None" is never a prediction.

| Level | Match requires | Role |
|---|---|---|
| pair | ID pair | context |
| pair + type | pair and relation type | **primary metric** |
| pair + novelty | pair and Novel/No | EVAL.md only |
| pair + type + novelty | all three | novelty column |
| macro-F1 | pair+type F1 per relation type, averaged over types present in the scored gold | macro column |

Macro-F1 averages 8 types on dev and 7 on test (no Drug_Interaction), so dev and test macro values are not comparable. Dev macro-F1 swings on single relations (Conversion n=1, Drug_Interaction n=2); do not tune on it.

## Scorer
`biored.eval` (check scorer) matches NCBI's `run_biored_eval.py` from `biored_re_source_code.tar`
on the BC8 FTP folder (SHA-256 74d71f00df3faf63d82418784f6f8a8543fb321df442237f3074ac584cc33021,
not redistributed). Like the official script it skips Disease–Disease, Gene–Variant and
Gene–OrganismTaxon relations in gold and predictions.

Agreement on dev (100 docs, 1,163 gold relations): identical TP/FP/FN at all four levels on
13 prediction sets. These are 11 controlled perturbations of the gold and both Phase 4 baseline
variants, read from their JSON-lines files (`scripts/check_scorers.py --pred-jsonl`,
`results/runs/scorer_agreement.json`). `score(gold, gold)` = 1.0 for all 21 metrics.

The no-filter baseline has the same pair-level counts as the every-BC8-candidate-pair set
(TP/FP/FN 1,163 / 5,884 / 0). This confirms that the project's candidate pairs and the pairs
BC8 scores are the same set once Gene–Variant is excluded.

## Predictions
Every system's predictions are stored in `results/preds/<run>.jsonl`, one line per relation
with `pmid`, `e1`, `e2`, `type` and `novel` (IDs and labels only, committed). Both scorers and
the bootstrap read these files. A PMID with no lines has no predictions.

## Test scoring
The official script cannot read the test gold (direction lines, relation IDs with no mention),
so test scores come from `biored.eval`. It scores 6,035 gold relations; the 17 that no
pair-based model can predict count as false negatives (recall ceiling 0.9972).
Final test predictions are also submitted to the BC8 post-task CodaLab Test phase as an
external check; both numbers are reported.

All predictions pass `validate_preds()` before scoring. Test is scored once per final system,
after models and prompts are frozen.

## Uncertainty
Paired bootstrap over abstracts: 10,000 resamples, seed 0, 95% percentile intervals.
Models are compared on the same resampled abstracts.

## Headline comparisons (fixed before any test results)
Primary metric on test, paired bootstrap:
1. Best encoder (chosen on dev) vs Claude Haiku
2. Best encoder vs Claude Sonnet
3. Best encoder vs Claude Opus

Three comparisons: report 98.3% intervals (Bonferroni) alongside the 95% ones.
Everything else is exploratory, including Qwen, thinking settings, per-type F1, the
learning curve, the PMID < 36,000,000 vs ≥ 36,000,000 contamination split, and the
baseline variant choice below.

## Baseline (Phase 4)
`biored.models.baseline.CooccurrenceBaseline` (`scripts/run_baseline.py`). It predicts every
candidate pair with the relation type and novelty most common in train for that entity-type
pair. The train majority labels are:
- Chemical–Disease: Positive_Correlation
- Chemical–Chemical: Negative_Correlation
- all other pairs: Association
- novelty: Novel for every pair.

The same-sentence variant keeps only pairs whose entities have mentions in a shared sentence.
It uses a rough regex splitter, and the title counts as one sentence.

Dev results (pair + type unless noted):

| Variant | Predictions | P | R | F1 | 95% CI | Pair F1 | Type+novelty F1 | Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| all candidate pairs | 7,591 | 0.102 | 0.618 | 0.1752 | [0.1395, 0.2163] | 0.2833 | 0.1306 | 0.0603 |
| **same sentence (chosen)** | 2,991 | 0.212 | 0.493 | **0.2967** | [0.2501, 0.3497] | 0.4858 | 0.2108 | 0.0989 |

Same-sentence minus all-pairs: +0.1216, paired 95% CI [+0.0825, +0.1570]. None of the
10,000 resamples had a difference ≤ 0. The chosen variant is frozen in
`configs/baseline.yaml` (`same_sentence_only: true`). It is the floor every model must beat
and is scored on test once, alongside the final systems.

Reading the numbers:
- **Cross-sentence relations are common.** The same-sentence filter drops 61% of predictions
  but only 225 of 1,163 gold pairs: pair recall falls from 1.000 to 0.807. About 19% of dev
  gold relations therefore have no sentence where both entities appear (by this splitter).
  These are relations a sentence-level model cannot find.
- **Majority types add a little.** They add +0.020 pair+type F1 over labelling every pair
  Association (0.175 vs 0.155; 719 vs 635 true positives). All of the gain comes from the
  Chemical–Disease and Chemical–Chemical rules.
- **Macro-F1 is low by construction.** The baseline only ever predicts three of the eight
  types.
- **Novelty is constant.** The baseline predicts Novel everywhere, so its pair+novelty true
  positives are exactly the Novel gold pairs it reaches.

## Dev-set precision
A near-trivial predictor (every candidate pair = Association/Novel) gets dev pair+type F1 0.155,
95% CI [0.121, 0.193]. The baseline CIs are 0.077 and 0.100 wide. With 100 abstracts, dev
differences of a few points are not evidence; use dev for choosing, test for claims.
