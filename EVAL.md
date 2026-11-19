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

## Encoder (Phase 6)
`biored.models.encoder.DocRE` (`scripts/train_encoder.py`, `configs/encoder.yaml`): BiomedBERT-base
(`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`), typed entity markers, one pass
per abstract (max 512 tokens), pair vectors from logsumexp-pooled entity markers, a 9-way type head
and a novelty head. 10 epochs, batch 4 abstracts, learning rate 3e-5 (encoder) / 1e-4 (heads),
novelty loss weight 0.5, fp16 on a Kaggle T4. Trained on train only; all choices made on dev.

**neg_keep** (share of None pairs kept per epoch), seed 13, dev:

| neg_keep | Pair+type F1 | Pair F1 |
|---|---|---|
| **0.25 (chosen)** | **0.4647** | 0.6291 |
| 0.5 | 0.4542 | 0.6261 |
| 1.0 | 0.4545 | 0.6252 |

The three are within about 1 point, which is within noise on 100 abstracts; 0.25 is taken as the highest.

**Seeds**, neg_keep 0.25, dev. Each seed keeps its best epoch by pair+type F1 and its own
none_margin from a dev sweep over -0.8..0.4 (step 0.05):

| Seed | Best epoch | none_margin | Pair+type F1 | Pair F1 | Type+novelty F1 | Macro-F1 |
|---|---|---|---|---|---|---|
| 13 | 7 | 0.2 | 0.4647 | 0.6291 | 0.3936 | 0.3885 |
| **21 (headline)** | 9 | -0.2 | **0.4716** | 0.6389 | 0.3906 | 0.3777 |
| 42 | 10 | 0.25 | 0.4749 | 0.6534 | 0.4069 | 0.4548 |
| **Mean ± sd** | | | **0.4704 ± 0.0052** | 0.6404 ± 0.0122 | 0.3970 ± 0.0087 | 0.4070 ± 0.0417 |

The headline seed is the median seed by dev pair+type F1. Its none_margin (-0.2) is stored in
`configs/encoder.yaml`; Phase 10 serves this seed as int8.

**Truncation** (train + dev, 512 tokens after marking): 15.5% of abstracts are truncated and 1.4%
of reachable gold relations are lost. Test truncation is measured in Phase 8.

Reading the numbers:
- **Seeds agree.** The pair+type F1 spread is 1.0 point.
- **Macro-F1 is unstable** (spread ~0.08) because rare types have only 1–9 dev relations.
- **The margin barely matters.** For every seed, F1 varies by about 1 point anywhere between -0.2
  and 0.4. Seed 21's -0.2 is an interior peak: F1 falls steadily below it.
- **Finding pairs is the weaker part.** Type F1 is about 74% of pair F1, a ratio close to published
  pair-classifier models, so most of the gap is in pair detection. Truncation and the single pass per
  abstract are likely causes.
- **Against other systems on dev** (exploratory; dev CIs are about ±4–5 points): well above the
  baseline (0.2967) and below the frozen Claude prompt (Haiku 0.507–0.525, Sonnet 0.538–0.553,
  `PROMPT_LOG.md`). Test decides the headline comparisons.
- Seed 42's best epoch was the last (10). The plan fixes 10 epochs, so more training was not tried.
- Seed 21's dev metrics were recomputed on CPU in full precision during the margin re-sweep
  (`scripts/resweep_margin.py`): pair+type F1 0.4716, against 0.4712 in fp16 on GPU.
- The neg_keep 0.5 and 1.0 runs used the first margin sweep (-0.2..0.4) and were not re-swept.