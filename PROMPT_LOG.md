   # Frontier prompt log (Phase 5, dev only, haiku-off)

   | Version | k | Change | Pair+type F1 | Kept |
   |---|---|---|---|---|
   | v1 | 0 | baseline prompt | 0.446 | – |
   | v1 | 4 | few-shot examples | 0.466 | – |
   | v2 | 4 / 3 / 5 | Association rule filled in | 0.434 and 0.444 (reps) / 0.468 / 0.446 | k=3 |
   | v3 | 3 | include background relations | 0.467 | no |
   | v4 | 3 | per-pair label rules from the guideline | **0.507** (rep 1: 0.493) | **yes** |
   | v5 | 3 | v4 + inclusion rules from guideline §4 | 0.462 | no |

   Frozen: prompt_v4.md, k=3 (configs/frontier_frozen.yaml, tag frontier-freeze).

   ## Dev results, frozen prompt

   | Arm | Pair+type F1 | Failed (refusals) | Cost / 100 abstracts |
   |---|---|---|---|
   | haiku-off | 0.507 | 0 | $0.09 |
   | haiku-on | 0.525 | 0 | $0.12 |
   | sonnet-off | 0.553 | 0 | $0.80 (batch) |
   | sonnet-on | 0.538 | 0 | $0.81 (batch) |
   | opus-low | 0.538 | 6 (all bio refusals) | $2.19 (batch) |
   | opus-med | 0.550 | 5 (all bio refusals) | $1.66 (batch) |

   Refused abstracts score as zero predictions; no fallback model is used.