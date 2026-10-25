"""Prove the check scorer (biored.eval) and NCBI's official script agree, then bootstrap.

Usage:
  python scripts/check_scorers.py data/kaggle/biored_test_orig.pubtator            # dev, synthetic predictions
  python scripts/check_scorers.py data/kaggle/biored_test_orig.pubtator --pred F    # also a real prediction file

Each synthetic prediction set is the gold with one controlled change, aimed at one way two
scorers can disagree (pair order, type, novelty, excluded type pairs, "None" lines, ...).
Agreement means identical TP/FP/FN at all four levels, which is stricter than F1 to 4 decimals.

PubTator files go to results/raw/ (git-ignored). The summary in results/runs/ holds IDs and
numbers only. Exit code 1 on any disagreement.
"""
import argparse
import dataclasses
import json
import random
import sys
from itertools import combinations
from pathlib import Path

from biored.data import load_pubtator
from biored.eval import (
    BC8_PAIRS,
    LEVELS,
    REL_TYPES,
    bootstrap,
    count_matrix,
    entity_types,
    load_preds,
    pair_key,
    score,
)
from biored.official import run_official, write_pubtator


def _with(r, **kw):
    """Copy a Relation with some fields changed (works for dataclasses and NamedTuples)."""
    return r._replace(**kw) if hasattr(r, "_replace") else dataclasses.replace(r, **kw)


def candidate_pairs(d, allowed=BC8_PAIRS):
    ents = sorted(d.entities.items())
    return [(a, b) for (a, ea), (b, eb) in combinations(ents, 2)
            if frozenset({ea.type, eb.type}) in allowed]


def perturbations(docs, seed=0):
    rng = random.Random(seed)
    gold = {d.pmid: list(d.relations) for d in docs}
    template = next(r for rs in gold.values() for r in rs)

    def new(a, b, t="Association", n="Novel"):
        return _with(template, e1=a, e2=b, type=t, novel=n)

    def per_doc(fn):
        return {d.pmid: fn(d, list(gold[d.pmid])) for d in docs}

    def false_pos(d, rs):
        taken = {pair_key(r) for r in rs}
        free = [p for p in candidate_pairs(d) if p not in taken]
        return rs + [new(a, b) for a, b in rng.sample(free, min(3, len(free)))]

    gene_variant = {frozenset({"GeneOrGeneProduct", "SequenceVariant"})}
    v = {
        "identity": gold,
        "empty": {},
        "reversed": per_doc(lambda d, rs: [_with(r, e1=r.e2, e2=r.e1) for r in rs]),
        "drop_30pct": per_doc(lambda d, rs: [r for r in rs if rng.random() > 0.3]),
        "type_flip_20pct": per_doc(lambda d, rs: [
            _with(r, type=rng.choice([t for t in REL_TYPES if t != r.type])) if rng.random() < 0.2 else r
            for r in rs]),
        "novelty_flip_20pct": per_doc(lambda d, rs: [
            _with(r, novel="No" if r.novel == "Novel" else "Novel") if rng.random() < 0.2 else r
            for r in rs]),
        "false_pos": per_doc(false_pos),
        "gene_variant_extra": per_doc(lambda d, rs: rs + [
            new(a, b) for a, b in candidate_pairs(d, gene_variant)]),
        "none_lines": per_doc(lambda d, rs: rs + [
            new(a, b, t="None") for a, b in candidate_pairs(d)[:2] if (a, b) not in {pair_key(r) for r in rs}]),
        # Every BC8 candidate pair as Association/Novel: a stand-in until the Phase 4 baseline exists.
        "all_association": per_doc(lambda d, rs: [new(a, b) for a, b in candidate_pairs(d)]),
    }
    mixed = {}
    for pm in gold:
        rs = v["type_flip_20pct"][pm]
        nov = {pair_key(r): r.novel for r in v["novelty_flip_20pct"][pm]}
        rs = [_with(r, novel=nov.get(pair_key(r), r.novel)) for r in rs if rng.random() > 0.2]
        extra = [r for r in v["false_pos"][pm] + v["gene_variant_extra"][pm] if pair_key(r) not in {pair_key(x) for x in gold[pm]}]
        mixed[pm] = [_with(r, e1=r.e2, e2=r.e1) for r in rs] + extra
    v["mixed"] = mixed
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--pred", action="append", default=[], help="PubTator prediction file (repeatable)")
    ap.add_argument("--raw", default="results/raw/scorer_check")
    ap.add_argument("--out", default="results/runs/scorer_agreement.json")
    ap.add_argument("--pred-jsonl", action="append", default=[],
                    help="JSON-lines predictions written by save_preds (repeatable)")
    args = ap.parse_args()

    docs = load_pubtator(args.gold)
    gold = {d.pmid: list(d.relations) for d in docs}
    types = entity_types(docs)
    raw = Path(args.raw)
    gold_clean = raw / "gold_clean.pubtator"
    write_pubtator(args.gold, gold, gold_clean)

    variants = perturbations(docs)
    for p in args.pred:
        pred_docs = load_pubtator(p)
        variants[f"file:{Path(p).name}"] = {d.pmid: list(d.relations) for d in pred_docs}

    for p in args.pred_jsonl:
        variants[f"jsonl:{Path(p).stem}"] = load_preds(p)

    summary, failures = {}, []
    print(f"{'predictions':24} {'level':18} {'check tp/fp/fn':>18} {'official tp/fp/fn':>18}   F1")
    for name, pred in variants.items():
        path = raw / f"{name.replace(':', '_')}.pubtator"
        write_pubtator(args.gold, pred, path)
        official = run_official(gold_clean, path)
        summary[name] = {}
        for lvl in LEVELS:
            mine = [int(x) for x in count_matrix(gold, pred, lvl, types).sum(0)]
            theirs = [official[lvl][k] for k in ("tp", "fp", "fn")]
            ok = mine == theirs
            f = 2 * mine[0] / max(2 * mine[0] + mine[1] + mine[2], 1)
            print(f"{name:24} {lvl:18} {'/'.join(map(str, mine)):>18} {'/'.join(map(str, theirs)):>18}"
                  f"   {f:.4f}{'' if ok else '  <-- MISMATCH'}")
            summary[name][lvl] = {"check": mine, "official": theirs, "official_f1": official[lvl]["f1"]}
            if not ok:
                failures.append(f"{name} {lvl}: check {mine} vs official {theirs}")

    s = score(gold, gold, types)
    perfect = all(abs(x - 1.0) < 1e-12 for x in s.values())
    print(f"\nscore(gold, gold) == 1.0 for all {len(s)} metrics: {perfect}")
    if not perfect:
        failures.append(f"score(gold, gold) not perfect: {s}")

    base = next((k for k in variants if k.startswith(("file:", "jsonl:"))), "all_association")
    ca = count_matrix(gold, variants[base], "pair_type", types)
    b = bootstrap(ca)
    print(f"bootstrap ({base}, pair+type, {len(ca)} docs): F1 {b['f1']:.4f}, "
          f"95% CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}], width {b['ci95'][1] - b['ci95'][0]:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"gold": Path(args.gold).name, "n_docs": len(docs), "agreement": summary,
         "bootstrap": {base: b}, "failures": failures}, indent=2))
    print("\n" + ("SCORERS AGREE" if not failures else "FAILED:\n  " + "\n  ".join(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
