"""Check scorer for the BC8 BioRED Subtask 1 metrics, plus a paired bootstrap.

Matches NCBI's run_biored_eval.py (biored_re_source_code.tar on the BC8 FTP folder):
- a relation is an unordered ID pair, plus its type and/or novelty depending on the level
- relations whose entity types form one of EXCLUDED_PAIRS are dropped from gold AND predictions
- predictions are sets, so two lines for the same pair and type count once

Deliberate differences (they only matter on test, never on dev):
- the official script crashes on a gold relation whose ID has no mention; here it is a false negative
- the official script gives "-" the type of the last unnormalized mention; here "-" has no type
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biored.data import Relation

REL_TYPES = (
    "Association",
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
    "Negative_Correlation",
    "Positive_Correlation",
)
NOVELTY = ("Novel", "No")

# The eight entity-type pairs BC8 scores (CodaLab competition overview).
BC8_PAIRS = {
    frozenset(p)
    for p in [
        ("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"),
        ("ChemicalEntity", "DiseaseOrPhenotypicFeature"),
        ("DiseaseOrPhenotypicFeature", "SequenceVariant"),
        ("ChemicalEntity", "ChemicalEntity"),
        ("ChemicalEntity", "GeneOrGeneProduct"),
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
        ("ChemicalEntity", "SequenceVariant"),
        ("SequenceVariant", "SequenceVariant"),
    ]
}

# Type pairs the official script skips in gold and predictions alike.
# frozenset({"X"}) is the X-X pair.
EXCLUDED_PAIRS = {
    frozenset({"DiseaseOrPhenotypicFeature"}),
    frozenset({"GeneOrGeneProduct", "SequenceVariant"}),
    frozenset({"GeneOrGeneProduct", "OrganismTaxon"}),
}

LEVELS = {
    "pair": lambda r: (),
    "pair_type": lambda r: (r.type,),
    "pair_novelty": lambda r: (r.novel,),
    "pair_type_novelty": lambda r: (r.type, r.novel),
}


def pair_key(r) -> tuple[str, str]:
    """Unordered ID pair, ordered by string comparison as the official script does."""
    return (r.e1, r.e2) if r.e1 <= r.e2 else (r.e2, r.e1)


def entity_types(docs) -> dict[str, dict[str, str]]:
    """{pmid: {entity_id: entity_type}}, needed to apply EXCLUDED_PAIRS."""
    return {d.pmid: {i: e.type for i, e in d.entities.items()} for d in docs}


def _scored(rels, types, only_type=None):
    """Drop "None" predictions, excluded type pairs and, for per-type F1, other types."""
    out = []
    for r in rels:
        if r.type == "None" or (only_type is not None and r.type != only_type):
            continue
        if (
            types is not None
            and frozenset({types.get(r.e1), types.get(r.e2)}) in EXCLUDED_PAIRS
        ):
            continue
        out.append(r)
    return out


def doc_counts(gold, pred, level, types=None, only_type=None):
    """(tp, fp, fn) for one document. `types` is that document's {id: type} map."""
    f = LEVELS[level]
    g = {(pair_key(r), *f(r)) for r in _scored(gold, types, only_type)}
    p = {(pair_key(r), *f(r)) for r in _scored(pred, types, only_type)}
    return len(g & p), len(p - g), len(g - p)


def count_matrix(
    gold: dict,
    pred: dict,
    level: str,
    types: dict | None = None,
    only_type: str | None = None,
) -> np.ndarray:
    """One row per gold document, sorted by PMID: [tp, fp, fn].

    A document missing from `pred` has no predictions. Predictions for a PMID that is not
    in `gold` raise instead of being silently dropped.
    """
    extra = set(pred) - set(gold)
    if extra:
        raise KeyError(
            f"predictions for {len(extra)} PMIDs not in gold, e.g. {sorted(extra)[:3]}"
        )
    rows = [
        doc_counts(
            gold[pm],
            pred.get(pm, []),
            level,
            None if types is None else types.get(pm, {}),
            only_type,
        )
        for pm in sorted(gold)
    ]
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def f1(c: np.ndarray) -> np.ndarray:
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def score(gold: dict, pred: dict, types: dict | None = None) -> dict[str, float]:
    """All metrics in EVAL.md. Build gold as {d.pmid: d.relations for d in docs} so documents
    without gold relations still count their false positives, and types as entity_types(docs)."""
    out = {}
    for lvl in LEVELS:
        tp, fp, fn = (int(x) for x in count_matrix(gold, pred, lvl, types).sum(0))
        out[f"micro_f1_{lvl}"] = float(f1(np.array([tp, fp, fn])))
        out[f"precision_{lvl}"] = tp / (tp + fp) if tp + fp else 0.0
        out[f"recall_{lvl}"] = tp / (tp + fn) if tp + fn else 0.0
    per_type = {}
    for t in REL_TYPES:
        c = count_matrix(gold, pred, "pair_type", types, only_type=t).sum(0)
        if c[0] + c[2] > 0:  # the type occurs in the scored gold
            per_type[t] = float(f1(c))
    out["macro_f1_type"] = float(np.mean(list(per_type.values()))) if per_type else 0.0
    out.update({f"f1_{t}": v for t, v in per_type.items()})
    return out


def validate_preds(pred: dict, docs) -> None:
    """Raise on predictions the official scorer would mis-handle without complaint.
    Run it on every model's output before scoring (Phases 4-7)."""
    ents = {d.pmid: d.entities for d in docs}
    for pm, rels in pred.items():
        if pm not in ents:
            raise ValueError(f"unknown PMID {pm}")
        keys = [pair_key(r) for r in rels]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{pm}: more than one relation for the same pair")
        for r in rels:
            if r.type not in REL_TYPES:
                raise ValueError(
                    f"{pm}: type {r.type!r}; leave out 'None' pairs instead of writing them"
                )
            if r.novel not in NOVELTY:
                raise ValueError(f"{pm}: novelty {r.novel!r}")
            if r.e1 == r.e2 or r.e1 not in ents[pm] or r.e2 not in ents[pm]:
                raise ValueError(
                    f"{pm}: {r.e1}-{r.e2} is not a pair of two annotated entities"
                )


def bootstrap(
    ca: np.ndarray, cb: np.ndarray | None = None, n: int = 10_000, seed: int = 0
) -> dict:
    """ca, cb: count_matrix outputs for the same documents in the same order."""
    if cb is not None and ca.shape != cb.shape:
        raise ValueError("ca and cb must cover the same documents")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ca), size=(n, len(ca)))
    fa = f1(ca[idx].sum(axis=1))
    out = {"f1": float(f1(ca.sum(0))), "ci95": np.percentile(fa, [2.5, 97.5]).tolist()}
    if cb is not None:
        d = fa - f1(cb[idx].sum(axis=1))
        out["diff"] = out["f1"] - float(f1(cb.sum(0)))
        out["diff_ci95"] = np.percentile(d, [2.5, 97.5]).tolist()
        out["p_two_sided"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out


def save_preds(pred: dict, path) -> None:
    """One JSON line per relation: pmid, e1, e2, type, novel. IDs and labels only, so safe to
    commit under results/preds/. Pairs are written in pair_key order so git diffs stay stable."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for pm in sorted(pred):
            for r in sorted(pred[pm], key=lambda r: (pair_key(r), r.type, r.novel)):
                e1, e2 = pair_key(r)
                f.write(
                    json.dumps(
                        {
                            "pmid": pm,
                            "e1": e1,
                            "e2": e2,
                            "type": r.type,
                            "novel": r.novel,
                        }
                    )
                    + "\n"
                )


def load_preds(path) -> dict:
    """Inverse of save_preds: {pmid: [Relation]}. A document with no lines has no predictions,
    which count_matrix already treats as an empty list."""
    pred: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                pred.setdefault(x["pmid"], []).append(
                    Relation(x["e1"], x["e2"], x["type"], x["novel"])
                )
    return pred
