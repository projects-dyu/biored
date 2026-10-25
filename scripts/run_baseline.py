import argparse
from pathlib import Path

import yaml

from biored.data import load_config, load_splits
from biored.eval import bootstrap, count_matrix, entity_types, save_preds, score, validate_preds
from biored.models.baseline import CooccurrenceBaseline
from biored.track import log_run

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/data.yaml")
ap.add_argument("--root", default=None)
args = ap.parse_args()

splits = load_splits(load_config(args.config, args.root))
dev = splits["dev"]
gold = {d.pmid: list(d.relations) for d in dev}
types = entity_types(dev)

model = CooccurrenceBaseline().fit(splits["train"])
print("majority label per entity-type pair (from train):")
for c in sorted(model.type_):
    print(f"  {c[0]}–{c[1]}: {model.type_[c]} / {model.novel_[c]}")

metrics, counts = {}, {}
for same in (False, True):
    name = f"baseline_dev_same{int(same)}"
    pred = {d.pmid: model.predict(d, same) for d in dev}
    validate_preds(pred, dev)
    save_preds(pred, f"results/preds/{name}.jsonl")
    m = score(gold, pred, types)
    m["n_pred"] = sum(len(rs) for rs in pred.values())
    log_run(name, {"model": "cooccurrence", "split": "dev", "same_sentence_only": same}, m)
    c = count_matrix(gold, pred, "pair_type", types)
    b = bootstrap(c)
    metrics[same], counts[same] = m, c
    print(f"{name}: P={m['precision_pair_type']:.3f} R={m['recall_pair_type']:.3f} "
          f"F1={m['micro_f1_pair_type']:.4f} 95% CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] "
          f"| pair F1={m['micro_f1_pair']:.4f} novelty F1={m['micro_f1_pair_type_novelty']:.4f} "
          f"macro={m['macro_f1_type']:.4f} n_pred={m['n_pred']}")

d = bootstrap(counts[True], counts[False])   # paired: same1 minus same0, exploratory
print(f"same1 − same0 pair+type F1: {d['diff']:+.4f}, "
      f"95% CI [{d['diff_ci95'][0]:+.4f}, {d['diff_ci95'][1]:+.4f}], p={d['p_two_sided']:.3f}")

best = max((False, True), key=lambda s: metrics[s]["micro_f1_pair_type"])
cfg = {
    "model": "cooccurrence",
    "same_sentence_only": best,
    "chosen_on": "dev",
    "metric": "micro_f1_pair_type",
    "dev_micro_f1_pair_type": {f"same{int(s)}": round(metrics[s]["micro_f1_pair_type"], 4)
                               for s in (False, True)},
    "dev_diff_same1_minus_same0_ci95": [round(x, 4) for x in d["diff_ci95"]],
}
Path("configs/baseline.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"chosen: same_sentence_only={best} -> configs/baseline.yaml")
