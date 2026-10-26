"""scripts/type_pair_table.py — relation types seen in train per entity-type pair, as Markdown for the prompt."""
import argparse
from collections import Counter, defaultdict

from biored.data import load_config, load_splits

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/data.yaml")
ap.add_argument("--root", default=None)
args = ap.parse_args()
train = load_splits({"train": load_config(args.config, args.root)["train"]})["train"]

by_pair = defaultdict(Counter)
for d in train:
    for r in d.relations:
        if r.e1 in d.entities and r.e2 in d.entities and r.e1 != r.e2:
            by_pair[tuple(sorted((d.entities[r.e1].type, d.entities[r.e2].type)))][r.type] += 1
print("| Entity pair | Relation types in the training data (count) |\n|---|---|")
for (a, b), c in sorted(by_pair.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"| {a} - {b} | " + ", ".join(f"{t} ({n})" for t, n in c.most_common()) + " |")