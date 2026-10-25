# scripts/label_stats.py
"""Relation type, entity-pair and novelty counts per split, as Markdown tables.

Tables go to stdout; relations whose entity types look wrong (an ID with no
matching mention, or an OrganismTaxon/CellLine argument) go to stderr so they
can be explained in DATA_CARD.md.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

K = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/kaggle")
SPLITS = {
    "train": ["biored_train.pubtator", "biored_dev_orig.pubtator"],
    "dev": ["biored_test_orig.pubtator"],
    "test": ["bc8_test.pubtator"],
}
UNEXPECTED = {"?", "OrganismTaxon", "Organism", "CellLine"}

for split, files in SPLITS.items():
    docs, mentions = set(), 0
    rtype, pair, novel = Counter(), Counter(), Counter()
    odd = []
    for fn in files:
        id2types = defaultdict(set)
        for line in (K / fn).read_text(encoding="utf-8").splitlines():
            if "|t|" in line:
                docs.add(line.split("|t|")[0])
                id2types = defaultdict(set)
                continue
            f = line.split("\t")
            if len(f) >= 6 and f[1].isdigit():
                mentions += 1
                for i in f[5].replace(";", ",").split(","):
                    id2types[i.strip()].add(f[4])
            elif len(f) >= 5 and not f[1].isdigit():  # skips 4-field direction lines
                rtype[f[1]] += 1
                novel[f[4]] += 1
                ta = "/".join(sorted(id2types.get(f[2], {"?"})))
                tb = "/".join(sorted(id2types.get(f[3], {"?"})))
                a, b = sorted((ta, tb))
                pair[f"{a}-{b}"] += 1
                if UNEXPECTED & set(a.split("/") + b.split("/")):
                    odd.append(f"{split}\t{f[0]}\t{f[1]}\t{f[2]}={ta}\t{f[3]}={tb}")
    print(
        f"\n## {split}: docs={len(docs)} mentions={mentions} relations={sum(rtype.values())}"
    )
    for name, c in [
        ("Relation type", rtype),
        ("Entity pair", pair),
        ("Novelty", novel),
    ]:
        print(f"\n| {name} | n |\n|---|---|")
        for k, v in c.most_common():
            print(f"| {k} | {v} |")
    for row in odd:
        print("ODD", row, file=sys.stderr)
