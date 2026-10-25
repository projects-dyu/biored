# scripts/split_checks.py
"""PMID ranges per split and PMID overlap between splits (leakage check).

PMIDs are assigned in roughly increasing order over time, so the range is a
quick proxy for publication date; ~36,000,000 was reached around Aug-Sep 2022.
Exact dates need PubMed metadata.
"""

import sys
from pathlib import Path

K = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/kaggle")
SPLITS = {
    "train": ["biored_train.pubtator", "biored_dev_orig.pubtator"],
    "dev": ["biored_test_orig.pubtator"],
    "test": ["bc8_test.pubtator"],
}
CUTOFF = 36_000_000

pmids = {}
for split, files in SPLITS.items():
    ids = set()
    for fn in files:
        for line in (K / fn).read_text(encoding="utf-8").splitlines():
            if "|t|" in line:
                ids.add(line.split("|t|")[0])
    pmids[split] = ids
    nums = sorted(int(i) for i in ids if i.isdigit())
    old = sum(n < CUTOFF for n in nums)
    print(
        f"{split}: n={len(ids)} min={nums[0]} median={nums[len(nums) // 2]} max={nums[-1]} "
        f"below_{CUTOFF}={old}"
    )

names = list(pmids)
for i, a in enumerate(names):
    for b in names[i + 1 :]:
        shared = sorted(pmids[a] & pmids[b])
        print(
            f"overlap {a}/{b}: {len(shared)}"
            + (f" e.g. {shared[:5]}" if shared else "")
        )

train_max = max(int(i) for i in pmids["train"] | pmids["dev"] if i.isdigit())
newer = sum(int(i) > train_max for i in pmids["test"] if i.isdigit())
print(
    f"test PMIDs newer than every train/dev PMID ({train_max}): {newer}/{len(pmids['test'])}"
)
