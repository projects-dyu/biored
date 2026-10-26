"""scripts/show_errors.py — usage: python scripts/show_errors.py results/preds/<run>.jsonl [seed]"""
import random, sys
from biored.data import load_config, load_splits
from biored.eval import load_preds, pair_key

dev = load_splits({"dev": load_config("configs/data.yaml", None)["dev"]})["dev"]
pred = load_preds(sys.argv[1])
errs = []
for d in dev:
    g = {pair_key(r): r for r in d.relations}
    p = {pair_key(r): r for r in pred.get(d.pmid, [])}
    for k in g.keys() | p.keys():
        if (g.get(k) and p.get(k) and (g[k].type, g[k].novel) == (p[k].type, p[k].novel)):
            continue
        errs.append((d, k, g.get(k), p.get(k)))
for d, (a, b), gr, pr in random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0).sample(errs, min(20, len(errs))):
    nm = lambda i: d.entities[i].mentions[0].text if i in d.entities else i
    lab = lambda r: f"{r.type}/{r.novel}" if r else "none"
    print(f"{d.pmid}  {nm(a)} — {nm(b)}   gold {lab(gr)}   pred {lab(pr)}")
