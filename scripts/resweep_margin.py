"""Re-sweep none_margin on dev for saved encoder runs over a wider grid. No retraining.

For each run it rebuilds the model from the saved weights, recomputes dev logits, checks that the
recorded dev F1 is reproduced at the recorded margin, sweeps the grid, and rewrites
results/runs/<run>.json (metrics, none_margin, grid) and results/preds/<run>.jsonl.
Loads train (for the candidate type pairs) and dev only; never touches test.

Usage:
  python scripts/resweep_margin.py --weights /kaggle/input/<weights dataset> \
      --root /kaggle/input/datasets/alanzee/biored-re-splits \
      encoder_dev_base_nk0.25_s13 encoder_dev_base_nk0.25_s21 encoder_dev_base_nk0.25_s42
Weights are read from <weights>/<run>.pt, or <weights>/<run>/model.pt.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from biored.data import allowed_type_pairs, load_config, load_splits
from biored.eval import entity_types, save_preds, score, validate_preds
from biored.models.encoder import MARKERS, DocRE, collate, decode, encode

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--weights", required=True)
ap.add_argument("--data-config", default="configs/data.yaml")
ap.add_argument("--root", default=None)
ap.add_argument("--lo", type=float, default=-0.8)
ap.add_argument("--hi", type=float, default=0.4)
ap.add_argument("--step", type=float, default=0.05)
ap.add_argument("--tol", type=float, default=0.002,
                help="allowed gap to the recorded F1; CPU fp32 vs Kaggle fp16 can differ slightly")
args = ap.parse_args()

dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = load_config(args.data_config, args.root)
splits = load_splits({s: paths[s] for s in ("train", "dev")})       # test is never loaded
allowed = allowed_type_pairs(splits["train"])
eligible = {t for pair in allowed for t in pair}
dev = splits["dev"]
gold, types = {d.pmid: list(d.relations) for d in dev}, entity_types(dev)
n = round((args.hi - args.lo) / args.step) + 1
margins = [round(args.lo + args.step * i, 2) + 0.0 for i in range(n)]

for name in args.runs:
    rec_path = Path(f"results/runs/{name}.json")
    rec = json.loads(rec_path.read_text())
    cfg = rec["params"]
    amp = dev_.type == "cuda" and str(cfg.get("fp16", True)) != "False"
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.add_special_tokens({"additional_special_tokens": MARKERS})     # same vocab as training
    w = Path(args.weights) / f"{name}.pt"
    if not w.exists():
        w = Path(args.weights) / name / "model.pt"
    model = DocRE(cfg["model"], len(tok))
    model.load_state_dict(torch.load(w, map_location="cpu"))
    model.to(dev_).eval()

    feats = [encode(d, tok, eligible, allowed, cfg.get("typed", True), cfg.get("max_len", 512)) for d in dev]
    logits = []
    with torch.no_grad():
        for i in range(0, len(feats), 8):                              # same batching as training
            chunk = feats[i:i + 8]
            b = collate(chunk, tok.pad_token_id)
            with torch.autocast(dev_.type, dtype=torch.float16, enabled=amp):
                tl, nl = model(**{k: b[k].to(dev_) for k in ("input_ids", "attention_mask", "ent_pos", "ent_mask", "pairs")})
            for j, x in enumerate(chunk):
                k = len(x["pairs"])
                logits.append((tl[j, :k].float().cpu(), nl[j, :k].float().cpu()))

    def predict(margin):
        return {d.pmid: decode(tl, nl, x["pair_ids"], margin) for d, x, (tl, nl) in zip(dev, feats, logits)}

    old_m, old_f1 = rec["metrics"]["none_margin"], rec["metrics"]["micro_f1_pair_type"]
    check = score(gold, predict(old_m), types)["micro_f1_pair_type"]
    print(f"{name}: recorded F1(type) {old_f1:.4f} at margin {old_m}, reproduced {check:.4f}")
    if abs(check - old_f1) > args.tol:
        raise SystemExit("  does not reproduce: wrong weights or code version; nothing written")

    sweep = {m: score(gold, predict(m), types)["micro_f1_pair_type"] for m in margins}
    margin = max(sweep, key=lambda m: (round(sweep[m], 6), -abs(m)))    # ties: margin closest to 0
    pred = predict(margin)
    validate_preds(pred, dev)
    rec["metrics"].update(score(gold, pred, types))
    rec["metrics"].update(none_margin=margin, n_pred=sum(len(r) for r in pred.values()))
    rec["params"]["margin_grid"] = [args.lo, args.hi, args.step]
    rec_path.write_text(json.dumps(rec, indent=2))
    save_preds(pred, f"results/preds/{name}.jsonl")
    print(f"  sweep: { {m: round(f, 4) for m, f in sweep.items()} }")
    print(f"  new none_margin={margin}: F1(type)={rec['metrics']['micro_f1_pair_type']:.4f}"
          + ("  <- still at the edge of the grid" if margin in (margins[0], margins[-1]) else ""))
    del model
    torch.cuda.empty_cache()