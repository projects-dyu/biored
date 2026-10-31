"""Phase 6: train the BiomedBERT encoder on train, pick the epoch and none_margin on dev.
Loads train and dev only; never touches test.

Writes
  <out>/enc_<tag>_nk<neg_keep>_s<seed>/   weights (model.pt), tokenizer with markers, run_config.json
  results/preds/<run>.jsonl               dev predictions (save_preds format, IDs and labels only)
  results/runs/<run>.json + MLflow        log_run record

Usage:
  python scripts/train_encoder.py --config configs/encoder.yaml --seed 13 \
      [--root /kaggle/input/biored-re-splits] [--out /kaggle/working/artifacts] [--neg-keep 0.25]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from biored.data import allowed_type_pairs, candidate_pairs, load_config, load_splits
from biored.eval import entity_types, save_preds, score, validate_preds
from biored.models.encoder import LABELS, MARKERS, NOVELTY, DocRE, collate, decode, encode, loss_fn, mark
from biored.track import log_run

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/encoder.yaml")
ap.add_argument("--data-config", default="configs/data.yaml")
ap.add_argument("--root", default=None, help="data folder, e.g. the Kaggle dataset path")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--neg-keep", type=float, default=None, help="overrides the config (dev sweep)")
ap.add_argument("--out", default="artifacts")
ap.add_argument("--epochs", type=int, default=None, help="overrides the config (smoke tests)")
ap.add_argument("--limit", type=int, default=None, help="first N train/dev docs (smoke tests, not logged)")
args = ap.parse_args()

# 1. config and seeds
cfg = yaml.safe_load(open(args.config))
if args.neg_keep is not None:
    cfg["neg_keep"] = args.neg_keep
if args.epochs is not None:
    cfg["epochs"] = args.epochs
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = dev_.type == "cuda" and cfg.get("fp16", True)

paths = load_config(args.data_config, args.root)
splits = load_splits({s: paths[s] for s in ("train", "dev")})       # test is never loaded
train, dev = splits["train"], splits["dev"]
if args.limit:
    train, dev = train[: args.limit], dev[: args.limit]
if cfg.get("train_frac", 1.0) < 1.0:                                 # learning-curve ablation
    train = random.Random(0).sample(train, round(len(train) * cfg["train_frac"]))
allowed = allowed_type_pairs(train)
eligible = {t for pair in allowed for t in pair}
typed, max_len = cfg.get("typed", True), cfg.get("max_len", 512)

tok = AutoTokenizer.from_pretrained(cfg["model"])
tok.add_special_tokens({"additional_special_tokens": MARKERS})
model = DocRE(cfg["model"], len(tok)).to(dev_)

gold = {d.pmid: list(d.relations) for d in dev}
types = entity_types(dev)
dev_feats = [encode(d, tok, eligible, allowed, typed, max_len) for d in dev]

# Truncation, measured once on train + dev (test is measured in Phase 8)
n_long = n_rel = n_lost = 0
for d in train + dev:
    n_long += len(tok(mark(d, eligible, typed)[0])["input_ids"]) > max_len
    kept = set(encode(d, tok, eligible, allowed, typed, max_len)["pair_ids"])
    cands = set(candidate_pairs(d, allowed))
    reachable = [r.key for r in d.relations if r.key in cands]       # loader-unreachable ones excluded
    n_rel += len(reachable)
    n_lost += sum(k not in kept for k in reachable)
trunc = {"trunc_doc_share": n_long / len(train + dev), "trunc_rel_lost_share": n_lost / max(n_rel, 1)}
print(f"truncation (train+dev): {trunc}")


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


@torch.no_grad()
def dev_logits():
    model.eval()
    out = []
    for chunk in batches(dev_feats, 8):
        b = collate(chunk, tok.pad_token_id)
        with torch.autocast(dev_.type, dtype=torch.float16, enabled=amp):
            tl, nl = model(**{k: b[k].to(dev_) for k in ("input_ids", "attention_mask", "ent_pos", "ent_mask", "pairs")})
        for i, x in enumerate(chunk):
            k = len(x["pairs"])
            out.append((tl[i, :k].float().cpu(), nl[i, :k].float().cpu()))
    model.train()
    return out


def predict(logits, margin):
    return {d.pmid: decode(tl, nl, x["pair_ids"], margin)
            for d, x, (tl, nl) in zip(dev, dev_feats, logits)}


# 3. optimizer: two learning rates, linear schedule with warmup, fp16, clipping
enc_params = [p for n, p in model.named_parameters() if n.startswith("enc.")]
new_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["lr_encoder"]},
                         {"params": new_params, "lr": cfg["lr_head"]}],
                        weight_decay=cfg.get("weight_decay", 0.01))
steps = cfg["epochs"] * math.ceil(len(train) / cfg["batch_size"])
sched = get_linear_schedule_with_warmup(opt, int(cfg.get("warmup", 0.1) * steps), steps)
scaler = torch.amp.GradScaler(dev_.type, enabled=amp)

best = {"f1": -1.0}
t0 = time.time()
for epoch in range(cfg["epochs"]):
    # 2. fresh negatives every epoch, shuffled by document
    rng = random.Random(args.seed * 1000 + epoch)
    feats = [encode(d, tok, eligible, allowed, typed, max_len, cfg["neg_keep"], rng) for d in train]
    rng.shuffle(feats)
    total, n = 0.0, 0
    for chunk in batches(feats, cfg["batch_size"]):
        b = {k: v.to(dev_) for k, v in collate(chunk, tok.pad_token_id).items()}
        if (b["labels"] == -100).all():
            continue                                  # no pairs: cross-entropy would be NaN
        with torch.autocast(dev_.type, dtype=torch.float16, enabled=amp):
            tl, nl = model(b["input_ids"], b["attention_mask"], b["ent_pos"], b["ent_mask"], b["pairs"])
        loss = loss_fn(tl, nl, b["labels"], b["novel"], cfg.get("nov_weight", 0.5))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total, n = total + loss.item(), n + 1
    # 4. keep the epoch with the best dev micro-F1 (pair + type)
    logits = dev_logits()
    f1 = score(gold, predict(logits, 0.0), types)["micro_f1_pair_type"]
    print(f"epoch {epoch + 1}: loss={total / max(n, 1):.4f} dev F1(type)={f1:.4f} [{time.time() - t0:.0f}s]")
    if f1 > best["f1"]:
        best = {"f1": f1, "epoch": epoch + 1, "logits": logits,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

# 5. sweep none_margin on dev, using the best epoch's logits
margins = [round(-0.2 + 0.05 * i, 2) + 0.0 for i in range(13)]   # -0.2 ... 0.4
sweep = {m: score(gold, predict(best["logits"], m), types)["micro_f1_pair_type"] for m in margins}
margin = max(sweep, key=sweep.get)
pred = predict(best["logits"], margin)
validate_preds(pred, dev)
metrics = {**score(gold, pred, types), **trunc, "best_epoch": best["epoch"], "none_margin": margin,
           "n_pred": sum(len(r) for r in pred.values()), "train_minutes": (time.time() - t0) / 60}
print(f"none_margin sweep: {sweep}")
print(f"best epoch {best['epoch']}, none_margin={margin}: dev F1(type)={metrics['micro_f1_pair_type']:.4f} "
      f"novelty F1={metrics['micro_f1_pair_type_novelty']:.4f} macro={metrics['macro_f1_type']:.4f}")
if args.limit:
    raise SystemExit("smoke test, not saved or logged")

# 6. save weights, tokenizer, config, dev predictions and the run record
name = f"encoder_dev_{cfg.get('tag', 'base')}_nk{cfg['neg_keep']}_s{args.seed}"
out = Path(args.out) / name
out.mkdir(parents=True, exist_ok=True)
torch.save(best["state"], out / "model.pt")
tok.save_pretrained(out)
run_cfg = {**cfg, "seed": args.seed, "none_margin": margin, "best_epoch": best["epoch"],
           "labels": LABELS, "novelty": list(NOVELTY), "eligible": sorted(eligible),
           "allowed": sorted(allowed)}
(out / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
save_preds(pred, f"results/preds/{name}.jsonl")
log_run(name, {**{k: v for k, v in cfg.items()}, "seed": args.seed, "split": "dev"}, metrics)
print("saved", out)
