"""One frontier arm on dev. Loads train and dev only.
Usage: python scripts/run_frontier.py --arm haiku-off --prompt configs/prompt_v1.md [--k 4] [--batch] [--rep 1] [--limit 5]"""
import argparse, time
from collections import Counter
from pathlib import Path
import anthropic, yaml
from biored.data import allowed_type_pairs, load_config, load_splits
from biored.eval import entity_types, save_preds, score, validate_preds
from biored.frontier.prompt import build_params, parse, prompt_hash
from biored.track import log_run

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--rep", type=int, default=0)
ap.add_argument("--batch", action="store_true")
ap.add_argument("--limit", type=int)
args = ap.parse_args()

paths = load_config("configs/data.yaml", None)
splits = load_splits({s: paths[s] for s in ("train", "dev")})
train, dev = splits["train"], splits["dev"][: args.limit or None]
eligible = {t for pair in allowed_type_pairs(train) for t in pair}

cfg = yaml.safe_load(open("configs/frontier_arms.yaml"))
arm = {**cfg["arms"][args.arm], "k": args.k}
system = Path(args.prompt).read_text(encoding="utf-8")
ph = prompt_hash(system, arm)                       # computed before cache_1h, so batch and sync match
arm["cache_1h"] = args.batch
params = {d.pmid: build_params(arm, d, train, system, eligible) for d in dev}
client = anthropic.Anthropic(max_retries=6)

msgs = {}
if args.batch:
    b = client.messages.batches.create(requests=[{"custom_id": pm, "params": p} for pm, p in params.items()])
    print("batch", b.id)
    while client.messages.batches.retrieve(b.id).processing_status != "ended":
        time.sleep(60)
    msgs = {r.custom_id: r.result.message for r in client.messages.batches.results(b.id)
            if r.result.type == "succeeded"}
else:
    for pm, p in params.items():
        try:
            msgs[pm] = client.messages.create(**p)
        except anthropic.BadRequestError:
            raise
        except anthropic.APIError as e:
            print(pm, type(e).__name__)

pred, tok = {}, Counter()
failed = dropped = kept = 0
for d in dev:
    if d.pmid not in msgs:
        pred[d.pmid] = []; failed += 1; continue
    rels, info = parse(msgs[d.pmid], d, eligible)
    pred[d.pmid] = rels
    failed += info["failed"]; dropped += info["dropped"]; kept += len(rels)
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        tok[k] += info["usage"].get(k) or 0

p = cfg["prices"][arm["model"]]
cost = (tok["input_tokens"] * p["input"] + tok["output_tokens"] * p["output"]
        + tok["cache_creation_input_tokens"] * p["cache_write_1h" if args.batch else "cache_write_5m"]
        + tok["cache_read_input_tokens"] * p["cache_read"]) / 1e6 * (0.5 if args.batch else 1.0)

validate_preds(pred, dev)
m = score({d.pmid: list(d.relations) for d in dev}, pred, entity_types(dev))
m |= {"failed_rate": failed / len(dev), "dropped_rate": dropped / max(kept + dropped, 1),
      "cost_usd": cost, **{f"tok_{k}": v for k, v in tok.items()}}
print(f"{args.arm} [{ph}] F1(type)={m['micro_f1_pair_type']:.4f} failed={failed} dropped={dropped} "
      f"${cost:.3f} {dict(tok)}")
if args.limit:
    raise SystemExit("smoke test, not logged")

name = f"frontier_dev_{args.arm}_{ph}_rep{args.rep}"
save_preds(pred, f"results/preds/{name}.jsonl")
log_run(name, {"arm": args.arm, **cfg["arms"][args.arm], "k": args.k, "prompt": Path(args.prompt).name,
               "prompt_hash": ph, "rep": args.rep, "batch": args.batch, "split": "dev"}, m)
