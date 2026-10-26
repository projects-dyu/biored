from __future__ import annotations
import hashlib, json
from collections import Counter
from biored.data import Relation
from biored.eval import NOVELTY, REL_TYPES                      

SCHEMA = {
    "type": "object",
    "properties": {"relations": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "entity1_id": {"type": "string"},
            "entity2_id": {"type": "string"},
            "type": {"type": "string", "enum": list(REL_TYPES)},
            "novel": {"type": "string", "enum": list(NOVELTY)},
        },
        "required": ["entity1_id", "entity2_id", "type", "novel"],
        "additionalProperties": False}}},
    "required": ["relations"],
    "additionalProperties": False,
}

def entity_table(doc, eligible: set[str]) -> str:
    rows = [f"{eid}\t{e.type}\t" + " ; ".join(sorted({m.text for m in e.mentions}))   
            for eid, e in sorted(doc.entities.items()) if e.type in eligible]
    return "id\ttype\tmentions\n" + "\n".join(rows)

def render_doc(doc, eligible: set[str], with_answer: bool = False) -> str:
    s = f"<abstract>\n{doc.text}\n</abstract>\n<entities>\n{entity_table(doc, eligible)}\n</entities>"
    if with_answer:
        rels = sorted((r for r in doc.relations if r.e1 != r.e2),    
                      key=lambda r: (r.type, r.key))
        ans = {"relations": [{"entity1_id": r.e1, "entity2_id": r.e2, "type": r.type, "novel": r.novel}
                             for r in rels]}
        s += f"\n<answer>\n{json.dumps(ans)}\n</answer>"
    return s

def _profile(doc) -> Counter:
    return Counter(e.type for e in doc.entities.values())

def _overlap(a: Counter, b: Counter) -> float:
    keys = a.keys() | b.keys()
    return sum(min(a[k], b[k]) for k in keys) / max(1, sum(max(a[k], b[k]) for k in keys))

def pick_examples(target, pool, k: int) -> list:
    tp = _profile(target)
    key = lambda d: (-_overlap(tp, _profile(d)), hashlib.md5(d.pmid.encode()).hexdigest())
    return sorted(pool, key=key)[:k]

def build_params(arm: dict, doc, pool, system: str, eligible: set[str]) -> dict:
    k = arm.get("k", 4)
    shots = pick_examples(doc, pool, k) if k else []
    examples = "\n\n".join(f"<example>\n{render_doc(s, eligible, True)}\n</example>" for s in shots)
    intro = f"Worked examples:\n\n{examples}\n\n" if shots else ""
    cache = {"type": "ephemeral", **({"ttl": "1h"} if arm.get("cache_1h") else {})}
    params = {
        "model": arm["model"],
        "max_tokens": arm.get("max_tokens", 16000),
        "system": [{"type": "text", "text": system, "cache_control": cache}],
        "messages": [{"role": "user", "content":
            f"{intro}Now extract every relation for this abstract.\n\n{render_doc(doc, eligible)}"}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": arm["effort"]},
    }
    if "thinking" in arm:
        params["thinking"] = arm["thinking"]
    return params

def parse(message, doc, eligible: set[str]) -> tuple[list, dict]:
    info = {"stop_reason": message.stop_reason, "usage": message.usage.model_dump()}
    text = next((b.text for b in message.content if b.type == "text"), None)
    try:
        items = json.loads(text)["relations"]
    except (TypeError, ValueError, KeyError):
        return [], {**info, "failed": True, "dropped": 0}
    canon_t = {t.lower(): t for t in REL_TYPES}
    canon_n = {n.lower(): n for n in NOVELTY}
    out, seen, dropped = [], set(), 0
    for it in items:
        a, b = it["entity1_id"].strip(), it["entity2_id"].strip()
        t, n = canon_t.get(it["type"].strip().lower()), canon_n.get(it["novel"].strip().lower())
        ok = a != b and t and n and all(x in doc.entities and doc.entities[x].type in eligible for x in (a, b))
        if not ok or tuple(sorted((a, b))) in seen:
            dropped += 1
            continue
        seen.add(tuple(sorted((a, b))))
        out.append(Relation(a, b, t, n))
    return out, {**info, "failed": False, "dropped": dropped, "kept": len(out)}

def prompt_hash(system: str, arm: dict) -> str:
    blob = json.dumps({"system": system, "schema": SCHEMA, "arm": arm}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
