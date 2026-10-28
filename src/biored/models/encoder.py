from __future__ import annotations
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from biored.data import Doc, Relation, REL_TYPES, NOVELTY, candidate_pairs

LABELS = ["None"] + REL_TYPES
SHORT = {"GeneOrGeneProduct": "GENE", "DiseaseOrPhenotypicFeature": "DISEASE",
         "ChemicalEntity": "CHEM", "SequenceVariant": "VAR"}
TAGS = [*SHORT.values(), "ENT"]
MARKERS = [f"[{t}]" for t in TAGS] + [f"[/{t}]" for t in TAGS]

def mark(doc: Doc, eligible: set[str], typed: bool = True):
    """Wrap every eligible mention once. Returns the marked text and, per entity,
    the character offsets of its opening markers."""
    by_span = defaultdict(list)
    for eid, ent in doc.entities.items():
        if ent.type in eligible:
            for m in ent.mentions:
                by_span[(m.start, m.end)].append(eid)
    parts, cursor, length, opens = [], 0, 0, defaultdict(list)
    for (s, e), eids in sorted(by_span.items()):
        if s < cursor:                    # overlapping mention: skipped (count how often)
            continue
        tag = SHORT.get(doc.entities[eids[0]].type, "ENT") if typed else "ENT"
        before = doc.text[cursor:s] + " "
        parts.append(before)
        length += len(before)
        for eid in eids:
            opens[eid].append(length)
        chunk = f"[{tag}] {doc.text[s:e]} [/{tag}] "
        parts.append(chunk)
        length += len(chunk)
        cursor = e
    parts.append(doc.text[cursor:])
    return "".join(parts), dict(opens)

def encode(doc, tok, eligible, allowed, typed=True, max_len=512, neg_keep=None, rng=None):
    text, opens = mark(doc, eligible, typed)
    enc = tok(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    start_to_tok = {s: i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s}
    ents = sorted(opens)
    index = {eid: i for i, eid in enumerate(ents)}
    ent_pos = [[start_to_tok[c] for c in opens[eid] if c in start_to_tok] for eid in ents]
    gold = {r.key: r for r in doc.relations}
    pairs, labels, novel = [], [], []
    for a, b in candidate_pairs(doc, allowed):
        if a not in index or b not in index or not ent_pos[index[a]] or not ent_pos[index[b]]:
            continue                       # an entity was cut off by truncation
        r = gold.get((a, b))
        if r is None and neg_keep is not None and rng.random() > neg_keep:
            continue                       # downsample negatives (training only)
        pairs.append((index[a], index[b]))
        labels.append(LABELS.index(r.type) if r else 0)
        novel.append(NOVELTY.index(r.novel) if r else -100)   # -100 = ignored by the loss
    return {"input_ids": enc["input_ids"], "ent_pos": ent_pos, "pairs": pairs, "labels": labels,
            "novel": novel, "pair_ids": [(ents[i], ents[j]) for i, j in pairs]}

def collate(batch, pad_id):
    B, T = len(batch), max(len(x["input_ids"]) for x in batch)
    E = max(1, max(len(x["ent_pos"]) for x in batch))
    M = max([1] + [len(p) for x in batch for p in x["ent_pos"]])
    P = max(1, max(len(x["pairs"]) for x in batch))
    out = {"input_ids": torch.full((B, T), pad_id), "attention_mask": torch.zeros(B, T, dtype=torch.long),
           "ent_pos": torch.zeros(B, E, M, dtype=torch.long), "ent_mask": torch.zeros(B, E, M, dtype=torch.bool),
           "pairs": torch.zeros(B, P, 2, dtype=torch.long),
           "labels": torch.full((B, P), -100), "novel": torch.full((B, P), -100)}
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        out["input_ids"][i, :n] = torch.tensor(x["input_ids"])
        out["attention_mask"][i, :n] = 1
        for j, p in enumerate(x["ent_pos"]):
            out["ent_pos"][i, j, :len(p)] = torch.tensor(p, dtype=torch.long)
            out["ent_mask"][i, j, :len(p)] = True
        k = len(x["pairs"])
        if k:
            out["pairs"][i, :k] = torch.tensor(x["pairs"])
            out["labels"][i, :k] = torch.tensor(x["labels"])
            out["novel"][i, :k] = torch.tensor(x["novel"])
    return out

class DocRE(nn.Module):
    def __init__(self, name: str, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.enc.resize_token_embeddings(vocab_size)
        h = self.enc.config.hidden_size
        self.mlp = nn.Sequential(nn.Linear(4 * h, h), nn.GELU(), nn.Dropout(dropout))
        self.type_head = nn.Linear(h, len(LABELS))
        self.nov_head = nn.Linear(h, len(NOVELTY))

    def forward(self, input_ids, attention_mask, ent_pos, ent_mask, pairs):
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # [B,T,h]
        B, E, M = ent_pos.shape
        h = H.size(-1)
        g = torch.gather(H, 1, ent_pos.reshape(B, E * M, 1).expand(-1, -1, h)).reshape(B, E, M, h)
        g = g.float().masked_fill(~ent_mask.unsqueeze(-1), -1e4)
        ent = torch.logsumexp(g, dim=2)                                                       # [B,E,h]
        ea = torch.gather(ent, 1, pairs[..., :1].expand(-1, -1, h))
        eb = torch.gather(ent, 1, pairs[..., 1:].expand(-1, -1, h))
        cls = H[:, :1].float().expand(-1, pairs.size(1), -1)
        x = self.mlp(torch.cat([ea + eb, ea * eb, (ea - eb).abs(), cls], dim=-1))
        return self.type_head(x), self.nov_head(x)

def loss_fn(type_logits, nov_logits, labels, novel, nov_weight=0.5):
    loss = F.cross_entropy(type_logits.reshape(-1, len(LABELS)).float(), labels.reshape(-1))
    if (novel != -100).any():
        loss = loss + nov_weight * F.cross_entropy(nov_logits.reshape(-1, 2).float(), novel.reshape(-1))
    return loss

def decode(type_logits, nov_logits, pair_ids, none_margin=0.0) -> list[Relation]:
    """One document. Predict a relation when the best type beats None by none_margin (tuned on dev)."""
    probs, nov = type_logits.softmax(-1), nov_logits.argmax(-1)
    out = []
    for (a, b), p, n in zip(pair_ids, probs, nov):
        best = int(p[1:].argmax()) + 1
        if p[best] > p[0] + none_margin:
            out.append(Relation(a, b, LABELS[best], NOVELTY[int(n)]))
    return out
