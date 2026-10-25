# src/biored/models/baseline.py
"""Co-occurrence baseline: every candidate pair gets the relation type and novelty that are most
common in train for its entity-type pair. The variant same_sentence_only=True keeps only pairs
whose entities have mentions in a shared sentence."""
from collections import Counter, defaultdict

from biored.data import Relation, allowed_type_pairs, candidate_pairs, sentence_spans


def type_pair(doc, a, b):
    return tuple(sorted((doc.entities[a].type, doc.entities[b].type)))


def sentence_index(doc):
    """{entity_id: indices of the sentences its mentions start in}."""
    spans = sentence_spans(doc)
    return {eid: {i for m in e.mentions for i, (s, t) in enumerate(spans) if s <= m.start < t}
            for eid, e in doc.entities.items()}


class CooccurrenceBaseline:
    def fit(self, train):
        types, novel = defaultdict(Counter), defaultdict(Counter)
        for d in train:
            for r in d.relations:
                if r.e1 in d.entities and r.e2 in d.entities:
                    c = type_pair(d, r.e1, r.e2)
                    types[c][r.type] += 1
                    novel[c][r.novel] += 1
        self.allowed_ = allowed_type_pairs(train)
        self.type_ = {c: n.most_common(1)[0][0] for c, n in types.items()}
        self.novel_ = {c: n.most_common(1)[0][0] for c, n in novel.items()}
        return self

    def predict(self, doc, same_sentence_only=False):
        sents = sentence_index(doc) if same_sentence_only else None
        out = []
        for a, b in candidate_pairs(doc, self.allowed_):
            if sents is not None and not (sents[a] & sents[b]):
                continue
            c = type_pair(doc, a, b)
            out.append(Relation(a, b, self.type_[c], self.novel_[c]))
        return out
