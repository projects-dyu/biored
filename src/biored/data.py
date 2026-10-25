"""src/biored/data.py"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

REL_TYPES = [
    "Association",
    "Positive_Correlation",
    "Negative_Correlation",
    "Bind",
    "Cotreatment",
    "Comparison",
    "Drug_Interaction",
    "Conversion",
]
NOVELTY = ["No", "Novel"]
TYPE_ALIASES = {"Organism": "OrganismTaxon"}

_HEADER = re.compile(r"^([^|\s]+)\|([ta])\|(.*)$")
_ID_SEP = re.compile(r"[,;]")


@dataclass
class Mention:
    start: int  # character offsets into Doc.text
    end: int
    text: str


@dataclass
class Entity:
    id: str
    type: str
    mentions: list[Mention] = field(default_factory=list)


@dataclass(frozen=True)
class Relation:
    e1: str
    e2: str
    type: str
    novel: str  # "Novel" or "No"

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.e1, self.e2)))


@dataclass
class Doc:
    pmid: str
    text: str
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    title_end: int = 0
    n_mention_lines: int = 0  # raw count, comparable with inventory.py
    directions: list[tuple[str, ...]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def norm_type(t: str) -> str:
    t = t.strip()
    return TYPE_ALIASES.get(t, t)


def split_ids(raw: str) -> list[str]:
    """'D001,D002' or 'D001;D002' -> two IDs. Never split on '|': variant IDs contain it."""
    return [x.strip() for x in _ID_SEP.split(raw) if x.strip() and x.strip() != "-"]


def _add_mention(
    doc: Doc, raw_id: str, etype: str, start: int, end: int, text: str
) -> None:
    etype = norm_type(etype)
    ids = split_ids(raw_id)
    if not ids:
        doc.warnings.append(f"no_id\t{start}-{end}\t{etype}")
        return
    for eid in ids:
        ent = doc.entities.setdefault(eid, Entity(eid, etype))
        if ent.type != etype:
            doc.warnings.append(
                f"type_conflict\t{eid}\t{ent.type} vs {etype} (kept {ent.type})"
            )
        if all((m.start, m.end) != (start, end) for m in ent.mentions):
            ent.mentions.append(Mention(start, end, text))


def _add_relation(doc: Doc, rtype: str, e1: str, e2: str, novel: str) -> None:
    rtype, e1, e2, novel = rtype.strip(), e1.strip(), e2.strip(), novel.strip()
    if rtype not in REL_TYPES:
        doc.warnings.append(f"unknown_rel_type\t{rtype}")
    if novel not in NOVELTY:
        doc.warnings.append(f"unknown_novelty\t{novel}")
    if e1 == e2:
        doc.warnings.append(f"self_relation\t{e1}")
    doc.relations.append(Relation(e1, e2, rtype, novel))


def load_pubtator(path: str | Path) -> list[Doc]:
    """Line by line, grouped by the PMID|t| header, so missing blank lines can't merge documents."""
    docs: list[Doc] = []
    cur: Doc | None = None
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        m = _HEADER.match(line)
        if m:
            pmid, kind, body = m.groups()
            if kind == "t":
                cur = Doc(pmid, body, title_end=len(body))
                docs.append(cur)
            else:
                assert cur is not None and cur.pmid == pmid, (
                    f"{path}:{n}: abstract without its title"
                )
                cur.text = f"{cur.text} {body}"  # PubTator abstract offsets start at len(title) + 1
            continue
        f = line.split("\t")
        assert cur is not None and f[0] == cur.pmid, (
            f"{path}:{n}: line for {f[0]} outside its document"
        )
        if len(f) >= 5 and f[1].isdigit() and f[2].isdigit():
            cur.n_mention_lines += 1
            _add_mention(
                cur, f[5] if len(f) > 5 else "", f[4], int(f[1]), int(f[2]), f[3]
            )
        elif len(f) >= 5:
            _add_relation(cur, f[1], f[2], f[3], f[4])
        elif len(f) == 4:
            cur.directions.append(tuple(f[1:]))
        else:
            cur.warnings.append(f"unparsed_line\t{line[:80]}")
    return docs


def load_bioc_xml(path: str | Path) -> list[Doc]:
    from bioc import biocxml  # imported here so the serving image doesn't need bioc

    with open(path, encoding="utf-8") as fp:
        collection = biocxml.load(fp)
    docs = []
    for d in collection.documents:
        size = max(p.offset + len(p.text or "") for p in d.passages)
        chars = [" "] * size
        for p in d.passages:
            chars[p.offset : p.offset + len(p.text or "")] = list(p.text or "")
        doc = Doc(d.id, "".join(chars), title_end=len(d.passages[0].text or ""))
        for p in d.passages:
            for a in p.annotations:
                doc.n_mention_lines += 1
                for loc in a.locations:
                    _add_mention(
                        doc,
                        a.infons.get("identifier", ""),
                        a.infons.get("type", ""),
                        loc.offset,
                        loc.offset + loc.length,
                        a.text,
                    )
        for r in list(d.relations) + [r for p in d.passages for r in p.relations]:
            inf = r.infons
            if not {"entity1", "entity2", "type"} <= inf.keys():
                doc.warnings.append(f"relation_infons\t{sorted(inf)}")
                continue
            _add_relation(
                doc, inf["type"], inf["entity1"], inf["entity2"], inf.get("novel", "No")
            )
        docs.append(doc)
    return docs


def load_any(path: str | Path) -> list[Doc]:
    return (
        load_bioc_xml(path)
        if str(path).lower().endswith(".xml")
        else load_pubtator(path)
    )


def load_config(
    path: str | Path = "configs/data.yaml", root: str | Path | None = None
) -> dict[str, list[str]]:
    """Read split -> file names. `root` overrides the config's root, e.g. /kaggle/input/biored-re-splits."""
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    base = Path(root or cfg.get("root", "."))
    return {s: [str(base / f) for f in cfg[s]] for s in ("train", "dev", "test")}


def load_splits(paths: dict[str, list[str]]) -> dict[str, list[Doc]]:
    splits = {
        name: [d for p in files for d in load_any(p)] for name, files in paths.items()
    }
    seen: dict[str, str] = {}
    for name, docs in splits.items():
        for d in docs:
            assert d.pmid not in seen, f"{d.pmid} is in both {seen[d.pmid]} and {name}"
            seen[d.pmid] = name
    return splits


def type_combo(doc: Doc, a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((doc.entities[a].type, doc.entities[b].type)))


def allowed_type_pairs(train: list[Doc]) -> set[tuple[str, str]]:
    return {
        type_combo(d, r.e1, r.e2)
        for d in train
        for r in d.relations
        if r.e1 in d.entities and r.e2 in d.entities
    }


def candidate_pairs(doc: Doc, allowed: set[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (a, b)
        for a, b in combinations(sorted(doc.entities), 2)
        if type_combo(doc, a, b) in allowed
    ]


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def sentence_spans(doc: Doc) -> list[tuple[int, int]]:
    """Rough splitter, title kept as its own sentence. Good enough for error analysis."""
    spans, start = [(0, doc.title_end)], doc.title_end + 1
    for m in _SENT_END.finditer(doc.text, start):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(doc.text)))
    return spans
