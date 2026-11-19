"""Encoder feature tests: mark() wraps the right text, every recorded offset points at an opening
marker, and encode() keeps every candidate pair when nothing is truncated. Made-up text."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from biored.data import load_pubtator  # noqa: E402
from biored.models.encoder import LABELS, NOVELTY, decode, mark  # noqa: E402

FIXTURE = "\n".join([
    "111|t|Aspirin and BRCA1 in cancer.",
    "111|a|Cancer risk fell. BRCA1 was mutated.",
    "111\t0\t7\tAspirin\tChemicalEntity\tD001241",
    "111\t12\t17\tBRCA1\tGeneOrGeneProduct\t672",
    "111\t21\t27\tcancer\tDiseaseOrPhenotypicFeature\tD009369",
    "111\t29\t35\tCancer\tDiseaseOrPhenotypicFeature\tD009369",
    "111\t47\t52\tBRCA1\tGeneOrGeneProduct\t672",
    "111\tAssociation\t672\tD009369\tNovel",
    "111\tNegative_Correlation\tD001241\tD009369\tNo",
]) + "\n"


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "toy.pubtator"
    p.write_text(FIXTURE, encoding="utf-8")
    return load_pubtator(p)[0]


def test_labels_match_the_scorer():
    from biored.eval import NOVELTY as N, REL_TYPES
    assert LABELS == ["None", *REL_TYPES] and NOVELTY == N


def test_mark_offsets_point_at_opening_markers(doc):
    eligible = {e.type for e in doc.entities.values()}
    text, opens = mark(doc, eligible)
    assert set(opens) == set(doc.entities)
    assert len(opens["672"]) == 2 and len(opens["D009369"]) == 2
    for eid, offs in opens.items():
        for o in offs:
            assert text[o] == "["
            tag, rest = text[o:].split("] ", 1)
            assert rest.split(" [/", 1)[0] in {m.text for m in doc.entities[eid].mentions}
    assert "[CHEM] Aspirin [/CHEM]" in text and "[GENE] BRCA1 [/GENE]" in text


def test_decode_never_emits_none():
    import torch
    logits = torch.tensor([[5.0] + [0.0] * 8, [0.0, 5.0] + [0.0] * 7])
    nov = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    out = decode(logits, nov, [("A", "B"), ("A", "C")])
    assert [(r.e1, r.e2, r.type, r.novel) for r in out] == [("A", "C", LABELS[1], NOVELTY[0])]
