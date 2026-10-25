"""Baseline and prediction-file tests on a made-up two-document PubTator file (safe to commit)."""
import pytest

from biored.data import load_pubtator
from biored.eval import entity_types, load_preds, pair_key, save_preds, score, validate_preds
from biored.models.baseline import CooccurrenceBaseline

# Abstract starts at len(title) + 1. Doc 222's two entities sit in different sentences.
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
    "",
    "222|t|Drug helps.",
    "222|a|Tumor grew.",
    "222\t0\t4\tDrug\tChemicalEntity\tD000001",
    "222\t12\t17\tTumor\tDiseaseOrPhenotypicFeature\tD000002",
]) + "\n"


@pytest.fixture
def docs(tmp_path):
    p = tmp_path / "toy.pubtator"
    p.write_text(FIXTURE, encoding="utf-8")
    return load_pubtator(p)


@pytest.fixture
def model(docs):
    return CooccurrenceBaseline().fit(docs[:1])


def test_majority_labels(model):
    assert model.type_[("ChemicalEntity", "DiseaseOrPhenotypicFeature")] == "Negative_Correlation"
    assert model.novel_[("ChemicalEntity", "DiseaseOrPhenotypicFeature")] == "No"
    assert model.type_[("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct")] == "Association"


def test_reproduces_its_own_train_gold(docs, model):
    gold, pred = {"111": docs[0].relations}, {"111": model.predict(docs[0])}
    assert score(gold, pred, entity_types(docs[:1]))["micro_f1_pair_type_novelty"] == 1.0


def test_valid_and_same_sentence_filters(docs, model):
    for same in (False, True):
        validate_preds({d.pmid: model.predict(d, same) for d in docs}, docs)
    assert {pair_key(r) for r in model.predict(docs[1])} == {("D000001", "D000002")}
    assert model.predict(docs[1], same_sentence_only=True) == []   # title vs abstract


def test_save_load_round_trip(docs, model, tmp_path):
    pred = {d.pmid: model.predict(d) for d in docs}
    save_preds(pred, tmp_path / "p.jsonl")
    back = load_preds(tmp_path / "p.jsonl")
    as_set = lambda p: {pm: {(pair_key(r), r.type, r.novel) for r in rs} for pm, rs in p.items() if rs}
    assert as_set(back) == as_set(pred)
