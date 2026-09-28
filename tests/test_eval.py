"""Метрики eval: сопоставление пунктов и recall пункта на k и на 1."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_eval import Question, SliceStats, _first_section_rank, _section_matches

from gost_rag.models import RetrievedChunk

DOC = "ГОСТ 24705-2004"


def _chunk(sections: list[str], designation: str = DOC) -> RetrievedChunk:
    return RetrievedChunk(
        point_id="p",
        text="",
        payload={"designation": designation, "sections": sections},
    )


def _question(section: str) -> Question:
    return Question(id="q", question="?", gold_designation=DOC, gold_section=section)


def test_section_matches_exact_and_nested():
    assert _section_matches("4.1", "4.1")
    assert _section_matches("4.1.3", "4.1")
    assert _section_matches("1.2", "1")


def test_section_does_not_match_by_string_prefix():
    # Раньше startswith засчитывал «4.12» эталону «4.1» и «10.2» эталону «1».
    assert not _section_matches("4.12", "4.1")
    assert not _section_matches("10.2", "1")


def test_first_section_rank_skips_other_documents():
    chunks = [
        _chunk(["4.1"], designation="ГОСТ 5264-80"),
        _chunk(["3.2", "3.3"]),
        _chunk(["3.9", "4.1.2"]),
    ]
    assert _first_section_rank(chunks, _question("4.1")) == 3


def test_first_section_rank_falls_back_to_single_section():
    # Индекс, собранный до появления поля ``sections``.
    chunk = RetrievedChunk(point_id="p", text="", payload={"designation": DOC, "section": "4.1"})
    assert _first_section_rank([chunk], _question("4.1")) == 1


def test_section_at_1_is_stricter_than_section_recall():
    stats = SliceStats("rerank")
    stats.update([_chunk(["4.1"])], _question("4.1"), k=10)
    stats.update([_chunk(["3.1"]), _chunk(["4.1"])], _question("4.1"), k=10)
    stats.update([_chunk(["3.1"])], _question("4.1"), k=10)

    assert stats.section_recall == 2 / 3
    assert stats.section_at_1 == 1 / 3


# --------------------------------------------------------------------------- #
# Проверка ответов
# --------------------------------------------------------------------------- #


def test_answer_matching_ignores_case_spaces_separator_and_lookalikes():
    from run_eval import answer_is_correct

    # Кириллическая «М» против латинской, точка против запятой, пробелы.
    assert answer_is_correct("Болт M 6 [S1]", ["М6"])
    assert answer_is_correct("d2 = 9.026 мм", ["9,026"])
    assert answer_is_correct("Жёлто-зелёная изоляция", [["зелено-желт", "желто-зелен"]])


def test_answer_number_must_match_whole_value():
    from run_eval import answer_is_correct

    assert not answer_is_correct("катет 17 мм", ["7 мм"])
    assert answer_is_correct("катет 7 мм", ["7 мм"])


def test_all_expected_items_required():
    from run_eval import answer_is_correct

    assert not answer_is_correct("по ГОСТ 9150", ["9150", "8724"])
    assert answer_is_correct("что угодно", []) is None


# --------------------------------------------------------------------------- #
# Порог отказа
# --------------------------------------------------------------------------- #


def _record(qid, score, *, oos, split="tune", named=False):
    return {
        "id": qid,
        "top_score": score,
        "out_of_corpus": oos,
        "split": split,
        "named_missing_only": named,
    }


def test_named_missing_standard_is_refused_at_any_threshold():
    from run_eval import refused_at

    record = _record("q", 0.99, oos=True, named=True)
    assert refused_at(record, 0.0)


def test_gate_stats_counts_answers_and_refusals():
    from run_eval import gate_stats

    records = [
        _record("a", 0.9, oos=False),
        _record("b", 0.05, oos=False),
        _record("c", 0.5, oos=True),
        _record("d", 0.01, oos=True),
    ]
    stats = gate_stats(records, 0.1)
    assert stats.answer_rate == 0.5
    assert stats.refusal_rate == 0.5


def test_threshold_is_chosen_on_tune_split_only():
    from run_eval import choose_threshold

    records = [
        _record("a", 0.6, oos=False, split="tune"),
        _record("b", 0.4, oos=True, split="tune"),
        # На test всё наоборот: если бы порог выбирался и по нему, он был бы другим.
        _record("c", 0.05, oos=False, split="test"),
        _record("d", 0.95, oos=True, split="test"),
    ]
    assert choose_threshold(records, thresholds=(0.0, 0.5, 0.9)) == 0.5


def test_split_is_stable_and_respects_explicit_value():
    from run_eval import split_of

    assert split_of(Question(id="x", question="?", split="test")) == "test"
    assert split_of(Question(id="same", question="?")) == split_of(
        Question(id="same", question="!")
    )


def test_rescore_counts_polite_refusal_and_not_its_keywords():
    from run_eval import rescore_generation

    questions = [
        Question(id="in", question="?", gold_designation=DOC, answer_contains=["прочност"]),
        Question(id="oos", question="?", out_of_corpus=True),
    ]
    polite = "Во фрагментах ответа нет [S1]. Упомянуты расчёты прочности [S1]."
    generation = {
        "modes": {"rag": {}},
        "answers": [
            {"id": "in", "rag": {"answer": polite, "insufficient": False}},
            {"id": "oos", "rag": {"answer": polite, "insufficient": False}},
        ],
    }
    mode = rescore_generation(generation, questions)["modes"]["rag"]
    assert mode["refusal_rate"] == 1.0
    assert mode["answer_rate"] == 0.0
    # Эталонное слово есть, но ответ — отказ: верным он не считается.
    assert mode["accuracy"] == 0.0


def test_rescore_drops_numbers_that_live_only_in_designations():
    from run_eval import rescore_generation

    questions = [Question(id="in", question="?", gold_designation=DOC, answer_contains=["М10"])]
    answer = "Болт М10 [S1], знак — по ГОСТ 21130-75; сопротивление 0,2 Ом [S1]."
    generation = {
        "modes": {"rag": {}},
        "answers": [
            {
                "id": "in",
                "rag": {
                    "answer": answer,
                    "insufficient": False,
                    "ungrounded_numbers": ["21130", "75", "0,2"],
                },
            }
        ],
    }
    result = rescore_generation(generation, questions)["answers"][0]["rag"]
    assert result["ungrounded_numbers"] == ["0,2"]
