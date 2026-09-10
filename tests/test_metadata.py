"""Тесты извлечения обозначений ГОСТ и связки с реестром."""

from __future__ import annotations

from pathlib import Path

import pytest

from gost_rag.ingest.metadata import (
    build_metadata,
    find_designation,
    load_registry,
    make_doc_id,
    parse_doc_type,
    parse_year,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ГОСТ 14634-93", "ГОСТ 14634-93"),
        ("ГОСТ 2.109-73 ЕСКД", "ГОСТ 2.109-73"),
        ("ГОСТ Р 1.2-2016", "ГОСТ Р 1.2-2016"),
        ("гост 14634-93", "ГОСТ 14634-93"),
        ("ГОСТ  14634 - 93", "ГОСТ 14634-93"),
        ("ГОСТ 14634–93", "ГОСТ 14634-93"),  # en dash
        ("СТО 1234-2015", "СТО 1234-2015"),
        ("РД 50-34-88", "РД 50-34"),  # первый разделитель выигрывает
    ],
)
def test_designation_variants(text, expected):
    assert find_designation(text) == expected


def test_international_designation():
    assert find_designation("ГОСТ Р ИСО 9001-2015") == "ГОСТ Р ИСО 9001-2015"
    assert find_designation("ГОСТ Р ISO 9001-2015") == "ГОСТ Р ИСО 9001-2015"


def test_designation_keeps_two_digit_year_verbatim():
    # «ГОСТ 14634-93» — каноничная запись; разворачивать её в «-1993» нельзя,
    # иначе обозначение перестанет совпадать с реестром и запросом пользователя.
    assert find_designation("ГОСТ 1234-15") == "ГОСТ 1234-15"


def test_year_field_expands_two_digit_year():
    assert parse_year("ГОСТ 14634-93") == 1993
    assert parse_year("ГОСТ 1234-15") == 2015


def test_no_designation_returns_none():
    assert find_designation("Руководство по эксплуатации станка") is None
    assert find_designation("") is None


def test_parse_year_and_type():
    assert parse_year("ГОСТ 14634-93") == 1993
    assert parse_year("ГОСТ Р ИСО 9001-2015") == 2015
    assert parse_year(None) is None
    assert parse_doc_type("ГОСТ Р 1.2-2016") == "ГОСТ Р"
    assert parse_doc_type("ГОСТ 14634-93") == "ГОСТ"


def test_doc_id_is_slugified_and_stable():
    a = make_doc_id("ГОСТ 14634-93", Path("x.pdf"))
    b = make_doc_id("ГОСТ 14634-93", Path("y.pdf"))
    assert a == b
    assert " " not in a
    assert make_doc_id(None, Path("Руководство станка.pdf")) == "руководство-станка"


# ---------- реестр ----------


@pytest.fixture
def registry_file(tmp_path: Path) -> Path:
    path = tmp_path / "documents.csv"
    path.write_text(
        "designation,title,year,status,source_url,replaced_by\n"
        "ГОСТ 14634-93,Ленты стальные холоднокатаные,1993,действующий,https://example.org/1,\n"
        "ГОСТ 1050-88,Прокат сортовой,1988,заменён,https://example.org/2,ГОСТ 1050-2013\n",
        encoding="utf-8",
    )
    return path


def test_registry_roundtrip(registry_file):
    registry = load_registry(registry_file)
    assert len(registry) == 2
    assert registry["ГОСТ 14634-93"]["status"] == "действующий"


def test_missing_registry_is_not_an_error(tmp_path):
    assert load_registry(tmp_path / "absent.csv") == {}


def test_metadata_prefers_filename_over_page_text(registry_file):
    registry = load_registry(registry_file)
    meta = build_metadata(
        Path("ГОСТ 14634-93.pdf"),
        first_page_text="ГОСТ 9999-99\nКакой-то другой стандарт",
        registry=registry,
    )
    assert meta.designation == "ГОСТ 14634-93"
    assert meta.status == "действующий"
    assert meta.title == "Ленты стальные холоднокатаные"
    assert meta.source_url == "https://example.org/1"


def test_metadata_falls_back_to_page_text(registry_file):
    meta = build_metadata(
        Path("scan_001.pdf"),
        first_page_text="ГОСТ 14634-93\nЛенты стальные",
        registry=load_registry(registry_file),
    )
    assert meta.designation == "ГОСТ 14634-93"


def test_status_unknown_when_absent_from_registry():
    meta = build_metadata(Path("ГОСТ 9999-99.pdf"), "", registry={})
    assert meta.status == "неизвестно"
    assert meta.year == 1999


def test_replaced_standard_carries_successor(registry_file):
    meta = build_metadata(Path("ГОСТ 1050-88.pdf"), "", registry=load_registry(registry_file))
    assert meta.status == "заменён"
    assert meta.replaced_by == "ГОСТ 1050-2013"


def test_status_spelling_without_yo_is_accepted(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(
        "designation,title,year,status,source_url,replaced_by\n"
        "ГОСТ 380-88,Сталь углеродистая,1988,отменен,,\n",
        encoding="utf-8",
    )
    meta = build_metadata(Path("ГОСТ 380-88.pdf"), "", registry=load_registry(path))
    assert meta.status == "отменён"


def test_title_guessed_from_first_page_when_registry_silent():
    meta = build_metadata(
        Path("ГОСТ 9999-99.pdf"),
        "ГОСТ 9999-99\n\nЛенты стальные холоднокатаные\nТехнические условия",
        registry={},
    )
    assert meta.title == "Ленты стальные холоднокатаные"


def test_payload_has_status_for_citation():
    meta = build_metadata(Path("ГОСТ 9999-99.pdf"), "", registry={})
    assert "status" in meta.as_payload()
