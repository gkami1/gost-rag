"""Тесты доступности OCR и сборки строк.

Бинарь Tesseract в тестах не используется: список языков подменяется. Иначе
тесты падали бы на машинах без OCR, а свойство «тесты не ходят наружу» — важнее.
"""

from __future__ import annotations

import pytest

from gost_rag.ingest import ocr as ocr_module


@pytest.fixture
def installed(monkeypatch):
    """Подменить список языков, установленных в Tesseract."""

    def set_langs(langs: list[str] | None):
        class FakePytesseract:
            @staticmethod
            def get_languages(config=""):
                if langs is None:
                    raise OSError("tesseract is not installed or it's not in your PATH")
                return langs

        monkeypatch.setitem(__import__("sys").modules, "pytesseract", FakePytesseract)

    return set_langs


def test_single_language_present(installed):
    installed(["eng", "osd", "rus"])
    assert ocr_module.ocr_available("rus") is True


def test_single_language_absent(installed):
    installed(["eng", "osd"])
    assert ocr_module.ocr_available("rus") is False


def test_combined_languages_are_split_on_plus(installed):
    """`rus+eng` — штатный синтаксис Tesseract, а не имя языка.

    Если сравнивать строку целиком, проверка провалится и конвейер молча
    выбросит все сканы — потеря данных с одним лишь предупреждением в логе.
    """
    installed(["eng", "osd", "rus"])
    assert ocr_module.ocr_available("rus+eng") is True


def test_combined_languages_fail_if_any_part_missing(installed):
    installed(["eng", "osd", "rus"])
    assert ocr_module.ocr_available("rus+deu") is False


def test_three_way_combination(installed):
    installed(["eng", "osd", "rus", "deu"])
    assert ocr_module.ocr_available("rus+eng+deu") is True


def test_empty_language_is_rejected(installed):
    installed(["eng", "osd", "rus"])
    assert ocr_module.ocr_available("") is False
    assert ocr_module.ocr_available("+") is False


def test_missing_binary_reports_unavailable(installed):
    installed(None)
    assert ocr_module.ocr_available("rus") is False


# --------------------------------------------------------------------------- #
# Сборка строк из результата Tesseract
# --------------------------------------------------------------------------- #


def test_reflow_groups_words_into_lines():
    data = {
        "text": ["Радиус", "гибки", "составляет", "2S"],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 2, 2],
    }
    assert ocr_module._reflow(data) == "Радиус гибки\nсоставляет 2S"


def test_reflow_skips_empty_tokens():
    data = {
        "text": ["Радиус", "", "  ", "гибки"],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 1],
    }
    assert ocr_module._reflow(data) == "Радиус гибки"


def test_reflow_keeps_blocks_in_order():
    data = {
        "text": ["второй", "первый"],
        "block_num": [2, 1],
        "par_num": [1, 1],
        "line_num": [1, 1],
    }
    assert ocr_module._reflow(data) == "первый\nвторой"
