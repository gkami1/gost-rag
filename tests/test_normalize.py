"""Тесты нормализации текста PDF: переносы, колонтитулы, нумерация пунктов."""

from __future__ import annotations

from gost_rag.ingest.normalize import (
    clean_page_text,
    extract_clause,
    find_repeated_lines,
    strip_repeated_lines,
)


def test_hyphen_wrap_is_joined():
    text = "шерохова-\nтость поверхности"
    assert clean_page_text(text) == "шероховатость поверхности"


def test_hyphen_wrap_with_indentation():
    assert clean_page_text("допус-\n   каемое отклонение") == "допускаемое отклонение"


def test_real_hyphenated_word_survives():
    # Дефис не на переносе строки трогать нельзя.
    assert clean_page_text("технико-экономический") == "технико-экономический"


def test_soft_hyphen_and_nbsp_removed():
    assert clean_page_text("тол­щина") == "толщина"
    assert clean_page_text("3 мм") == "3 мм"


def test_spaced_caps_heading_collapsed():
    assert "ОБЩИЕ" in clean_page_text("О Б Щ И Е  Т Р Е Б О В А Н И Я")


def test_multiple_spaces_and_blank_lines_collapsed():
    assert clean_page_text("а    б\n\n\n\nв") == "а б\n\nв"


def test_crlf_normalised():
    assert clean_page_text("а\r\nб") == "а\nб"


# ---------- нумерация пунктов ----------


def test_extract_clause_variants():
    assert extract_clause("3.2.1 Радиус гибки") == "3.2.1"
    assert extract_clause("4 Технические требования") == "4"
    assert extract_clause("5.1. Маркировка") == "5.1"


def test_extract_clause_rejects_plain_text_and_bare_numbers():
    assert extract_clause("Радиус гибки составляет") is None
    assert extract_clause("3.2.1") is None  # без текста это, скорее всего, номер в таблице


def test_extract_clause_rejects_table_rows():
    """Строки таблиц ГОСТ 24705-2004: начинаются с числа, но пунктами не являются.

    Взято из реального индекса: пока такие строки считались пунктами, чанк
    получал `section="6"` от номера строки, и цитата ссылалась на пункт, в
    котором этого текста нет.
    """
    assert extract_clause("6 68,103 65,505 64,639 4 69,402") is None
    assert extract_clause("1 5,350 4,917 4,773 6 0,75 5,513") is None
    assert extract_clause("24 1,5 23,026 22,376 22,160") is None


def test_extract_clause_ignores_units_without_text():
    # «мм» — единица измерения в строке таблицы, а не название пункта.
    assert extract_clause("6 68,103 мм") is None


def test_extract_clause_accepts_short_but_real_headings():
    assert extract_clause("7 Маркировка") == "7"
    assert extract_clause("2 Нормативные ссылки") == "2"


def test_extract_clause_rejects_absurdly_long_numbers():
    """Искажённый текстовый слой таблицы ГОСТ 5264-80 дал «пункт» 1777771."""
    assert extract_clause("1777771 hxW T i (УУа.И.) 11 +1,0") is None


def test_extract_clause_keeps_two_digit_clauses():
    # Пункты 11 и 16 — настоящие, в корпусе они действительно встречаются.
    assert extract_clause("11. (Исключен, Изм. № 1).") == "11"
    assert extract_clause("16. При подготовке кромок предельные отклонения") == "16"


def test_clause_line_is_not_treated_as_running_header():
    # Даже если нумерованный пункт повторяется на каждой странице, вырезать его нельзя.
    pages = [
        f"3.1 Общие требования\nсодержание раздела номер {i} и его описание" for i in range(10)
    ]
    repeated = find_repeated_lines(pages)
    assert not any("общие требования" in line for line in repeated)


# ---------- колонтитулы ----------


def _pages_with_header(n: int) -> list[str]:
    return [f"ГОСТ 14634-93\nсодержательный текст страницы {i}\nСтр. {i}" for i in range(1, n + 1)]


def test_running_header_and_footer_detected():
    pages = _pages_with_header(10)
    repeated = find_repeated_lines(pages)
    # Номер страницы нормализуется, поэтому «Стр. 1»..«Стр. 10» — одна строка.
    assert any("гост" in line for line in repeated)
    assert any("стр." in line for line in repeated)


def test_page_numbers_do_not_block_footer_detection():
    pages = _pages_with_header(10)
    repeated = find_repeated_lines(pages)
    cleaned = strip_repeated_lines(pages[3], repeated)
    assert "Стр. 4" not in cleaned
    assert "содержательный текст" in cleaned


def test_short_document_keeps_everything():
    # На 3 страницах повтор может быть случайным — не вырезаем.
    pages = _pages_with_header(3)
    assert find_repeated_lines(pages) == set()


def test_body_text_is_never_stripped():
    pages = _pages_with_header(10)
    repeated = find_repeated_lines(pages)
    for i, page in enumerate(pages, start=1):
        assert f"содержательный текст страницы {i}" in strip_repeated_lines(page, repeated)


def test_strip_is_noop_without_repeats():
    assert strip_repeated_lines("а\nб", set()) == "а\nб"


def test_repeated_line_in_body_is_kept():
    # Повторяющаяся строка вдали от края страницы — это содержание, не колонтитул.
    body = "\n".join(["шапка", "a", "b", "c", "повтор", "d", "e", "f", "подвал"])
    repeated = {"повтор"}
    assert "повтор" in strip_repeated_lines(body, repeated)


def test_table_caption_and_units_are_not_running_headers():
    """ГОСТ 24705-2004: табл. 1 на 13 страницах, у каждой сверху подпись и единицы."""
    pages = [
        f"ГОСТ 24705—2004\nПродолжение таблицы 1\nВ миллиметрах\n| {n} | 1 |\nтекст {n}"
        for n in range(8)
    ]
    repeated = find_repeated_lines(pages)
    assert strip_repeated_lines(pages[3], repeated).startswith(
        "Продолжение таблицы 1\nВ миллиметрах"
    )
    assert "ГОСТ 24705—2004" not in strip_repeated_lines(pages[3], repeated)


def test_header_with_dash_variants_collapses_to_one():
    pages = [
        (f"С. {n} ГОСТ 5264-80" if n % 2 else f"ГОСТ 5264—80 С. {n}") + f"\nтекст страницы {n}"
        for n in range(8)
    ]
    repeated = find_repeated_lines(pages)
    assert all("ГОСТ" not in strip_repeated_lines(page, repeated) for page in pages)
