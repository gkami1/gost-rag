"""OCR-фоллбэк для отсканированных страниц.

Многие ГОСТы существуют только как сканы, поэтому без OCR корпус получается
неполным. Tesseract — внешний бинарник; если его нет, ingestion не падает,
а помечает страницы как нераспознанные и продолжает работу (см. ``ocr_available``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gost_rag.logging import get_logger

log = get_logger(__name__)

#: Слова с уверенностью ниже порога не учитываем при усреднении — это обычно мусор.
_MIN_WORD_CONF = 0.0


@dataclass(slots=True)
class OcrResult:
    text: str
    confidence: float | None


def configure_tesseract(cmd: str | None) -> None:
    if cmd:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = cmd


def ocr_available(lang: str = "rus") -> bool:
    """Проверить, что бинарник есть и нужные языки установлены.

    Tesseract допускает комбинацию языков через ``+`` (``rus+eng``) — это штатный
    приём для ГОСТов, где в русский текст вкраплены латинские обозначения (Ra, S,
    M10). Сравнивать такую строку со списком языков целиком нельзя: проверка
    провалится, ingestion сочтёт OCR недоступным и молча выбросит все сканы.
    """
    try:
        import pytesseract

        langs = pytesseract.get_languages(config="")
    except Exception as exc:
        log.warning("tesseract_unavailable", error=str(exc))
        return False

    requested = [part for part in lang.split("+") if part]
    missing = [part for part in requested if part not in langs]
    if not requested or missing:
        log.warning(
            "tesseract_lang_missing",
            lang=lang,
            missing=missing or [lang],
            available=sorted(langs)[:10],
        )
        return False
    return True


def deskew(image: np.ndarray) -> np.ndarray:
    """Выровнять наклон скана.

    Сканы ГОСТов часто перекошены на 1–2°, и Tesseract на них заметно теряет
    точность. Угол оцениваем по минимальному охватывающему прямоугольнику
    тёмных пикселей; повороты больше 15° игнорируем — это почти всегда ошибка
    оценки, а не реальный перекос.
    """
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    threshold = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = cv2.findNonZero(threshold)
    if coords is None:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.1 or abs(angle) > 15:
        return gray

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def render_page(pdf_path: Path, page_no: int, dpi: int = 300) -> np.ndarray:
    """Отрисовать страницу PDF в градациях серого (page_no — с единицы)."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(page_no - 1)
        pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)


def ocr_page(pdf_path: Path, page_no: int, *, lang: str = "rus", dpi: int = 300) -> OcrResult:
    """Распознать одну страницу. При ошибке возвращает пустой результат."""
    import pytesseract

    try:
        image = deskew(render_page(pdf_path, page_no, dpi=dpi))
    except Exception as exc:
        log.warning("render_failed", path=str(pdf_path), page=page_no, error=str(exc))
        return OcrResult(text="", confidence=None)

    try:
        data = pytesseract.image_to_data(
            image, lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT
        )
    except Exception as exc:
        log.warning("ocr_failed", path=str(pdf_path), page=page_no, error=str(exc))
        return OcrResult(text="", confidence=None)

    words = [
        (word, float(conf))
        for word, conf in zip(data["text"], data["conf"], strict=False)
        if word.strip() and float(conf) > _MIN_WORD_CONF
    ]
    if not words:
        return OcrResult(text="", confidence=None)

    text = _reflow(data)
    confidence = round(sum(conf for _, conf in words) / len(words), 2)
    return OcrResult(text=text, confidence=confidence)


def _reflow(data: dict) -> str:
    """Собрать слова обратно в строки по номерам блок/абзац/строка от Tesseract."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    return "\n".join(" ".join(words) for _, words in sorted(lines.items()))
