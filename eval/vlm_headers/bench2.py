"""Вариант 2: VLM читает шапку по одной ячейке, пути собираются по геометрии сетки;
легенда — отдельный вызов по странице с чертежом."""
import base64, io, json, re, sys, time
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pdfplumber, pymupdf
from PIL import Image
from gost_rag.ingest.loaders import _render_gray
from gost_rag.ingest.tables import find_tables, table_rows, grid_cells, split_header, _index
from bench import CASES, TRUTH, LEGEND, score

CELL_PROMPT = ("Это одна ячейка шапки таблицы из ГОСТа. Текст может быть повёрнут на 90°. "
               "Перепиши текст ячейки точно, по-русски, одной строкой. Нижние индексы пиши "
               "обычными цифрами или через подчёркивание: R1, D2, d_f. Пустая ячейка — пустая строка.")
LEGEND_PROMPT = ("На изображении — страница ГОСТа с чертежом и таблицей. В шапке таблицы есть "
                 "обозначения: {symbols}. По чертежу и тексту страницы определи, что означает "
                 "каждое (например «радиус скругления проточки», «ширина проточки»). Ответ по-русски, "
                 "кратко. Если смысл обозначения на странице не виден — пропусти его.")
CELL_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
LEG_SCHEMA = {"type": "object", "properties": {"legend": {"type": "object",
              "additionalProperties": {"type": "string"}}}, "required": ["legend"]}

def b64(img):
    buf = io.BytesIO(); img.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()

def chat(model, prompt, images, schema, num_predict=200):
    body = {"model": model, "stream": False, "format": schema, "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
            "messages": [{"role": "user", "content": prompt, "images": [b64(i) for i in images]}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = json.loads(r.read())["message"]["content"]
            break
        except Exception as exc:
            print("    [ошибка]", attempt, exc, flush=True)
    if raw is None:
        return {"text": "?", "legend": {}}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("    [невалидный JSON]", raw[:80].replace(chr(10), " "), flush=True)
        return {"text": "?", "legend": {}}

def prepare_cell(img, h_pt, w_pt):
    """Повёрнутый текст ГОСТа читается снизу вверх: высокую узкую ячейку ставим ровно.
    Мелкую — увеличиваем и кладём на белое поле: Qwen-VL режет картинку на патчи 28 px."""
    if h_pt > 1.6 * w_pt:
        img = img.rotate(-90, expand=True)
    if min(img.size) < 96:
        k = 96 / min(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    if max(img.size) > 448:  # каждые 28x28 px — токен; крупная ячейка стоила >1000 токенов
        k = 448 / max(img.size)
        img = img.resize((max(28, round(img.width * k)), max(28, round(img.height * k))), Image.LANCZOS)
    canvas = Image.new("RGB", (img.width + 32, img.height + 32), "white")
    canvas.paste(img, (16, 16))
    return canvas

def header_cells(doc, pno, idx):
    path = Path(f"data/raw/{doc}.pdf")
    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[pno - 1]
        gray = _render_gray(path, pno)
        tables = [t for t in sorted(find_tables(page, gray), key=lambda t: t.bbox[1])
                  if table_rows(page, t, require_data=True)]
        t = tables[idx]
        h = split_header(grid_cells(t))
        col_edges = sorted({round(c[0], 1) for c in t.cells} | {round(t.bbox[2], 1)})
        cells = []
        for row in t.rows[:h]:
            for bbox in row.cells:
                if bbox and bbox not in cells:
                    cells.append(bbox)
    return path, col_edges, cells

def run(model, cid, doc, pno, idx, legend_page):
    path, edges, cells = header_cells(doc, pno, idx)
    pdf = pymupdf.open(str(path))
    texts = {}
    t0 = time.time()
    for bbox in cells:
        x0, top, x1, bottom = bbox
        pix = pdf[pno - 1].get_pixmap(dpi=200, clip=pymupdf.Rect(x0 + 1, top + 1, x1 - 1, bottom - 1))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = prepare_cell(img, bottom - top, x1 - x0)
        texts[bbox] = chat(model, CELL_PROMPT, [img], CELL_SCHEMA, num_predict=80).get("text", "").strip()
        print("    cell", [round(v) for v in bbox], repr(texts[bbox]), flush=True)
    columns = []
    for j in range(len(edges) - 1):
        center = (edges[j] + edges[j + 1]) / 2
        parts = []
        for bbox in sorted(cells, key=lambda b: b[1]):
            if bbox[0] - 0.5 <= center <= bbox[2] + 0.5 and texts[bbox] and (not parts or parts[-1] != texts[bbox]):
                parts.append(texts[bbox])
        columns.append(" / ".join(parts))
    t_cells = time.time() - t0
    symbols = sorted({c.split(" / ")[-1] for c in columns if 0 < len(c.split(" / ")[-1]) <= 4})
    t1 = time.time()
    legend = {}
    if symbols:
        pix = pdf[legend_page - 1].get_pixmap(dpi=110)
        page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        legend = chat(model, LEGEND_PROMPT.format(symbols=", ".join(symbols)), [page_img],
                      LEG_SCHEMA, num_predict=400).get("legend", {})
    got = {"columns": columns, "legend": legend}
    s = score(cid, got)
    s.update(cells=len(cells), secs_cells=round(t_cells), secs_legend=round(time.time() - t1))
    return got, s

LEGEND_PAGE = {"10549-t2": 3, "10549-t3cont": 5, "5264-t3": 7, "24705-t1": 6}

if __name__ == "__main__":
    model = sys.argv[1]
    chat(model, CELL_PROMPT, [Image.new("RGB", (64, 64), "white")], CELL_SCHEMA)  # прогрев
    results = {}
    for cid, doc, pno, idx in CASES:
        got, s = run(model, cid, doc, pno, idx, LEGEND_PAGE[cid])
        results[cid] = {"score": s, "got": got}
        print(cid, s, flush=True)
    json.dump(results, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
