"""Таблицы для сравнения VLM: грид, шапка, первая строка данных, растр шапки."""
from pathlib import Path
import pdfplumber, pymupdf
from PIL import Image
from gost_rag.ingest.loaders import _render_gray
from gost_rag.ingest.tables import find_tables, table_rows, grid_cells, split_header

CASES = [  # (id, документ, страница, номер таблицы на странице сверху)
    ("10549-t2", "ГОСТ 10549-80", 3, 0),
    ("10549-t3cont", "ГОСТ 10549-80", 5, 0),
    ("5264-t3", "ГОСТ 5264-80", 7, 0),
    ("24705-t1", "ГОСТ 24705-2004", 6, 0),
]

def load_case(doc, pno, idx, dpi=200):
    path = Path(f"data/raw/{doc}.pdf")
    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[pno - 1]
        gray = _render_gray(path, pno)
        tables = sorted(find_tables(page, gray), key=lambda t: t.bbox[1])
        tables = [t for t in tables if table_rows(page, t, require_data=True)]
        print("   data tables on page:", len(tables))
        t = tables[idx]
        g = grid_cells(t)
        h = split_header(g)
        rows = table_rows(page, t, require_data=True)
        x0, top, x1, bottom = t.bbox
        hdr_bottom = t.rows[h].bbox[1] if h < len(t.rows) else bottom
        # строка данных целиком — для привязки столбцов
        data_bottom = t.rows[h].bbox[3] if h < len(t.rows) else bottom
        pw, ph = page.width, page.height
    doc_ = pymupdf.open(str(path))
    z = dpi / 72
    pix = doc_[pno - 1].get_pixmap(dpi=dpi, clip=pymupdf.Rect(x0 - 2, top - 2, x1 + 2, data_bottom + 2))
    crop = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    pix2 = doc_[pno - 1].get_pixmap(dpi=60)
    page_img = Image.frombytes("RGB", (pix2.width, pix2.height), pix2.samples)
    return dict(ncols=len(g[0]), header=rows.header if rows else None,
                first=rows.body[0] if rows else None, crop=crop, page=page_img)

if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1])
    for cid, doc, pno, idx in CASES:
        c = load_case(doc, pno, idx)
        c["crop"].save(out / f"case_{cid}.png")
        print(cid, "cols", c["ncols"], "crop", c["crop"].size)
        print("  first:", c["first"])
