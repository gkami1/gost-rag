"""Сравнение локальных VLM на шапках таблиц: точность столбцов, легенда, время."""
import base64, io, json, re, sys, time, difflib
import urllib.request
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from cases import CASES, load_case

TRUTH = {
 "10549-t2": ["Шаг резьбы P","Сбег x, не более / нормальный","Сбег x, не более / уменьшенный",
   "Недорез a, не более / нормальный","Недорез a, не более / уменьшенный",
   "Проточка / Тип 1 / нормальная / f","Проточка / Тип 1 / нормальная / R","Проточка / Тип 1 / нормальная / R1",
   "Проточка / Тип 1 / узкая / f","Проточка / Тип 1 / узкая / R","Проточка / Тип 1 / узкая / R1",
   "Проточка / Тип 2 / f","Проточка / Тип 2 / R2","Проточка / d_f",
   "Фаска z / при сопряжении с внутренней резьбой с проточкой типа 2","Фаска z / для всех других случаев"],
 "10549-t3cont": ["Обозначение размера резьбы","Число шагов на длине 25,4 мм",
   "Сбег x, не более, при угле заборной части инструмента / 20°","Сбег x, не более, при угле заборной части инструмента / 30°",
   "Недорез a, не более / нормальный","Недорез a, не более / уменьшенный",
   "Проточка / нормальная / f","Проточка / нормальная / R","Проточка / нормальная / R1",
   "Проточка / узкая / f","Проточка / узкая / R","Проточка / узкая / R1","Проточка / d_f","Фаска z"],
 "5264-t3": ["Условное обозначение сварного соединения",
   "Конструктивные элементы / подготовленных кромок свариваемых деталей","Конструктивные элементы / сварного шва",
   "s","R","e, не более","g / Номин.","g / Пред. откл."],
 "24705-t1": ["Номинальный диаметр резьбы D, наружный диаметр резьбы d","Шаг P","Средний диаметр D2, d2",
   "Внутренний диаметр D1, d1","Внутренний диаметр по дну впадины d3"],
}
#: что легенда обязана сказать (подстрока в описании), по нормализованному символу
LEGEND = {"10549-t2": {"r": "радиус", "r1": "радиус", "r2": "радиус", "f": "ширин", "df": "диаметр"},
          "10549-t3cont": {"r": "радиус", "r1": "радиус", "f": "ширин", "df": "диаметр"},
          "5264-t3": {"r": "радиус"}, "24705-t1": {}}

SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
def norm(s):
    s = s.translate(SUB).lower().replace("ё", "е").replace("_", "")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"[^\w/°,.+]", "", s)
    return s
def leaf(s): return norm(s).split("/")[-1]

PROMPT = """На первом изображении — шапка таблицы из ГОСТа и начало её тела. На втором — вся страница в низком разрешении: по чертежу и тексту видно, что означают буквенные обозначения.

В таблице ровно {n} столбцов. Первая строка данных по столбцам слева направо: {first}

Верни JSON:
{{"columns": [ровно {n} строк — полный путь заголовка каждого столбца слева направо, от верхнего уровня шапки к нижнему, уровни через " / ", текст как напечатан; повёрнутый текст тоже читай; нижние индексы пиши обычными цифрами: R1, D2, d_f],
 "legend": {{"обозначение": "что оно означает, по-русски, кратко"}}}}

Не придумывай. Нечитаемый заголовок — "?". В legend — только обозначения из шапки, смысл которых виден на чертеже или в тексте страницы."""

SCHEMA = {"type": "object", "properties": {
  "columns": {"type": "array", "items": {"type": "string"}},
  "legend": {"type": "object", "additionalProperties": {"type": "string"}}},
  "required": ["columns", "legend"]}

def b64(img, fmt="PNG"):
    buf = io.BytesIO(); img.save(buf, fmt); return base64.b64encode(buf.getvalue()).decode()

def ask(model, case):
    body = {"model": model, "stream": False, "format": SCHEMA, "think": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1500},
            "messages": [{"role": "user",
                          "content": PROMPT.format(n=case["ncols"], first=" | ".join(case["first"])),
                          "images": [b64(case["crop"]), b64(case["page"])]}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    return out, time.time() - t

def score(cid, got):
    truth = TRUTH[cid]
    cols = got.get("columns") or []
    count_ok = len(cols) == len(truth)
    leaf_ok = path_ok = 0
    for i, t in enumerate(truth):
        g = cols[i] if i < len(cols) else ""
        leaf_ok += leaf(g) == leaf(t)
        path_ok += difflib.SequenceMatcher(None, norm(g), norm(t)).ratio() >= 0.85
    leg = {re.sub(r"[^\w]", "", k.translate(SUB).lower().replace("_", "")): v.lower()
           for k, v in (got.get("legend") or {}).items()}
    need = LEGEND[cid]
    leg_ok = sum(1 for k, word in need.items() if word in leg.get(k, ""))
    return dict(count_ok=count_ok, leaf=f"{leaf_ok}/{len(truth)}", path=f"{path_ok}/{len(truth)}",
                legend=f"{leg_ok}/{len(need)}", leaf_n=leaf_ok, path_n=path_ok, leg_n=leg_ok,
                total=len(truth), leg_total=len(need))

if __name__ == "__main__":
    model = sys.argv[1]
    cases = {cid: load_case(doc, p, i) for cid, doc, p, i in CASES}
    # прогрев: загрузка весов не должна попасть во время первой таблицы
    t = time.time(); w, _ = ask(model, cases["24705-t1"])
    print(f"warmup {time.time()-t:.0f}s tokens={w.get('eval_count')}", w["message"]["content"][:400].replace(chr(10), " "), flush=True)
    results = {}
    for cid, case in cases.items():
        try:
            out, secs = ask(model, case)
            raw = out["message"]["content"]
            got = json.loads(raw)
        except Exception as exc:
            print(cid, "ERROR", exc, "| eval_count", out.get("eval_count"), "|", raw[:300].replace(chr(10), " "), flush=True)
            results[cid] = {"error": str(exc), "raw": raw}; continue
        s = score(cid, got); s["secs"] = round(secs)
        results[cid] = {"score": s, "got": got}
        print(cid, s, flush=True)
    json.dump(results, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
