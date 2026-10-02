"""Замер качества baseline: поиск, порог отказа и — по флагу — сами ответы.

Смысл этого скрипта — зафиксировать числа ДО того, как появятся multi-query,
HyDE и семантический чанкинг. Иначе улучшения нечем будет подтвердить: любая
надстройка «кажется лучше», пока её не с чем сравнить.

Три отдельных вопроса, которые раньше смешивались в одно число:

1. **Ранжирование** — где в выдаче нужный документ и пункт. Срезы dense, rrf и
   rerank; rerank здесь — порядок кросс-энкодера БЕЗ порога. Раньше порог
   срезал выдачу внутри среза, и «вклад реранкера −0,136» на деле измерял
   отказы, а не ранжирование: MRR совпадал с recall до третьего знака.
2. **Порог отказа** — отвечает ли система на вопросы из корпуса и молчит ли на
   остальных. Скоры кросс-энкодера сохраняются в JSON, и порог перебирается
   без моделей (``--scores-from``): прогон на CPU идёт часами, подбор порога —
   секунды. Порог выбирается на части ``tune``, а качество отказов
   показывается на части ``test`` — подбирать и мерить на одних и тех же шести
   вопросах значит мерить подгонку.
3. **Ответы** (``--generate``, платно: вызывает LLM) — содержит ли ответ
   эталонные значения и нет ли в нём чисел, которых нет в источниках. Два
   режима: RAG и «весь корпус в контексте». Корпус из четырёх стандартов
   целиком влезает в окно модели, и RAG обязан обыгрывать этот вариант, чтобы
   оправдать свою сложность.

Запуск:
  uv run python eval/run_eval.py --k 10 --json eval/baseline.json
  uv run python eval/run_eval.py --scores-from eval/baseline.json      # только порог
  uv run python eval/run_eval.py --k 10 --generate --json eval/baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gost_rag.config import get_settings
from gost_rag.logging import configure_logging
from gost_rag.models import RetrievedChunk

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")

#: Сигналы отказа и пороги, которые для них перебираются: сигмоида
#: кросс-энкодера и косинус лучшего плотного попадания (считается всегда и
#: бесплатно, поэтому годится для CPU, где реранкер выключен).
SIGNALS = {
    "top_score": (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
    "dense_top": (0.0, 0.45, 0.48, 0.5, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.6),
}
THRESHOLDS = SIGNALS["top_score"]


@dataclass
class Question:
    id: str
    question: str
    gold_designation: str | None = None
    gold_section: str | None = None
    out_of_corpus: bool = False
    note: str | None = None
    #: Что обязано быть в ответе. Элемент — строка или список равноправных
    #: вариантов («зелено-жёлтая» / «жёлто-зелёная»).
    answer_contains: list[str | list[str]] = field(default_factory=list)
    #: ``tune`` или ``test``; без явного значения — по хэшу id (см. ``split_of``).
    split: str | None = None


def split_of(question: Question) -> str:
    """Часть набора: подбор порога (tune) или его проверка (test).

    Детерминированно по хэшу id, чтобы новые вопросы распределялись сами и
    разбиение не менялось от перестановки вопросов в файле.
    """
    if question.split in {"tune", "test"}:
        return question.split
    return "tune" if hashlib.sha1(question.id.encode()).digest()[0] % 2 == 0 else "test"


# --------------------------------------------------------------------------- #
# Ранжирование
# --------------------------------------------------------------------------- #


@dataclass
class SliceStats:
    """Накопитель метрик одного среза выдачи."""

    name: str
    doc_hits: int = 0
    section_hits: int = 0
    section_top1_hits: int = 0
    section_total: int = 0
    reciprocal_ranks: list[float] = field(default_factory=list)
    total: int = 0

    def update(self, chunks: list[RetrievedChunk], question: Question, k: int) -> None:
        self.total += 1
        top = chunks[:k]

        rank = _first_rank(top, question.gold_designation)
        if rank is not None:
            self.doc_hits += 1
            self.reciprocal_ranks.append(1.0 / rank)
        else:
            self.reciprocal_ranks.append(0.0)

        if question.gold_section:
            self.section_total += 1
            section_rank = _first_section_rank(top, question)
            if section_rank is not None:
                self.section_hits += 1
                # Recall пункта на k=10 насыщается: 10 чанков по нескольку пунктов
                # покрывают заметную долю корпуса. Различает варианты поиска
                # только то, попал ли пункт в первый чанк — его модель цитирует первым.
                if section_rank == 1:
                    self.section_top1_hits += 1

    @property
    def recall(self) -> float:
        return self.doc_hits / self.total if self.total else 0.0

    @property
    def section_recall(self) -> float | None:
        return self.section_hits / self.section_total if self.section_total else None

    @property
    def section_at_1(self) -> float | None:
        return self.section_top1_hits / self.section_total if self.section_total else None

    @property
    def mrr(self) -> float:
        return (
            sum(self.reciprocal_ranks) / len(self.reciprocal_ranks)
            if self.reciprocal_ranks
            else 0.0
        )


def _matches_doc(chunk: RetrievedChunk, designation: str | None) -> bool:
    if not designation:
        return False
    found = (chunk.payload.get("designation") or "").upper()
    target = designation.upper()
    # Год в запросе эталона может быть опущен: «ГОСТ 380» матчит «ГОСТ 380-2005».
    return found == target or found.startswith(f"{target}-")


def _first_rank(chunks: list[RetrievedChunk], designation: str | None) -> int | None:
    for index, chunk in enumerate(chunks, start=1):
        if _matches_doc(chunk, designation):
            return index
    return None


def _section_matches(found: str, gold: str) -> bool:
    """Пункт совпадает с эталоном или вложен в него — по сегментам номера.

    Простой ``startswith`` засчитывал эталону «4.1» пункт «4.12», а эталону
    «1» — любой из «10.x»; вложенный «4.1.3» при этом честно относится к «4.1».
    """
    return found == gold or found.startswith(f"{gold}.")


def _first_section_rank(chunks: list[RetrievedChunk], question: Question) -> int | None:
    """Позиция первого чанка нужного документа, в текст которого попал эталонный пункт.

    Смотрим на ``sections`` — все пункты чанка, а не только тот, которым он
    подписан: 800-токенный чанк охватывает несколько пунктов.
    """
    if not question.gold_section:
        return None
    for index, chunk in enumerate(chunks, start=1):
        if not _matches_doc(chunk, question.gold_designation):
            continue
        sections = chunk.payload.get("sections") or [chunk.payload.get("section") or ""]
        if any(_section_matches(s, question.gold_section) for s in sections):
            return index
    return None


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Question(**item) for item in data.get("questions", [])]


# --------------------------------------------------------------------------- #
# Порог отказа
# --------------------------------------------------------------------------- #


def refused_at(record: dict, threshold: float, key: str = "top_score") -> bool:
    """Отказался бы граф на этом вопросе при данном пороге сигнала ``key``.

    Две причины, как в графе: вопрос называет только отсутствующие документы,
    или лучший скор (кросс-энкодера или плотного поиска) ниже порога.
    """
    if record.get("named_missing_only"):
        return True
    top = record.get(key)
    return top is None or top < threshold


@dataclass
class GateStats:
    threshold: float
    answered_in: int = 0
    total_in: int = 0
    refused_out: int = 0
    total_out: int = 0

    @property
    def answer_rate(self) -> float | None:
        return self.answered_in / self.total_in if self.total_in else None

    @property
    def refusal_rate(self) -> float | None:
        return self.refused_out / self.total_out if self.total_out else None

    @property
    def balanced(self) -> float | None:
        """Среднее двух долей: пропустить вопрос из корпуса так же плохо, как
        ответить на вопрос вне его, и перевес вопросов одного вида не маскирует
        провал на другом."""
        if self.answer_rate is None or self.refusal_rate is None:
            return None
        return (self.answer_rate + self.refusal_rate) / 2


def gate_stats(
    records: list[dict], threshold: float, split: str | None = None, key: str = "top_score"
) -> GateStats:
    stats = GateStats(threshold)
    for record in records:
        if split and record["split"] != split:
            continue
        refused = refused_at(record, threshold, key)
        if record["out_of_corpus"]:
            stats.total_out += 1
            stats.refused_out += refused
        else:
            stats.total_in += 1
            stats.answered_in += not refused
    return stats


def choose_threshold(records: list[dict], thresholds=THRESHOLDS, key: str = "top_score") -> float:
    """Порог с лучшей сбалансированной точностью на части tune.

    При равенстве — меньший: ложный отказ на вопросе из корпуса хотя бы виден
    пользователю, но при прочих равных лишний раз молчать незачем.
    """
    best, best_score = thresholds[0], -1.0
    for threshold in thresholds:
        score = gate_stats(records, threshold, split="tune", key=key).balanced
        if score is not None and score > best_score:
            best, best_score = threshold, score
    return best


# --------------------------------------------------------------------------- #
# Ответы
# --------------------------------------------------------------------------- #

#: Кириллица, похожая на латиницу, — к латинице: «М10» и «M10», «0,1 Ом» и
#: «0,1 Om» должны совпадать, как бы их ни набрал слой распознавания или модель.
_LOOKALIKES = str.maketrans(
    "АВЕКМНОРСТХІУаеорсхуі−–—",
    "ABEKMHOPCTXIYaeopcxyi---",
)


def normalize_answer(text: str) -> str:
    # «ё» — до замены похожих букв: иначе из «ё» получилась бы кириллическая «е»
    # среди уже латинских.
    text = text.replace("ё", "е").replace("Ё", "Е").translate(_LOOKALIKES).casefold()
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    return re.sub(r"\s+", "", text)


def _contains(answer: str, expected: str) -> bool:
    """Подстрока с границей числа: эталон «7мм» не должен совпадать с «17мм»."""
    needle = normalize_answer(expected)
    if not needle:
        return True
    guard = r"(?<![\d,])" if needle[0].isdigit() else ""
    return re.search(guard + re.escape(needle), normalize_answer(answer)) is not None


def answer_is_correct(answer: str, expected: list[str | list[str]]) -> bool | None:
    """Все эталонные значения есть в ответе; None — эталона у вопроса нет."""
    if not expected:
        return None
    for item in expected:
        variants = item if isinstance(item, list) else [item]
        if not any(_contains(answer, v) for v in variants):
            return False
    return True


@dataclass
class GenStats:
    name: str
    answered_in: int = 0
    total_in: int = 0
    correct: int = 0
    graded: int = 0
    ungrounded: int = 0
    refused_out: int = 0
    total_out: int = 0

    def update(self, question: Question, result: dict) -> None:
        if question.out_of_corpus:
            self.total_out += 1
            self.refused_out += bool(result["insufficient"])
            return
        self.total_in += 1
        self.answered_in += not result["insufficient"]
        if result.get("correct") is not None:
            self.graded += 1
            # Эталонное слово в ответе «во фрагментах этого нет, но вот что есть»
            # — не верный ответ: засчитывается только ответ, который не отказ.
            self.correct += bool(result["correct"]) and not result["insufficient"]
        self.ungrounded += bool(result.get("ungrounded_numbers"))


def _run_generation(question: Question, chunks, llm, settings, missing=None, alternatives=None):
    """Один ответ через те же узлы, что в графе: generate -> verify_citations."""
    from gost_rag.graph.nodes import make_generate_node, verify_citations
    from gost_rag.graph.state import ungrounded_numbers, used_indices

    state = {
        "question": question.question,
        "reranked": chunks,
        "messages": [],
        "missing_designations": missing or [],
        "designation_alternatives": alternatives or {},
    }
    raw = make_generate_node(llm, settings)(state)["answer"]
    verified = verify_citations({**state, "answer": raw})
    cited = [chunks[i - 1] for i in used_indices(raw, len(chunks))]
    return {
        "answer": raw,
        "insufficient": verified["insufficient"],
        "warnings": verified["warnings"],
        "ungrounded_numbers": ungrounded_numbers(raw, cited, question.question) if cited else [],
        "correct": answer_is_correct(raw, question.answer_contains),
    }


def rescore_generation(generation: dict, questions: list[Question]) -> dict:
    """Пересчитать метрики ответов по сохранённым текстам — без вызова LLM.

    Правила проверки меняются чаще, чем стоит перезапускать генерацию: отказ
    «во фрагментах ответа нет» со ссылкой на фрагмент, числа внутри обозначений
    стандартов. Ссылки и числа вне обозначений от этого не меняются, поэтому
    новый список чисел — это старый минус те, что встречаются только в
    обозначениях.
    """
    from gost_rag.graph.state import NUMBER_RE, says_no_answer
    from gost_rag.retrieval.filters import LOOSE_DESIGNATION_RE

    by_id = {q.id: q for q in questions}
    modes = {name: GenStats(name) for name in generation["modes"]}
    answers: list[dict] = []
    for entry in generation["answers"]:
        question = by_id.get(entry["id"])
        if question is None:
            continue
        rescored_entry = {"id": entry["id"]}
        for name in modes:
            result = dict(entry[name])
            answer = result.get("answer", "")
            result["insufficient"] = bool(result["insufficient"]) or says_no_answer(answer)
            outside = {
                m.group(0).lstrip("-+±")
                for m in NUMBER_RE.finditer(LOOSE_DESIGNATION_RE.sub(" ", answer))
            }
            result["ungrounded_numbers"] = [
                n for n in result.get("ungrounded_numbers") or [] if n in outside
            ]
            result["correct"] = (
                answer_is_correct(answer, question.answer_contains) if answer else False
            )
            modes[name].update(question, result)
            rescored_entry[name] = result
        answers.append(rescored_entry)
    return {"modes": _mode_dicts(modes), "answers": answers}


def _mode_dicts(modes: dict[str, GenStats]) -> dict:
    return {
        name: {
            "answer_rate": s.answered_in / s.total_in if s.total_in else None,
            "accuracy": s.correct / s.graded if s.graded else None,
            "graded": s.graded,
            "ungrounded_rate": s.ungrounded / s.total_in if s.total_in else None,
            "refusal_rate": s.refused_out / s.total_out if s.total_out else None,
        }
        for name, s in modes.items()
    }


def _all_chunks(client, settings) -> list[RetrievedChunk]:
    """Весь корпус в порядке документов — контекст для режима «без поиска»."""
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            settings.collection_name, limit=512, offset=offset, with_payload=True
        )
        points.extend(batch)
        if offset is None:
            break
    points.sort(key=lambda p: (p.payload.get("designation") or "", p.payload["chunk_index"]))
    return [
        RetrievedChunk(point_id=str(p.id), text=p.payload.get("text", ""), payload=p.payload)
        for p in points
    ]


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка baseline")
    parser.add_argument("--k", type=int, default=10, help="Глубина, на которой считать recall/MRR")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Вызвать LLM и проверить ответы (RAG и «весь корпус в контексте»). Платно.",
    )
    parser.add_argument(
        "--scores-from",
        type=Path,
        help="Не запускать модели: пересчитать порог по скорам из прошлого --json",
    )
    parser.add_argument(
        "--no-full-context",
        action="store_true",
        help="С --generate: не гонять режим «весь корпус в контексте» (дорогой, от поиска "
        "не зависит — достаточно одного прогона)",
    )
    parser.add_argument("--json", type=Path, help="Куда сохранить результат в JSON")
    args = parser.parse_args()

    if args.scores_from:
        previous = json.loads(args.scores_from.read_text(encoding="utf-8"))
        print(render_gates(previous["per_question"], get_settings()))
        if "generation" in previous:
            rescored = rescore_generation(previous["generation"], load_questions(args.questions))
            print(_render_generation(rescored))
            if args.json:
                previous["generation"] = rescored
                args.json.write_text(
                    json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        return 0

    settings = get_settings()
    configure_logging("WARNING")

    from gost_rag.ingest.embed import get_embedder
    from gost_rag.ingest.index import count_points, get_client
    from gost_rag.retrieval.filters import known_designations, resolve_designations
    from gost_rag.retrieval.rerank import get_reranker, passes_dense_gate, select_context
    from gost_rag.retrieval.store import dense_only_search, hybrid_search

    questions = load_questions(args.questions)
    client = get_client(settings)
    if count_points(client, settings) == 0:
        print("Индекс пуст — сначала выполните индексацию (scripts/ingest.py).")
        return 1

    embedder = get_embedder()
    reranker = get_reranker()
    available = known_designations(client, settings)
    _warn_about_missing_gold([q for q in questions if not q.out_of_corpus], available)

    use_reranker = settings.rerank_candidates > 0
    slice_names = ("dense", "rrf", "rerank") if use_reranker else ("dense", "rrf")
    slices = {name: SliceStats(name) for name in slice_names}
    records: list[dict] = []
    kept_by_id: dict[str, list[RetrievedChunk]] = {}
    matches = {}

    for number, question in enumerate(questions, start=1):
        # Прогон на CPU идёт часами; без прогресса не отличить работу от зависания.
        print(f"[{number}/{len(questions)}] {question.id}", file=sys.stderr, flush=True)
        match = resolve_designations(question.question, available)
        matches[question.id] = match
        record = {
            "id": question.id,
            "split": split_of(question),
            "out_of_corpus": question.out_of_corpus,
            "gold": question.gold_designation,
            "named": match.mentioned,
            "named_missing_only": match.only_missing,
            "top_score": None,
        }
        if match.only_missing:
            # Граф в этом случае не ищет вовсе — и оценка тоже.
            records.append(record)
            kept_by_id[question.id] = []
            continue

        embedding = embedder.encode_one(question.question)
        dense = dense_only_search(client, embedding, settings, query_filter=match.filter)
        dense_top = dense[0].fusion_score if dense else None
        rrf = hybrid_search(client, embedding, settings, query_filter=match.filter)
        if use_reranker:
            # Порядок кросс-энкодера без порога — отдельно от решения об отказе.
            pool = list(rrf[: settings.rerank_candidates])
            ranked = reranker.rerank(question.question, pool, top_n=len(pool), threshold=-1.0)
        else:
            ranked = list(rrf)
        top = ranked[0].rerank_score if use_reranker and ranked else None
        # Контекст для генерации — ровно так, как его соберёт граф.
        kept_by_id[question.id] = (
            select_context(question.question, list(rrf), reranker, settings)
            if passes_dense_gate(dense_top, settings)
            else []
        )

        rrf_rank = {c.point_id: i for i, c in enumerate(rrf, start=1)}
        record.update(
            {
                # Откуда в RRF пришли фрагменты контекста: по этим позициям видно,
                # сколько кандидатов реранкеру нужно на самом деле.
                "kept_rrf_ranks": [rrf_rank.get(c.point_id) for c in kept_by_id[question.id]],
                "top_score": top,
                "dense_top": dense_top,
                "top_found": ranked[0].designation if ranked else None,
                "scores": [
                    {
                        "designation": c.designation,
                        "score": round(c.rerank_score if use_reranker else c.fusion_score, 4),
                    }
                    for c in ranked[: args.k]
                ],
            }
        )
        if not question.out_of_corpus:
            slices["dense"].update(dense, question, args.k)
            slices["rrf"].update(rrf, question, args.k)
            if use_reranker:
                slices["rerank"].update(ranked, question, args.k)
            record.update(
                {
                    "rank_rrf": _first_rank(rrf[: args.k], question.gold_designation),
                    "rank_rerank": _first_rank(ranked[: args.k], question.gold_designation),
                    "section_rank_rrf": _first_section_rank(rrf[: args.k], question),
                    "section_rank_rerank": _first_section_rank(ranked[: args.k], question),
                }
            )
        records.append(record)

    in_corpus = [q for q in questions if not q.out_of_corpus]
    report = [_render_ranking(slices, records, args.k, len(in_corpus))]
    report.append(render_gates(records, settings))

    generation: dict | None = None
    if args.generate:
        generation = _generate_all(
            questions, kept_by_id, matches, client, settings, full_context=not args.no_full_context
        )
        report.append(_render_generation(generation))

    print("\n".join(report))

    if args.json:
        tuned = choose_threshold(records)
        tuned_dense = choose_threshold(records, SIGNALS["dense_top"], key="dense_top")
        payload = {
            "k": args.k,
            "rerank_candidates": settings.rerank_candidates,
            "rerank_threshold": settings.rerank_threshold,
            "min_dense_score": settings.min_dense_score,
            "tuned_dense_threshold": tuned_dense,
            "tuned_dense_on_test": _gate_dict(
                gate_stats(records, tuned_dense, split="test", key="dense_top")
            ),
            "slices": {
                name: {
                    "recall": s.recall,
                    "section_recall": s.section_recall,
                    "section_at_1": s.section_at_1,
                    "mrr": s.mrr,
                }
                for name, s in slices.items()
            },
            "refusal_rate": _graph_gate(records, settings).refusal_rate,
            "answer_rate": _graph_gate(records, settings).answer_rate,
            "tuned_threshold": tuned,
            "tuned_on_test": _gate_dict(gate_stats(records, tuned, split="test")),
            "per_question": records,
        }
        if generation is not None:
            payload["generation"] = generation
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    client.close()
    return 0


def _generate_all(questions, kept_by_id, matches, client, settings, full_context=True) -> dict:
    from gost_rag.llm.client import build_chat_model

    llm = build_chat_model(settings, streaming=False)
    corpus = _all_chunks(client, settings) if full_context else []
    modes = {"rag": GenStats("rag")}
    if full_context:
        modes["full_context"] = GenStats("full_context")
    answers: list[dict] = []

    for number, question in enumerate(questions, start=1):
        print(f"[gen {number}/{len(questions)}] {question.id}", file=sys.stderr, flush=True)
        match = matches[question.id]
        chunks = kept_by_id.get(question.id) or []
        if chunks:
            rag = _run_generation(
                question, chunks, llm, settings, match.missing, match.alternatives
            )
        else:
            rag = {"answer": "", "insufficient": True, "warnings": [], "correct": False}
        modes["rag"].update(question, rag)
        entry = {"id": question.id, "rag": rag}
        if full_context:
            entry["full_context"] = _run_generation(question, corpus, llm, settings)
            modes["full_context"].update(question, entry["full_context"])
        answers.append(entry)

    return {"modes": _mode_dicts(modes), "answers": answers}


def _warn_about_missing_gold(questions: list[Question], available: set[str]) -> None:
    """Предупредить, если эталонных документов нет в индексе.

    Без этой проверки нулевой recall выглядит как провал поиска, хотя на деле
    означает, что в корпус просто не загрузили нужные стандарты.
    """
    missing = sorted(
        {
            q.gold_designation
            for q in questions
            if q.gold_designation and not any(_matches(q.gold_designation, d) for d in available)
        }
    )
    if missing:
        print("ВНИМАНИЕ: эталонных документов нет в индексе — метрики по ним будут нулевыми:")
        for designation in missing:
            print(f"  - {designation}")
        print()


def _matches(target: str, found: str) -> bool:
    return found.upper() == target.upper() or found.upper().startswith(f"{target.upper()}-")


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def _gate_dict(stats: GateStats) -> dict:
    return {
        "threshold": stats.threshold,
        "answer_rate": stats.answer_rate,
        "refusal_rate": stats.refusal_rate,
        "balanced": stats.balanced,
        "in_corpus": stats.total_in,
        "out_of_corpus": stats.total_out,
    }


def _render_ranking(slices: dict[str, SliceStats], records: list[dict], k: int, n: int) -> str:
    lines = [
        "",
        f"Ранжирование: вопросов в корпусе {n}, k={k} (rerank — порядок без порога)",
        "",
        f"{'срез':<10} {'recall@k':>10} {'MRR@k':>10} {'пункт@k':>10} {'пункт@1':>10}",
        "-" * 54,
    ]
    for name, stats in slices.items():
        lines.append(
            f"{name:<10} {stats.recall:>10.3f} {stats.mrr:>10.3f} "
            f"{_fmt(stats.section_recall):>10} {_fmt(stats.section_at_1):>10}"
        )
    lines += [
        "",
        f"Вклад гибрида (rrf - dense) по MRR: {slices['rrf'].mrr - slices['dense'].mrr:+.3f}",
    ]
    if "rerank" in slices:
        lift = slices["rerank"].mrr - slices["rrf"].mrr
        lines.append(f"Вклад реранкера (rerank - rrf) по MRR: {lift:+.3f}")
    else:
        lines.append("Реранкер выключен (RERANK_CANDIDATES=0).")
    missed = [r for r in records if not r["out_of_corpus"] and r.get("rank_rerank") is None]
    if missed:
        lines += ["", "Нужного документа нет в топ-k:"]
        lines += [f"  {r['id']}: ожидался {r['gold']}, найден {r.get('top_found')}" for r in missed]
    return "\n".join(lines)


def _graph_gate(records: list[dict], settings) -> GateStats:
    """Отказы при текущих настройках — обоими сигналами сразу, как в графе."""
    stats = GateStats(settings.min_dense_score)
    for record in records:
        refused = refused_at(record, settings.min_dense_score, "dense_top")
        if settings.rerank_candidates > 0:
            refused = refused or refused_at(record, settings.rerank_threshold)
        if record["out_of_corpus"]:
            stats.total_out += 1
            stats.refused_out += refused
        else:
            stats.total_in += 1
            stats.answered_in += not refused
    return stats


def render_gates(records: list[dict], settings) -> str:
    """Таблицы порогов по обоим сигналам — какие есть в записях."""
    parts = []
    if any(r.get("dense_top") is not None for r in records):
        parts.append(
            render_gate(records, settings.min_dense_score, key="dense_top", title="плотный косинус")
        )
    if any(r.get("top_score") is not None for r in records):
        parts.append(render_gate(records, settings.rerank_threshold, title="кросс-энкодер"))
    return "\n".join(parts)


def render_gate(
    records: list[dict], current: float, key: str = "top_score", title: str = "кросс-энкодер"
) -> str:
    """Таблица порогов по частям tune/test и итог для выбранного на tune порога."""
    thresholds = SIGNALS[key]
    tuned = choose_threshold(records, thresholds, key=key)
    lines = [
        "",
        f"Порог отказа, сигнал — {title} "
        "(доля ответов на вопросы из корпуса / доля отказов вне корпуса)",
        "",
        f"{'порог':>7} {'tune: ответ':>12} {'tune: отказ':>12} {'test: ответ':>12} "
        f"{'test: отказ':>12}",
        "-" * 59,
    ]
    for threshold in sorted({*thresholds, current}):
        tune = gate_stats(records, threshold, "tune", key=key)
        test = gate_stats(records, threshold, "test", key=key)
        mark = []
        if threshold == current:
            mark.append("текущий")
        if threshold == tuned:
            mark.append("лучший на tune")
        lines.append(
            f"{threshold:>7.2f} {_fmt(tune.answer_rate):>12} {_fmt(tune.refusal_rate):>12} "
            f"{_fmt(test.answer_rate):>12} {_fmt(test.refusal_rate):>12}"
            + (f"  <- {', '.join(mark)}" if mark else "")
        )
    test = gate_stats(records, tuned, "test", key=key)
    counts = gate_stats(records, tuned, key=key)
    lines += [
        "",
        f"Порог, выбранный на tune: {tuned:.2f}. На test: отвечает на "
        f"{test.answered_in} из {test.total_in} вопросов из корпуса, отказывает на "
        f"{test.refused_out} из {test.total_out} вне корпуса.",
        f"Вопросов всего: в корпусе {counts.total_in}, вне корпуса {counts.total_out} — "
        "каждый вопрос вне корпуса двигает долю отказов на "
        + (f"{100 / counts.total_out:.0f} п.п." if counts.total_out else "—"),
    ]
    named = [r for r in records if r["out_of_corpus"] and r.get("named_missing_only")]
    if named:
        lines.append(
            f"Из них {len(named)} называют отсутствующий стандарт и отклоняются "
            "до поиска — порог на них не влияет."
        )
    return "\n".join(lines)


def _render_generation(generation: dict) -> str:
    lines = [
        "",
        "Ответы (LLM): RAG против «весь корпус в контексте»",
        "",
        f"{'режим':<14} {'отвечает':>9} {'верно':>9} {'чисел вне':>10} {'отказ вне':>10}",
        f"{'':<14} {'':>9} {'':>9} {'источника':>10} {'корпуса':>10}",
        "-" * 56,
    ]
    for name, m in generation["modes"].items():
        lines.append(
            f"{name:<14} {_fmt(m['answer_rate']):>9} {_fmt(m['accuracy']):>9} "
            f"{_fmt(m['ungrounded_rate']):>10} {_fmt(m['refusal_rate']):>10}"
        )
    graded = next(iter(generation["modes"].values()))["graded"]
    lines.append(f"«верно» — по {graded} вопросам с эталоном answer_contains.")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
