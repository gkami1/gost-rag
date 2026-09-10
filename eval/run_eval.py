"""Замер качества baseline: recall, MRR, вклад реранкера, доля отказов.

Смысл этого скрипта — зафиксировать числа ДО того, как появятся multi-query,
HyDE и семантический чанкинг. Иначе улучшения нечем будет подтвердить: любая
надстройка «кажется лучше», пока её не с чем сравнить.

Метрики считаются на трёх срезах, чтобы видеть, где именно теряется документ:
  * dense  — только плотный поиск;
  * rrf    — гибрид после слияния;
  * rerank — после кросс-энкодера.

Запуск: uv run python eval/run_eval.py [--k 10] [--no-generate]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gost_rag.config import get_settings
from gost_rag.logging import configure_logging
from gost_rag.models import RetrievedChunk

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")


@dataclass
class Question:
    id: str
    question: str
    gold_designation: str | None = None
    gold_section: str | None = None
    out_of_corpus: bool = False
    note: str | None = None


@dataclass
class SliceStats:
    """Накопитель метрик одного среза выдачи."""

    name: str
    doc_hits: int = 0
    section_hits: int = 0
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
            if any(
                _matches_doc(c, question.gold_designation)
                and (c.payload.get("section") or "").startswith(question.gold_section)
                for c in top
            ):
                self.section_hits += 1

    @property
    def recall(self) -> float:
        return self.doc_hits / self.total if self.total else 0.0

    @property
    def section_recall(self) -> float | None:
        return self.section_hits / self.section_total if self.section_total else None

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


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Question(**item) for item in data.get("questions", [])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка baseline-поиска")
    parser.add_argument("--k", type=int, default=10, help="Глубина, на которой считать recall/MRR")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Не вызывать LLM — считать только метрики поиска (без затрат на API)",
    )
    parser.add_argument("--json", type=Path, help="Куда сохранить результат в JSON")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")

    from gost_rag.ingest.embed import get_embedder
    from gost_rag.ingest.index import count_points, get_client
    from gost_rag.retrieval.filters import build_filter, known_designations
    from gost_rag.retrieval.rerank import get_reranker
    from gost_rag.retrieval.store import dense_only_search, hybrid_search

    questions = load_questions(args.questions)
    in_corpus = [q for q in questions if not q.out_of_corpus]
    out_corpus = [q for q in questions if q.out_of_corpus]

    client = get_client(settings)
    if count_points(client, settings) == 0:
        print("Индекс пуст — сначала выполните индексацию (scripts/ingest.py).")
        return 1

    embedder = get_embedder()
    reranker = get_reranker()
    available = known_designations(client, settings)

    _warn_about_missing_gold(in_corpus, available)

    slices = {name: SliceStats(name) for name in ("dense", "rrf", "rerank")}
    per_question: list[dict] = []

    for question in in_corpus:
        embedding = embedder.encode_one(question.question)
        query_filter = build_filter(question.question, available)

        dense = dense_only_search(client, embedding, settings, query_filter=query_filter)
        rrf = hybrid_search(client, embedding, settings, query_filter=query_filter)
        reranked = reranker.rerank(question.question, list(rrf), top_n=args.k)

        slices["dense"].update(dense, question, args.k)
        slices["rrf"].update(rrf, question, args.k)
        slices["rerank"].update(reranked, question, args.k)

        per_question.append(
            {
                "id": question.id,
                "gold": question.gold_designation,
                "rank_rrf": _first_rank(rrf[: args.k], question.gold_designation),
                "rank_rerank": _first_rank(reranked, question.gold_designation),
                "top_found": reranked[0].designation if reranked else None,
            }
        )

    refusals = _measure_refusals(out_corpus, embedder, reranker, client, settings, available)

    report = _render(slices, per_question, refusals, args.k, len(in_corpus))
    print(report)

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "k": args.k,
                    "slices": {
                        name: {
                            "recall": s.recall,
                            "section_recall": s.section_recall,
                            "mrr": s.mrr,
                        }
                        for name, s in slices.items()
                    },
                    "refusal_rate": refusals,
                    "per_question": per_question,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    client.close()
    return 0


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


def _measure_refusals(
    questions: list[Question], embedder, reranker, client, settings, available
) -> float | None:
    """Доля вопросов вне корпуса, на которых поиск честно не дал источников."""
    if not questions:
        return None

    from gost_rag.retrieval.filters import build_filter
    from gost_rag.retrieval.store import hybrid_search

    refused = 0
    for question in questions:
        embedding = embedder.encode_one(question.question)
        candidates = hybrid_search(
            client,
            embedding,
            settings,
            query_filter=build_filter(question.question, available),
        )
        # Отказ определяется тем же порогом, что и в графе: если после реранкера
        # не осталось ничего, узел guard уводит запрос в refuse и LLM не вызывается.
        if not reranker.rerank(question.question, list(candidates)):
            refused += 1
    return refused / len(questions)


def _render(
    slices: dict[str, SliceStats], per_question: list[dict], refusals: float | None, k: int, n: int
) -> str:
    lines = [
        "",
        f"Оценка baseline: вопросов в корпусе {n}, k={k}",
        "",
        f"{'срез':<10} {'recall@k':>10} {'MRR@k':>10} {'recall пункта':>15}",
        "-" * 48,
    ]
    for name in ("dense", "rrf", "rerank"):
        stats = slices[name]
        section = stats.section_recall
        lines.append(
            f"{name:<10} {stats.recall:>10.3f} {stats.mrr:>10.3f} "
            f"{('—' if section is None else f'{section:.3f}'):>15}"
        )

    lift = slices["rerank"].mrr - slices["rrf"].mrr
    lines += [
        "",
        f"Вклад гибрида (rrf - dense) по MRR: {slices['rrf'].mrr - slices['dense'].mrr:+.3f}",
        f"Вклад реранкера (rerank - rrf) по MRR: {lift:+.3f}",
    ]
    if refusals is not None:
        lines.append(f"Отказы на вопросах вне корпуса: {refusals:.0%}")

    missed = [q for q in per_question if q["rank_rerank"] is None]
    if missed:
        lines += ["", "Не найдено:"]
        lines += [f"  {q['id']}: ожидался {q['gold']}, найден {q['top_found']}" for q in missed]

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
