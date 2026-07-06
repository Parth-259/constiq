"""Hand-written benchmark runner for the RAG pipeline.

Integration eval, not a unit test: run against a live local backend.

    .venv/bin/python -m backend.eval.run_eval [--base-url http://localhost:8000]

Uses substring matching against human-written expected phrases rather than an
LLM-as-judge — deterministic, free, and debuggable at hackathon scale.
"""
import argparse
import json
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def run(base_url: str) -> dict:
    benchmark = json.loads(BENCHMARK_PATH.read_text())
    results = []
    for entry in benchmark:
        try:
            resp = requests.post(
                f"{base_url}/api/ask",
                json={"project_id": entry["project_id"], "question": entry["question"]},
                timeout=60,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            results.append({**entry, "error": str(exc), "answer_correct": False, "source_correct": False})
            continue

        answer = (body.get("answer") or "").lower()
        answer_correct = all(p.lower() in answer for p in entry["expected_answer_contains"])
        source_pages = [s.get("page_number") for s in body.get("sources", [])]
        source_correct = entry["expected_source_page"] in source_pages
        results.append(
            {
                **entry,
                "returned_answer": body.get("answer"),
                "returned_source_pages": source_pages,
                "answer_correct": answer_correct,
                "source_correct": source_correct,
            }
        )

    total = len(results)
    summary = {
        "total_questions": total,
        "answer_correct_pct": round(100 * sum(r["answer_correct"] for r in results) / total, 1) if total else 0.0,
        "source_correct_pct": round(100 * sum(r["source_correct"] for r in results) / total, 1) if total else 0.0,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    logger.info("=== ConstructIQ eval: %s questions ===", total)
    logger.info("answer correct: %s%%   source correct: %s%%", summary["answer_correct_pct"], summary["source_correct_pct"])
    for r in results:
        status = "PASS" if r["answer_correct"] else "FAIL"
        logger.info("[%s] %s", status, r["question"])
        if not r["answer_correct"]:
            logger.info("   expected phrases: %s", r["expected_answer_contains"])
            logger.info("   got: %s", (r.get("returned_answer") or r.get("error", ""))[:200])
    logger.info("Full results written to %s", RESULTS_PATH)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    run(args.base_url)
