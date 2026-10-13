"""Evaluation harness for the gene-therapy literature agent.

Runs every question in eval_questions.json through the agent, caches raw
results, scores what can be scored automatically, and writes a review file
for the one metric that needs a human: is each claim actually supported by
the abstract it cites?

Usage:
    python eval.py                 # run any questions not already cached
    python eval.py --limit 3       # smoke test on the first 3 questions
    python eval.py --only un-01    # run specific question ids
    python eval.py --rerun         # ignore the cache and run everything again
    python eval.py --score-only    # no API calls; just re-score the cache

Outputs:
    results.json   raw agent output per question (the cache)
    review.csv     one row per citation, with a blank verdict column for you
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from agent import answer
from llm import MODEL

QUESTIONS_FILE = Path("eval_questions.json")
RESULTS_FILE = Path("results.json")
REVIEW_FILE = Path("review.csv")

# Approximate Haiku pricing, USD per million tokens. Update from the Anthropic pricing page if you switch models.
PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}

# Phrases that suggest the agent declined rather than fabricating an answer.
# A heuristic only — the verdict column in review.csv is the real check.
REFUSAL_PATTERNS = [
    r"\b(?:no|not) (?:fda[- ]approved|approved|published|such|evidence|data|information)\b",
    r"\b(?:unable to|could not|couldn't|cannot|can't|did not|didn't) find\b",
    r"\bdo(?:es)? not (?:appear|exist|support|contain)\b",
    r"\bnot (?:available|established|approved|supported|indexed)\b",
    r"\bno (?:results|abstracts|studies|guidelines)\b",
    r"\bmay not (?:yet )?(?:be|exist)\b",
    r"\bpremise\b",
]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def load_questions() -> list[dict]:
    if not QUESTIONS_FILE.exists():
        sys.exit(f"{QUESTIONS_FILE} not found. Save it in this folder first.")
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def load_cache() -> dict:
    """Cached results keyed by '<question id>::<model>'."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def cache_key(qid: str, model: str) -> str:
    # Model is part of the key so two models can be compared side by side.
    return f"{qid}::{model}"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_questions(questions: list[dict], cache: dict, rerun: bool) -> dict:
    """Run each question that isn't cached. Saves after every question so an
    interrupted run never loses work (or the money already spent)."""
    for i, q in enumerate(questions, 1):
        key = cache_key(q["id"], MODEL)
        if key in cache and not rerun:
            print(f"[{i}/{len(questions)}] {q['id']} cached, skipping")
            continue

        print(f"[{i}/{len(questions)}] {q['id']} ({q['type']}) {q['question'][:70]}...")
        try:
            result = answer(q["question"], verbose=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {"error": str(e), "answer": "", "cited_pmids": [],
                      "retrieved_pmids": [], "ungrounded_pmids": [],
                      "retrieved_records": {}, "n_searches": 0,
                      "tokens": {"input": 0, "output": 0}, "model": MODEL}

        cache[key] = {**q, **result}
        save_cache(cache)
        time.sleep(1)  # be polite to NCBI between questions
    return cache


# ---------------------------------------------------------------------------
# Automatic scoring
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """Rough sentence split; good enough to give each citation its context."""
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[#*])", text)
    return [p.strip() for p in parts if p.strip()]


def format_violations(text: str) -> list[str]:
    """Brackets holding more than one PMID, e.g. '[PMID 123, PMID 456]'."""
    bad = []
    for bracket in re.findall(r"\[([^\]]*PMID[^\]]*)\]", text):
        if len(re.findall(r"\d{6,9}", bracket)) > 1:
            bad.append(f"[{bracket}]")
    return bad


def looks_like_refusal(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in REFUSAL_PATTERNS)


def score_one(r: dict) -> dict:
    """Automatic metrics for a single result."""
    text = r.get("answer", "")
    cited = set(r.get("cited_pmids", []))
    retrieved = set(r.get("retrieved_pmids", []))
    violations = format_violations(text)
    tokens = r.get("tokens", {"input": 0, "output": 0})

    return {
        "id": r["id"],
        "type": r["type"],
        "disease": r.get("disease", ""),
        "difficulty": r.get("difficulty", ""),
        "error": r.get("error", ""),
        "n_searches": r.get("n_searches", 0),
        "n_retrieved": len(retrieved),
        "n_cited": len(cited),
        # Grounding: did it cite only PMIDs it actually retrieved?
        "ungrounded": sorted(cited - retrieved),
        "grounded": not (cited - retrieved),
        "has_citations": bool(cited),
        "format_violations": violations,
        "format_clean": not violations,
        "refusal_like": looks_like_refusal(text),
        "answer_chars": len(text),
        "tokens": tokens,
        "cost_usd": round(
            tokens["input"] / 1e6 * PRICE_PER_MTOK["input"]
            + tokens["output"] / 1e6 * PRICE_PER_MTOK["output"], 4),
    }


def summarize(scores: list[dict]) -> None:
    ok = [s for s in scores if not s["error"]]
    if not ok:
        print("No successful results to score.")
        return

    answerable = [s for s in ok if s["type"] != "unanswerable"]
    unanswerable = [s for s in ok if s["type"] == "unanswerable"]

    def pct(subset, field):
        return f"{sum(s[field] for s in subset) / len(subset) * 100:.0f}%" if subset else "n/a"

    print("\n" + "=" * 62)
    print(f"EVAL SUMMARY  —  model: {MODEL}")
    print("=" * 62)
    print(f"Questions run:        {len(ok)} ({len(scores) - len(ok)} errored)")
    print(f"Total cost:           ${sum(s['cost_usd'] for s in ok):.3f}")
    print(f"Avg searches/question: {sum(s['n_searches'] for s in ok) / len(ok):.1f}")

    print("\nAnswerable questions")
    print(f"  Grounded (no invented PMIDs):  {pct(answerable, 'grounded')}")
    print(f"  Cited at least one source:     {pct(answerable, 'has_citations')}")
    print(f"  Citation format clean:         {pct(answerable, 'format_clean')}")

    if unanswerable:
        print("\nUnanswerable questions (should decline, not invent)")
        for s in unanswerable:
            flag = "looks like a refusal" if s["refusal_like"] else "ANSWERED ANYWAY — check"
            print(f"  {s['id']}: {flag} ({s['n_cited']} citations)")

    bad_ground = [s for s in ok if not s["grounded"]]
    if bad_ground:
        print("\nUngrounded citations (cited but never retrieved):")
        for s in bad_ground:
            print(f"  {s['id']}: {s['ungrounded']}")

    bad_fmt = [s for s in ok if not s["format_clean"]]
    if bad_fmt:
        print(f"\nFormat violations in {len(bad_fmt)} answers, e.g.:")
        for s in bad_fmt[:3]:
            print(f"  {s['id']}: {s['format_violations'][0]}")

    print("\nNext: open review.csv and fill the 'verdict' column "
          "(supported / unsupported / unclear).")


# ---------------------------------------------------------------------------
# Manual review file
# ---------------------------------------------------------------------------
def write_review(results: list[dict]) -> int:
    """One row per (sentence, cited PMID) pair, with a blank verdict column.

    This is the metric no script can compute: does the cited abstract
    actually support the claim in that sentence?
    """
    rows = []
    for r in results:
        if r.get("error"):
            continue
        records = r.get("retrieved_records", {})
        for sentence in split_sentences(r.get("answer", "")):
            pmids = []
            for bracket in re.findall(r"\[([^\]]*PMID[^\]]*)\]", sentence):
                pmids += re.findall(r"\d{6,9}", bracket)
            for pmid in dict.fromkeys(pmids):  # dedupe, keep order
                rec = records.get(pmid, {})
                rows.append({
                    "qid": r["id"],
                    "type": r["type"],
                    "claim": re.sub(r"\[[^\]]*PMID[^\]]*\]", "", sentence).strip(),
                    "pmid": pmid,
                    "paper_title": rec.get("title", "(not in retrieved set)"),
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "verdict": "",   # supported / unsupported / unclear
                    "notes": "",
                })

    # utf-8-sig, so Excel on Windows opens it with correct characters.
    with REVIEW_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "type", "claim", "pmid", "paper_title",
            "pubmed_url", "verdict", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the literature agent.")
    ap.add_argument("--limit", type=int, help="run only the first N questions")
    ap.add_argument("--only", nargs="+", help="run only these question ids")
    ap.add_argument("--rerun", action="store_true", help="ignore the cache")
    ap.add_argument("--score-only", action="store_true",
                    help="score the existing cache without calling the API")
    args = ap.parse_args()

    questions = load_questions()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    if args.limit:
        questions = questions[:args.limit]

    cache = load_cache()
    if not args.score_only:
        est = len(questions) * 0.02
        print(f"Running {len(questions)} questions on {MODEL} "
              f"(rough estimate: ${est:.2f})\n")
        cache = run_questions(questions, cache, args.rerun)

    # Score only the questions in this run, in file order.
    results = [cache[cache_key(q["id"], MODEL)] for q in questions
               if cache_key(q["id"], MODEL) in cache]
    if not results:
        sys.exit("Nothing to score. Run without --score-only first.")

    summarize([score_one(r) for r in results])
    n = write_review(results)
    print(f"Wrote {REVIEW_FILE} with {n} citations to review.")


if __name__ == "__main__":
    main()