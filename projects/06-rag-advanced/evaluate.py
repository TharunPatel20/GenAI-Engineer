"""Project 06: Measure retrieval (hit rate) and run a simple groundedness heuristic.

Run:  python evaluate.py

Keyword search needs no API key. Vector and hybrid search need GEMINI_API_KEY in .env.
The groundedness check is a SIMPLE HEURISTIC, not a full evaluation.
"""

import json
import os
from pathlib import Path

from hybrid_retrieval import HybridRetriever, tokenize

PROJECT_DIR = Path(__file__).parent
QUESTIONS_FILE = PROJECT_DIR / "eval_questions.json"
TOP_K = 3
MODES = ["keyword", "vector", "hybrid"]

# Common words that carry little meaning. The groundedness check ignores them.
STOPWORDS = {
    "about", "after", "also", "because", "before", "could", "from", "have", "into",
    "only", "other", "should", "their", "there", "these", "they", "this", "those",
    "through", "what", "when", "where", "which", "while", "will", "with", "would",
    "your", "that", "than", "then", "were", "been", "being", "does", "more",
}

# Two hard-coded answers for the groundedness demo. They are NOT model output.
DEMO_QUESTION = "What does order code ACR-4471 include?"
DEMO_ANSWERS = {
    "supported by the FAQ": (
        "Order code ACR-4471 is the Starter Bundle. It includes the lamp, a spare shade, "
        "and a 1.5 m USB-C cable."
    ),
    "invented details": (
        "Order code ACR-4471 includes a lifetime extended warranty and free same-day "
        "delivery worldwide."
    ),
}


def hit(expected_phrase, chunks):
    # A hit means the expected phrase appears in at least one of the top-k chunks (ignoring case).
    return any(expected_phrase.lower() in c.page_content.lower() for c in chunks)


def key_terms(answer):
    # Key terms: longer words, plus codes and numbers. Stopwords and very short tokens are skipped.
    terms = set()
    for token in tokenize(answer):
        if token in STOPWORDS or len(token) < 2:
            continue
        if len(token) >= 5 or any(ch.isdigit() for ch in token):
            terms.add(token)
    return terms


def groundedness_check(answer, retrieved_chunks, min_support=0.8):
    # SIMPLE HEURISTIC, NOT A FULL EVALUATION.
    # It flags an answer when too many of its key terms are missing from the retrieved text.
    # It cannot tell whether a term is used with the right meaning, so read the flags with care.
    context_terms = set()
    for chunk in retrieved_chunks:
        context_terms.update(tokenize(chunk.page_content))
    terms = key_terms(answer)
    if not terms:
        return {"flagged": False, "support": None, "missing": []}
    missing = sorted(t for t in terms if t not in context_terms)
    support = 1 - len(missing) / len(terms)
    return {"flagged": support < min_support, "support": support, "missing": missing}


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    retriever = HybridRetriever()

    modes = MODES
    if not os.getenv("GEMINI_API_KEY", "").strip():
        modes = ["keyword"]
        print("GEMINI_API_KEY is not set. Only keyword search is measured.\n")

    # Step 1: For each mode, record whether the expected phrase is in the top-k chunks.
    results = {}
    for mode in modes:
        results[mode] = []
        for item in questions:
            chunks = retriever.search(item["question"], k=TOP_K, mode=mode)
            results[mode].append(hit(item["expected_phrase"], chunks))

    # Step 2: Print a small table.
    total = len(questions)
    print(f"Retrieval hit rate (top {TOP_K}), {total} questions")
    print(f"{'Mode':<10}{'Hits':>8}{'Rate':>8}")
    for mode in modes:
        hits = sum(results[mode])
        print(f"{mode:<10}{hits:>5}/{total:<2}{hits / total:>8.0%}")

    # Step 3: List the questions that the last measured mode missed.
    last_mode = modes[-1]
    missed = [q["question"] for q, ok in zip(questions, results[last_mode]) if not ok]
    print(f"\nMissed by {last_mode}:")
    for question in missed or ["(none)"]:
        print(f" - {question}")

    # Step 4: Groundedness heuristic on two hard-coded answers. Context comes from keyword search,
    # so this step makes no API calls.
    context = retriever.search(DEMO_QUESTION, k=TOP_K, mode="keyword")
    print(f"\nGroundedness heuristic (demo, not a full evaluation). Question: {DEMO_QUESTION}")
    for label, answer in DEMO_ANSWERS.items():
        check = groundedness_check(answer, context)
        status = "FLAGGED" if check["flagged"] else "ok"
        print(f" - {label}: {status}, support {check['support']:.0%}, missing terms: {check['missing']}")


if __name__ == "__main__":
    main()
