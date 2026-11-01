"""
Score LoreLoop against a fixed question set, so changes to prompts, models or
chunking can be compared with numbers instead of gut feel.

    python evaluate.py eval_attention_paper.json
    python evaluate.py eval_attention_paper.json --model mistral --k 5 --chunk-size 800

Each case may set:
  expect_source    citations that count as finding the right chunk, e.g. "1706.03762, p. 5"
  expect_in_answer words/numbers the answer must contain; "a|b" accepts either
Results are printed and saved to eval_results/ (gitignored).
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import agent

sys.stdout.reconfigure(encoding="utf-8")  # ✓/✗ crash Windows' cp1252 when output is piped to a file

DEFAULTS = {"model": "llama3", "k": 3, "chunk_size": 1000, "chunk_overlap": 50, "max_iterations": 3}


def mentions(text: str, needle: str) -> bool:
    """Whole-token, case-insensitive match: '8' won't match '2018', 'p. 5' won't match 'p. 50'."""
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text, re.I) is not None


def score_case(case: dict, result: dict, seconds: float) -> dict:
    answer = result["final_answer"]
    retrieved = [s for h in result["run_history"] for s in h["sources"]]
    expect_src, expect_ans = case.get("expect_source", []), case.get("expect_in_answer", [])
    return {
        "question": case["question"],
        "answer": answer,
        # None = case doesn't check this, so it's left out of that rate
        "retrieval_hit": any(mentions(s, e) for s in retrieved for e in expect_src) if expect_src else None,
        "answer_match": all(any(mentions(answer, alt) for alt in w.split("|")) for w in expect_ans) if expect_ans else None,
        "verified": not result["failed"] and not answer.startswith("⚠️"),
        "attempts": result["iterations"],
        "seconds": round(seconds, 1),
    }


def summarize(rows: list) -> dict:
    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(100 * sum(vals) / len(vals)) if vals else None
    return {
        "retrieval_hit_%": rate("retrieval_hit"),
        "answer_match_%": rate("answer_match"),
        "verified_%": rate("verified"),
        "avg_attempts": round(sum(r["attempts"] for r in rows) / len(rows), 2),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Score LoreLoop on a question set.")
    ap.add_argument("eval_file")
    ap.add_argument("--model")
    ap.add_argument("--k", type=int)
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--chunk-overlap", type=int)
    ap.add_argument("--max-iterations", type=int)
    args = ap.parse_args()

    with open(args.eval_file, encoding="utf-8") as f:
        spec = json.load(f)
    cfg = {**DEFAULTS, **spec.get("settings", {})}
    cfg.update({k: v for k, v in vars(args).items() if k in cfg and v is not None})

    print(f"Loading {len(spec['sources'])} source(s)...")
    docs = [d for s in spec["sources"] for d in agent.load_source(s)]
    rag_app, num_chunks = agent.build_pipeline(docs, cfg["chunk_size"], cfg["chunk_overlap"], cfg["k"], cfg["model"])
    print(f"Indexed {num_chunks} chunks | settings: {cfg}\n")

    mark = {True: "✓", False: "✗", None: "-"}
    rows = []
    for i, case in enumerate(spec["cases"], 1):
        start = time.time()
        row = score_case(case, agent.run_query(rag_app, case["question"], cfg["max_iterations"]), time.time() - start)
        rows.append(row)
        print(f"{i:>2}. retrieval {mark[row['retrieval_hit']]}  answer {mark[row['answer_match']]}  "
              f"verified {mark[row['verified']]}  {row['attempts']} try  {row['seconds']:>5}s  {row['question']}")

    summary = summarize(rows)
    print("\n" + "  ".join(f"{k}: {v}" for k, v in summary.items()))

    os.makedirs("eval_results", exist_ok=True)
    name = os.path.splitext(os.path.basename(args.eval_file))[0]
    out = os.path.join("eval_results", f"{name}_{cfg['model']}_{datetime.now():%Y%m%d-%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"settings": cfg, "summary": summary, "cases": rows}, f, ensure_ascii=False, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
