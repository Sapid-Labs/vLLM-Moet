#!/usr/bin/env python3
"""Build the v3 weak-band prompt set from OpenHermes-2.5 (Sparkulator round 3).

Targets the categories where the epoch-3 dspark draft is WEAK per the s16g probe
(creative 0.675 ... summarize 0.721), weighted toward the weakest. Emits train +
holdout jsonl (holdout never trained; used in the final A/B).

Category sourcing:
  creative_writing : airoboros2.2 category in {writing, roleplay, rp,
                     stylized_response, joke, wordgame, gtkm}
  chat_open        : LMSys Chatbot Arena + lmsys1m (real user chat), airoboros 'general'
  summarize_rewrite: any source, prompt starts with/contains summarize/rewrite/
                     paraphrase/condense/"explain ... to a" verbs
  code_explain     : glaive-code-assist prompts asking explanation (explain/what does/
                     why/how does/difference) — not generation
  code_gen         : glaive-code-assist remainder (write/implement/create)
  structured_factual: airoboros trivia + UnnaturalInstructions (small anchor slice)

Same hygiene as build_reasoning_prompts.py: first human turn only, dedupe by
200-char prefix, length bounds, deterministic seed.
"""
import argparse
import json
import random
import re
import sys

SUMMARIZE_RE = re.compile(
    r"^(please\s+)?(summari[sz]e|rewrite|paraphrase|condense|shorten)\b|"
    r"\bexplain\b.{0,80}\bto a (five|5|child|beginner|layman)", re.I | re.S)
CODE_EXPLAIN_RE = re.compile(
    r"\b(explain|what does|what is the difference|why does|how does|walk me through|"
    r"can you describe)\b", re.I)

CREATIVE_CATS = {"writing", "roleplay", "rp", "stylized_response", "joke",
                 "wordgame", "gtkm"}


def first_human(conv):
    for turn in conv or []:
        if turn.get("from") in ("human", "user"):
            v = (turn.get("value") or "").strip()
            return v or None
    return None


def classify(rec, prompt):
    src = (rec.get("source") or "").lower()
    cat = (rec.get("category") or "").lower()
    if "airoboros" in src and cat in CREATIVE_CATS:
        return "creative_writing"
    if "lmsys" in src or "chatbot arena" in src:
        return "chat_open"
    if "airoboros" in src and cat == "general":
        return "chat_open"
    if SUMMARIZE_RE.search(prompt):
        return "summarize_rewrite"
    if "glaive-code" in src:
        return "code_explain" if CODE_EXPLAIN_RE.search(prompt) else "code_gen"
    if "airoboros" in src and cat == "trivia":
        return "structured_factual"
    if "unnaturalinstructions" in src:
        return "structured_factual"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/joemuller/dspark-distill-data/"
                    "openhermes-src/openhermes2_5.json")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-holdout", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # weak-weighted mix, 4800 total
    QUOTA = {
        "creative_writing": 1200,
        "chat_open": 1000,
        "summarize_rewrite": 800,
        "code_explain": 800,
        "code_gen": 600,
        "structured_factual": 400,
    }

    data = json.load(open(args.src))
    by_cat = {c: [] for c in QUOTA}
    seen = set()
    for rec in data:
        p = first_human(rec.get("conversations"))
        if not p or not (args.min_chars <= len(p) <= args.max_chars):
            continue
        cat = classify(rec, p)
        if cat is None:
            continue
        key = p[:200]
        if key in seen:
            continue
        seen.add(key)
        by_cat[cat].append(p)
    print("available:", {c: len(v) for c, v in by_cat.items()}, file=sys.stderr)

    rng = random.Random(args.seed)
    train, holdout = [], []
    for cat, quota in QUOTA.items():
        pool = by_cat[cat]
        rng.shuffle(pool)
        take = pool[:quota]
        if len(take) < quota:
            print(f"WARN: {cat} only {len(take)}/{quota}", file=sys.stderr)
        n_hold = max(1, int(len(take) * args.holdout_frac))
        for i, p in enumerate(take):
            row = {"id": f"{cat}_{i}", "source": f"weak-{cat}", "prompt": p}
            (holdout if i < n_hold else train).append(row)
    rng.shuffle(train)
    for path, rows in ((args.out_train, train), (args.out_holdout, holdout)):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    def counts(rows):
        out = {}
        for r in rows:
            out[r["source"]] = out.get(r["source"], 0) + 1
        return out
    print(f"train {len(train)} -> {args.out_train} {counts(train)}", file=sys.stderr)
    print(f"holdout {len(holdout)} -> {args.out_holdout} {counts(holdout)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
