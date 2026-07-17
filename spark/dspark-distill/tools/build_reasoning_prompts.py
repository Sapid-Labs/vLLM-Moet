#!/usr/bin/env python3
"""Extract reasoning/technical/code prompts from OpenHermes-2.5 by source tag.

Domains (Joe's pick): technical explanation/derivation + code + general reasoning.
Math is explicitly excluded. Output: ShareGPT-style prompt jsonl (first human turn only),
tagged with domain, deterministic sampling.
"""
import argparse, glob, json, os, random, sys

# source-substring -> domain bucket. Checked case-insensitively, first match wins.
BUCKETS = [
    ("code",       ["glaive-code", "code-assistant", "code_", "codealpaca", "code "]),
    ("derivation", ["camel"]),  # CamelAI physics/chem/bio = technical derivation/explanation
    ("reasoning",  ["airoboros", "cot_", "cot-", "evol", "platypus", "caseus"]),
]
EXCLUDE = ["math", "gsm", "metamath"]  # no math per Joe

def domain_of(source):
    s = (source or "").lower()
    if any(x in s for x in EXCLUDE):
        return None
    for dom, subs in BUCKETS:
        if any(x in s for x in subs):
            return dom
    return None

def first_human(conv):
    for turn in conv:
        if turn.get("from") in ("human", "user"):
            v = (turn.get("value") or "").strip()
            return v or None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-domain", type=int, default=1600)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=4000)  # keep prompts reasonable
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src_dir, "*.json")) +
                   glob.glob(os.path.join(args.src_dir, "**", "*.json"), recursive=True))
    files = [f for f in files if os.path.getsize(f) > 10_000_000]  # the big data file(s)
    if not files:
        print("no large .json found in", args.src_dir, file=sys.stderr); sys.exit(1)
    print("reading:", files, file=sys.stderr)

    by_dom = {d: [] for d, _ in BUCKETS}
    seen = set()
    n_read = 0
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for rec in data:
            n_read += 1
            dom = domain_of(rec.get("source"))
            if not dom:
                continue
            p = first_human(rec.get("conversations") or [])
            if not p or not (args.min_chars <= len(p) <= args.max_chars):
                continue
            key = p[:200]
            if key in seen:
                continue
            seen.add(key)
            by_dom[dom].append(p)
    print("read", n_read, "records; per-domain available:",
          {d: len(v) for d, v in by_dom.items()}, file=sys.stderr)

    rng = random.Random(args.seed)
    out = []
    for dom, prompts in by_dom.items():
        rng.shuffle(prompts)
        take = prompts[:args.per_domain]
        for i, p in enumerate(take):
            out.append({"id": f"{dom}_{i}", "source": f"reasoning-{dom}", "prompt": p})
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print("wrote", len(out), "prompts ->", args.out,
          "| domain counts:", {d: sum(1 for r in out if r["source"] == f"reasoning-{d}") for d, _ in BUCKETS},
          file=sys.stderr)

if __name__ == "__main__":
    main()
