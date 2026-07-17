#!/usr/bin/env python3
"""sparkbench — the frozen Sparkulator A/B benchmark.

Runs every prompt in sparkbench-prompts.jsonl (greedy, 300 tok, streamed) against a
served glm-5.2 endpoint and reports, per prompt and per category:
  - decode tok/s   = usage.completion_tokens / (t_last_chunk - t_first_chunk)
                     (usage-based — NEVER stream-chunk counts; MTP/dspark bundle
                      tokens per chunk)
  - pos-0/1/.. acceptance and tok/verify from /metrics deltas:
        per_pos_total{position=N} / num_drafts_total      (drafts, NOT draft tokens)

Protocol (session-18): fresh cluster + serve, warm >= 8 runs (--warmup 8), and A/B
candidates back-to-back in the SAME session on this same file. Compare nothing else.

Usage:
  python3 sparkbench.py --label epoch-3 --warmup 8 --out /path/epoch3.json
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def read_metrics(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
        txt = r.read().decode()
    out = {"drafts": 0.0, "accepted": 0.0, "per_pos": {}}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = float(line.split()[-1])
        elif "spec_decode_num_accepted_tokens_per_pos_total" in line:
            m = re.search(r'position="(\d+)"', line)
            if m:
                out["per_pos"][int(m.group(1))] = float(line.split()[-1])
    return out


def stream_run(base, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_first = t_last = None
    completion = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                completion = obj["usage"]["completion_tokens"]
            if obj.get("choices"):
                delta = obj["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
    rate = completion / (t_last - t_first) if t_first and t_last > t_first else 0.0
    return completion, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--prompts", default=os.path.join(HERE, "sparkbench-prompts.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]

    for i in range(args.warmup):
        p = prompts[i % len(prompts)]["prompt"]
        _, rate = stream_run(args.base, args.model, p, args.max_tokens)
        print(f"[warm {i + 1}/{args.warmup}] {rate:.1f} tok/s", file=sys.stderr, flush=True)

    rows = []
    for i, item in enumerate(prompts):
        m0 = read_metrics(args.base)
        toks, rate = stream_run(args.base, args.model, item["prompt"], args.max_tokens)
        m1 = read_metrics(args.base)
        drafts = m1["drafts"] - m0["drafts"]
        row = {
            "category": item["category"],
            "prompt": item["prompt"],
            "tokens": toks,
            "decode_tok_s": round(rate, 2),
        }
        if drafts > 0:
            row["tok_per_verify"] = round(
                1 + (m1["accepted"] - m0["accepted"]) / drafts, 3)
            for pos in sorted(m1["per_pos"]):
                row[f"pos{pos}"] = round(
                    (m1["per_pos"][pos] - m0["per_pos"].get(pos, 0)) / drafts, 3)
        rows.append(row)
        pos_str = " ".join(f"pos{k}={row.get(f'pos{k}', '-')}" for k in (0, 1))
        print(f"[{args.label} {i + 1}/{len(prompts)}] {item['category']:18s} "
              f"{rate:5.1f} tok/s  {pos_str}", file=sys.stderr, flush=True)

    bycat = collections.defaultdict(list)
    for r in rows:
        bycat[r["category"]].append(r)
    summary = {}
    for cat, rs in sorted(bycat.items()):
        summary[cat] = {
            "n": len(rs),
            "decode_tok_s": round(sum(r["decode_tok_s"] for r in rs) / len(rs), 2),
            "pos0": round(sum(r.get("pos0", 0) for r in rs) / len(rs), 3),
            "pos1": round(sum(r.get("pos1", 0) for r in rs) / len(rs), 3),
            "tok_per_verify": round(
                sum(r.get("tok_per_verify", 0) for r in rs) / len(rs), 3),
        }
    overall = {
        "decode_tok_s_mean": round(sum(r["decode_tok_s"] for r in rows) / len(rows), 2),
        "pos0_mean": round(sum(r.get("pos0", 0) for r in rows) / len(rows), 3),
        "pos1_mean": round(sum(r.get("pos1", 0) for r in rows) / len(rows), 3),
        "tok_per_verify_mean": round(
            sum(r.get("tok_per_verify", 0) for r in rows) / len(rows), 3),
        "n_prompts": len(rows),
    }
    result = {"label": args.label, "overall": overall, "by_category": summary,
              "rows": rows}
    print(json.dumps({"label": args.label, "overall": overall,
                      "by_category": summary}, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
