#!/usr/bin/env python3
"""Streaming decode benchmark: TTFT + steady-state tok/s (excludes prefill).

usage: bench_decode.py [--port 8000] [--model glm-5.2] [--tokens 256] [--runs 3]
"""
import argparse
import json
import time
import urllib.request


def run_once(port, model, tokens, prompt):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps({
            "model": model, "temperature": 0, "max_tokens": tokens,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    stamps = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            delta = ev["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                stamps.append(now)
    if len(stamps) < 10:
        return ttft, 0.0, len(stamps)
    # steady-state: drop the first 10% of tokens
    k = max(1, len(stamps) // 10)
    span = stamps[-1] - stamps[k]
    return ttft, (len(stamps) - 1 - k) / span if span > 0 else 0.0, len(stamps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    prompts = [
        "Describe the water cycle in detail.",
        "Explain how a transistor works, step by step.",
        "Tell the history of the printing press.",
        "Walk through how TCP congestion control works.",
        "Explain photosynthesis at a molecular level.",
    ]
    rates = []
    for i in range(args.runs):
        ttft, rate, n = run_once(args.port, args.model, args.tokens,
                                 prompts[i % len(prompts)] + f" (run {i})")
        rates.append(rate)
        print(f"run {i}: ttft {ttft:.2f}s, steady decode {rate:.2f} tok/s "
              f"({n} chunks)")
    rates.sort()
    print(f"MEDIAN steady decode: {rates[len(rates)//2]:.2f} tok/s")


if __name__ == "__main__":
    main()
