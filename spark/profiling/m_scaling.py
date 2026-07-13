#!/usr/bin/env python3
"""M-scaling probe: per-stream steady decode tok/s at concurrency 1/2/4/8.

The core question for the 20 tok/s goal (spark/GOAL.md): is single-stream
decode bandwidth-bound or latency/M-starved? If per-stream throughput RISES as
we add concurrent streams, the memory system was starved at batch 1 (more
in-flight work hides latency) -> the win is amortizing plane reads over more
tokens (tree speculation). If per-stream FALLS, we were already saturating
bandwidth and speculation won't help.

Non-destructive: hits the running server over the API. usage-token based, so
MTP token-bundling doesn't skew it.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

PROMPT = ("Write a detailed technical essay about why memory bandwidth, not "
          "compute, limits large language model inference on unified-memory "
          "systems. Cover the roofline model, batch size effects, and "
          "mixture-of-experts models.\n\n")


def one_stream(port, model, tokens, out, idx):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps({
            "model": model, "temperature": 0.7, "seed": 42 + idx,
            "max_tokens": tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT}],
        }).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    stamps = []
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            if not ev["choices"]:
                continue
            d = ev["choices"][0].get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                stamps.append(now)
    # steady-state: drop first 10% of wall after ttft
    n = usage["completion_tokens"] if usage else len(stamps)
    if len(stamps) < 12:
        out[idx] = (0.0, n)
        return
    k = max(1, len(stamps) // 10)
    span = stamps[-1] - stamps[k]
    # scale chunk-count rate to true token count (MTP bundles tokens/chunk)
    rate = (len(stamps) - 1 - k) / span if span > 0 else 0.0
    scale = n / max(1, len(stamps))
    out[idx] = (rate * scale, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--levels", default="1,2,4,8")
    args = ap.parse_args()

    print(f"# M-scaling probe  model={args.model} tokens={args.tokens}")
    print(f"{'N':>3} {'per-stream tok/s (median)':>26} {'aggregate tok/s':>16} "
          f"{'per-stream spread':>18}")
    for N in [int(x) for x in args.levels.split(",")]:
        out = [None] * N
        threads = [threading.Thread(target=one_stream,
                                    args=(args.port, args.model, args.tokens, out, i))
                   for i in range(N)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        rates = [o[0] for o in out if o and o[0] > 0]
        toks = sum(o[1] for o in out if o)
        if not rates:
            print(f"{N:>3}  (no valid streams)")
            continue
        med = statistics.median(rates)
        agg = toks / wall
        spread = f"{min(rates):.1f}-{max(rates):.1f}"
        print(f"{N:>3} {med:>26.2f} {agg:>16.1f} {spread:>18}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
