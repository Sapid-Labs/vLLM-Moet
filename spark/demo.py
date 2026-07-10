#!/usr/bin/env python3
"""Recordable inference demo: live token stream + speed stats.

usage: demo.py [prompt] [--tokens N] [--model glm-5.2] [--port 8000]
"""
import argparse
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default=(
    "Explain, in three short paragraphs, why mixture-of-experts models can "
    "be compressed to 2 bits per expert weight with little quality loss."))
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--model", default="glm-5.2")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()

banner = f"GLM-5.2 · 753B params · 2x NVIDIA DGX Spark · 2-bit MoE experts" \
    if args.model == "glm-5.2" else f"{args.model}"
print(f"\033[1m{banner}\033[0m")
print(f"\033[2m> {args.prompt}\033[0m\n")

req = urllib.request.Request(
    f"http://127.0.0.1:{args.port}/v1/chat/completions",
    json.dumps({
        "model": args.model, "temperature": 0, "max_tokens": args.tokens,
        "stream": True,
        "messages": [{"role": "user", "content": args.prompt}],
    }).encode(),
    {"Content-Type": "application/json"})

t0 = time.time()
ttft = None
n = 0
thinking = False
with urllib.request.urlopen(req, timeout=1800) as r:
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[6:])["choices"][0].get("delta", {})
        piece = d.get("content") or ""
        reason = d.get("reasoning_content") or ""
        if piece or reason:
            if ttft is None:
                ttft = time.time() - t0
            n += 1
        if reason and not thinking:
            sys.stdout.write("\033[2m")   # dim the thinking
            thinking = True
        if piece and thinking:
            sys.stdout.write("\033[0m\n\n")
            thinking = False
        sys.stdout.write(reason or piece)
        sys.stdout.flush()

wall = time.time() - t0
decode = (n - 1) / (wall - ttft) if n > 1 and wall > ttft else 0
print(f"\n\n\033[1m--- {n} tokens · TTFT {ttft:.1f}s · "
      f"{decode:.1f} tok/s decode · {wall:.1f}s total ---\033[0m")
