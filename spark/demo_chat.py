#!/usr/bin/env python3
"""Interactive streaming demo against the GLM-5.2 dspark serve on :8000.

Stdlib only. Streams tokens to the terminal, then prints a stats line:
decode tok/s (measured client-side) and speculative-decode acceptance
(delta of the server's /metrics counters across the request).

usage: python3 spark/demo_chat.py [--host spark-05cc] [--max-tokens 512] [-t 0.6]
       one-shot: python3 spark/demo_chat.py "your prompt here"
"""

import argparse
import json
import sys
import time
import urllib.request

def get_spec_counters(base):
    counters = {}
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:spec_decode_num_") and "created" not in line:
                    name, val = line.rsplit(" ", 1)
                    key = name.split("{")[0]
                    pos = '"position":' in name or 'position="' in name
                    if pos:
                        key += "_pos" + name.split('position="')[1].split('"')[0]
                    counters[key] = counters.get(key, 0.0) + float(val)
    except OSError:
        pass
    return counters

def chat(base, messages, max_tokens, temperature):
    body = json.dumps({
        "model": "glm-5.2",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    before = get_spec_counters(base)
    t0 = time.time()
    first = None
    usage = None
    text = []
    in_think = False
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                think = delta.get("reasoning_content")
                piece = delta.get("content")
                if think:
                    if not in_think:
                        sys.stdout.write("\x1b[2m")  # dim the thinking
                        in_think = True
                    sys.stdout.write(think)
                elif piece:
                    if in_think:
                        sys.stdout.write("\x1b[0m\n---\n")
                        in_think = False
                    sys.stdout.write(piece)
                    text.append(piece)
                if (think or piece) and first is None:
                    first = time.time()
                sys.stdout.flush()
    if in_think:
        sys.stdout.write("\x1b[0m")
    print()
    after = get_spec_counters(base)
    t = time.time()
    comp = usage["completion_tokens"] if usage else 0
    gen_t = max(t - (first or t0), 1e-9)
    stats = [f"{comp} tok", f"ttft {(first or t) - t0:.2f}s",
             f"decode {comp / gen_t:.1f} tok/s"]
    drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - \
        before.get("vllm:spec_decode_num_drafts_total", 0)
    accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - \
        before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    if drafts > 0:
        stats.append(f"spec: {1 + accepted / drafts:.2f} tok/verify "
                     f"({int(accepted)}/{int(drafts)} extra accepted)")
    print(f"\x1b[36m[{' | '.join(stats)}]\x1b[0m")
    return "".join(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", help="one-shot prompt (omit for REPL)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    if args.prompt:
        chat(base, [{"role": "user", "content": " ".join(args.prompt)}],
             args.max_tokens, args.temperature)
        return

    print(f"GLM-5.2 @ {base} — dspark speculative decode demo. "
          "Ctrl-D or 'exit' to quit.\n(first runs after a fresh serve are "
          "slow: mmap'd expert planes warm into page cache over ~10 turns)")
    messages = []
    while True:
        try:
            prompt = input("\n\x1b[1myou>\x1b[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": prompt})
        reply = chat(base, messages, args.max_tokens, args.temperature)
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
