#!/usr/bin/env python3
"""Correctness battery for MTP-under-PP debugging.

Greedy spec decode must be output-invariant: these prompts make corruption
obvious (exact arithmetic, named facts, long prose that shows dropped or
repeated tokens).

usage: mtp_correctness_battery.py [--model ds4-pp2] [--port 8000]
"""
import argparse
import json
import re
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ds4-pp2")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()


def ask(prompt, tokens):
    req = {"model": args.model, "temperature": 0, "max_tokens": tokens,
           "messages": [{"role": "user", "content": prompt}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        json.dumps(req).encode(), {"Content-Type": "application/json"}),
        timeout=1800))
    return r["choices"][0]["message"]["content"]


checks = []

t = ask("What is 17 * 23? Answer with just the number.", 160)
checks.append(("arithmetic 17*23=391", "391" in t, t[-80:]))

t = ask("What is 23 * 19? Answer with just the number.", 160)
checks.append(("arithmetic 23*19=437", "437" in t, t[-80:]))

t = ask("Name the capital of Australia in one word.", 120)
checks.append(("fact Canberra", "Canberra" in t, t[-60:]))

t = ask("Write two paragraphs about the history of lighthouses.", 320)
words = re.findall(r"[a-zA-Z']+", t)
# repetition detector: any 4-gram appearing 3+ times
grams = {}
loop = False
for i in range(len(words) - 4):
    g = " ".join(words[i:i + 4]).lower()
    grams[g] = grams.get(g, 0) + 1
    if grams[g] >= 3:
        loop = True
        break
# mangling detector: single-letter fragments that are not a/i
frags = sum(1 for w in words if len(w) == 1 and w.lower() not in "ai")
checks.append(("prose no repetition-loop", not loop, f"loops={loop}"))
checks.append(("prose no mangled fragments", frags <= 2, f"frags={frags}"))

ok = all(c[1] for c in checks)
for name, passed, detail in checks:
    print(f"{'PASS' if passed else 'FAIL'}  {name}   [{detail}]")
print("VERDICT:", "CLEAN" if ok else "CORRUPT")
