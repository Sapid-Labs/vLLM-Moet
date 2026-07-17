#!/usr/bin/env python3
"""Streamed greedy decode-rate measurement against localhost:8000 (glm-5.2)."""
import json
import sys
import time
import urllib.request

PROMPTS = [
    "Explain how a transformer language model generates text, step by step.",
    "Write a Python function that merges two sorted lists and explain its complexity.",
    "Natalia sold clips to 48 friends in April, and half as many in May. How many total? Show your reasoning.",
    "Describe the tradeoffs between tensor parallelism and pipeline parallelism for LLM inference.",
]


def run(prompt: str, max_tok: int = 300) -> tuple[int, float]:
    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t_first = None
    t_last = None
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
    if t_first is None or t_last == t_first:
        return completion, 0.0
    return completion, completion / (t_last - t_first)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for i in range(n):
        toks, rate = run(PROMPTS[i % len(PROMPTS)])
        print(f"run {i+1:2d}: {toks} tok  {rate:.1f} tok/s", flush=True)
