#!/usr/bin/env python3
"""Capture per-token expert routing at native k=8 across diverse prompt domains.

Sends /v1/completions requests (server must run --enable-return-routed-experts),
decodes the base64 npy routing arrays (tokens-1, layers, top_k), and saves one
.npy per request into OUTDIR plus a manifest.
"""
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "http://localhost:8000/v1/completions"
OUTDIR = Path(__file__).parent / "routing_capture"
OUTDIR.mkdir(exist_ok=True)

PROMPTS = {
    "essay_bandwidth": "Write a detailed technical essay about why memory bandwidth, not compute, limits large language model inference on unified-memory systems. Cover the roofline model, batch size effects, and mixture-of-experts models.\n\n",
    "code_python": "Write a Python module implementing an LRU cache with TTL expiry, generics type hints, thread safety, and full docstrings. Then write pytest tests for it.\n\n```python\n",
    "code_rust": "Implement a lock-free multi-producer single-consumer ring buffer in Rust with atomics. Explain each memory ordering choice in comments.\n\n```rust\n",
    "math_reasoning": "Solve step by step, showing all work: A train leaves city A at 9:15 traveling 84 km/h. Another leaves city B (312 km away) at 9:45 traveling toward it at 96 km/h. At what time do they meet? Then compute 24*17, 391/17, and the prime factorization of 1001.\n\n",
    "factual_qa": "Answer these factual questions with brief explanations: What is the capital of Australia? Who wrote One Hundred Years of Solitude? What year did the Berlin Wall fall? What is the boiling point of water at the top of Mount Everest and why? Name the four largest moons of Jupiter.\n\n",
    "fiction": "Write the opening chapter of a mystery novel set in a remote Norwegian fishing village in winter, where the local lighthouse keeper has vanished leaving only a half-finished chess game.\n\n",
    "json_structured": "Generate a JSON array of 15 fictional employee records with fields: id, name, department, salary, hire_date, skills (array), manager_id (nullable). Output only valid JSON.\n\n",
    "dialogue_support": "Write a realistic customer support chat transcript where a customer can't log into their bank app after a phone upgrade, and the agent walks them through diagnosis to resolution, including 2FA re-enrollment.\n\n",
    "translation_style": "Translate the following into French, then German, then Japanese, preserving tone: 'The committee regrets to inform you that your proposal, while ambitious, exceeds the scope of this year's funding cycle.' Then explain the key translation choices for each language.\n\n",
    "science_explain": "Explain how CRISPR-Cas9 gene editing works at the molecular level, including guide RNA design, PAM sequences, double-strand break repair pathways, and why off-target effects occur.\n\n",
    "legal_brief": "Draft a short memorandum analyzing whether a software company can enforce a non-compete clause against a remote employee living in California when the contract specifies Delaware law.\n\n",
    "recipe_howto": "Write a complete recipe for sourdough bread from starter maintenance through baking, with a troubleshooting section covering dense crumb, gummy interior, and flat loaves.\n\n",
}

MAX_TOKENS = 400


def run_one(name: str, prompt: str) -> dict:
    body = json.dumps({
        "model": "glm-5.2",
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.load(r)
    wall = time.time() - t0
    choice = resp["choices"][0]
    usage = resp["usage"]
    b64 = choice.get("routed_experts")
    rec = {
        "name": name,
        "wall_s": round(wall, 2),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "has_routing": b64 is not None,
    }
    if b64 is not None:
        arr = np.load(io.BytesIO(base64.b64decode(b64)))
        np.save(OUTDIR / f"{name}.npy", arr)
        rec["shape"] = list(arr.shape)
        rec["dtype"] = str(arr.dtype)
    (OUTDIR / f"{name}.txt").write_text(choice["text"])
    return rec


def main():
    names = sys.argv[1:] or list(PROMPTS)
    manifest_path = OUTDIR / "manifest.jsonl"
    for name in names:
        rec = run_one(name, PROMPTS[name])
        with manifest_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
