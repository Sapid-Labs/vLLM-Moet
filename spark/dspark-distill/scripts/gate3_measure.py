#!/usr/bin/env python3
"""GATE 3: measure dspark decode tok/s + per-position acceptance on reasoning prompts.

Streams each completion (temp 0, 300 tok), computes decode tok/s = completion_tokens /
(t_last - t_first_token), and reads /metrics spec-decode counters before/after the run
for tok/verify and per-position acceptance.
"""
import argparse, json, re, sys, time, urllib.request

def metrics(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
        txt = r.read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"(\S+?)(\{.*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if "spec_decode" in name:
            out[name + labels] = float(val)
    return out

def stream_chat(base, model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except Exception:
                continue
            delta = d["choices"][0].get("delta", {}).get("content")
            if delta:
                if t_first is None:
                    t_first = time.time()
                n += 1
    t_last = time.time()
    dec = n / (t_last - t_first) if t_first and t_last > t_first else 0
    return n, dec, (t_first - t0 if t_first else 0)

def sum_field(m, substr):
    return sum(v for k, v in m.items() if substr in k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=0, help="run this many prompts first, untimed")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    prompts = json.load(open(args.prompts))

    for i in range(args.warmup):
        p = prompts[i % len(prompts)]
        n, dec, ttft = stream_chat(args.base, args.model, p, args.max_tokens)
        print(f"[warm {i+1}] {n} tok, {dec:.1f} tok/s", file=sys.stderr, flush=True)

    m0 = metrics(args.base)
    rates = []
    for i, p in enumerate(prompts):
        n, dec, ttft = stream_chat(args.base, args.model, p, args.max_tokens)
        rates.append(dec)
        print(f"[{args.label} {i+1}/{len(prompts)}] {n} tok  decode {dec:.1f} tok/s  ttft {ttft:.1f}s",
              file=sys.stderr, flush=True)
    m1 = metrics(args.base)

    drafts = sum_field(m1, "num_draft_tokens_total") - sum_field(m0, "num_draft_tokens_total")
    accepted = sum_field(m1, "num_accepted_tokens_total") - sum_field(m0, "num_accepted_tokens_total")
    # per-position accepted counts
    perpos = {}
    for k in m1:
        if "per_pos" in k or "position" in k:
            mpos = re.search(r'(?:position|pos)="?(\d+)"?', k)
            if mpos:
                perpos[int(mpos.group(1))] = m1[k] - m0.get(k, 0)
    rates.sort()
    med = rates[len(rates)//2] if rates else 0
    mean = sum(rates)/len(rates) if rates else 0
    tok_per_verify = 1 + accepted/drafts if drafts else 0
    print(json.dumps({
        "label": args.label, "n_prompts": len(prompts),
        "decode_tok_s_mean": round(mean,2), "decode_tok_s_median": round(med,2),
        "decode_tok_s_min": round(min(rates),2) if rates else 0, "decode_tok_s_max": round(max(rates),2) if rates else 0,
        "drafts": int(drafts), "accepted": int(accepted), "tok_per_verify": round(tok_per_verify,3),
        "per_pos_accepted": {k: int(v) for k,v in sorted(perpos.items())},
    }, indent=2))

if __name__ == "__main__":
    main()
