#!/usr/bin/env python3
"""Generate the target's greedy completions for reasoning prompts → ShareGPT jsonl.

Stdlib only (ThreadPoolExecutor + urllib). Resumable: skips ids already in --out.
Output record matches the magpie raw schema:
  {"id","source","conversations":[{"from":"human","value":..},{"from":"gpt","value":..}],"finish_reason"}
"""
import argparse, json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

def call(base, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"]["content"], ch.get("finish_reason", "stop")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts)]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"total={len(rows)} done={len(done)} todo={len(todo)}", file=sys.stderr, flush=True)

    lock = threading.Lock()
    out_f = open(args.out, "a")
    n_ok = [0]; n_err = [0]; t0 = time.time(); tok = [0]

    def work(r):
        for attempt in range(args.retries):
            try:
                comp, fr = call(args.base, args.model, r["prompt"], args.max_tokens, args.timeout)
                return r, comp, fr, None
            except Exception as e:
                if attempt == args.retries - 1:
                    return r, None, None, str(e)
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            r, comp, fr, err = fut.result()
            if err or not comp:
                n_err[0] += 1
                if n_err[0] <= 10:
                    print("ERR", r["id"], err, file=sys.stderr, flush=True)
                continue
            rec = {"id": r["id"], "source": r["source"],
                   "conversations": [{"from": "human", "value": r["prompt"]},
                                     {"from": "gpt", "value": comp}],
                   "finish_reason": fr}
            with lock:
                out_f.write(json.dumps(rec) + "\n"); out_f.flush()
                n_ok[0] += 1; tok[0] += len(comp.split())
                if n_ok[0] % 50 == 0:
                    el = time.time() - t0
                    print(f"ok={n_ok[0]} err={n_err[0]} ~words={tok[0]} "
                          f"rate={n_ok[0]/el:.2f} req/s elapsed={el:.0f}s", file=sys.stderr, flush=True)
    out_f.close()
    print(f"DONE ok={n_ok[0]} err={n_err[0]}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
