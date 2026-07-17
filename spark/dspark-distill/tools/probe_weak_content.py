#!/usr/bin/env python3
"""Find where the epoch-3 dspark draft is WEAK: per-prompt pos-0/1 acceptance across
diverse content categories. Reads /metrics per_pos_total + num_drafts_total before/after
each request to get that request's acceptance. Ranks categories by acceptance.
"""
import json, time, urllib.request, sys, collections

PROMPTS = {
 "code_gen": [
   "Write a Python function that merges two sorted linked lists into one sorted list.",
   "Implement a debounce decorator in Python with a configurable delay.",
   "Write a SQL query to find the second-highest salary in an employees table.",
   "Write a Rust function that returns the nth Fibonacci number iteratively.",
 ],
 "code_explain": [
   "Explain what this does: `df.groupby('k')['v'].transform(lambda x: x - x.mean())`.",
   "Explain the difference between `git rebase` and `git merge` and when to use each.",
   "Walk through how a Bloom filter works and its false-positive tradeoff.",
   "Explain Python's GIL and how it affects multithreaded CPU-bound code.",
 ],
 "math_derivation": [
   "Derive the quadratic formula from ax^2 + bx + c = 0 by completing the square.",
   "Compute the derivative of f(x) = x^3 ln(x) and show each step.",
   "A fair coin is flipped 10 times. What is the probability of exactly 6 heads? Show work.",
   "Solve the recurrence T(n) = 2T(n/2) + n using the master theorem, with steps.",
 ],
 "science_derivation": [
   "Derive the equation for the period of a simple pendulum from first principles.",
   "Explain why the sky is blue in terms of Rayleigh scattering, step by step.",
   "Balance this reaction and explain: C3H8 + O2 -> CO2 + H2O.",
   "Explain how a transistor amplifies a signal, at the level of electrons.",
 ],
 "logic_reasoning": [
   "Three switches control one bulb in another room; you may enter once. How do you find which switch?",
   "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Explain.",
   "You have 8 balls, one heavier. With a balance scale and 2 weighings, find it. Explain.",
   "A bat and ball cost $1.10; the bat costs $1 more than the ball. How much is the ball? Explain.",
 ],
 "factual_short": [
   "What is the capital of Australia?",
   "Who wrote the novel 'Pride and Prejudice'?",
   "What year did the Berlin Wall fall?",
   "What is the chemical symbol for gold?",
 ],
 "creative_writing": [
   "Write the opening paragraph of a mystery novel set on a rainy night.",
   "Write a short poem about the ocean at dawn.",
   "Invent a dialogue between a robot and a cat who just met.",
   "Describe a bustling alien marketplace in vivid sensory detail.",
 ],
 "summarize_rewrite": [
   "Summarize the plot of Romeo and Juliet in three sentences.",
   "Rewrite this formally: 'hey can u send me the thing when u get a sec thx'.",
   "Explain photosynthesis to a five-year-old.",
   "Condense the causes of World War I into a short bulleted list.",
 ],
 "structured_output": [
   "Return a JSON object describing a book with title, author, year, and genres (array).",
   "Make a markdown table comparing REST and GraphQL across 4 dimensions.",
   "List the first 10 prime numbers, comma-separated, nothing else.",
   "Output a YAML config for a web server with port, host, and 3 routes.",
 ],
 "chat_open": [
   "I'm feeling overwhelmed at work lately. Any advice?",
   "What's a good weekend project for someone learning electronics?",
   "Recommend three sci-fi books and say why in one line each.",
   "How do you stay motivated when learning something hard?",
 ],
}

BASE="http://localhost:8000"

def metrics():
    t=urllib.request.urlopen(f"{BASE}/metrics",timeout=10).read().decode()
    d=p0=p1=0
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_drafts_total") and "created" not in l: d=float(l.split()[-1])
        elif "per_pos_total" in l and 'position="0"' in l: p0=float(l.split()[-1])
        elif "per_pos_total" in l and 'position="1"' in l: p1=float(l.split()[-1])
    return d,p0,p1

def run(prompt, max_tokens=300):
    body=json.dumps({"model":"glm-5.2","messages":[{"role":"user","content":prompt}],
                     "temperature":0,"max_tokens":max_tokens,"stream":False}).encode()
    req=urllib.request.Request(f"{BASE}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req,timeout=600).read()

def main():
    # warm
    for _ in range(4): run(PROMPTS["math_derivation"][0])
    rows=[]
    for cat, ps in PROMPTS.items():
        for p in ps:
            d0,a0,b0=metrics(); run(p); d1,a1,b1=metrics()
            dd=d1-d0
            if dd<=0: continue
            pos0=(a1-a0)/dd; pos1=(b1-b0)/dd
            rows.append((cat,pos0,pos1,p))
            print(f"{cat:18s} pos0={pos0:.3f} pos1={pos1:.3f} | {p[:45]}",file=sys.stderr,flush=True)
    # per-category means
    bycat=collections.defaultdict(list)
    for cat,p0,p1,_ in rows: bycat[cat].append((p0,p1))
    summary={c:{"pos0":round(sum(x[0] for x in v)/len(v),3),
                "pos1":round(sum(x[1] for x in v)/len(v),3),"n":len(v)} for c,v in bycat.items()}
    ranked=sorted(summary.items(), key=lambda kv: kv[1]["pos0"])
    print("\n=== per-category pos-0 acceptance (ascending = weakest first) ===")
    for c,s in ranked: print(f"  {c:18s} pos0={s['pos0']:.3f} pos1={s['pos1']:.3f} n={s['n']}")
    allp0=[r[1] for r in rows]
    print(f"\noverall pos0 mean={sum(allp0)/len(allp0):.3f} min={min(allp0):.3f} max={max(allp0):.3f} spread={max(allp0)-min(allp0):.3f}")
    json.dump({"rows":[{"cat":c,"pos0":p0,"pos1":p1,"prompt":pr} for c,p0,p1,pr in rows],"summary":summary},
              open("/home/joemuller/weak-content-probe.json","w"),indent=2)

if __name__=="__main__": main()
