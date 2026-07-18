# s19 (2026-07-18) — Sparkulator workstream 0 (confidence-gated dynamic K): VERDICT NULL

> Previous: 12-sparkulator-v4-design.md
> Design doc: `../SPARKULATOR-DESIGN.md` (§3 = this workstream; verdict noted there too)

## STATUS

**Workstream 0 is a measured NULL — dynamic K loses on this stack at every
operating point tried. epoch-3 dspark K=3 stays champion (24.28/24.40).**
Code is built, deployed to both venvs, and inert at the default τ=0 (verified:
τ=0 reproduces baseline). Branch `spark-gb10`. Resting serve restored (§4b
fast build, epoch-3 K=3, no gating, :8000).

## DONE (all same-session A/B, fresh cluster+serve each, warm 8, frozen sparkbench)

| config | tok/s mean | pos0 | tok/verify | records |
|---|---|---|---|---|
| τ=0 (gating off) — baseline repro | **24.40** | 0.742 | 2.640 | `../dspark-distill/tools/sparkbench-tau0-s19.json` |
| τ=0.5 | 18.77 | 0.633* | 2.253 | `sparkbench-tau05-s19.json` |
| τ=0.5 + min_k=1 (no count-0 cliff) | 19.76 | 0.733 | 2.395 | `sparkbench-tau05-mink1-s19.json` |

\* artifact — see gotcha 3.

Implementation (deployed to both venvs, committed in `dspark-port/new/`):
- `qwen3_dspark.py`: confidence head (`Linear(6144+256→1)`, ckpt weights
  `confidence_head.proj.*`, input `[hidden; markov_embed]`, trained target
  `1 − d_TV`) now instantiated + loaded (was skipped).
- `dspark.py`: `VLLM_DSPARK_CONF_TAU` (float, 0=off) truncates each proposal at
  the first position with `sigmoid(conf) < τ`; `VLLM_DSPARK_CONF_MIN_K` floors
  the count. `DSPARK_CONF` log line every 500 steps with per-pos conf means.
- `gpu_model_runner.py`: dspark-with-τ rides the ngram-gpu valid-count trim path
  (async D2H counts + `update_scheduler_for_invalid_drafts` shrinks next step's
  scheduled verify slots). Gated on `_dspark_dynamic_k`; dormant at τ=0.

## WHY IT LOSES (the finding that matters for v4)

Decode here is bandwidth-bound on dense/attention bytes (~74% of a decode
step) which are paid **once per step regardless of K**. So the marginal verify
cost of a draft position is far below the ~32 ms/pos the design assumed, while
even a 0.4–0.5-acceptance position carries real expected-token value. Numbers:
- Gating that can reach count-0 is catastrophic: those steps fall to the
  ~15 tok/s no-spec floor (~24% of steps at τ=0.5 → −5.6 tok/s).
- Best case for trimming (min_k=1, deep positions only): still −4.6 tok/s.
  tok/verify fell 2.64→2.395 — real lost acceptances, not harness overhead.
- Bracket ⇒ no τ crosses into profit. **Structural, not a tuning miss.**

**Consequence for SPARKULATOR-DESIGN §2:** the fallback "if v4 gains are
pos-0-only, dynamic-K still monetizes them" is DEAD. v4 must pay through
acceptance lift alone. GATE C's "K sweep + dynamic-K τ" simplifies to a K sweep.

## NEXT

v4 training (design doc §5) when Joe green-lights: trainer surgery in the
pinned v05 clone → GATE A step-0 equivalence → per-epoch live probes. The
confidence head should still be retrained there (cheap, and calibration was
never cleanly measured — see gotcha 3), but not as a dynamic-K play.

## HOW TO RESUME / RE-RUN

- Serve §4b + gating: add `VLLM_DSPARK_CONF_TAU=0.5 VLLM_DSPARK_CONF_MIN_K=1`
  to the RUNBOOK §4b command line. τ unset/0 = stock dspark.
- Bench: `~/venvs/vllm-moet/bin/python sparkbench.py --label X --warmup 8 --out X.json`
  in `spark/dspark-distill/tools/`.
- Conf snapshots: `grep DSPARK_CONF /tmp/ray/session_latest/logs/worker-*.out`
  (both nodes; line fires at step 1 then every 500).

## GOTCHAS / KEY FACTS

1. Env vars DO propagate to Ray workers on both nodes (`get_driver_env_utils`
   forwards ~all of os.environ); the APIServer "Unknown vLLM environment
   variable" warning is cosmetic. Verified both ranks log identical conf values
   (no rank divergence — both ranks must gate identically or TP hangs).
2. Sentinel-padding instead of real trimming would save nothing — a padded
   position still runs the verify forward. The scheduler-side trim
   (ngram-gpu machinery) is the only way to actually shed verify compute.
3. **Metrics artifact:** sparkbench pos-N acceptance = `per_pos_total[N] /
   num_drafts_total`; steps whose proposal was truncated before position N sit
   in the denominator but can never accept at N. So cycle-2's pos0 0.633 and
   cycle-3's pos1 0.399 do NOT prove the confidence head is miscalibrated —
   selection quality was never cleanly measured. Correct per-proposal
   acceptance needs a proposed-at-N denominator the metrics don't expose.
4. `pkill -f` self-match strikes again (CLAUDE.md warns re ib_write_bw): a
   `pkill -9 -f "bin/vllm serve"` from a tool shell matches the *calling*
   wrapper and kills it. Kill by PID.
5. vLLM graceful shutdown after SIGTERM can hang >2 min; SIGKILL + fresh
   `start-ray-cluster.sh` (it force-stops both sides) is the reliable reset.

## CORRECTNESS GATES (all passed)

- τ=0 reproduces epoch-3 baseline (24.40 vs record 24.28, boot noise).
- Gating live on both ranks with identical values (worker logs).
- min_k=1 pos-0 acceptance 0.733 ≈ baseline 0.742 (pos-0 untouched, as designed).
- Greedy spec decode is lossless by construction — no quality gate needed for a
  proposal-length-only change (GSM8K skipped: nothing ships).

## LINKS

- Design: `../SPARKULATOR-DESIGN.md` (v4 = the remaining bet)
- Records: `../dspark-distill/tools/sparkbench-tau0-s19.json`,
  `sparkbench-tau05-s19.json`, `sparkbench-tau05-mink1-s19.json`
- Session tracker: `/sessions/sparkulator-dynamic-k.json`
