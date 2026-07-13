# 02 — GLM-5.2 single-stream decode → 10 tok/s: top-k reduction + recovery finetune

## Target

GLM-5.2 single-stream decode **≥ 10 tok/s** on the two DGX Sparks (TP2), at
**acceptable quality**. Speed alone is solved (top-k=4 crosses 10); the open
problem is recovering quality at low top-k. Ends when a config sustains
≥10 tok/s and passes `spark/mtp_correctness_battery.py --model glm-5.2`.

## State (2026-07-12, session 6 — post-reboot re-baseline; k=4 = 8.5–9.8 sustained)

Node `.1` was rebooted before this session (the session-5 gate). Findings:

- **INFRA: RoCEv2 GID indexes are NOT stable across reboots/link events.**
  `.2`'s v2 GID index moved 5→6 (it wasn't even `.2` that rebooted); the
  hardcoded `NCCL_IB_GID_INDEX=5` in `start-ray-cluster.sh` made NCCL die at
  the very first init allreduce with "unhandled system error". **FIXED:** the
  script now resolves each node's RoCEv2 GID index for its fabric IP from
  sysfs at start time (prints `== RoCEv2 GID indexes: .1=X .2=Y ==`). If NCCL
  ever fails at init again, check that banner first.
- **Fresh k=8 sustained = 3.1 tok/s (3.18, 3.08 — flat/settled), LOWER than
  the pre-reboot 4.0–4.2.** Power 22–27 W both nodes (memory-stalled
  signature), no Firefox, no desktop hogs. So the reboot did NOT lift the
  k=8 floor; either `.2` (3 d uptime) is now the drag or this is the
  documented prompt-domain spread (measurement domain: MoE-bandwidth essay,
  temp 0.7 seed 42 — `scratchpad/measure_sustained.sh` protocol: 3×400 settle,
  then 2×(512+1024 differential)).
- **Fresh k=4 sustained = 9.80 / 8.52 tok/s — best sustained k=4 yet** (was
  7.7–7.9 flat in session 5). Brushes the target but does not reliably clear
  10. Note k4/k8 ratio = ~2.9×, far above the 1.8× bytes prediction — k=4
  likely gets a superlinear boost from higher resident-tier hit rate (4×78
  expert working set fits better). Worth understanding; it means k=8's low
  reading and k=4's high reading may both be residency effects, not noise.
- **Quality at k=4 remains broken: battery verdict CORRUPT** (repetition loop
  in the prose probe) even after ~2400 tok settle — though settle was in the
  essay domain, and verdicts are settle/domain-sensitive (session-5 caveat).
  **Finetune remains the gate for k=4 quality; speed-wise k=4 is now
  borderline-sufficient.**
- Server left RUNNING at **k=4** (for continued probing); shipped/usable
  config remains k=6 — reboot to k=6 if the cluster is needed for real use.

## Prior state (2026-07-11, session 5 — sustained-rate curve fill)

**Headline: NO top-k sustains ≥10 tok/s on the current node state.** Session 1–4
burst numbers overestimated; the curve was re-measured with a sustained protocol
(1024-tok runs + 512→1024 differentials) after ≥1200-tok settle:

| top-k | burst diff | sustained | quality (probes / battery) |
|-------|-----------|-----------|----------------------------|
| 8 | 4.2 | — | coherent (prior sessions) |
| 6 | 5.9 (prior) | — | minor damage (prior) |
| 5 | 6.3–7.0 | 5.5–7.5 (climbing) | battery CLEAN; fact probe rambles |
| 4 | 9.9–12.4 | **7.7–7.9 (flat, 4 runs)** | battery CLEAN; fact probe rambles |
| 3 | 8.3–8.8 | 7.4–11.4 (unstable) | **broken**: math garbled, fact hallucinated |

Key findings this session:

- **Sustained k=4 = 7.8 tok/s, not 10+.** Physically consistent: 7.8/4.2 = 1.86×
  vs bytes-ratio prediction (8+1shared)/(4+1shared) = 1.8×. The old 9.08/16.5
  bursts were 384-tok differential artifacts. **Target ≥10 is NOT met by k=4.**
- **Sustained rate is prompt-domain-dependent.** The resident-expert tier warms
  per domain: k=3 hit 11.44 tok/s late in a long-exposed essay prompt, then fell
  to 7.44 on a fresh prompt domain, recovering to 8.78 on its repeat. ±30%
  spread within minutes on one config. Settle protocol must match the eval
  domain, and "settled" is never global.
- **`mtp_correctness_battery.py` is too shallow as the quality gate**: verdict
  CLEAN at k=5, k=4, AND k=3, while free-form probes show clear degradation at
  k≤5 and outright breakage at k=3 (e.g. "24×17 = 408 (4 carry 11...) 3411";
  Canberra answer wrapped in hallucinated population/flower facts). Need a
  stronger battery (exact-answer scoring on longer generations) before any
  finetune go/no-go.
- **Quality verdicts are also settle-sensitive, not just speed.** At k=6,
  temperature=0, the battery went CORRUPT (repetition loop, right after boot)
  → CLEAN (after ~1200-tok settle) on identical prompts/config. Outputs at t=0
  differ run-to-run (near-tie logit flips cascading over long generations), so
  a single battery run — especially cold — is meaningless. Run quality evals
  only post-settle, ideally ×2.
- **Node `.1` at 5+ days uptime remains the confound** — bandwidth variance
  blurs adjacent-k differences (k=5 late-run 7.48 ≈ k=4's 7.72, which bytes
  physics says shouldn't happen). **Reboot `.1` before trusting any absolute
  number**, then re-baseline k=8 and k=4. If a fresh k=8 baseline recovers to
  ~5.2 (best seen in session 4), predicted fresh k=4 ≈ 9.4 — still short of 10;
  k=3 ≈ 11.7 but quality is unusable naive. **The finetune (next-steps #1) is
  therefore confirmed as the only route to ≥10 tok/s at acceptable quality**,
  with target k=4 (k=3 likely too damaged even as a finetune target, and k=4
  only clears 10 post-reboot if the fresh baseline holds — verify first).
- Server left at **k=6** (usable shipped config) after the session.

## Prior state (2026-07-11, sessions 1–4)

- **TP2 serving works and is stable.** FULL cudagraphs across both nodes, fixed
  by NCCL 2.30.7 (see `01-*.md` for that whole saga — `VLLM_NCCL_SO_PATH` baked
  into `start-ray-cluster.sh` COMMON; the hang is gone). Settled decode:
  **~4.0 tok/s at the model's native top-k=8**, coherent output.
- **The speed lever is identified and proven: reduce routed top-k.** Clean curve
  (settle-first protocol, Firefox closed, 2026-07-11):

  | top-k | tok/s | ms/tok | vs k8 | quality |
  |-------|-------|--------|-------|---------|
  | 8 | 4.0 | 242 | — | all probes coherent |
  | 6 | 5.9 | 178 | +48% | code/reason OK; 24×17 arithmetic breaks |
  | 4 | ~9–16 | 110 | +130–300% | COLLAPSE: hallucinations, broken facts |

  Set via `--hf-overrides '{"num_experts_per_tok": N}'`. **k=4 hits the target
  but naive quality is unusable; k=6 is a near-free +48% with minor quality
  cost.** The wall is quality, not speed.
- **Serve scripts:** `spark/serve-glm52-tp2.sh` (TP2, add `--hf-overrides` for
  top-k), `spark/serve-glm52-tp2-mtp.sh` (TP2+MTP), `spark/serve-glm52-pp2.sh`
  (stable PP2 fallback, 4.9–5.5 single / 13–15 aggregate). TP2 planes on both
  nodes at `~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2` (2-bit, rank0 .1/rank1 .2).
- **Current running server:** k=6 TP2 (left by session 5 as the usable config).

## Attempts log (what was tried across sessions 1–4, ranked outcomes)

1. **TP2 FULL cudagraph hang** — root cause NCCL 2.28.9 graph-replay deadlock on
   GB10+CX7; **FIXED by NCCL 2.30.7**. Ruled out: NCCL_GRAPH_MIXING_SUPPORT=1,
   splitting_ops (can't pull collectives out of a full decode graph). Detail in
   `01-*.md`.
2. **MTP / speculative decode (k=1)** — **net LOSS −17%** (4.30 vs 5.19) despite
   93.5% acceptance. Bandwidth-bound verify reads ~2× expert bytes for ~1.9×
   tokens. Workload sweep: utility <1 on prose/code, approaches ~1 only on
   json/repetition. Code is WORST (high acceptance but high expert diversity →
   high read cost). Cascade (arXiv 2506.20675) would only damage-limit, not
   reach target. **Parked.**
3. **pool-REAP (e.g. 0xSero/GLM-5.2-REAP-504B, 168/256 experts)** — pruning the
   POOL keeps top-8, so it does **not** cut per-token bytes → no direct speedup.
   Only value here = raise expert overlap to make MTP marginally profitable.
   GGUF is llama.cpp (incompatible with our 2-bit plane path); NVFP4 safetensors
   base would need dequant→re-prepack. **Not the speed lever.**
4. **NVFP4 quantization** — experts are ALREADY served at **2-bit** planes
   (below NVFP4's 4-bit); NVFP4 would ~2× the bytes AND not fit RAM (~207 GB/
   rank > 128). Wrong direction for speed. **Rejected.**
5. **RTX 5090 for spec decode** — 1.79 TB/s but only 32 GB; can't hold the 194 GB
   target model, so can't run the verify (the expensive half). Drafter isn't the
   bottleneck. **Rejected.**
6. **top-k reduction** — **WORKS** (see State curve). The path to the target.
   Blocked only by quality at low k. **← ACTIVE THREAD.**

## Established facts (with evidence)

- **Bottleneck is memory bandwidth, not compute/launch/network.** Decode is
  ~193 ms/token at k=8 reading ~4.5 GB/rank/token of 2-bit expert planes from
  unified LPDDR5X (~273 GB/s). GPU 96% "util" but only ~17–20 W and 0%
  mem-controller util during decode = stalled on ATS, not computing.
- **GB10 = unified LPDDR5X, no GDDR/HBM.** nvidia-smi reports `[N/A]` GPU mem
  (shared 127 GB). This is why residency can't rescue MTP and why bytes/token is
  everything.
- **Experts are served at 2-bit** (custom moe_w2 planes, `N*K/4` bytes; 97 GB/
  rank; prepack from the FP8 checkpoint). Per-token read = top_k × expert_bytes
  × layers, LINEAR in top_k → why top-k reduction scales speed.
- **Top-k is cleanly reducible:** `num_experts_per_tok` via `--hf-overrides`;
  kernel reads `top_k = topk_ids.shape[1]` dynamically (`moe_w2_cubit.py:1121`,
  no k=8 hardcode); router renormalizes (`norm_topk_prob=True`); GLM has 1
  always-on shared expert (quality floor, holds to ~k=6). n_group=1 so no
  grouping constraint. Config: 256 experts, top-8, sigmoid/noaux_tc routing,
  routed_scaling_factor=2.5, 78 layers (first 3 dense).
- **Measurement settling effect (cost us a session):** `warm-planes.sh` only
  fills the file cache; the moe_w2 resident-expert tier (`mark_seen`/
  `ensure_resident`) is populated by INFERENCE. Cold tier ≈2.6, settled ≈4.2
  tok/s — proven by two identical back-to-back k=8 runs (2.63 then 4.23).
  Desktop Firefox on `.1` steals shared LPDDR5X bandwidth (−40%).

## Next steps (ranked, updated session 6)

0. ~~Reboot `.1`, re-baseline k=8 and k=4 sustained~~ **DONE session 6:
   k=8 = 3.1 (low, see State), k=4 = 8.5–9.8.** Open sub-question: why is
   k4/k8 ≈ 2.9× (superlinear vs 1.8× bytes)? Residency hypothesis — if
   real, a smaller expert working set (REAP-pruned pool, or pinned resident
   planes) could compound with top-k reduction.
0b. **Build a real quality battery** (exact-answer scoring, longer generations,
   multi-domain) — current battery passes k=3, which is visibly broken.
1. **Low-k recovery finetune (THE path to the target), target k=4.**
   **STARTED (session 5): the REAP→GLM port is DONE** — `~/Dev/reap` branch
   `add-glm_moe_dsa-support` (commit ec1ad70): MODEL_ATTRS + observer registry
   entries for GlmMoeDsaForCausalLM, bias-aware top-k selection in the fused
   observer path (sigmoid+e_score_correction_bias, matches real routing; also
   improves hy_v3), tiny-model smoke tests pass (and hy_v3 smokes still pass).
   Run tests with `PYTHONPATH=src ~/venvs/vllm-moet/bin/python3.12
   tests/test_glm_moe_dsa_*_smoke.py` (needs transformers ≥5.x for
   glm_moe_dsa; reap's pinned 4.55 venv does NOT have it).
   Next on this track: (a) layer-wise streaming calibration of GLM-5.2-FP8
   (194 GB > 128 GB RAM — use the existing per-layer disk streaming + the
   data-parallel observer merge to split the calib set across both Sparks);
   (b) run the observer at BOTH k=8 and k=4 (`num_experts_per_tok` override at
   load) — the k=4-selected set vs k=8-selected set tells you which experts
   lose traffic and where routing damage concentrates; (c) then design the
   recovery step (KD from k=8 teacher at fixed k=4 routing, saliency-weighted). Adapt GLM-5.2 to route
   well at top-k=4 (or 5) so quality survives at ≥10 tok/s. Naive drop+renorm is
   insufficient below k≈6. Approach: short distillation/finetune with the target
   k fixed (KD from the k=8 teacher, or continued pretraining on a calibration
   set with k=4 routing). Your REAP saliency machinery (`~/Dev/reap`) measures
   which experts matter — reuse it to inform/regularize. NOTE: reap has hy_v3
   support, NOT GlmMoeDsa — needs a GLM port (like the hy3 one).
2. ~~Map k=5 and k=3 + battery~~ **DONE session 5** (see State table).
3. ~~Re-measure k=4 steady-state~~ **DONE session 5: 7.8 tok/s sustained.**
4. ~~Confirm settled rates under sustained load~~ **DONE — sustained < burst;
   rates are prompt-domain-dependent (see findings).**
5. If quality can't be recovered at k=4: settle for **k=6 (~5.9 tok/s, usable)**
   as the shipped config and treat 10 tok/s as finetune-gated.

## Resume mechanics

- **Bring up cluster:** `bash spark/start-ray-cluster.sh` (carries NCCL 2.30.7
  via VLLM_NCCL_SO_PATH + GIDs + graph-mixing flag; raylet MOE env must be
  EMPTY — verify with
  `tr '\0' '\n' < /proc/$(pgrep -f 'raylet '|head -1)/environ | grep MOE`).
- **Serve at top-k N:** `bash spark/serve-glm52-tp2.sh --hf-overrides '{"num_experts_per_tok": N}'`.
- **MEASUREMENT PROTOCOL (do not skip — see settling fact):** after health,
  `bash spark/warm-planes.sh ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2 --peer`,
  THEN run ~600 tok of diverse generation to settle the resident tier, THEN
  measure decode via usage-token differential (32 vs 384, non-stream, rate =
  Δtokens/Δwall). Reusable script:
  `/tmp/.../scratchpad/measure_topk2.sh <k>` (or re-author; it settles then
  measures ×2 + 4 quality probes). Close Firefox on `.1` first.
- **Sanity that speed isn't a mirage:** during decode check `power.draw`
  (>~30 W = working; ~17–20 W = memory-stalled) — nvidia-smi util alone lies on
  unified memory.
- **Correctness gate:** `spark/mtp_correctness_battery.py --model glm-5.2`.
- **Node `.1` was up 5 days with recurring memory-bandwidth degradation** (k=8
  measured 2.6–5.2 across the session); a fresh reboot of `.1` + closing the
  desktop session gives the most stable bandwidth. sudo needs the user's TTY
  (`!` prefix).
- **Venv-vs-repo drift still unfolded** (fold into `spark/*.patch` when
  convenient): fp8.py `_moe_w2_active` guard, moe_w2_cubit geometry+tp_rank
  validation. NCCL_GRAPH_MIXING_SUPPORT=1 in start-ray-cluster.sh is harmless
  (ruled out, droppable).
- **Footguns:** `pkill -f <pat>` kills your own shell; `pkill -x firefox` is
  safe. Serve boots purge page cache. Each top-k value needs a full reboot
  (config set at load).
