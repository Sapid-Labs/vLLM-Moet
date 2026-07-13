# 02 — GLM-5.2 single-stream decode → 10 tok/s: top-k reduction + recovery finetune

## Target

GLM-5.2 single-stream decode **≥ 10 tok/s** on the two DGX Sparks (TP2), at
**acceptable quality**. Speed alone is solved (top-k=4 crosses 10); the open
problem is recovering quality at low top-k. Ends when a config sustains
≥10 tok/s and passes `spark/mtp_correctness_battery.py --model glm-5.2`.

## State (2026-07-13, session 8 — **MTP RETRY on pruned planes: 15.0 tok/s sustained, battery CLEAN** — spec decode verdict REVERSED, now +31%)

Retried MTP speculative decode (k=1, GLM's native next-n drafter) on top of the
session-7b pruned-resident config. The session-1–4 verdict (−17%, "parked") was
measured on unpruned 97 GB planes where verify's extra expert reads amplified
page-cache thrash. On fully-resident pruned planes the economics flip:

- **Sustained decode 15.09 / 15.02 / 15.01 tok/s** (512→1024 differential,
  essay domain, post-settle — same protocol as the 11.44 baseline), **fault-free**
  (1–36 MiB NVMe per pair once settled). +31% over non-MTP 11.44.
- **Acceptance 67.6%** (1885/2789 drafts, temp 0.7 essay). Ideal speedup at
  that acceptance is 1.68×; measured 1.31× — the gap is verify's extra
  expert-plane reads (~28% per-step cost), affordable now that reads come from
  page cache instead of NVMe.
- **Battery VERDICT: CLEAN** (arithmetic ×2, fact, prose ×2 all PASS) +
  free-form probes clean (24×17=408, 391/17=23, 1001=7·11·13, Rayleigh prose).
- **Serve (new shipped config):**
  `VLLM_MOE_W2_PREPACKED_DIR=$HOME/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208
  MTP_K=1 bash spark/serve-glm52-tp2-mtp.sh` — the MTP script now respects a
  pre-set PREPACKED_DIR (same one-line fix as the plain script). Pruned planes
  include layer 78 (the MTP drafter's MoE) UNPRUNED (keep=256), so the drafter
  routes over the full pool; only the 75 main MoE layers are 256→208.
- Untested upside: `MTP_K=2` (drafter reused autoregressively). At 67.6%
  pos-0 acceptance the marginal token is ~0.4/step against a 3rd position of
  expert reads — plausibly small gain, needs a reboot to test.
- Server left RUNNING in this config (MTP k=1, pruned planes, native k=8, mmap).

## Prior state (2026-07-13, session 7b — **TARGET MET: 11.44 tok/s sustained at NATIVE k=8, battery CLEAN**, via frequency-pruned 208-expert planes)

The residency smoke test didn't just validate the hypothesis — it hit the
full goal. Config: pool-pruned planes (coldest-48-per-layer by measured k=8
traffic, `spark/routing/keep208.json`), native k=8 routing, TP2 FULL
cudagraphs.

- **Sustained decode 11.44 / 11.45 tok/s** (512→1024 differential ×2, essay
  domain, post-settle — the exact protocol that gave k=8=3.1 and k=4=7.8-9.8
  before). Settle runs dead-flat (400 tok in 35.3 s ×3). **1024-tok
  generations no longer thrash**: NVMe reads 0–9 MiB per differential pair
  (was 1–2 GiB). Planes 79 GB/rank < ~88 GB cache ⇒ fully resident, as
  predicted.
- **Battery VERDICT: CLEAN** (arithmetic/fact/prose all PASS) + free-form
  probes clean (24×17=408, 391/17=23, 1001=7·11·13, Rayleigh prose). This is
  NATIVE k=8 routing minus 5.9% slot traffic — categorically gentler than
  the k=4 collapse. (Battery is still the shallow one; deeper eval optional.)
- **How pruning works at runtime** (`spark/pool-prune-runtime.patch`, applied
  to BOTH venvs): planes row-compacted to keep-list order
  (`spark/routing/prune_planes.py` — row-select from existing planes, no
  requant, ~15 min); loader reads meta "keep", masks
  `e_score_correction_bias` of pruned experts by −1e9 (selection-only, gate
  weights renormalize over kept — REAP-style pool prune), remaps ids→compact
  rows via LUT gather in `_moe_w2_forward_timed` (cudagraph-safe). Verified:
  0 routed ids outside keep set on live capture.
- **Serve:** `VLLM_MOE_W2_PREPACKED_DIR=$HOME/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208
  bash spark/serve-glm52-tp2.sh --enable-return-routed-experts` (script now
  respects a pre-set PREPACKED_DIR). Planes dir exists on both nodes; boot
  logs "POOL-PRUNED 256->208" ×75 on both ranks.
- **Caveat:** expert selection is frequency-based (smoke-test quality). REAP
  saliency selection (next steps) may pick a better set; current quality
  already gates CLEAN, so that's now an upside option, not a blocker.
- **THP tier tried (session 7b) — NO GAIN: 11.47/11.43 tok/s, identical to
  mmap.** `VLLM_MOE_W2_PLANES_THP=1` boots fine post-prune (110/121 GB used)
  but only ~half the planes get 2 MiB backing (AnonHugePages 38/49 GB of
  79 — fragmentation under pressure), and decode rate doesn't move. At
  87 ms/tok the effective plane-read rate is ~52 GB/s in BOTH configs, so
  fault-free decode is no longer bound by the 4 KiB-vs-2 MiB ATS penalty
  (contradicts the extrapolated "170 GB/s ⇒ well past 10" projection —
  something shared binds first). **Shipped config = mmap** (same speed, far
  more headroom; page cache degrades gracefully instead of OOMing).
- Server left RUNNING on pruned planes (mmap config), native k=8, capture on.

## Prior state (2026-07-13, session 7 — k=8 traffic CDF MEASURED: routing is much flatter than assumed; prune ratio picked at 48/layer)

Ran session-6b next-step #1: rebooted serve at NATIVE k=8 with
`--enable-return-routed-experts` (flag verified end-to-end; capturer hooks
`router.set_capture_fn`, works under FULL cudagraphs + moe_w2 — confirmed
non-null on a live request). Captured routing for **5,293 tokens across 12
domains** (essay/code×2/math/factual/fiction/json/dialogue/translation/
science/legal/recipe, 400 gen tok each, temp 0.7 seed 42). Data + scripts:
`spark/routing/` (capture_routing.py, analyze_routing.py,
`capture-k8-20260713/` incl. per-domain npy + summed counts). Findings:

- **The "90% of tokens through 20% of experts" code comment is WRONG for
  real traffic.** Measured CDF (per-(layer,expert) slot counts, 75 MoE
  layers × 256 experts): 50% of traffic needs **27.4%** of cells, 90% needs
  **72.6%**, 99% needs 93.5%. Only 5/19200 cells never routed. Routing is
  FLAT, not skewed.
- **Prune sweep (coldest-N-per-layer):** N=32 → 84.9 GB/rank, loses 3.2% of
  routed slots; **N=48 → 78.8 GB/rank (residency target), loses 5.9%**
  (worst layer 12.0%); N=56 → 75.8 GB, 7.4% (worst 14.3%). So full
  residency costs ~6% of routed traffic exposure — NOT negligible, but well
  inside what REAP papers/community prunes survive (0xSero pruned 34%), and
  slot counts OVERSTATE damage: they weight the 8th-gate slot same as the
  1st, and pruned tokens renormalize onto kept experts rather than vanish.
- **Cold set is domain-stable:** every domain sends 5.9–9.3% of its traffic
  to the global-cold-56 set (no domain catastrophically depends on it).
  Frequency-only pruning is therefore *plausible* as a residency smoke test,
  but the 6-7% exposure says use REAP saliency for the real selection.
- **Native k=8 quality re-confirmed clean** during capture: 24×17=408,
  391/17=23, 1001=7×11×13 all correct, prose coherent (the exact probes k≤4
  fails).
- **Routing-array format gotcha:** shape is (tokens-1, **78**, 8) — the
  first 3 rows are the DENSE layers, all-zero placeholders. Strip
  `[:, 3:, :]` before counting or expert 0 gets 8×tokens phantom hits.
- **Decision: target prune = 48/layer (keep 208), planes → ~78.8 GB/rank**;
  post-prune THP budget 78.8+32 ≈ 111 < 121 GB also fits. Next: REAP
  saliency calibration to choose WHICH 48 (per layer), not hit counts.
- Server left RUNNING at native k=8 with routing capture on.

## Prior state (2026-07-12, session 6b — RESIDENCY ANOMALY SOLVED: the bottleneck is page-cache thrash, not LPDDR bandwidth)

Chased why k4/k8 ≈ 2.9× (superlinear vs the 1.8× bytes prediction). Answer, with
direct evidence:

- **Planes (97 GB/rank) don't fit in page cache (~88 GB available)** → decode
  continuously faults cold experts from NVMe, and **tok/s tracks fault volume
  almost linearly**. Measured at k=4, one prompt repeated 6× (256 tok each):
  disk 2011→1331→1186→1074→695→**19 MiB** as rate went 7.25→…→**11.50 tok/s
  (incl prefill)**. Fresh prompt domain: ~14 MiB/token faulted; "settled"
  domains get re-evicted by any excursion to another domain. This one
  mechanism explains the settling effect, the ±30% prompt-domain spread, the
  k=8 uptime "degradation", AND the superlinear k=4 ratio (k=4's per-domain
  working set mostly fits; k=8's cannot).
- **Fault-free k=4 clears the target** (11.5 tok/s incl prefill on 256-tok
  runs) — but only in a bounded domain: 512/1024-tok generations expand the
  union working set past cache again (6.6 GiB faulted, back to 7.5 sustained).
  The residency deficit is only ~5–15 GB.
- **Why ATS reads are slow even when cached** (from moe_w2_cubit.py comments,
  measured by the author): file-backed 4 KiB pages make GPU ATS reads
  **10–25× slower** than raw (`decode moe_w2_mm 7.4 ms median vs sub-ms`);
  2 MiB THP pages restore ~170 GB/s (`VLLM_MOE_W2_PLANES_THP=1`) but require
  full anon residency — currently impossible (97 planes + 32 other > 121).
  The old "26 ms warm reads" estimate was the THP number; 193 ms/tok reality
  is 4 KiB pages + faults.
- **STRATEGY SHIFT — pool-REAP is back, and it may beat the k=4 finetune.**
  Session-5 verdict #3 rejected pool pruning because it doesn't cut per-token
  bytes. But the bottleneck is *residency*, not bytes: pruning ~20–25% of the
  coldest experts (256→~200) shrinks planes 97→~78 GB/rank → fully
  cache-resident → fault-free at any generation length and prompt domain, at
  NATIVE k=8 routing (tiny quality cost vs the k=4 collapse). And post-prune,
  THP anon residency fits (78+32≈110 < 121) → ~170 GB/s reads → k=8 projected
  well past 10 tok/s. Code comment corroborates skew: "GLM routes ~90% of
  tokens through ~20% of experts".
- Also catalogued (moe_w2_delta.py): a GPU **base cache** exists
  (`VLLM_MOE_W2_BASE_CACHE_GB`, pinned-host planes + GPU slot pool +
  miss-replay) but is unusable at current plane size (pinned 97 GB + pool
  doesn't fit); delta tier disabled (`VLLM_MOE_W2_DELTA_GB=0`). Routing can be
  captured per-token via `--enable-return-routed-experts` (returns base64 npy
  `(tokens-1, layers, top_k)` per completion) — the tool for measuring
  cold-expert traffic share before choosing a prune ratio.
- Server left RUNNING at k=4.

## Prior state (2026-07-12, session 6 — post-reboot re-baseline; k=4 = 8.5–9.8 sustained)

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

## Next steps (ranked, updated session 8 — 15 tok/s shipped; all items are upside)

-1. **MTP_K=2 sweep** (cheap, one reboot): measure acceptance at pos 1 and
   whether the marginal ~0.4 tok/step beats the 3rd position's read cost.
   Also re-measure acceptance per domain — 67.6% was essay/temp-0.7 only.

## Old next steps (session 7b)

0. ~~THP tier attempt~~ **DONE session 7b: NO GAIN (11.47 = 11.44), mmap
   stays the shipped config** (see State). If more speed is ever needed,
   first profile what binds fault-free decode at 87 ms/tok (effective plane
   read ~52 GB/s « 273 raw) — page size isn't it.
1. **REAP saliency calibration of GLM-5.2 to pick the 48-per-layer prune set.**
   `~/Dev/reap` branch `add-glm_moe_dsa-support` (ec1ad70) is ported and
   smoke-tested. Needs: layer-wise disk streaming (194 GB model > 128 GB RAM),
   calib set split across both Sparks, additive observer-stat merge. Compare
   the REAP-picked set against the frequency-cold set
   (`spark/routing/capture-k8-20260713/counts_layer_expert.npy`) as a sanity
   check — large disagreement means slot counts were misleading, small means
   either works.
   - Optional fast pre-check: frequency-only prune of the cold-48 as a pure
     residency smoke test (re-prepack, verify zero NVMe faults + speed at
     k=8) before investing in calibration. Quality of that artifact is NOT
     trustworthy; it only tests the residency→speed hypothesis.
2. **After pruning lands:** re-prepack planes (`spark/prepack_planes.py`
   variant that drops pruned experts + remaps router indices), verify full
   residency (zero `nvme0n1` sectors during decode — the smoking-gun metric),
   then `VLLM_MOE_W2_PLANES_THP=1` for the ~170 GB/s tier.
3. Keep the k=4 recovery finetune parked unless pruned-k=8 misses 10 tok/s.

## Old next steps (session 6b)

~~NEW #1: Measure cold-expert traffic share at k=8~~ **DONE session 7** (see
State: CDF flat, prune=48/layer picked, 5.9% slot-traffic exposure).
NEW #2 (now #2 above): after pruning lands: re-prepack planes, verify full
residency (zero `nvme0n1` sectors during decode), then
`VLLM_MOE_W2_PLANES_THP=1` for the 170 GB/s tier.

## Old next steps (session 6)

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
