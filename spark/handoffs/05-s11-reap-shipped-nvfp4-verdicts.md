# s11 (2026-07-14) — REAP shipped + NVFP4 verdicts; MTP drafter measured near-optimal

> Handoff journey file (split out of the old stacked HANDOFF.md on 2026-07-17).
> Previous: 02-*.md (how we got to 15) · Next: 06-s12-*.md

## STATUS (2026-07-14, session 11)

**REAP (saliency expert-prune) shipped + coherence-validated for GLM-5.2.** Also
uncovered two serving-quality problems in the shipped speed stack (below).

- **Server UP now:** **big-3 NVFP4 + REAP planes + MTP** (`MODEL=$M/nvfp4_big3_overlay`,
  `VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj`, `moe_w2_planes_tp2_p208_reap`,
  MTP on), TP2, healthy on :8000. Verified **fully coherent + correct** on chat:
  320-tok transformer explanation clean at greedy, "all but 9 sheep" →9, GSM
  Natalia. **REAP does not break the model, on the config you actually ship.**
- **✅ RESOLVED — the "NVFP4 incoherent" scare was the FULL cut, not big-3.**
  The **full** `nvfp4_dense_overlay` cut (all 10 targets incl. dense MLP
  `gate/up/down_proj`) degrades on sustained generation — coherent for a while
  then token-repetition / "</think>" loops (greedy fast, sampled ~250 tok then
  collapses). Reproduces on frequency AND REAP planes, nodes byte-identical
  (not a desync) → it's the aggressive quant (rel-L1 ~0.09), not the prune, not
  MTP. The **big-3 cut** (`o_proj,q_b_proj,kv_b_proj`, ~18 tok/s, the config
  actually demoed) is fully coherent. **Do not ship the full cut** without a
  real quant fix + eval; big-3 is the safe NVFP4 config. FP8 (no NVFP4) is also
  clean.
- **MTP note:** the "Paris1 and1..." garbage-draft interleave appeared only on
  the FULL-cut degraded target; on big-3 MTP is clean. So MTP is fine — its
  earlier garbage was downstream of the full-cut target collapse, not an MTP bug.
- **HF model updated (session 11):** `sapidlabs/GLM-5.2-NVFP4-attn-experimental`
  now ships the **big-3 stable** cut (4 shards `nvfp4-dense-000{0..3}`; deleted
  the 6 full-cut shards; corrected card: title, `VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj`,
  coherence-validated + task-battery-pending). Commit `eb2ae11`.
- **MTP drafter — MEASURED (session 11), verdict: drafter is near-optimal, both
  levers small.** Big-3+REAP build, warmed (planes resident), greedy, same prompt:
  | K | tok/s | tok/verify | acceptance |
  |---|------|-----------|-----------|
  | 1 | 18.5 | 1.85 | 84.6% (sampled 83.8%) |
  | 2 | 19.0 | 2.52 | pos0 85.3% / pos1 66.4% |
  - **Depth (K≥2) is a DEAD END here (~1.03×):** K=2 drafts 1.36× more tok/verify
    but the extra MTP-head forward — a FULL decoder layer (layer 78: eh_proj +
    enorm/hnorm + full attn + 256-expert MoE + shared_head, ~9.7B params, run
    serially/autoregressively with its own TP2 all-reduce) — eats the gain.
  - **Accuracy fine-tune (K=1) ≈ +4% only:** 84%→~92% accept caps at 1.92/1.85.
    Not worth training a 9.7B MoE layer for. (MTP already gives ~1.62× over the
    ~11.4 non-MTP floor; the ~12% gap to the 1.84× ceiling is MTP-forward overhead,
    a serving/kernel cost, not a drafter-accuracy cost.)
  - **Recommendation: do NOT build the drafter trainer** — ROI too low.
  - **TP-free draft TESTED — negative result.** `draft_tensor_parallel_size=1`
    (added `DRAFT_TP` env knob to `serve-glm52-tp2-mtp.sh`; it's a supported vLLM
    config, no surgery) boots + is coherent + same 84.6% acceptance, but runs
    **~17.6 tok/s (~5% SLOWER** than TP2-draft's 18.5). So the ~12 ms/draft
    overhead is NOT the TP all-reduce — splitting the draft across both GPUs
    (TP2) beats running it on one + syncing. The overhead is launch/orchestration
    (Python spec-decode loop, sample/accept/reject, KV bookkeeping), which neither
    TP-free nor expert-count touches (MoE reads top-k regardless of pool). **The
    cheap "cheaper-forward" levers are exhausted.** Reducing MTP overhead further
    = deep vLLM spec-decode work (cudagraph the whole draft+verify loop), low ROI.
  - **Net: MTP at K=1 (18.5 tok/s, 84% accept) is near-optimal for this setup.**
    Bigger throughput levers live elsewhere (NVFP4 breadth capped at big-3 by
    quality; the non-MTP floor). Shipped/best MTP config = plain K=1 (no DRAFT_TP).
- **Code pushed:** `Sapid-Labs/vLLM-Moet` `spark-gb10`. REAP tooling:
  `~/Dev/reap` `add-glm_moe_dsa-support` (commits `02b838a`,`7f9f567`,`fd2b7f4`).
- **NEXT:** (a) REAP-vs-frequency quality A/B on the clean FP8 stack (GSM8K-50) —
  does REAP actually buy back quality; (b) root-cause NVFP4 collapse + MTP garbage
  (both block the 20 tok/s config); then drafter → full battery.

### (prev) STATUS session 10 — NVFP4 ~20 tok/s
Full NVFP4 attn+shared reached ~20 tok/s greedy (throughput only; see the NVFP4
quality caveat above). `sapidlabs/GLM-5.2-NVFP4-attn-experimental` pushed.

## NEXT (agreed plan — order matters)

**REAP → drafter → eval**, with cheap sanity checks between (full battery only at
the end; it's slow). Rationale: the drafter is *fit to* the target's output
distribution, and REAP changes which experts are active → do all target-changing
steps first, fit the drafter last against the frozen model.

1. **REAP (do first)** — replace the frequency-based expert prune (drop 48
   coldest by traffic) with saliency-based REAP at the same 208 experts, to buy
   back quality. Quality move, NOT speed (same bytes/token).
   - **Pipeline (confirmed session 11):** REAP observer → per-layer expert
     saliency → top-208-per-layer keep-list in `spark/routing/keep208.json`
     format → `spark/routing/prune_planes.py <src> <dst> <keep.json>` (row-select,
     no requant, ~15 min) on the FULL `moe_w2_planes_tp2` (still on disk, 97 GB)
     → serve with the new p208 dir. `spark/prepack_planes.py` packs all experts;
     it is `prune_planes.py` that applies the keep-list. Run prune_planes on BOTH
     nodes' tp2 planes (same keep.json; rank-agnostic row-select).
   - **CODE DONE (session 11, reap `add-glm_moe_dsa-support` @ 02b838a):** GLM-5.2
     REAP support is complete. Observer/prune registries were already there; the
     missing piece was the **disk-streaming FP8 path** — the 357B model is >> RAM
     so calibration must stream layer-by-layer from disk, but `disk_stream` had no
     GLM converter and no FP8 dequant (it cast raw float8 bytes, ignoring
     `weight_scale_inv`). Added block-wise FP8 dequant + `glm_converter` (fuses
     per-expert gate/up/down into native batched params); 4 GLM smoke tests pass
     (incl. a synthetic real-layout FP8 checkpoint matched to bf16 ref, rel-L1
     <5%); hy3 tests still green. Env: run reap with **`~/venvs/hf/bin/python`
     `PYTHONPATH=src`** (reap not pip-installed; that venv has pytest+transformers
     w/ GlmMoeDsa).
   - **VALIDATED ON REAL MODEL (session 11):** the streaming observer runs
     end-to-end on GLM-5.2-FP8 (all 78 blocks: data→FP8 dequant→DSA forward→
     saliency). Server was torn down (both nodes, GPUs freed 0.0/2.0). Reap deps
     (`accelerate datasets scikit-learn matplotlib seaborn`) installed into the
     **CUDA `vllm-moet` venv** on BOTH nodes (the `hf` venv is CPU-only torch —
     do NOT use it for the real run). Peer synced: reap `src/scripts/tests`
     rsync'd to `spark-c84b:~/Dev/reap` + deps installed + imports verified.
     Extra reap fixes this session (commits `02b838a`, `7f9f567`, `fd2b7f4`):
     FP8 dequant+glm_converter; DP `data_shard_index/count`+`save_raw_state`;
     **DSA cross-layer top-k threading** (GLM "shared" attn layers crash the
     isolated block-replay without it). Disk-bound: each MoE block re-reads
     ~19 GB (dequant), so a full pass is multi-hour; DP halves the compute part.
   - **RUN THE OBSERVER (both nodes, DP; from `~/Dev/reap`, env
     `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src`, python `~/venvs/vllm-moet/bin/python`):**
     ```
     python -m reap.layerwise_prune \
       --model_name ~/models/hf/GLM-5.2-FP8 \
       --dataset_name theblackcat102/evol-codealpaca-v1 \
       --run_observer_only true --disk_stream true \
       --batches_per_category <N> --batch_size <B> --model_max_length 2048 \
       --seed 42 --output_file_name reap.pt \
       --data_shard_count 2 --data_shard_index <0 on .1 / 1 on .2>
     ```
     Output: `artifacts/GLM-5.2-FP8/evol-codealpaca-v1/layerwise/reap.shard<i>.raw.pt`
     (RAW state, trackers intact — needed for the merge).
   - **MERGE → KEEP-LIST → PRUNE PLANES:**
     ```
     python scripts/merge_observer_states.py \
       reap.shard0.raw.pt <copied-from-peer>reap.shard1.raw.pt --out reap.merged.pt
     python ~/Dev/vLLM-Moet/spark/routing/reap_keep_list.py \
       reap.merged.pt ~/Dev/vLLM-Moet/spark/routing/keep208_reap.json \
       --keep 208 --compare ~/Dev/vLLM-Moet/spark/routing/keep208.json
     # on BOTH nodes (rank-agnostic row-select, ~15 min):
     python ~/Dev/vLLM-Moet/spark/routing/prune_planes.py \
       ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2 \
       ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208_reap \
       ~/Dev/vLLM-Moet/spark/routing/keep208_reap.json
     ```
     Then serve with `VLLM_MOE_W2_PREPACKED_DIR=...moe_w2_planes_tp2_p208_reap`
     (else identical to the shipped serve cmd). Sanity-check: ~10-20 prompts
     coherence or GSM8K-50.
   - **RUN IN PROGRESS (session 11):** DP calibration launched both nodes with
     `batches_per_category 16 --batch_size 4 --model_max_length 1024 --seed 42`
     (→ 8 batches/node, ~65k tokens total), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
     ~2.2s/it, ~28s/MoE block → ~40-50 min/node. Writes `reap.shard0.raw.pt`
     (this node) / `reap.shard1.raw.pt` (peer).
   - **OOM GOTCHA:** `batch_size 8 --model_max_length 1024` OOMs the GB10 (~119 GB
     peak on the first MoE block: 256 experts dequant'd to bf16 ~19 GB + per-token
     MoE/attn intermediates over 8192 tokens). `bs4 seq1024` peaks ~55 GB — safe.
     If you push tokens, add batches (num_batches only affects time, not peak),
     don't raise batch_size/seq.
   - **DONE (session 11):** DP calibration completed both nodes (8 batches each,
     16 total). Merged → `spark/routing/keep208_reap.json` (top-208/layer;
     ~37 experts/layer differ from the frequency prune; merged coverage mean
     252/256 active). `prune_planes.py` ran on BOTH nodes →
     `moe_w2_planes_tp2_p208_reap` (79 GB, 75×208, present on .1 and .2).
     Observer artifacts: `~/Dev/reap/artifacts/GLM-5.2-FP8/evol-codealpaca-v1/
     layerwise/reap.{merged,shard0.raw,shard1.raw}.pt`.
   - **PEER checkout was stale** (missing `spark/routing/`) — copied
     `prune_planes.py`+`keep208_reap.json` there manually. If re-running on the
     peer, verify `~/Dev/vLLM-Moet/spark/routing/` exists first.
   - **SERVING the REAP build:** same NVFP4 serve cmd but
     `VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208_reap`. Ray must be up
     first (`bash spark/start-ray-cluster.sh`; the serve driver forwards the
     VLLM_MOE_W2_* env to workers — not baked in raylet). Sanity-check: greedy
     coherence on ~5-10 prompts, then GSM8K-50.
2. **Drafter (second, last model change)** — fine-tune the MTP head (layer 78)
   against the frozen NVFP4+REAP target to raise MTP acceptance (the real lever
   for effective throughput; sampled tok/s currently varies ~13-20 with
   acceptance). Bigger lift than REAP. Sanity-check after.
3. **Full quality battery (end)** — GPQA + GSM8K + IFEval + MMLU-Pro on the final
   build. This is the gate for any "same quality" public claim. Run it ALONE
   (never co-run — session-9 crash) and resumable (`--use_cache`), see below.

## HOW TO RESUME

**Serve the shipped NVFP4 build (2× Spark, TP2):**
```bash
cd ~/Dev/vLLM-Moet; M=~/models/hf/GLM-5.2-FP8
MODEL=$M/nvfp4_dense_overlay VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208 \
VLLM_NVFP4_DENSE=1 \
VLLM_NVFP4_TARGETS="o_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,fused_qkv_a_proj,gate_proj,up_proj,gate_up_proj,down_proj" \
MTP_K=1 VLLM_ENGINE_READY_TIMEOUT_S=2400 \
nohup bash spark/serve-glm52-tp2-mtp.sh > ~/serve-nvfp4.log 2>&1 &
```
Big-3-only cut = same but `VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj` +
`MODEL=$M/nvfp4_big3_overlay`. Plain FP8-attn baseline = drop `VLLM_NVFP4_DENSE`
+ `MODEL=$M` + `VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208`.

**Measure decode:** use **greedy (temperature 0)** for a clean number (~20);
sampled (temp>0) varies with MTP acceptance. Settle 3-5 warmups first (Marlin
FP4 kernels warm slowly). One-liner protocol in session-10 transcript / GOAL.md.

**Rebuild an overlay** (e.g. after REAP changes nothing here, but for new quant):
`python spark/prepack_nvfp4_linear.py --targets <basenames> --out <dir>` on BOTH
nodes (deterministic; peer needs the packer copied — it's at `~/prepack_nvfp4_linear.py`).

## GOTCHAS / KEY FACTS (hard-won this session)

- **NVFP4 code lives in site-packages** (`~/venvs/vllm-moet/.../vllm/`), captured
  in `patch/nvfp4-dense.patch` + `spark/nvfp4_dense_hook.py` (Dockerfile applies
  both). A fresh vllm install needs them applied. The PEER node's site-packages
  still has leftover debug prints (`NVFP4HOOKDBG`) — harmless, but re-sync the
  cleaned files if you rebuild: hook, parameter.py, deepseek_v2.py, deepseek_mtp.py.
- **Why the overlay works:** vLLM globs ALL *.safetensors (not just the index),
  so the overlay's NVFP4 tensor AND the symlinked original's fp8 tensor both load.
  Handled by `NVFP4SKIP2` guard (skip fp8→uint8-param) + `NVFP4ORPHAN` guards
  (skip fp8 `weight_scale_inv` with no nvfp4 param) in BOTH main + MTP loaders,
  stacked + non-stacked paths. Don't remove these.
- **GPU release between serves:** `kill -9` on the APIServer ORPHANS the Ray
  EngineCore + RayWorkerProc (one holds ~79 GB), which squat the GPUs → next boot
  fails "Cannot provide a placement group requiring 2.0 GPUs". Kill
  `EngineCore|RayWorkerProc|vllm serve` by PID on BOTH nodes, then `ray status`
  should show `0.0/2.0 GPU`.
- **Ray corrupts across many failed boots** (`ActorHandle ... across Ray
  sessions`) → `bash spark/start-ray-cluster.sh` (resolves RoCE GIDs, restarts
  clean).
- **Fresh NVFP4 compile > 600s** default engine-ready timeout → set
  `VLLM_ENGINE_READY_TIMEOUT_S=2400` (cached after first boot; ~4 min warm).
- **[CORRECTED session 12 — see CORRECTIONS above] GB10 DOES have FP4 tensor cores
  (`cutlass_fp4_supported()`=True, sm_121).** We use weight-only Marlin FP4 (4-bit read, bf16
  compute). Correct for bandwidth-bound decode. Adds a little dequant latency
  (measured 1.2× vs predicted 1.29× for big-3).
- **Never co-run evals** (session-9: IFEval co-ran with GPQA 5.5h → aiohttp
  "Session is closed" crash). Run alone, `--use_cache <dir>` + `--log_samples`
  (resumable), `max_length=8192` matching the server + `max_gen_toks=5000`
  (avoids the 8192-context overflow 400s), `num_concurrent=4`.

## CORRECTNESS GATES

- NVFP4 serve: coherent greedy output (verified — it explains memory-bound
  inference correctly) + greedy decode ≥ ~1.3× the FP8-attn baseline.
- After REAP / drafter: cheap sanity (coherence / GSM8K-50) each; full battery
  within noise of the 15 tok/s baseline (GSM8K ≥ ~90%) at the end.

## LINKS

- `spark/NVFP4-DENSE.md`, `spark/GOAL.md`, `spark/RUNBOOK.md`
- `spark/prepack_nvfp4_linear.py`, `spark/nvfp4_dense_hook.py`, `patch/nvfp4-dense.patch`
- HF: `sapidlabs/GLM-5.2-NVFP4-attn-experimental`,
  `sapidlabs/GLM-5.2-2bit-MoE-planes-pruned208-tp2` (2-bit planes)
- REAP tooling: `~/Dev/reap` (branch `add-glm_moe_dsa-support`)
