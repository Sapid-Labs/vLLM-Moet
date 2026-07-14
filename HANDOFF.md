# HANDOFF — GLM-5.2 on 2× DGX Spark (branch spark-gb10)

Short "resume here" pointer. Deep docs: `spark/GOAL.md` (20 tok/s plan),
`spark/NVFP4-DENSE.md` (the current lever, full build plan), `spark/RUNBOOK.md`
(serve/measure), `spark/handoffs/02-*.md` (how we got to 15 tok/s).

## STATUS (2026-07-13, session 9)

Shipped config = TP2 + pruned-208 2-bit experts + MTP k=1 = **15 tok/s**
(~180 GB total weights, ~90 GB/rank, 2× Spark). New goal: **20 tok/s at same
quality**. Session 9 established decode is **bandwidth-bound** (dense/attention =
74% of decode bytes; MLA attention alone ~60%) and set the lever: **NVFP4 the
dense/attention linears**, experts untouched. Two background jobs running:
GPQA-Diamond eval (~74% done) and a queued decode profiler (fires after GPQA).

## DONE / verified

- **Byte accounting** (from safetensors headers): per MoE-layer decode read =
  MLA attn 174.6 MB (60%) + routed-2bit 75.5 (26%) + shared 37.8 (13%) + gate 3.
  ~23 GB/token → 11.4 tok/s×23 ≈ 262 GB/s ≈ the 273 ceiling. Bandwidth-bound.
  Overturns the old "52 GB/s / 5× headroom" thesis (that counted only plane bytes).
- **NVFP4 recon** complete (`spark/NVFP4-DENSE.md`): weight-only W4A16 method
  `ModelOptNvFp4W4A16LinearMethod` + Marlin kernel already in-tree; extend
  `Fp8Config.get_quant_method` LinearBase branch; experts stay on the 2-bit hook.
- **Packer written + validated:** `spark/prepack_nvfp4_linear.py`. Real GLM-5.2
  attention/shared tensors → NVFP4 rel-L1 ~0.089, byte change 0.56× (matches the
  ~1.47× decode projection). Reads block-FP8, emits loader-matching tensors.
- **Not the donor:** `nvidia/GLM-5.2-NVFP4` quantizes the *experts* and keeps
  attention BF16 — useless as a drop-in; use it only as a BF16 requant *source*.
- GSM8K eval done: 96% flexible / 91% strict.

## NEXT (immediate, in order)

1. Wait for GPQA to finish → watcher reports score. Then the profiler auto-fires;
   report the 67 ms breakdown + **flat-vs-rising M-scaling** (the bandwidth-bound
   confirm; this is the go/no-go gate for NVFP4).
2. If flat/bandwidth-bound: wire the loader hook (extend `Fp8Config` LinearBase
   branch → NVFP4 W4A16 for the target prefixes), prototype in site-packages,
   then fold into `patch/vllm-moet-v0.24.0.patch`.
3. Build overlay: `python spark/prepack_nvfp4_linear.py --out
   $MODEL/nvfp4_dense_overlay`; mirror to peer. Serve from overlay + 2-bit planes.
4. Speed-measure (expect ~1.47× → ~18–22 tok/s), then full quality battery.

## HOW TO RESUME

- Model: `$HOME/models/hf/GLM-5.2-FP8` (dense FP8 + 2-bit planes
  `moe_w2_planes_tp2_p208`). Serve: `spark/serve-glm52-tp2-mtp.sh` (needs NCCL
  2.30.7 via `VLLM_NCCL_SO_PATH`, planes on both nodes, Ray up). See RUNBOOK §4.
- Served vllm = install at `~/venvs/vllm-moet/lib/python3.12/site-packages/vllm`
  (stock 0.24.0 + `patch/vllm-moet-v0.24.0.patch`). Repo has NO vllm/ dir; code
  changes = edit patch (prototype in site-packages first).
- Packer validate: `python spark/prepack_nvfp4_linear.py --verify --limit-layers 10`.

## GOTCHAS / KEY FACTS

- **Do NOT edit site-packages/vllm while the profiler is pending** — it re-imports
  on its eager reboot and restores the shipped config on exit.
- Stay on `Fp8Config`; switching to compressed-tensors config strands the 2-bit
  expert hook (only Fp8MoEMethod/ModelOptNvFp4FusedMoE carry it).
- NVFP4 needs weights serialized on disk (no on-the-fly quant). Overlay dir
  symlinks originals + adds NVFP4 shards + rewrites the index (drop replaced fp8).
- Eval footgun: `--max-model-len 8192` + `max_gen_toks 6000` overflows on long
  GPQA prompts (400s → scored wrong). Current GPQA is a slight floor + greedy, so
  NOT comparable to external temp-1.0 numbers. Rerun with `--max-model-len 12288`
  for a comparable cell.

## CORRECTNESS GATES

- Profiler M-scaling FLAT ⇒ bandwidth-bound ⇒ NVFP4 lever is valid (proceed).
- NVFP4 serve: speed ≥ ~1.4× baseline on the standard protocol (settle, ×2,
  usage-token differentials, power.draw > 30W) AND quality battery within noise
  of the 15 tok/s baseline (GSM8K ≥ ~90%).

## LINKS

- `spark/GOAL.md`, `spark/NVFP4-DENSE.md`, `spark/RUNBOOK.md`,
  `spark/prepack_nvfp4_linear.py`, `spark/prepack_planes.py` (expert planes),
  `spark/profile_decode.sh` (queued profiler).
