# NVFP4 dense/attention quant — implementation plan (20 tok/s lever)

Companion to `spark/GOAL.md`. This is the concrete build plan for the Tier-1
lever: NVFP4 the DENSE/attention linears (74% of decode bytes; MLA attention
alone ~60%), keep the routed experts on the untouched 2-bit plane path.
Projected ~1.47× decode → ~22 tok/s with MTP.

**Gate:** do NOT start executing (serving) until the queued decode profiler
confirms flat M-scaling (bandwidth-bound). Offline packing is safe anytime.
Do NOT edit `site-packages/vllm` while the profiler is pending — it re-imports
on its eager reboot. All work stays in the repo until the profiler is done.

---

## Recon summary (verified 2026-07-13, session 9)

- **Model code:** GLM-5.2 = `GlmMoeDsaForCausalLM` → served by
  `site-packages/vllm/model_executor/models/deepseek_v2.py` (DeepseekV2MLAAttention
  `:878`, DeepseekV2MoE `:246`, shared expert DeepseekV2MLP `:316`, FusedMoE `:326`).
  NOT laguna.py. The whole model receives ONE `vllm_config.quant_config`.
- **Experts → 2-bit:** env-gated hook (`VLLM_MOE_W2=1`) inside `Fp8MoEMethod`
  (`fp8.py:492`), keyed by layer-name regex `moe_w2_cubit.is_w2_layer`. Loads
  prepacked planes from `VLLM_MOE_W2_PREPACKED_DIR`. Orthogonal to linears.
- **Dense linears → FP8:** `Fp8Config.get_quant_method` (`fp8.py:179`) dispatches
  by layer TYPE: LinearBase→`Fp8LinearMethod`, RoutedExperts→`Fp8MoEMethod`,
  Attention→`Fp8KVCacheMethod`. An `ignored_layers`/`is_layer_skipped` prefix
  filter only picks quantized-vs-unquantized — NOT a different quant family.
  **This get_quant_method is the extension point.**
- **Weight-only NVFP4 already in-tree (fork addition):**
  `ModelOptNvFp4W4A16LinearMethod` (`modelopt.py:1243`) + `MarlinNvFp4LinearKernel`
  (`kernels/linear/nvfp4/marlin.py:18`). No activation scales / no input_scale.
  Expects on disk: `weight` uint8 `[No, Ki//2]`, `weight_scale` fp8-e4m3
  `[No, Ki//16]` (group_size 16, input-dim groups), `weight_scale_2` fp32 global
  (`amax/(6*448)=amax/2688`). Requires NVFP4 weights **serialized on disk**
  (`is_checkpoint_nvfp4_serialized`; no on-the-fly quant).
- **Producing NVFP4 without ModelOpt/llm-compressor:** only compressed-tensors
  0.17.0 installed, but it has all primitives (`pack_fp4_to_uint8`,
  `FLOAT_TO_E2M1`, `calculate_qparams`+`FP4_E2M1_DATA`, e4m3 group scales). We use
  a self-contained RTN packer (below) — no new dependency.

**Blockers noted:** single global quant_config (no two-config compose — the chosen
config must implement the mixed dispatch itself); NVFP4 needs offline packing;
the 2-bit expert hook lives only in Fp8MoEMethod/ModelOptNvFp4FusedMoE (so we
must stay on Fp8Config, NOT switch to compressed-tensors config, or the experts'
2-bit path is stranded).

---

## Design decision

**Keep `Fp8Config`. Extend its LinearBase branch to route the NVFP4 prefixes to a
weight-only NVFP4 method. Experts stay byte-for-byte on FP8→2-bit.**

Weights are delivered as an **overlay checkpoint dir** (no 737 GB rewrite):
symlink every original shard, add ~0.2 GB/layer of NVFP4 shards for the targeted
attention/shared tensors, rewrite `model.safetensors.index.json` to point those
tensors at the NVFP4 shards (dropping their fp8 `weight`/`weight_scale_inv`).
vLLM's linear loader TP-shards the FULL NVFP4 tensors on load — group-16 along the
input dim lands on clean byte+group boundaries for both column-parallel
(`q_b`,`kv_b`,`gate`,`up`) and row-parallel (`o_proj`,`down`) splits (e.g. o_proj
Ki=16384 → /2 ranks = 8192, a multiple of 32). So **no per-rank presharding**
(unlike the expert planes).

Targets (`TARGET_SUFFIXES` in the packer): `self_attn.{q_a,q_b,kv_a,kv_b,o}_proj`
+ `mlp.shared_experts.{gate,up,down}_proj`. Left alone: `indexer.*` (tiny),
`*_layernorm`, router gate, embeddings, lm_head, MTP layer.

---

## STATUS 2026-07-14 (session 10): NVFP4 big-3 WORKS (eager), full-graph compiling

**NVFP4 weight-only big-3 (`o_proj`,`q_b`,`kv_b`) attention runs and generates
COHERENT output** on 2× Spark (eager). Confirmed correct: memory-bandwidth essay
was accurate. Eager decode 6.3 tok/s (NOT comparable to the 15 tok/s FULL-graph
baseline — eager is slower for everyone). Full-graph boot recompiling now
(`VLLM_ENGINE_READY_TIMEOUT_S=2400`; fresh NVFP4 compile > default 600s timeout)
for the comparable number (~1.29× → ~19 tok/s expected).

**Serve (working):**
```
MODEL=$M/nvfp4_big3_overlay VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208 \
VLLM_NVFP4_DENSE=1 VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj MTP_K=1 \
VLLM_ENGINE_READY_TIMEOUT_S=2400 bash spark/serve-glm52-tp2-mtp.sh
```

**Integration bugs fixed (site-packages patches — MUST fold into
`patch/vllm-moet-v0.24.0.patch`):**
1. `fp8.py` `Fp8Config.get_quant_method` — NVFP4 hook call, placed AFTER the
   `is_layer_skipped` block (was dead code inside the if → never fired).
2. `layers/quantization/utils/nvfp4_dense_hook.py` — new module (from spark/).
3. `model_executor/parameter.py` — `NVFP4SKIP2` guard in the named-param
   `load_{column,row,merged}_parallel_weight`: skip fp8 tensor loaded into a
   uint8 (NVFP4) param. Needed because vLLM globs ALL *.safetensors so the
   overlay's NVFP4 tensor AND the symlinked original's fp8 tensor both load.
4. `model_executor/models/deepseek_v2.py` (~1572) + `deepseek_mtp.py` (~463) —
   `if name not in params_dict: continue` (`NVFP4ORPHAN`): skip the orphan fp8
   `weight_scale_inv` (no NVFP4 param). MTP has its OWN loader → needs it too.
5. Env vars `VLLM_NVFP4_DENSE`/`VLLM_NVFP4_TARGETS` propagate to Ray workers via
   driver-forward (setdefault) — no raylet baking needed; but the hook only
   fires once the fp8.py placement (#1) is correct.

**Gotchas learned:** ~~GB10 has NO native FP4 MMA~~ **[CORRECTED session 12: GB10
DOES have FP4 tensor cores — `cutlass_fp4_supported()`=True on sm_121.]** We use
weight-only Marlin FP4 anyway because it's **faster for M=1 decode** (measured:
Marlin 1.4–3.7× faster than Cutlass W4A4 at M=1; Cutlass's per-call activation
fp4-quant overhead dominates at M=1 and only pays off batched, M≳64). Perfect
for memory-bound decode: 4-bit read, bf16 compute. Overlay approach works but
relies on the glob-duplicate skip guards (#3/#4) — a cleaner future design is
moe_w2-style: load fp8 then swap from a prepacked dir. Ray cluster corrupts
across many failed boots (ActorHandle-across-sessions) → restart with
start-ray-cluster.sh. Debug prints (`NVFP4HOOKDBG`) still in the hook — remove.

## Status (original plan)

- [x] **Packer written + validated:** `spark/prepack_nvfp4_linear.py`. CPU round-trip
  rel-L1 ~0.089 synthetic; on REAL layers 10 & 40 all 16 target tensors rel-L1
  0.089–0.091, byte change **0.56×** (0.41→0.23 GB on those tensors) — matches the
  ~1.47× decode projection. Reads block-FP8 (128×128 `weight_scale_inv`), dequants,
  requants to W4A16 NVFP4, emits `weight`(uint8)/`weight_scale`(e4m3)/
  `weight_scale_2`(fp32) with names matching the loader.
- [x] **Loader hook written:** `spark/nvfp4_dense_hook.py` (env-gated
  `VLLM_NVFP4_DENSE=1`, prefix-keyed, mirrors the moe_w2 hook). Constructs a
  cached `ModelOptNvFp4Config(quant_method="W4A16_NVFP4",
  is_checkpoint_nvfp4_serialized=True)` and returns `ModelOptNvFp4W4A16LinearMethod`
  for attention (`q_a/q_b/kv_a/kv_b/o_proj`, +fused `qkv`/`fused_qkv_a_proj`) and
  shared-expert (`gate/up/down/gate_up_proj`) prefixes; excludes indexer, norms,
  MTP drafter. **APPLY (after profiler done):**
  1. `cp spark/nvfp4_dense_hook.py <site-packages>/vllm/model_executor/layers/quantization/utils/`
  2. Insert into `fp8.py` `Fp8Config.get_quant_method`, LinearBase branch, right
     after the `is_layer_skipped`→`UnquantizedLinearMethod()` block:
     ```python
     from vllm.model_executor.layers.quantization.utils import nvfp4_dense_hook
     _m = nvfp4_dense_hook.maybe_linear_method(prefix)
     if _m is not None:
         return _m
     ```
  3. Then fold both into `patch/vllm-moet-v0.24.0.patch`.
  **VERIFY on first serve:** vLLM fusion of shared `gate_proj+up_proj`→`gate_up_proj`
  and any `q/kv` fusion — the stacked weight_loader must map the overlay's separate
  NVFP4 tensors into the fused params (weight_scale_2 shape = #partitions). If a
  fused prefix isn't matched or a stacked-load shape errors, adjust the regex /
  packer tensor split.
- [ ] **Build overlay:** `python spark/prepack_nvfp4_linear.py --out
  $MODEL/nvfp4_dense_overlay` (full run, all 78 layers; ~read 21 GB fp8, write
  ~9 GB NVFP4; CPU, non-destructive). Mirror to peer node (both ranks load it).
- [ ] **Serve + speed-measure** on the standard protocol → confirm ~1.47× (~18–22
  tok/s). Serve from the overlay dir; experts still from the 2-bit plane dir.
- [ ] **Quality battery** (GSM8K + GPQA + IFEval + MMLU-Pro). Gate: within noise
  of the 15 tok/s baseline.

---

## Staging (source precision)

1. **Speed cut — source from local FP8** (done-ready): no download; source
   precision doesn't change byte size or kernel path, only quality. Proves the
   1.47×. rel-L1 already 0.089 from FP8.
2. **Quality cut — source from BF16** (if the FP8-sourced quality battery
   regresses): pull BF16 attention/shared tensors from `nvidia/GLM-5.2-NVFP4`
   (those tensors are BF16 there — see GOAL.md) or the ZAI base, requant. Cleaner
   block scales, no FP8→NVFP4 double-rounding.
3. **De-risk cut — big-3 only** (`o_proj`,`q_b`,`kv_b` = ~85% of the win): if the
   full set moves quality, restrict `TARGET_SUFFIXES` to those and keep
   `q_a`/`kv_a`/shared at FP8.

## Gotchas

- Marlin W4A16 needs `Ki % 16 == 0` per shard (holds for all targets at TP2).
- Overlay index must DROP the fp8 `weight`+`weight_scale_inv` for replaced tensors
  or the loader sees duplicate names.
- The loader hook must key off prefix at BUILD time (before weights load); simplest
  is a static NVFP4-prefix list in config.json read by the extended Fp8Config.
- Keep experts on Fp8Config — switching to compressed-tensors config strands the
  2-bit expert hook (blocker #2).
