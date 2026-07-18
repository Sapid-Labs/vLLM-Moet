# s13 (2026-07-16) — dspark external speculator ported; fast-build perf diagnosis (revised in s14)

> Previous: 06-s12-*.md · Next: 08-s14-*.md

## STATUS (2026-07-16, session 13) — DSpark EXTERNAL speculator ported + working; fast-build PERF blocked (diagnosed) — **DIAGNOSIS REVISED IN SESSION 14 ABOVE**

RESUME POINTER for the dspark-speculator effort. One-line: `DSparkProposer` port is
DONE and generates correctly on 2 Sparks; on the **fast build** it currently
REGRESSES throughput (context-KV precompute is 3-4s/step), root-caused to two
fixable causes. Branch `spark-gb10`; venv `~/venvs/vllm-moet` (0.24.0); port files
in `dspark-port/` (apply.sh re-installs into a venv — RUN ON BOTH NODES).

### DONE (verified)
- **Ported `dspark` speculative decoding** (external RedHatAI/GLM-5.2-speculator.dspark,
  a `speculators`-format DSparkDraftModel, verifier zai-org/GLM-5.2-FP8) into the fork:
  - `transformers_utils/configs/speculators/algos.py`: `register_speculator("dspark")`
    (emits arch `Qwen3DSparkModel`, `dspark_bonus_anchor=True`, `dflash_config.mask_token_id`,
    aux/target layers).
  - `config/speculative.py`: `DSparkModelTypes`, method dispatch, `use_dspark()`,
    aux-hidden-states + parallel_drafting for dspark.
  - `model_executor/models/{registry.py,qwen3_dspark.py}`: `Qwen3DSparkModel`
    (DFlashQwen3 backbone + Markov head; from the 0.25.2 reference).
  - `v1/spec_decode/dspark.py`: **`DSparkProposer(DFlashProposer)`** — the real work.
    Overrides `_create_draft_vllm_config` (draft KV → "auto", DSA target's fp8_ds_mla
    has no non-causal backend), `set_inputs_first_pass` (store bonus token),
    `_sample_draft_tokens` (GREEDY sequential Markov; probabilistic/gumbel = TODO).
    Has env-gated `VLLM_DSPARK_PROFILE` timing (not forwarded to Ray workers — hardcode
    `_DSPARK_PROFILE=True` to re-profile; worker logs in `/tmp/ray/session_latest/logs/`).
  - `v1/spec_decode/llm_base_proposer.py`: `model_returns_tuple()` add "dspark";
    combine-hidden-states branch add "dspark".
  - `v1/worker/gpu_model_runner.py`: import + `elif use_dspark(): DSparkProposer(...)`.
  - `v1/spec_decode/dflash.py`: `__init__` assert accepts "dspark".
- **WORKS on GLM-5.2-FP8** (2 Sparks, eager): coherent gen, **83-100% acceptance**,
  mean accept length 3.5-4.0. Serve: `serve-glm52-tp2-dspark.sh --eager`.
- **Fast build (NVFP4 overlay + p208 REAP + top-k4) + dspark FULL graphs**: loads &
  generates, moderate acceptance (mean ~3, avg 60-70% — draft trained on full FP8 so
  it diverges from the pruned/NVFP4 target). BUT throughput **~2.3 tok/s vs 21.8
  baseline** = a REGRESSION, not a win.
- **Profiled the regression** (decisive): per draft step —
  `precompute_and_store_context_kv` **3-4.7s** (99.5%), draft forward **10ms**, Markov
  **5ms**. Recompilation ruled out. So it's ONE slow method, NOT verify bandwidth,
  NOT my code. **Refutes the MTP-DRAFT-COST bandwidth worry for this case.**
- **Confirmed cause #1 (~half):** per-step `VLLM_MOE_W2_FADVISE_GLOB` over
  `$MODEL/*.safetensors` — disabling it (`VLLM_MOE_W2_FADVISE_GLOB=`) dropped precompute
  to ~1-2s. (serve script now honors an override.)

### NEXT (the 2-step fix to try for a 25 tok/s answer)
1. **Kill per-step fadvise:** find where the W2 plane machinery calls fadvise; gate it
   to load-time / target-only so it never fires on the drafter's context-KV precompute.
2. **Kill the ~1-2s residual:** nsys/torch-profile a single `precompute_and_store_context_kv`
   (runs EAGER, cudagraph=NONE). Prime suspect: the eager **NVFP4 dense hook on the
   DRAFT's** `o_proj/q_b_proj/kv_b_proj` (`VLLM_NVFP4_TARGETS` matches the draft's own
   layer names) — try excluding the draft from NVFP4, or not applying NVFP4 to the draft.
   Variance (953-1983ms) still smells I/O/contention.
   Goal: precompute → ~ms → a spec step ≈ one verify forward → with mean-accept ~2-3,
   plausibly BEATS 21.8. If it can't get under native-MTP, report that honestly.
3. Then benchmark 3-way (dspark vs native MTP vs no-draft) + publish to ~/Dev/howtospark.

### HOW TO RESUME
- Re-apply port to a fresh venv (BOTH NODES): `dspark-port/apply.sh <vllm-site-packages>`
  then rsync each edited file to `192.168.100.2:<same path>`.
- Cheap sanity: `~/venvs/vllm-moet/bin/python -c "from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig as C; print(C.from_pretrained('RedHatAI/GLM-5.2-speculator.dspark').architectures)"` → `['Qwen3DSparkModel']`.
- Serve FP8 dspark: `bash spark/serve-glm52-tp2-dspark.sh --eager` (endpoint :8000, model `glm-5.2`).
- Serve FAST build dspark (FULL graphs): `MODEL=$HOME/models/hf/GLM-5.2-FP8/nvfp4_big3_overlay
  VLLM_MOE_W2_PREPACKED_DIR=$HOME/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208
  VLLM_NVFP4_DENSE=1 VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj VLLM_ENGINE_READY_TIMEOUT_S=2400
  bash spark/serve-glm52-tp2-dspark.sh --hf-overrides '{"num_experts_per_tok":4}'`
  (add `--eager` to skip graphs / for fast boot; add `VLLM_MOE_W2_FADVISE_GLOB=` to halve precompute).
- Ray corrupts across boots — if "ActorHandle across sessions", `ray stop` both nodes + `bash spark/start-ray-cluster.sh`.
- Speculator lives on BOTH nodes: `~/models/hf/GLM-5.2-speculator.dspark` (7GB).

### GOTCHAS / KEY FACTS
- **PATCH BOTH NODES' venvs** — Ray runs workers on head + worker, each has its own
  site-packages. A head-only patch fails on the worker ("Unknown method dspark").
- venv is NOT editable-installed; the repo `vllm/` tree lacks this code. `dspark-port/` is the tracked copy.
- The 0.24 fork uses `v1/spec_decode/` **Proposer** classes; the 0.25.2 reference uses
  `v1/worker/gpu/spec_decode/` **Speculator** classes — DIFFERENT frameworks. The port
  targets the Proposer one; the `gpu/spec_decode/dspark/*` files I copied first are DEAD (ignore).
- `kill`/`pkill -f 'vllm serve'` self-matches the shell (exit 144). Use `pgrep -f '[b]in/vllm serve'`.
- Foreground `sleep` is blocked in the Claude harness (exit 144).

### CORRECTNESS GATES
- Coherent output on FP8 (got "Paris"/"2+2=4"/reasoning); SpecDecoding metrics show
  acceptance >0 (drafts being accepted). Bad = empty/garbage output or accept ~0.
- A throughput WIN means fast-build dspark tok/s > native-MTP (21.8). Currently 2.3 (fail).

### LINKS
- Port: `dspark-port/` (new/, orig/, apply.sh). Serve: `spark/serve-glm52-tp2-dspark.sh`.
- Speculator: https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark
- dspark-capable ref image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2, has dspark+GlmMoeDsa).
- Site benchmarks repo: ~/Dev/howtospark (models/glm-5.2.md). IQ1_S llama.cpp 2-Spark
  GLM-5.2 benchmark (separate track, DONE+published): howtospark commit `1ab303c`.

