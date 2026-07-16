# HANDOFF — DSpark draft self-distillation (squeeze GLM-5.2 dspark acceptance)

Resume-here pointer for the dspark draft fine-tune. Deep docs alongside:
`PLAN.md` (design + decisions), `FINDINGS.md` (full investigation + rule-outs).
Convention: `~/CLAUDE.md` → "Session handoffs".

## STATUS (2026-07-16, session 15) — pipeline BUILT + root-caused; blocked on speculators version drift; next = pin the trainer version

Goal: raise the dspark draft's acceptance on the shipped fast-build target
(dspark K=2 + REAP p208 + NVFP4 big-3 + top-k4, ~21.8 tok/s) from measured
pos-0/1 ≈ 68/45% toward the card's 83/72%, for an estimated +2-4 tok/s
(→ ~24-25 ceiling at K=2-3). The whole offline pipeline works; it is blocked
by ONE thing: the trainer (speculators repo HEAD) is a drifted implementation
of the dspark forward vs the checkpoint's training version, so a correctly
loaded checkpoint scores pos-0 0.35 at step-0 instead of the card's 0.83.
**Fix = run the trainer at the checkpoint's era (~commit 21033a7), not HEAD.**

## DONE (verified)
- **Hidden-states extraction server on the fork**: `spark/serve-glm52-tp2-hidden.sh`
  (extract_hidden_states + ExampleHiddenStatesConnector, chunked-prefill off,
  eager). Needs TWO venv patches, tracked in `patches/`, applied+synced BOTH nodes:
  1. `deepseek_v2.py`: emit pre-norm final hidden when aux id == num_layers (78),
     so the trainer gets the last-layer state for tv/ce targets.
  2. `example_hidden_states_connector.py`: `.pop(req_id, None)` — aborted requests
     (client timeouts) otherwise KeyError-kill the EngineCore.
- **Cache generated**: 1988 samples (~3M tok, ~205 GB) at
  `spark-c84b:~/dspark-distill-data/prepared/hidden_states/hs_<idx>.safetensors`,
  format `{hidden_states:[seq,6,6144], token_ids:[seq]}` (5 aux layers
  [8,23,39,55,70] + last layer 78). Prepared arrow dataset (2000 magpie/ultrachat
  samples, GLM assistant-pattern) at `~/dspark-distill-data/prepared` on both nodes.
- **Trainer runs** on the worker: `--from-pretrained` epoch-3, single-GPU bf16,
  Muon, sdpa/flex both work, ~100 tok/s, mem < 100 GB at max-anchors 1024.
- **Root cause found — all alternatives RULED OUT** (see FINDINGS.md): draft
  weights byte-match checkpoint; tv/ce targets correct (0.607 = card 0.613); aux
  collection point correct (live-proposer fc-input diff vs cache = fp8 noise only,
  magnitudes match); loss_mask correct (81%, clean split); attn impl irrelevant.
  The ONLY thing left is the speculators forward rewrite between the checkpoint's
  `0.5.0.dev38` and installed HEAD `0.7.0.dev102` (`sample_from_anchor` #760,
  default sliding-window #749, `create_block_mask` compile #731, argmax cleanup
  #730 — ALL dated 2026-07-13, after the checkpoint was trained).
- **Version target identified**: `21033a7` (2026-07-08) — after dspark-add
  (ff71b1e, 06-29), before the 07-13 forward rewrites; has offline gen +
  --from-pretrained + dspark core; lacks sample_from_anchor. Statically verified.

## NEXT (the concrete plan — do in order)
1. **Pin the trainer version + confirm step-0.** In a SEPARATE clone (don't
   disturb the HEAD install the extract server's tooling uses — though the serve
   itself is vLLM, not speculators):
   ```
   git clone ~/Dev/speculators ~/Dev/speculators-v05 && cd ~/Dev/speculators-v05
   git checkout 21033a7
   ~/venvs/vllm-moet/bin/pip install --no-deps -e .   # NOTE: breaks HEAD install;
   #   reinstall HEAD (`pip install --no-deps -e ~/Dev/speculators`) before using
   #   the extract/gen scripts again, OR use a second venv for the trainer.
   ```
   Then the STEP-0 GATE on the worker (serve must be DOWN — needs the GPU):
   ```
   ssh spark-c84b '~/venvs/vllm-moet/bin/python ~/Dev/speculators-v05/scripts/train.py \
     --verifier-name-or-path ~/models/hf/GLM-5.2-FP8 --speculator-type dspark \
     --from-pretrained ~/models/hf/GLM-5.2-speculator.dspark \
     --data-path ~/dspark-distill-data/prepared-mini \
     --hidden-states-path ~/dspark-hs-fp8test \
     --save-path /tmp/t --epochs 1 --lr 0 --scheduler-type none \
     --total-seq-len 4096 --max-anchors 1024 --draft-attn-impl sdpa \
     --loss-fn "{\"ce\":0.1,\"tv\":0.9}" --on-missing skip --log-freq 1 \
     --num-workers 2 --trust-remote-code'
   ```
   PASS = pos_0_acc ≈ 0.80-0.83 (vs the broken 0.35 at HEAD). If still 0.35,
   BISECT `ff71b1e..a0be7bb` (git bisect on the step-0 metric); the offending
   commit is likely a0be7bb/4fd632c/f9e9685 (all 07-13). Do NOT proceed to a real
   fine-tune until this gate passes — a fine-tune from a broken baseline is worthless.
2. **Fine-tune on the EXISTING cache** (fast-build hidden states, magpie/ultrachat
   content): same command, drop `--hidden-states-path`/`--lr 0`, use
   `--data-path ~/dspark-distill-data/prepared` (the 1988-cache), `--lr 1e-5`,
   `--epochs 1-2`, `--save-path ~/dspark-distill-data/ckpt-v1`. This is the
   pure "align to fast-build target" run. Expected: small gain (target-alignment
   was measured to cost ≤1pt, so this mostly re-teaches familiar content).
3. **Generate the reasoning-heavy slice** (the real lever — task #7). Use the
   fast-build target's OWN greedy outputs on reasoning/technical prompts (the
   content where dspark is weak — the 300-tok explanation/derivation style from
   the ablation prompt set). Serve fast-build normally, generate ~1-2M tok of
   completions, format as jsonl conversations, `prepare_data.py`, then the
   hidden-states server + offline gen. Mix ~30-50% with magpie. Re-fine-tune.
4. **Deploy + A/B.** Point `--speculative-config`'s `model` at the new checkpoint
   dir (may need conversion to the speculators HF layout the fork's loader
   expects — verify `SpeculatorsConfig.from_pretrained(ckpt).architectures ==
   ['Qwen3DSparkModel']`). Serve dspark K=2, warm ≥10 runs, read pos-0/1
   acceptance from /metrics, measure tok/s with `spark/demo_chat.py`. Re-sweep
   K=2 vs K=3 (higher acceptance may re-open K=3).

## HOW TO RESUME
- **Baseline serve (the thing to beat, also the current live server)**: §4b of
  `spark/RUNBOOK.md` — dspark K=2 + REAP + NVFP4 + top-k4 ≈ 21.8 tok/s.
  `~/serve-dspark-best.log`. Tear it down before any trainer/gen run (GPU).
- **Extract server** (for new hidden states): `HIDDEN_STATES_PATH=<dir>
  MODEL=$M/nvfp4_big3_overlay VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208_reap
  VLLM_NVFP4_DENSE=1 VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj
  VLLM_MOE_W2_FADVISE_GLOB= bash spark/serve-glm52-tp2-hidden.sh
  --hf-overrides '{"num_experts_per_tok":4}'` (add nothing for plain-FP8).
  WARM with 2 completions before generating (cold prefill > 120s default).
- **Offline gen**: `~/Dev/speculators*/scripts/data_generation_offline.py
  --model glm-5.2 --endpoint http://localhost:8000/v1 --preprocessed-data
  <prepared> --output <hs-dir> --concurrency 4 --request-timeout 900`. Drain to
  worker with an rsync `--remove-source-files --min-size=1` loop (head disk ~97 GB).
- **Step-0 eval harness**: `~/dspark-distill-data/prepared-mini` (30 rows) +
  `~/dspark-hs-fp8test` (FP8 hidden states) + the train.py LR-0 command above.
- Speculator ckpt on BOTH nodes: `~/models/hf/GLM-5.2-speculator.dspark` (7.6 GB).

## GOTCHAS / KEY FACTS
- **speculators version tags are NOT chronological** (v0.6.0 dated 06-12,
  v0.5.0 older, but dspark-add ff71b1e is 06-29 AFTER both). Trust commit DATES
  and the step-0 gate, not the version string. Checkpoint self-reports 0.5.0.dev38.
- **The trainer forward ≠ the fork's inference proposer** are two codebases;
  BUT the original 0.5.x→fork pipeline empirically works (the shipped draft gets
  68% under the fork proposer), so training at the checkpoint's era and deploying
  on the fork is a proven recipe — the (b)≠(c) worry in FINDINGS is not blocking.
- **Ray does NOT forward driver env to workers** — the proposer/model runner runs
  on workers; env-gated debug (VLLM_DUMP_AUX etc.) must be hardcoded, not exported.
- **`--model` to the gen script is an equality check** vs the server's model list;
  pass the SERVED alias `glm-5.2`, not the path.
- **Cold page cache**: first ~10 runs after any serve boot are slow (planes fault
  in via mmap); never benchmark cold. Warm the extract server too before gen.
- **kill self-match**: `pgrep -f 'train.py'|kill` and `pkill -f 'vllm serve'`
  match their own launcher shell → exit 144. Use bracket class `[t]rain.py`,
  `[b]in/vllm serve`, and kill by PID.
- **Trainer is single-GPU** on the worker; the head can serve meanwhile IF the
  trainer only uses the worker — but TP2 serve needs BOTH GPUs, so serve must be
  fully down for any worker-GPU trainer/gen run.
- **`--scheduler-type` choices are linear|cosine|none** (NOT "constant").
- Aux last-layer patch: this fork collects aux at LAYER ENTRY inside the loop
  (aux id i = output of layer i-1); id==num_layers(78) is handled after the loop
  (pre-norm final). Verified correct via lm_head→0.607.

## CORRECTNESS GATES
- **GATE 1 (blocks everything)**: step-0 LR-0 on the version-pinned trainer must
  give pos_0_acc ≈ 0.80-0.83 on the FP8 cache. Until then the pipeline is wrong.
- **GATE 2**: after fine-tune, greedy coherence spot-check (target quality is
  unchanged by the draft — no battery needed).
- **GATE 3 (ship)**: dspark K=2 pos-0 acceptance ≥ ~75% AND ≥ 23 tok/s steady on
  the ablation prompt set, else keep epoch-3 and report honestly.

## LINKS
- `PLAN.md` (design), `FINDINGS.md` (investigation), `patches/` (2 venv patches).
- Ablation that motivates this: `spark/handoffs/03-dspark-acceptance-ablation.md`.
- Serve recipes: `spark/RUNBOOK.md` §4b (fast build) / §4c (this effort).
- Speculator: https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark
- Results log: `~/Dev/howtospark/models/glm-5.2.md`.
