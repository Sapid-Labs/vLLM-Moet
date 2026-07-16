# DSpark draft self-distillation against the fast-build target

Goal: raise the DSpark draft's acceptance on the SERVED target (fast build =
NVFP4 big-3 + REAP p208 + top-k4) and on representative long-form content,
from measured pos-0/pos-1 ≈ 68%/45% toward the model card's 83%/72%, worth an
estimated +2-4 tok/s at K=2-3 (21.8 → ~24-25 ceiling). Status/tables that
motivated this: `spark/handoffs/03-dspark-acceptance-ablation.md`.

## Why fine-tune is the only lever left (measured 2026-07-16)

Acceptance is IDENTICAL (±1pt) across every target ablation: freq vs REAP
planes, top-k4 vs k8, NVFP4 vs plain FP8 attention, pruned vs full 256-expert
planes. The draft was trained on magpie/ultrachat-style chat; our measurement
content (300-tok reasoning/explanations) is simply harder for it. So the win
comes from training on (a) OUR content mix, conditioned on (b) OUR target's
hidden states — not from un-modifying the target.

## Design decisions (firm)

1. **Fine-tune from the epoch-3 checkpoint**, not from scratch. We need a
   distribution shift, not a new model. Short run, low LR, small data.
2. **No response regeneration.** Reuse `mgoin/GLM-5.2-FP8-magpie-ultrachat`
   text (already GLM-5.2-generated) PLUS a reasoning-heavy slice we care
   about; the training signal that matters — target hidden states and target
   next-token distribution per position — is recomputed by prefill against
   OUR fast-build target regardless of who wrote the context text. (Risk:
   context text from FP8-full is slightly off-distribution for the fast
   build; second-order, accept it for v1.)
3. **The hidden-states server must be OUR fork** (vllm-moet 0.24 + planes +
   NVFP4 + dspark port) with the fast-build env — the whole point is
   distilling the served model. If speculators' launch_vllm.py can't wrap the
   fork, port its hidden-states hook into the fork (it already computes aux
   hidden states for the dspark proposer — the plumbing exists server-side).
4. **Two-phase on 2× GB10 if memory forces it**: phase A = serve fast build
   (both Sparks, prefill-only traffic) + cache hidden states to disk; phase B
   = tear down server, train the 7 GB draft from the disk cache (1 node, or
   DDP over the 200G fabric). Online co-located training would fight the
   plane page cache for RAM — avoid unless the trainer footprint is tiny.
   Disk math: 5 layers × 6144 × bf16 ≈ 60 KB/token → ~61 GB per 1M tokens.
   Head node has ~98 GB free, worker ~579 GB → cache lives on the WORKER.
5. **Data budget v1**: ~2-5M tokens (fine-tune, not the 250M-token from-
   scratch run). Prefill at a few hundred tok/s → hours, not days.
6. **Eval gates** (same 300-tok greedy prompt set as the ablation, plus a
   held-out slice):
   - acceptance A/B via /metrics deltas: ship only if pos-0 ≥ ~75%.
   - throughput: dspark K=2 fast build ≥ 23 tok/s steady.
   - quality is the TARGET's, unchanged by the draft — no battery needed;
     greedy-coherence spot check suffices.
   - Also re-sweep K=2 vs K=3 after: higher acceptance may flip K=3 back on.

## Recon results (2026-07-16, speculators @ c1dd464 — clone in scratchpad)

- **Our fork has everything server-side**: `extract_hidden_states` spec method,
  `ExampleHiddenStatesConnector` (kv_transfer), `return_token_ids` in the
  completion protocol. launch_vllm.py is just a wrapper that passes two JSON
  configs — we replicate them in our own serve script instead (need our
  planes/NVFP4/Ray env). It force-adds `--no-enable-chunked-prefill`.
- **Flow**: trainer/data-gen sends completions (token ids, max_tokens=1,
  `return_token_ids`); server writes per-sample safetensors
  `{hidden_states: [seq, 6, 6144], token_ids}` (5 aux layers + last layer) to
  `shared_storage_path` and returns the file path. Trainer and server need a
  SHARED FILESYSTEM. ~72 KiB/token on disk.
- **Fully offline supported**: `scripts/data_generation_offline.py` pre-fetches
  the cache (resumable, `--world-size/--rank` shardable). Train later with
  `--on-missing raise` and the server down. This is our two-phase plan.
- **Trainer**: single-GPU fine (`--nproc_per_node 1`), pure bf16, no grad
  checkpointing. Muon default optimizer (torch ≥2.9; venv has 2.11 ✓).
  Peak memory dominated by the T×V logits/softmax stack (T = max_anchors ×
  block_size, V=154880): at max-anchors 1024 → T=8192 → ~17-20 GB; keep
  max-anchors ≤1024 on GB10. `--draft-attn-impl sdpa` if flex-attn/compile
  misbehaves on sm_121.
- **Fine-tune**: `--from-pretrained RedHatAI/GLM-5.2-speculator.dspark`
  (mutually exclusive with the shape flags; heads' hyperparams come from the
  checkpoint). Trainer loads ONLY embed_tokens/lm_head/model.norm from the
  verifier dir (~5.7 GB bf16) — point it at the ORIGINAL GLM-5.2-FP8 dir
  (embed/lm_head/norm are not touched by our overlay).
- **Losses**: tv+ce computed in-trainer from verifier_lm_head over the LAST-
  layer hidden states the server returns → the distillation target is exactly
  OUR served model's distribution. Markov + confidence heads train jointly.
- **Venv**: install speculators into `~/venvs/vllm-moet` with `--no-deps`
  (its vllm dep is unpinned and must NOT drag a wheel onto aarch64);
  transformers 5.13.0 and datasets 5.0.0 already satisfy pins.
- **Disk**: hidden-state cache at 72 KiB/token → 1M tok ≈ 72 GB. Head has
  ~98 GB free (95% full!) — cache belongs on the WORKER (579 GB free) or an
  NFS/sshfs mount; decide in chunk 3. Data budget v1: 1-2M tokens.

## Chunks

1. **Env + hidden-states server bring-up** (this session): pip install
   speculators --no-deps; `spark/serve-glm52-tp2-hidden.sh` (fast-build env +
   extract_hidden_states + connector configs, chunked prefill off); smoke
   test: one completion → safetensors appears, shape [seq, 6, 6144], token
   ids echo. Find out which node the connector writes on under Ray TP2.
2. Data prep (CPU-only): download `mgoin/GLM-5.2-FP8-magpie-ultrachat` slice
   (+ optionally a reasoning-heavy set), prepare_data.py with the GLM
   assistant-pattern, output to worker disk.
3. Offline hidden-state cache gen: data_generation_offline.py against the
   server; storage per chunk-1 findings; ~1-2M tokens.
4. Trainer bring-up: single GB10, --from-pretrained epoch-3, low LR (~1e-5,
   NOT the 6e-4 from-scratch LR), --loss-fn '{"ce":0.1,"tv":0.9}', smoke 100
   steps → loss decreasing, memory < 100 GB.
5. Short fine-tune (1-2 epochs over the slice), checkpoints to worker disk.
6. Deploy + A/B: point --speculative-config model at the new checkpoint,
   measure acceptance/throughput on the ablation prompt set, re-sweep K=2/3,
   verdict + handoff/model-log updates. Gates: pos-0 ≥75%, ≥23 tok/s, else
   report honestly and keep epoch-3.
