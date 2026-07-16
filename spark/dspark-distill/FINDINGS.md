# DSpark self-distill — investigation findings (2026-07-16)

## Outcome: pipeline BLOCKED by a 3-way forward-implementation mismatch. Root
cause found; not a quick fix. Recommendation: do NOT fine-tune under repo-HEAD
speculators; the expected payoff (a few tok/s) doesn't justify reconciling it.

## What we built (works)
- `serve-glm52-tp2-hidden.sh`: extract_hidden_states server on the fork. Two
  venv patches needed (tracked in `patches/`, applied+synced both nodes):
  1. `deepseek_v2.py`: emit the pre-norm final hidden (aux id = num_layers=78).
  2. `example_hidden_states_connector.py`: tolerate aborted requests (client
     timeouts KeyError'd the EngineCore).
- Offline cache gen (`data_generation_offline.py`, --request-timeout 900),
  worker-drain rsync loop, 1988-sample cache (~205 GB) on spark-c84b.
- Trainer runs on the worker: `--from-pretrained` epoch-3, single-GPU bf16,
  Muon, sdpa or flex both work.

## The blocker (fully diagnosed)
At step-0 with the pretrained checkpoint + LR 0, the trainer should reproduce
the card's val accuracy (pos-0 ~0.83). It gives **pos-0 0.35, accept_rate 0.075**
— on BOTH the fast-build target AND a card-matched plain-FP8 target (0.355).
So it is NOT distribution shift; the fine-tune's premise ("recover acceptance")
would start from a broken baseline.

### Ruled out (all verified, not assumed)
- **Draft weights load correctly** — fc / decoder / hidden_norm / markov_w1,w2 /
  confidence all byte-match `model.safetensors` (norms compared).
- **TV/CE targets correct** — verifier lm_head over the cached last-layer hidden
  gives 0.607 next-token argmax match = the card's "full accuracy 0.613".
- **Aux collection point correct** — dumped the LIVE DSparkProposer's fc input
  (`aux_live_dump.pt`, via a one-shot dump patch in `llm_base_proposer.py`) and
  diffed vs the extract cache for identical tokens: per-layer magnitudes match
  (layer4 live 0.783 vs cache 0.791); the 18% element-wise diff grows with depth
  = FP8 batching nondeterminism, not a wrong layer/representation.
- **loss_mask correct** — 81% coverage, clean user(False)/assistant(True) split.
- **Attention impl** — flex (default) and sdpa both give ~0.32; not the cause.

### Root cause: speculators VERSION DRIFT
Checkpoint trained with **speculators 0.5.0.dev38** (from its config.json). We
installed repo **HEAD 0.7.0.dev102**. Between `v0.5.0..HEAD`, the training/forward
code changed enormously (2339 insertions): `dflash/core.py` +366, `metrics.py`
+250, `trainer.py` +467, `data.py` +169, and specifically forward-altering
commits — `sample_from_anchor logic to DFlash/DSpark (#760)`, `default sliding
window attention for dflash/dspark (#749)`, `compile create_block_mask (#731)`,
`Cleanup unused draft-token argmax from DFlash/DSpark forward (#730)`. HEAD's
trainer is a different implementation of the dspark forward than the one that
made these weights — hence 0.35 not 0.83.

### The deeper problem (why even pinning 0.5.x is risky)
There are THREE independent dspark forward implementations in play:
(a) the checkpoint's original trainer (speculators 0.5.0.dev38),
(b) whatever trainer we run to fine-tune,
(c) the FORK's inference `DSparkProposer` (v1/spec_decode/dspark.py, gets 68%).
For a fine-tune to help inference, (b) must match (c). But (c) is a bespoke port,
not speculators code. Even pinning (b)=0.5.x only reconciles (a)=(b); it does not
guarantee (b)=(c). Training under any speculators version to deploy under the
fork's proposer is an impedance mismatch with no guarantee the gradients improve
the deployed metric.

## Options from here
1. **STOP (recommended).** Ship dspark K=2 + REAP (21.8 tok/s), keep the pipeline
   documented. Payoff of the fine-tune was always modest (few tok/s, maybe none
   on familiar data); the 3-way forward mismatch raises cost far above that.
2. **Pin speculators==0.5.x**, re-verify step-0 reproduces ~0.83, then fine-tune —
   accepting risk (b)≠(c) may mean inference doesn't benefit. ~half-day, uncertain.
3. **Train against the fork's own proposer forward** (make (b)=(c) by writing a
   trainer around v1/spec_decode/dspark.py). Most correct, most work (days).

## Artifacts
- Cache: `spark-c84b:~/dspark-distill-data/prepared/hidden_states/` (1988 files).
- Dump patch + probe: `aux_live_dump.pt`, patch in `llm_base_proposer.py` (REVERT
  before shipping — it writes a file on first proposer call).
- Logs: `~/dspark-distill-data/train-*.log`, `~/serve-hidden*.log`.
