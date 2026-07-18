# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark proposer: DFlash parallel drafting + a sequential Markov logit-bias head.

DSpark reuses DFlashProposer's whole parallel-drafting machinery (context-KV
precompute, the 1+N query-block fused-input kernel, one non-causal forward) and
only changes how the N draft tokens are sampled from the block's hidden states:
instead of DFlash's single parallel argmax, it samples LEFT-TO-RIGHT, adding a
low-rank Markov transition bias derived from the previously sampled token at each
step (the sequential stage).

Ported from the speculators-format DSpark reference (vLLM 0.25.2
`v1/worker/gpu/spec_decode/dspark/`) onto this fork's `v1/spec_decode/` Proposer
framework. Greedy drafting only for now (sufficient for throughput benchmarking);
probabilistic/gumbel drafting is a TODO (see `_enable_probabilistic_draft_probs`).
"""

import os
import time
from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)

_DSPARK_PROFILE = os.environ.get("VLLM_DSPARK_PROFILE", "0") == "1"
# Confidence-gated dynamic K: if > 0, truncate each request's proposal at the
# first draft position whose confidence-head acceptance estimate falls below
# tau. The runner picks the per-request valid counts up via
# take_dspark_valid_counts() and trims the scheduled verify slots (the
# ngram-gpu scheduler-side trim path), saving the per-position verify cost.
_DSPARK_CONF_TAU = float(os.environ.get("VLLM_DSPARK_CONF_TAU", "0") or "0")
# Floor on the gated proposal length. min_k=1 avoids the count-0 cliff: a step
# that proposes nothing degenerates to the ~15 tok/s no-spec floor, which the
# s19 tau=0.5 run showed dominates any per-position verify savings.
_DSPARK_CONF_MIN_K = int(os.environ.get("VLLM_DSPARK_CONF_MIN_K", "0") or "0")


def _sync_ms(t0: float) -> float:
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        # Per-request bonus token ("prev" seed for the Markov head), captured in
        # set_inputs_first_pass. Speculators DSpark uses dspark_bonus_anchor=True
        # (the 1+N DFlash layout), so the bonus token is next_token_ids.
        self._dspark_prev: torch.Tensor | None = None
        # Lazily resolved on first sample (the draft model isn't loaded yet
        # here): tau > 0 AND the checkpoint actually has a confidence head.
        self._dspark_use_conf: bool | None = None
        self._dspark_valid_counts: torch.Tensor | None = None
        if self._enable_probabilistic_draft_probs:
            logger.warning(
                "DSpark probabilistic drafting is not implemented; falling back "
                "to greedy Markov drafting for the draft proposals."
            )

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()  # DFlash: non-causal attention
        # The DSA target serves with kv_cache_dtype=fp8_ds_mla (DeepSeek-MLA fp8),
        # but the dense non-causal draft has NO attention backend that supports
        # that combo. The draft's KV cache is tiny, so give it a plain dtype.
        return replace(
            base,
            cache_config=replace(base.cache_config, cache_dtype="auto"),
        )

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # Seed for the Markov head: the bonus token per request (target vocab).
        self._dspark_prev = next_token_ids
        return super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs=None,
    ):
        # Times the context-KV precompute (prime suspect for the per-step cost).
        if _DSPARK_PROFILE:
            t0 = time.perf_counter()
            out = super().build_model_inputs_first_pass(
                num_tokens, num_input_tokens, mm_embed_inputs
            )
            ctxkv_ms = _sync_ms(t0)
            logger.info(
                "DSPARK_PROF ctxkv+inputs num_ctx=%d ms=%.1f",
                self._dflash_num_context,
                ctxkv_ms,
            )
            # Mark the start of the draft-model forward (runs after this returns).
            self._dspark_t_fwd_start = time.perf_counter()
            return out
        return super().build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _DSPARK_PROFILE and getattr(self, "_dspark_t_fwd_start", None) is not None:
            logger.info(
                "DSPARK_PROF draft_fwd ms=%.1f", _sync_ms(self._dspark_t_fwd_start)
            )
        _t0 = time.perf_counter() if _DSPARK_PROFILE else None
        # hidden_states: [num_reqs * n, hidden], ordered (req, step) — the N draft
        # positions of every request's 1+N query block.
        n = self.num_speculative_tokens
        num_reqs = hidden_states.shape[0] // n

        # Draft-vocab base logits for all N positions in one shot.
        base_logits = self.model.compute_draft_logits(hidden_states)
        draft_vocab = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n, draft_vocab)

        assert self._dspark_prev is not None
        prev = self._dspark_prev[:num_reqs]  # bonus token per req (target vocab)

        if self._dspark_use_conf is None:
            self._dspark_use_conf = _DSPARK_CONF_TAU > 0 and getattr(
                self.model, "has_confidence_head", lambda: False
            )()
            if _DSPARK_CONF_TAU > 0:
                logger.info(
                    "DSpark confidence gating: tau=%.3f enabled=%s",
                    _DSPARK_CONF_TAU,
                    self._dspark_use_conf,
                )
        use_conf = self._dspark_use_conf
        hs_blocks = hidden_states.view(num_reqs, n, -1) if use_conf else None
        conf_cols: list[torch.Tensor] = []

        out = torch.empty(
            (num_reqs, n), dtype=torch.int64, device=hidden_states.device
        )
        for i in range(n):
            # Sequential stage: bias position i by the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            if use_conf:
                conf_cols.append(
                    self.model.confidence_logits(hs_blocks[:, i], markov_embed)
                )
            logits_i = base_logits[:, i] + bias
            # Greedy in draft space, then remap draft->target ids.
            draft_i = self.model.map_draft_to_target(logits_i.argmax(dim=-1))
            out[:, i] = draft_i
            prev = draft_i

        self._dspark_valid_counts = None
        if use_conf:
            conf = torch.stack(conf_cols, dim=1).float().sigmoid()  # [reqs, n]
            # Keep a prefix of positions: truncate at the first one whose
            # predicted acceptance is below tau (that position included).
            keep = (conf >= _DSPARK_CONF_TAU).to(torch.int32).cumprod(dim=1)
            self._dspark_valid_counts = (
                keep.sum(dim=1).clamp_(min=min(_DSPARK_CONF_MIN_K, n)).to(torch.int32)
            )
            # Periodic conf snapshot (cheap: one sync per 500 steps) to guide
            # the tau sweep without paying the full profile-mode sync cost.
            self._dspark_conf_step = getattr(self, "_dspark_conf_step", 0) + 1
            if self._dspark_conf_step % 500 == 1:
                logger.info(
                    "DSPARK_CONF step=%d per_pos_mean=%s counts=%s",
                    self._dspark_conf_step,
                    [round(x, 3) for x in conf.mean(dim=0).tolist()],
                    self._dspark_valid_counts.tolist(),
                )
            if _DSPARK_PROFILE:
                logger.info(
                    "DSPARK_PROF conf mean=%.3f per_pos=%s counts=%s",
                    conf.mean().item(),
                    [round(x, 3) for x in conf.mean(dim=0).tolist()],
                    self._dspark_valid_counts.tolist(),
                )

        if _DSPARK_PROFILE:
            logger.info(
                "DSPARK_PROF markov_sample n=%d reqs=%d ms=%.1f",
                n,
                num_reqs,
                _sync_ms(_t0),
            )
        # Flatten (req, step) -> matches the caller's .view(-1, n).
        return out.reshape(-1), None

    def take_dspark_valid_counts(self) -> torch.Tensor | None:
        """Per-request confidence-gated draft counts from the last sample.

        int32 [num_reqs], aligned with the batch order of the proposals; None
        when gating is off. Consumed once per step by the runner, which trims
        the scheduled verify slots on the next step.
        """
        counts = self._dspark_valid_counts
        self._dspark_valid_counts = None
        return counts
