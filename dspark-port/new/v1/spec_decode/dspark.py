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

        out = torch.empty(
            (num_reqs, n), dtype=torch.int64, device=hidden_states.device
        )
        for i in range(n):
            # Sequential stage: bias position i by the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            # Greedy in draft space, then remap draft->target ids.
            draft_i = self.model.map_draft_to_target(logits_i.argmax(dim=-1))
            out[:, i] = draft_i
            prev = draft_i

        if _DSPARK_PROFILE:
            logger.info(
                "DSPARK_PROF markov_sample n=%d reqs=%d ms=%.1f",
                n,
                num_reqs,
                _sync_ms(_t0),
            )
        # Flatten (req, step) -> matches the caller's .view(-1, n).
        return out.reshape(-1), None
