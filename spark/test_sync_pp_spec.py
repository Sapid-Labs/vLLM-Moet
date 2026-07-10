#!/usr/bin/env python3
"""Deterministic CPU-only repro of the sync-scheduler + PP + spec bug.

Under PP (batch queue), draft tokens are delivered asynchronously via
update_draft_token_ids. If they land while the request's previous step is
still in flight (num_computed_tokens == num_tokens), the scheduler schedules
a drafts-only step at positions whose input token doesn't exist yet, and
num_new_tokens goes negative on the following step.

Mirrors the live trace from GLM-5.2 PP2 + ngram on 2026-07-10.
"""
import sys

import torch

from vllm.config import (
    CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig,
    SpeculativeConfig, VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

BLOCK = 16


def make_sync_scheduler():
    model_config = ModelConfig(
        model="facebook/opt-125m", trust_remote_code=True, dtype="float16",
        seed=42, skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=8, max_num_batched_tokens=512, max_model_len=8192,
        enable_chunked_prefill=True, async_scheduling=False,
        is_encoder_decoder=False, watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto",
        enable_prefix_caching=False,
    )
    speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=2,
        prompt_lookup_max=4, prompt_lookup_min=2,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config, model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(
            pipeline_parallel_size=2, distributed_executor_backend="ray"
        ),
        speculative_config=speculative_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10000, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(
            ["layer"],
            FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1,
                              dtype=torch.float32),
        )],
    )
    cache_config.num_gpu_blocks = 10000
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config, kv_cache_config=kv_cache_config,
        block_size=BLOCK, log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def output_for(so, sampled_map):
    req_ids = list(so.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[sampled_map.get(r, []) for r in req_ids],
        logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
    )


def main():
    sched = make_sync_scheduler()
    init_none_hash(sha256)
    sp = SamplingParams(ignore_eos=True, max_tokens=32)
    sp.update_from_generation_config({}, 50256)
    req = Request(
        request_id="r0",
        prompt_token_ids=list(range(100, 126)),  # 26-token prompt, as live
        sampling_params=sp, pooling_params=None, mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )
    sched.add_request(req)
    rid = req.request_id

    def counters():
        return (req.num_tokens, req.num_tokens_with_spec,
                req.num_computed_tokens)

    # step 1: prefill scheduled (in flight)
    s1 = sched.schedule()
    assert s1.num_scheduled_tokens[rid] == 26, s1.num_scheduled_tokens

    # step 2: nothing to schedule while prefill in flight
    s2 = sched.schedule()
    assert rid not in s2.num_scheduled_tokens

    # prefill output arrives: sampled t0
    sched.update_from_output(s1, output_for(s1, {rid: [1000]}))

    # step 3: decode of t0's successor scheduled (in flight)
    s3 = sched.schedule()
    assert s3.num_scheduled_tokens[rid] == 1, s3.num_scheduled_tokens

    # RACE: ngram drafts from the prefill step arrive only now, while
    # step 3 is still in flight (post_step take_draft_token_ids under PP).
    sched.update_draft_token_ids(
        DraftTokenIds(req_ids=[rid], draft_token_ids=[[1001, 1002]])
    )

    # step 4: without the fix this schedules a drafts-only step (new=2)
    # at positions whose input token hasn't been sampled yet.
    s4 = sched.schedule()
    n4 = s4.num_scheduled_tokens.get(rid, 0)
    print(f"step4 scheduled={n4} counters={counters()}")

    if n4 == 0:
        # fixed behavior: wait for the in-flight step
        sched.update_from_output(s3, output_for(s3, {rid: [1003]}))
        s5 = sched.schedule()
        n5 = s5.num_scheduled_tokens.get(rid, 0)
        print(f"step5 scheduled={n5} counters={counters()}")
        assert n5 > 0, "request must resume after in-flight output"
        print("PASS: drafts deferred until in-flight step completed")
        return 0

    # buggy behavior: reproduce the negative-scheduling crash
    sched.update_from_output(s3, output_for(s3, {rid: [1003]}))
    s5 = sched.schedule()
    n5 = s5.num_scheduled_tokens.get(rid, 0)
    print(f"step5 scheduled={n5} counters={counters()}")
    if n5 <= 0 and rid in s5.num_scheduled_tokens:
        print(f"BUG REPRODUCED: num_scheduled_tokens={n5}")
        return 1
    # even if not negative, scheduling drafts during flight is the defect
    print("BUG (variant): drafts-only step scheduled during in-flight step")
    return 1


if __name__ == "__main__":
    sys.exit(main())
