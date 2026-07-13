#!/usr/bin/env python3
"""Standalone TP-allreduce latency at the decode message size, over the RoCE
fabric — to attribute the comm slice of the ~67 ms/token decode budget.

At decode, each of the ~78 attn layers + ~75 MoE layers does a TP2 allreduce of
[num_tokens, hidden] activations. Single-stream num_tokens=1, hidden=6144, bf16
=> ~12 KB: pure small-message latency regime. This times that exact allreduce
and reports the per-token comm estimate (x layers x reduces/layer).

Launch (from spark/profile_decode.sh, uses the serve NCCL env):
  rank0 (this node):  MASTER_ADDR=192.168.100.1 RANK=0 WORLD_SIZE=2 python allreduce_probe.py
  rank1 (peer, ssh):  MASTER_ADDR=192.168.100.1 RANK=1 WORLD_SIZE=2 python allreduce_probe.py
"""
import os
import torch
import torch.distributed as dist

HIDDEN = int(os.getenv("PROBE_HIDDEN", "6144"))
ITERS = int(os.getenv("PROBE_ITERS", "300"))
# GLM-5.2: 78 attn allreduces + 75 MoE allreduces per token (approx; dense
# layers also reduce). Report per-reduce and a per-token projection.
REDUCES_PER_TOKEN = int(os.getenv("PROBE_REDUCES", "153"))


def main():
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(0)
    x = torch.ones(1, HIDDEN, dtype=torch.bfloat16, device="cuda")

    for _ in range(20):  # warmup
        dist.all_reduce(x)
    torch.cuda.synchronize()

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        dist.all_reduce(x)
    e1.record()
    torch.cuda.synchronize()
    per = e0.elapsed_time(e1) / ITERS  # ms per allreduce

    if rank == 0:
        proj = per * REDUCES_PER_TOKEN
        print(f"[allreduce] size={HIDDEN}x bf16 (~{HIDDEN*2} B)  "
              f"per-reduce {per:.3f} ms  "
              f"x{REDUCES_PER_TOKEN} reduces/token = {proj:.1f} ms/token comm")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
