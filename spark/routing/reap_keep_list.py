#!/usr/bin/env python3
"""Build a per-layer expert keep-list from a REAP saliency observer report.

Saliency-based selection (keep the top-N experts per layer by REAP saliency),
the quality-preserving replacement for the frequency-coldest prune in
make_keep_list.py. Consumes the merged observer report produced by
`scripts/merge_observer_states.py` (or a single-node run's report): a dict
{layer_idx: {"reap": tensor[num_experts], ...}}, where `reap` is the tracked
mean of (expert_output_norm * router_weight) per expert. Higher = more salient
→ keep.

Output matches spark/routing/keep208.json exactly, so it is a drop-in input to
`spark/routing/prune_planes.py`:
    {"selection": "...", "experts_kept": K, "keep": {"<layer>": [sorted ids], ...}}
Layer keys are absolute decoder-layer indices (the observer state's block ids);
only layers carrying a `reap` tensor (the MoE layers) are emitted.

usage: reap_keep_list.py report.pt out.json [--keep 208] [--compare keep208.json]
"""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="merged observer report .pt (trackers->means)")
    ap.add_argument("out", help="output keep-list json")
    ap.add_argument("--keep", type=int, default=208,
                    help="experts to keep per MoE layer (default 208; prunes the rest)")
    ap.add_argument("--compare", default=None,
                    help="optional keep208.json to diff against (e.g. the frequency prune)")
    a = ap.parse_args()

    report = torch.load(a.report, weights_only=False)

    keep = {}
    total_experts = None
    for layer, st in report.items():
        if not isinstance(st, dict) or "reap" not in st:
            continue  # dense layer / no MoE
        sal = st["reap"]
        if not torch.is_tensor(sal):
            sal = torch.as_tensor(sal)
        sal = sal.float().flatten()
        E = sal.numel()
        total_experts = E if total_experts is None else total_experts
        assert E == total_experts, f"layer {layer}: {E} experts != {total_experts}"
        assert a.keep <= E, f"--keep {a.keep} > {E} experts in layer {layer}"
        # Highest-saliency experts are kept; ties broken by index for determinism.
        top = torch.topk(sal, a.keep, largest=True, sorted=False).indices
        keep[str(int(layer))] = sorted(int(i) for i in top.tolist())

    if not keep:
        raise SystemExit("no MoE layers with a 'reap' tensor found in the report")

    meta = {
        "prune_per_layer": total_experts - a.keep,
        "experts_kept": a.keep,
        "total_experts": total_experts,
        "selection": "REAP-saliency (top-k by mean expert_norm*router_weight)",
        "source_report": a.report,
        "keep": keep,
    }
    json.dump(meta, open(a.out, "w"))
    n_layers = len(keep)
    print(f"wrote {a.out}: keep {a.keep}/{total_experts} per layer "
          f"across {n_layers} MoE layers")

    if a.compare:
        other = json.load(open(a.compare))["keep"]
        common = sorted(set(keep) & set(other))
        if common:
            diffs = [len(set(keep[l]) ^ set(other[l])) // 2 for l in common]
            import statistics
            print(f"vs {a.compare}: mean {statistics.mean(diffs):.1f} experts "
                  f"changed/layer (max {max(diffs)}, min {min(diffs)}) over "
                  f"{len(common)} shared layers")


if __name__ == "__main__":
    main()
