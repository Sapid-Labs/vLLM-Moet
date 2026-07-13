#!/usr/bin/env python3
"""Build a per-layer expert keep-list from captured routing traffic counts.

Frequency-only selection (coldest-N pruned per layer) — intended for the
RESIDENCY SMOKE TEST, not the shipped artifact (REAP saliency should pick
the real set; see handoff 02).

usage: make_keep_list.py counts_layer_expert.npy out.json [--prune 48]
"""
import argparse
import json

import numpy as np

DENSE_LAYERS = 3  # counts rows are MoE layers only; absolute layer = row + 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("out")
    ap.add_argument("--prune", type=int, default=48)
    a = ap.parse_args()

    counts = np.load(a.counts)  # (moe_layers, 256)
    layers, E = counts.shape
    keep = {}
    lost = 0
    for l in range(layers):
        order = np.argsort(counts[l], kind="stable")  # coldest first
        kept = np.sort(order[a.prune:])
        keep[str(l + DENSE_LAYERS)] = kept.tolist()
        lost += counts[l, order[:a.prune]].sum()
    meta = {
        "prune_per_layer": a.prune,
        "experts_kept": E - a.prune,
        "selection": "frequency-coldest (smoke test, NOT REAP saliency)",
        "traffic_lost_frac": float(lost / counts.sum()),
        "keep": keep,
    }
    json.dump(meta, open(a.out, "w"))
    print(f"wrote {a.out}: keep {E - a.prune}/{E} per layer, "
          f"slot-traffic lost {lost / counts.sum() * 100:.3f}%")


if __name__ == "__main__":
    main()
