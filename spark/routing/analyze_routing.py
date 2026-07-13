#!/usr/bin/env python3
"""Build the per-(layer,expert) traffic CDF from captured routing arrays and
evaluate pool-prune candidates: for pruning the N coldest experts per layer,
report plane size and the share of routed traffic lost."""
import json
from pathlib import Path

import numpy as np

CAPDIR = Path(__file__).parent / "routing_capture"
NUM_EXPERTS = 256
PLANE_GB_PER_RANK = 97.0  # current tp2 planes, 256 experts
DENSE_LAYERS = 3  # layers 0-2 are dense; capture rows are all-zero placeholders

def main():
    files = sorted(CAPDIR.glob("*.npy"))
    if not files:
        raise SystemExit("no capture files")
    counts = None  # (layers, experts)
    total_tokens = 0
    per_domain = {}
    for f in files:
        arr = np.load(f)[:, DENSE_LAYERS:, :]  # (tokens-1, moe_layers, top_k)
        t, layers, k = arr.shape
        if counts is None:
            counts = np.zeros((layers, NUM_EXPERTS), dtype=np.int64)
        c = np.zeros((layers, NUM_EXPERTS), dtype=np.int64)
        for l in range(layers):
            c[l] = np.bincount(arr[:, l, :].ravel(), minlength=NUM_EXPERTS)
        counts += c
        total_tokens += t
        per_domain[f.stem] = c
        print(f"{f.stem}: {t} tokens, layers={layers}, k={k}")

    layers = counts.shape[0]
    total_slots = counts.sum()
    print(f"\ntotal tokens (prompt+gen): {total_tokens}, routed slots: {total_slots}")

    # Global CDF: sort all (layer,expert) cells by traffic desc
    flat = np.sort(counts.ravel())[::-1]
    cdf = np.cumsum(flat) / total_slots
    for frac in (0.5, 0.75, 0.9, 0.95, 0.99):
        n = int(np.searchsorted(cdf, frac)) + 1
        print(f"  {frac*100:.0f}% of traffic served by {n}/{counts.size} "
              f"(layer,expert) cells ({n/counts.size*100:.1f}%)")

    # Never-hit experts
    never = int((counts == 0).sum())
    print(f"  never-routed (layer,expert) cells: {never}/{counts.size} "
          f"({never/counts.size*100:.1f}%)")

    # Per-layer prune sweep: drop the N coldest experts in EACH layer
    print(f"\nprune sweep (coldest-N-per-layer, {layers} layers, "
          f"{NUM_EXPERTS} experts/layer):")
    print(f"{'N':>4} {'keep':>5} {'plane GB/rank':>14} {'traffic lost %':>15} "
          f"{'worst-layer lost %':>19}")
    for n_prune in (16, 32, 40, 48, 56, 64, 72, 80, 96):
        keep = NUM_EXPERTS - n_prune
        sorted_layer = np.sort(counts, axis=1)  # ascending per layer
        lost_per_layer = sorted_layer[:, :n_prune].sum(axis=1)
        lost = lost_per_layer.sum() / total_slots * 100
        layer_tot = counts.sum(axis=1)
        worst = (lost_per_layer / np.maximum(layer_tot, 1)).max() * 100
        gb = PLANE_GB_PER_RANK * keep / NUM_EXPERTS
        print(f"{n_prune:>4} {keep:>5} {gb:>14.1f} {lost:>15.3f} {worst:>19.2f}")

    # Cross-domain stability of the cold set (would a per-domain prune differ?)
    print("\ncold-set overlap across domains (N=56 coldest per layer, "
          "vs global cold set):")
    n_prune = 56
    global_cold = np.argsort(counts, axis=1)[:, :n_prune]
    global_cold_sets = [set(global_cold[l]) for l in range(layers)]
    for name, c in sorted(per_domain.items()):
        dom_cold = np.argsort(c, axis=1)[:, :n_prune]
        # traffic this domain sends to the GLOBAL cold set
        lost = sum(c[l, list(global_cold_sets[l])].sum() for l in range(layers))
        print(f"  {name:>20}: traffic to global-cold-set = "
              f"{lost / max(c.sum(), 1) * 100:.3f}%")

    np.save(CAPDIR / "counts_layer_expert.npy", counts)
    print(f"\nsaved counts -> {CAPDIR/'counts_layer_expert.npy'}")

if __name__ == "__main__":
    main()
