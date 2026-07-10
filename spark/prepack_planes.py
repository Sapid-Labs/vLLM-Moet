#!/usr/bin/env python3
"""Offline 2-bit plane prepack for vLLM-Moet on unified-memory hosts.

Converts a DeepSeek-V4-Flash-style FP4 (mxfp4) checkpoint's routed experts
to the moe_w2 fragment-major 2-bit planes ONCE, on disk. The serve-time
loader (VLLM_MOE_W2_PREPACKED_DIR) then reads planes directly — no host
staging, no per-layer GPU requant, no transient churn: on a 121 GiB GB10
the in-process conversion's memory overhead was the difference between
loading and OOM (see spark/README.md).

Uses the venv's own moe_w2_planes pack functions, so output is bit-identical
to what build_layer_planes produces at serve time.

usage: prepack_planes.py --model ~/models/hf/DeepSeek-V4-Flash [--out DIR]
Restartable: layers with existing outputs are skipped.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_codes,
    pack_fragment_major,
    pack_scales,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    out = args.out or os.path.join(model, "moe_w2_planes")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")

    wm = json.load(open(os.path.join(model, "model.safetensors.index.json"))
                   )["weight_map"]
    # layer -> expert -> {w1,w2,w3,(scales)} -> (file, tensor_name)
    layers: dict[int, dict[int, dict[str, str]]] = {}
    files_of: dict[str, set] = {}
    pat = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$")
    for name, f in wm.items():
        m = pat.match(name)
        if not m:
            continue
        li, ei, w, kind = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        layers.setdefault(li, {}).setdefault(ei, {})[f"{w}.{kind}"] = name
        files_of.setdefault(name, set()).add(f)

    file_of = {n: next(iter(fs)) for n, fs in files_of.items()}
    handles: dict[str, "safe_open"] = {}

    def get(name):
        f = file_of[name]
        if f not in handles:
            # keep at most 2 shard handles open (mmap)
            while len(handles) >= 2:
                handles.pop(next(iter(handles)))
            handles[f] = safe_open(os.path.join(model, f), framework="pt")
        t = handles[f].get_tensor(name)
        # checkpoint dtypes are I8 codes / F8_E8M0 scales — the pack
        # pipeline consumes raw bytes (bit-preserving view, like the staged
        # uint8 params in the serve-time path)
        return t.view(torch.uint8)

    for li in sorted(layers):
        dst = os.path.join(out, f"layer_{li:03d}")
        if os.path.exists(dst + ".meta.json"):
            print(f"layer {li}: exists, skip", flush=True)
            continue
        experts = layers[li]
        E = len(experts)
        # shapes from expert 0
        w1 = get(experts[0]["w1.weight"])          # [I, H/2] u8
        I, H2 = w1.shape
        H = H2 * 2
        N13, K13, N2, K2 = 2 * I, H, H, I
        planes13 = np.empty((E, N13 * K13 // 4), dtype=np.uint8)
        sc13 = np.empty((E, N13 * K13 // 32), dtype=np.uint8)
        planes2 = np.empty((E, N2 * K2 // 4), dtype=np.uint8)
        sc2 = np.empty((E, N2 * K2 // 32), dtype=np.uint8)

        for e in sorted(experts):
            t = experts[e]
            w13 = torch.cat((get(t["w1.weight"]), get(t["w3.weight"])), 0)
            s13 = torch.cat((get(t["w1.scale"]), get(t["w3.scale"])), 0)
            w2 = get(t["w2.weight"])
            s2 = get(t["w2.scale"])
            wg, sg = w13.to(dev), s13.to(dev)
            planes13[e] = pack_fragment_major(mxfp4_to_codes(wg)).cpu().numpy()
            sc13[e] = pack_scales(sg).cpu().numpy()
            wg, sg = w2.to(dev), s2.to(dev)
            planes2[e] = pack_fragment_major(mxfp4_to_codes(wg)).cpu().numpy()
            sc2[e] = pack_scales(sg).cpu().numpy()
            del wg, sg

        for arr, tag in ((planes13, "planes13"), (sc13, "sc13"),
                         (planes2, "planes2"), (sc2, "sc2")):
            np.save(f"{dst}.{tag}.npy", arr)
        json.dump(dict(E=E, N13=N13, K13=K13, N2=N2, K2=K2),
                  open(dst + ".meta.json", "w"))
        torch.cuda.empty_cache()
        print(f"layer {li}: packed E={E} "
              f"({(planes13.nbytes+sc13.nbytes+planes2.nbytes+sc2.nbytes)>>20}"
              f" MB)", flush=True)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
