#!/usr/bin/env python3
"""Offline 2-bit plane prepack for vLLM-Moet on unified-memory hosts.

Converts a checkpoint's routed experts to the moe_w2 fragment-major 2-bit
planes ONCE, on disk. The serve-time loader (VLLM_MOE_W2_PREPACKED_DIR) then
reads planes directly — no host staging, no per-layer GPU requant, no
transient churn: on a 121 GiB GB10 the in-process conversion's memory
overhead was the difference between loading and OOM (see spark/README.md).

Two checkpoint flavors, auto-detected from the index:
  mxfp4 : DS4-Flash-style  layers.L.ffn.experts.E.{w1,w2,w3}.{weight,scale}
  fp8   : GLM-5.2-FP8-style model.layers.L.mlp.experts.E.
          {gate_proj,up_proj,down_proj}.{weight,weight_scale_inv}

Uses the venv's own moe_w2_planes pack functions, so output is bit-identical
to what build_layer_planes{,_fp8} produce at serve time.

usage: prepack_planes.py --model DIR [--out DIR] [--layers 3-40]
Restartable: layers with existing outputs are skipped; layers whose shard
files are not yet downloaded are skipped with a notice (rerun later).
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
    fp8_block_to_codes_scales,
    mxfp4_to_codes,
    pack_fragment_major,
    pack_scales,
)

PAT_MXFP4 = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$")
PAT_FP8 = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$")
ROLE = {"w1": "gate", "w3": "up", "w2": "down",
        "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", default=None,
                    help="inclusive range like 3-40 (default: all)")
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    out = args.out or os.path.join(model, "moe_w2_planes")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")

    wm = json.load(open(os.path.join(model, "model.safetensors.index.json"))
                   )["weight_map"]
    fmt = "fp8" if any(PAT_FP8.match(n) for n in wm) else "mxfp4"
    pat = PAT_FP8 if fmt == "fp8" else PAT_MXFP4
    print(f"checkpoint format: {fmt}", flush=True)

    layers: dict[int, dict[int, dict[str, str]]] = {}
    file_of: dict[str, str] = {}
    for name, f in wm.items():
        m = pat.match(name)
        if not m:
            continue
        li, ei, w, kind = (int(m.group(1)), int(m.group(2)),
                           ROLE[m.group(3)], m.group(4))
        kind = "scale" if kind != "weight" else "weight"
        layers.setdefault(li, {}).setdefault(ei, {})[f"{w}.{kind}"] = name
        file_of[name] = f

    lsel = None
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        lsel = range(lo, hi + 1)

    handles: dict[str, object] = {}

    def get(name, as_bytes):
        f = file_of[name]
        if f not in handles:
            while len(handles) >= 2:
                handles.pop(next(iter(handles)))
            handles[f] = safe_open(os.path.join(model, f), framework="pt")
        t = handles[f].get_tensor(name)
        return t.view(torch.uint8) if as_bytes else t

    for li in sorted(layers):
        if lsel is not None and li not in lsel:
            continue
        dst = os.path.join(out, f"layer_{li:03d}")
        if os.path.exists(dst + ".meta.json"):
            print(f"layer {li}: exists, skip", flush=True)
            continue
        experts = layers[li]
        needed_files = {file_of[n] for t in experts.values()
                        for n in t.values()}
        missing = [f for f in needed_files
                   if not os.path.exists(os.path.join(model, f))]
        if missing:
            print(f"layer {li}: shards not downloaded yet "
                  f"({missing[0]}...), skip", flush=True)
            continue
        E = len(experts)
        as_bytes = fmt == "mxfp4"

        def expert_w13_w2(t):
            w13 = torch.cat((get(t["gate.weight"], as_bytes),
                             get(t["up.weight"], as_bytes)), 0)
            s13 = torch.cat((get(t["gate.scale"], as_bytes),
                             get(t["up.scale"], as_bytes)), 0)
            return (w13, s13, get(t["down.weight"], as_bytes),
                    get(t["down.scale"], as_bytes))

        w13, _, w2, _ = expert_w13_w2(experts[min(experts)])
        N13 = w13.shape[0]
        K13 = w13.shape[1] * (2 if fmt == "mxfp4" else 1)
        N2 = w2.shape[0]
        K2 = w2.shape[1] * (2 if fmt == "mxfp4" else 1)
        planes13 = np.empty((E, N13 * K13 // 4), dtype=np.uint8)
        sc13 = np.empty((E, N13 * K13 // 32), dtype=np.uint8)
        planes2 = np.empty((E, N2 * K2 // 4), dtype=np.uint8)
        sc2 = np.empty((E, N2 * K2 // 32), dtype=np.uint8)

        for e in sorted(experts):
            w13, s13, w2, s2 = expert_w13_w2(experts[e])
            for w, s, pl, sc in ((w13, s13, planes13, sc13),
                                 (w2, s2, planes2, sc2)):
                wg, sg = w.to(dev), s.to(dev)
                if fmt == "fp8":
                    codes, sbytes, _ = fp8_block_to_codes_scales(wg, sg)
                else:
                    codes, sbytes = mxfp4_to_codes(wg), sg
                pl[e] = pack_fragment_major(codes).cpu().numpy()
                sc[e] = pack_scales(sbytes).cpu().numpy()
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
