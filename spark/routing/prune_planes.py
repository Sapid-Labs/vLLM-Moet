#!/usr/bin/env python3
"""Build pool-pruned moe_w2 planes by row-selecting kept experts from an
existing prepacked planes dir (no re-quantization; pure disk copy).

Output meta gains a "keep" field (sorted original expert ids, row i of the
pruned planes = expert keep[i]); the runtime builds its id->row remap from it.

usage: prune_planes.py <src_planes_dir> <dst_planes_dir> <keep.json>
"""
import json
import os
import sys

import numpy as np

TAGS = ("planes13", "sc13", "planes2", "sc2")


def main():
    src, dst, keep_json = sys.argv[1], sys.argv[2], sys.argv[3]
    keep_all = json.load(open(keep_json))["keep"]
    os.makedirs(dst, exist_ok=True)
    metas = sorted(f for f in os.listdir(src) if f.endswith(".meta.json"))
    for mf in metas:
        li = int(mf.split("_")[1].split(".")[0])
        meta = json.load(open(os.path.join(src, mf)))
        stem = mf[: -len(".meta.json")]
        dmeta = os.path.join(dst, mf)
        if os.path.exists(dmeta):
            print(f"layer {li}: exists, skip", flush=True)
            continue
        keep = keep_all.get(str(li))
        if keep is None:  # layer not in keep list: copy through unpruned
            keep = list(range(meta["E"]))
        assert keep == sorted(keep) and keep[-1] < meta["E"]
        idx = np.asarray(keep)
        for tag in TAGS:
            arr = np.load(os.path.join(src, f"{stem}.{tag}.npy"), mmap_mode="r")
            np.save(os.path.join(dst, f"{stem}.{tag}.npy"), arr[idx])
        meta["E"] = len(keep)
        meta["keep"] = keep
        json.dump(meta, open(dmeta, "w"))
        print(f"layer {li}: kept {len(keep)}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
