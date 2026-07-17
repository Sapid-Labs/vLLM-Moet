#!/usr/bin/env python3
"""Assemble the combined magpie + weak-band training set on the worker (v3).

- Combined arrow = concat(magpie prepared [2000 rows], prepared-weak-sub [3200 rows]).
- Combined hidden_states/ = symlinks:
    magpie hs_j -> hs_j          (j = 0..1999; ~1988 present, gaps ok via --on-missing skip)
    weak   hs_i -> hs_{2000+i}   (i = 0..3199; hs_0 missing — skipped at extract)
  Trainer matches hidden states by ROW INDEX, so the offset must equal the
  magpie ROW count (2000), not its hs-file count.
"""
import glob
import os
import re

from datasets import load_from_disk, concatenate_datasets

BASE = os.path.expanduser("~/dspark-distill-data")
MAGPIE = f"{BASE}/prepared"
WEAK = f"{BASE}/prepared-weak-sub"
WEAK_HS = os.path.expanduser("~/dspark-hs-weak")
OUT = f"{BASE}/prepared-combined-v3"
OUT_HS = f"{OUT}/hidden_states"


def main():
    magpie = load_from_disk(MAGPIE)
    weak = load_from_disk(WEAK)
    n_mag = len(magpie)
    print(f"magpie rows={n_mag}  weak rows={len(weak)}")
    assert n_mag == 2000, f"expected magpie 2000 rows, got {n_mag}"

    combined = concatenate_datasets([magpie, weak])
    print(f"combined rows={len(combined)}")
    combined.save_to_disk(OUT)

    os.makedirs(OUT_HS, exist_ok=True)
    n_mag_hs = n_w_hs = 0
    for f in glob.glob(f"{MAGPIE}/hidden_states/hs_*.safetensors"):
        j = int(re.search(r"hs_(\d+)\.safetensors", f).group(1))
        link = f"{OUT_HS}/hs_{j}.safetensors"
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(f), link)
        n_mag_hs += 1
    for f in glob.glob(f"{WEAK_HS}/hs_*.safetensors"):
        i = int(re.search(r"hs_(\d+)\.safetensors", f).group(1))
        link = f"{OUT_HS}/hs_{n_mag + i}.safetensors"
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(f), link)
        n_w_hs += 1
    total = len(glob.glob(f"{OUT_HS}/hs_*.safetensors"))
    maxidx = max(int(re.search(r"hs_(\d+)", os.path.basename(l)).group(1))
                 for l in glob.glob(f"{OUT_HS}/hs_*.safetensors"))
    print(f"symlinked magpie hs={n_mag_hs}  weak hs={n_w_hs}  total={total}")
    print(f"max hs index={maxidx}  combined rows={len(combined)}  ok={maxidx < len(combined)}")


if __name__ == "__main__":
    main()
