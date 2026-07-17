#!/usr/bin/env python3
"""Assemble the combined magpie + reasoning training set on the worker.

- Combined arrow = concat(magpie prepared [2000 rows], reasoning-sub [3200 rows]).
- Combined hidden_states/ = symlinks:
    magpie   hs_j            -> hs_j            (j = 0..1999; ~1988 present, 12 gaps ok)
    reasoning hs_i           -> hs_{2000+i}     (i = 0..3199)
  Trainer matches hidden states by ROW INDEX, so the offset must equal the
  magpie row count (2000), NOT the magpie hs-file count.
"""
import glob, os, re, sys
from datasets import load_from_disk, concatenate_datasets

BASE = os.path.expanduser("~/dspark-distill-data")
MAGPIE = f"{BASE}/prepared"
REASON = f"{BASE}/prepared-reasoning-sub"
REASON_HS = os.path.expanduser("~/dspark-hs-reasoning-in")
OUT = f"{BASE}/prepared-combined"
OUT_HS = f"{OUT}/hidden_states"

def main():
    magpie = load_from_disk(MAGPIE)
    reason = load_from_disk(REASON)
    n_mag = len(magpie)
    print(f"magpie rows={n_mag}  reasoning rows={len(reason)}")
    assert n_mag == 2000, f"expected magpie 2000 rows, got {n_mag}"

    combined = concatenate_datasets([magpie, reason])
    print(f"combined rows={len(combined)}")
    combined.save_to_disk(OUT)

    os.makedirs(OUT_HS, exist_ok=True)
    # magpie hs: copy index unchanged
    n_mag_hs = 0
    for f in glob.glob(f"{MAGPIE}/hidden_states/hs_*.safetensors"):
        j = int(re.search(r"hs_(\d+)\.safetensors", f).group(1))
        link = f"{OUT_HS}/hs_{j}.safetensors"
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(f), link)
        n_mag_hs += 1
    # reasoning hs: offset by n_mag (2000)
    n_r_hs = 0
    for f in glob.glob(f"{REASON_HS}/hs_*.safetensors"):
        i = int(re.search(r"hs_(\d+)\.safetensors", f).group(1))
        link = f"{OUT_HS}/hs_{n_mag + i}.safetensors"
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(f), link)
        n_r_hs += 1
    total_links = len(glob.glob(f"{OUT_HS}/hs_*.safetensors"))
    print(f"symlinked magpie hs={n_mag_hs}  reasoning hs={n_r_hs}  total_links={total_links}")
    # sanity: max index < combined rows
    maxidx = max(int(re.search(r"hs_(\d+)", os.path.basename(l)).group(1))
                 for l in glob.glob(f"{OUT_HS}/hs_*.safetensors"))
    print(f"max hs index={maxidx}  combined rows={len(combined)}  ok={maxidx < len(combined)}")

if __name__ == "__main__":
    main()
