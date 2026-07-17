#!/usr/bin/env python3
"""Evict large data files from page cache via posix_fadvise(DONTNEED) — no sudo.
Frees cache so the model planes can stay resident. Owner-only, safe."""
import os, glob, sys

DIRS = [
    os.path.expanduser("~/dspark-distill-data/prepared/hidden_states"),
    os.path.expanduser("~/dspark-distill-data/prepared-combined/hidden_states"),
    os.path.expanduser("~/dspark-hs-reasoning-in"),
    os.path.expanduser("~/dspark-distill-data/openhermes-src"),
]
freed = 0
for d in DIRS:
    for f in glob.glob(os.path.join(d, "*")):
        if not os.path.isfile(f):
            continue
        try:
            fd = os.open(f, os.O_RDONLY)
            sz = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            freed += sz
        except Exception as e:
            print("skip", f, e, file=sys.stderr)
print(f"fadvise DONTNEED issued, ~{freed/1e9:.0f} GB of file pages hinted for eviction")
