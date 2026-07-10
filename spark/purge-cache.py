#!/usr/bin/env python3
"""Evict page cache for model files (no root): posix_fadvise(DONTNEED) every
file under the given dirs. vLLM's startup check wants free >= util*total,
and cudaMalloc can't reclaim page cache on GB10 — purge before launching."""
import os
import sys

for root in sys.argv[1:]:
    n = b = 0
    for dirpath, _, files in os.walk(os.path.expanduser(root)):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                n += 1
                b += sz
            except OSError:
                pass
    print(f"{root}: fadvised {n} files ({b >> 30} GiB)")
