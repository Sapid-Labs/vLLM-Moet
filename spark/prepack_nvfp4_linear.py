#!/usr/bin/env python3
"""Weight-only NVFP4 (W4A16) packer for GLM-5.2 DENSE/attention linears.

Part of the 20 tok/s goal (spark/GOAL.md): decode is bandwidth-bound and the
DENSE/attention FP8 reads are ~74% of the per-token bytes (MLA attention alone
~60%). This packs the attention + shared-expert linears from the block-FP8
checkpoint down to NVFP4 (4-bit e2m1 + per-16 e4m3 group scale + fp32 global),
~0.56 B/param vs 1.0 — the ~1.47x decode lever. Routed experts are UNTOUCHED
(they keep the 2-bit plane path).

Output is an OVERLAY model dir: every original shard is symlinked, the targeted
attention/shared tensors are replaced by NVFP4 tensors in new shards, and the
safetensors index is rewritten. vLLM's linear loader TP-shards the full NVFP4
tensors on load (group-16 along the input dim lands on clean byte+group
boundaries for both column- and row-parallel splits), so NO per-rank presharding
is needed here (unlike the expert planes).

Loader side (to be wired, see spark/NVFP4-DENSE.md): extend Fp8Config.
get_quant_method's LinearBase branch to return ModelOptNvFp4W4A16LinearMethod
for the NVFP4 prefixes. Tensor names/dtypes below match what that method's
create_weights expects (modelopt.py ~1307): weight uint8 [No, Ki//2],
weight_scale fp8-e4m3 [No, Ki//16], weight_scale_2 fp32 (amax/2688).

NVFP4 math validated on CPU (synthetic rel-L1 ~0.089; CT pack round-trip exact).

Usage:
  # validate error on one layer, write nothing:
  python spark/prepack_nvfp4_linear.py --verify --limit-layers 10
  # build the full overlay (attention + shared expert):
  python spark/prepack_nvfp4_linear.py --out $MODEL/nvfp4_dense_overlay
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# signed e2m1 levels (matches compressed_tensors FLOAT_TO_E2M1)
_E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.])
_LEVELS = torch.cat([-_E2M1.flip(0)[:-1], _E2M1])  # [-6..0..6], 15 distinct
FP8_MAX = 448.0
E2M1_MAX = 6.0
GROUP = 16
FP8_BLOCK = 128  # source block-fp8 granularity

# attention + shared-expert linears to quantize (o_proj is ~60% of the win).
# indexer.* and *_layernorm stay as-is (tiny / norm). q_a/kv_a are Replicated
# (not TP-sharded) but still quantizable.
TARGET_SUFFIXES = (
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)


def dequant_block_fp8(w_fp8: torch.Tensor, scale_inv: torch.Tensor,
                      block: int = FP8_BLOCK) -> torch.Tensor:
    """Block-FP8 -> fp32. scale_inv is one fp32 per [block x block] tile."""
    w = w_fp8.to(torch.float32)
    No, Ki = w.shape
    sr, sc = scale_inv.shape
    # expand the tile scales to full [No, Ki]
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, 0)[:No].repeat_interleave(block, 1)[:, :Ki]
    return w * s


def pack_fp4_to_uint8(codes_idx: torch.Tensor) -> torch.Tensor:
    """codes_idx: int tensor [No, Ki] of e2m1 code indices 0..15 (sign+mag in
    the standard e2m1 nibble encoding). Pack two per byte along Ki -> [No,Ki//2].
    """
    No, Ki = codes_idx.shape
    lo = codes_idx[:, 0::2].to(torch.uint8)
    hi = codes_idx[:, 1::2].to(torch.uint8)
    return (lo | (hi << 4)).contiguous()


# map signed level index (0..14 in _LEVELS) -> e2m1 4-bit nibble code.
# e2m1 nibble: bit3=sign, bits2-0=magnitude index into [0,.5,1,1.5,2,3,4,6].
def _level_to_nibble() -> torch.Tensor:
    mags = [0, .5, 1, 1.5, 2, 3, 4, 6.]
    out = []
    for v in _LEVELS.tolist():
        m = abs(v)
        mi = min(range(8), key=lambda i: abs(mags[i] - m))
        sign = 1 if v < 0 else 0
        # -0 collapses to +0 (nibble 0)
        out.append((sign << 3) | mi if not (m == 0) else 0)
    return torch.tensor(out, dtype=torch.long)


_NIBBLE = _level_to_nibble()


def quantize_nvfp4_w4a16(W: torch.Tensor, wg_override=None):
    """W: fp32 [No, Ki] (Ki % GROUP == 0). Returns:
      weight       uint8  [No, Ki//2]
      weight_scale e4m3   [No, Ki//GROUP]
      weight_scale_2 fp32 scalar (amax / (E2M1_MAX*FP8_MAX))

    wg_override: if given, use this global scale instead of W's own amax. Used to
    pack the two halves of a MergedColumnParallelLinear (shared-expert gate/up)
    against a SHARED weight_scale_2 -- the W4A16 loader collapses the per-shard
    weight_scale_2 via .max(), so mismatched scales corrupt the smaller-scale
    half (see modelopt.py process_weights_after_loading warning).
    """
    No, Ki = W.shape
    assert Ki % GROUP == 0, f"Ki={Ki} not divisible by {GROUP}"
    if wg_override is not None:
        wg = wg_override
    else:
        amax = W.abs().max()
        wg = (amax / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)      # global scale
    Wg = W.view(No, Ki // GROUP, GROUP)
    gamax = Wg.abs().amax(dim=-1, keepdim=True)                  # [No,G,1]
    gscale = (gamax / E2M1_MAX / wg).to(torch.float8_e4m3fn)     # e4m3 group scale
    deq = gscale.to(torch.float32) * wg                          # [No,G,1]
    q = (Wg / deq.clamp(min=1e-12)).clamp(-E2M1_MAX, E2M1_MAX)
    idx = (q.unsqueeze(-1) - _LEVELS).abs().argmin(-1)           # nearest level
    nibble = _NIBBLE[idx].view(No, Ki)                           # e2m1 nibble codes
    weight = pack_fp4_to_uint8(nibble)
    weight_scale = gscale.view(No, Ki // GROUP)
    # .clone() so a shared wg_override doesn't alias across gate/up (safetensors
    # save_file rejects tensors that share storage).
    weight_scale_2 = wg.to(torch.float32).reshape(()).clone()
    # error for --verify
    Wdq = (_LEVELS[idx] * deq).view(No, Ki)
    rel = ((Wdq - W).abs().mean() / W.abs().mean().clamp(min=1e-9)).item()
    return weight, weight_scale, weight_scale_2, rel


def iter_targets(index, limit_layers=None, suffixes=TARGET_SUFFIXES):
    for name, shard in index["weight_map"].items():
        if not name.endswith(".weight"):
            continue
        if not any(name.endswith(suf) for suf in suffixes):
            continue
        if limit_layers is not None:
            # name like model.layers.10.self_attn.o_proj.weight
            try:
                L = int(name.split(".layers.")[1].split(".")[0])
            except Exception:
                continue
            if L not in limit_layers:
                continue
        yield name, shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/hf/GLM-5.2-FP8"))
    ap.add_argument("--out", default=None, help="overlay dir to create")
    ap.add_argument("--verify", action="store_true",
                    help="report rel-L1 error only, write nothing")
    ap.add_argument("--limit-layers", default=None,
                    help="comma list of layer indices (for testing)")
    ap.add_argument("--shard-tensors", type=int, default=64,
                    help="tensors per output safetensors shard")
    ap.add_argument("--targets", default=None,
                    help="comma module basenames (e.g. o_proj,q_b_proj,kv_b_proj) "
                         "to restrict which linears get NVFP4; default = all "
                         "attention+shared. Big-3 (o_proj,q_b_proj,kv_b_proj) "
                         "avoids the merged-column load path.")
    args = ap.parse_args()

    lim = None
    if args.limit_layers:
        lim = {int(x) for x in args.limit_layers.split(",")}

    suffixes = TARGET_SUFFIXES
    if args.targets:
        want = {t.strip() for t in args.targets.split(",") if t.strip()}
        suffixes = tuple(s for s in TARGET_SUFFIXES
                         if s.rsplit(".", 2)[-2] in want)
        print(f"# restricted targets -> {suffixes}")

    idx = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))
    targets = list(iter_targets(idx, lim, suffixes))
    print(f"# {len(targets)} target tensors "
          f"({'verify' if args.verify else 'pack'})", flush=True)

    # open handles lazily per shard
    handles = {}

    def get(name, shard):
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.model, shard),
                                       framework="pt", device="cpu")
        return handles[shard].get_tensor(name)

    # Shared global scale for MergedColumnParallelLinear pairs. The W4A16 loader
    # keeps ONE weight_scale_2 per module (per-shard entries collapsed via .max(),
    # see modelopt.py process_weights_after_loading), so the halves of a merged
    # linear MUST be packed against a COMMON global scale -- otherwise the
    # smaller-scale half has its per-group scales computed against its own wg but
    # dequantized with the other's, i.e. a straight (wg_other/wg_own) error in the
    # weights. Measured disparities on GLM-5.2: gate/up 1.07-1.69x (mild), but
    # q_a/kv_a up to 15.9x -- catastrophic, and q_a feeds the whole query latent.
    # This is the likely root cause of the FULL-cut degradation (big-3 is clean at
    # the same rel-L1 0.09 precisely because none of big-3 is merged).
    MERGE_GROUPS = (
        # (module infix, checkpoint members fused into one merged linear)
        (".mlp.shared_experts.", ("gate_proj.weight", "up_proj.weight")),      # -> gate_up_proj
        (".self_attn.", ("q_a_proj.weight", "kv_a_proj_with_mqa.weight")),     # -> fused_qkv_a_proj
    )

    def merge_key(nm):
        for infix, members in MERGE_GROUPS:
            if infix in nm:
                for m in members:
                    if nm.endswith(m):
                        return nm[:-len(m)], members
        return None, None

    targ_names = {n for n, _ in targets}
    shared_wg = {}
    for name, shard in targets:
        k, members = merge_key(name)
        if k is None or k in shared_wg:
            continue
        pair = [k + m for m in members]
        # only share when BOTH halves are actually being packed
        if not all(p in targ_names for p in pair):
            continue
        amax = 0.0
        for p in pair:
            psc = p[:-len(".weight")] + ".weight_scale_inv"
            Wf = dequant_block_fp8(get(p, idx["weight_map"][p]),
                                   get(psc, idx["weight_map"][psc]))
            amax = max(amax, float(Wf.abs().max()))
        shared_wg[k] = torch.tensor(amax / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
    if shared_wg:
        print(f"# shared global scale for {len(shared_wg)} merged pairs "
              f"(gate_up / fused_qkv_a)")

    errs = []
    new_tensors = {}   # name -> tensor (for output shards)
    orig_bytes = 0
    new_bytes = 0
    for name, shard in targets:
        w = get(name, shard)
        sname = name[:-len(".weight")] + ".weight_scale_inv"
        sc = get(sname, idx["weight_map"][sname])
        Wf = dequant_block_fp8(w, sc)
        wg_ov = shared_wg.get(merge_key(name)[0])
        weight, wscale, wscale2, rel = quantize_nvfp4_w4a16(Wf, wg_override=wg_ov)
        errs.append((name, rel))
        orig_bytes += w.numel() + sc.numel() * 4
        new_bytes += weight.numel() + wscale.numel() + 4
        if not args.verify:
            base = name[:-len(".weight")]
            new_tensors[base + ".weight"] = weight
            new_tensors[base + ".weight_scale"] = wscale
            new_tensors[base + ".weight_scale_2"] = wscale2
        print(f"  {name:55s} rel-L1={rel:.4f} "
              f"{tuple(w.shape)}->uint8{tuple(weight.shape)}", flush=True)

    import statistics
    rels = [r for _, r in errs]
    print(f"\n# rel-L1 err: mean={statistics.mean(rels):.4f} "
          f"max={max(rels):.4f} (n={len(rels)})")
    print(f"# byte change on targeted tensors: {orig_bytes/1e9:.2f} GB -> "
          f"{new_bytes/1e9:.2f} GB ({new_bytes/max(orig_bytes,1):.2f}x)")

    if args.verify:
        return

    # ---- write overlay dir: symlink originals, add NVFP4 shards, rewrite index ----
    out = args.out
    os.makedirs(out, exist_ok=True)
    replaced = set()
    for name, _ in targets:
        base = name[:-len(".weight")]
        replaced.update({base + ".weight", base + ".weight_scale_inv"})
    # symlink every original shard + aux files
    for f in os.listdir(args.model):
        if f == "model.safetensors.index.json":
            continue
        src = os.path.join(args.model, f)
        dst = os.path.join(out, f)
        if not os.path.exists(dst) and os.path.isfile(src):
            os.symlink(src, dst)
    # write new NVFP4 shards
    items = list(new_tensors.items())
    new_map = {}
    for i in range(0, len(items), args.shard_tensors * 3):
        chunk = dict(items[i:i + args.shard_tensors * 3])
        shard = f"nvfp4-dense-{i//(args.shard_tensors*3):04d}.safetensors"
        save_file(chunk, os.path.join(out, shard))
        for k in chunk:
            new_map[k] = shard
    # rewrite index: drop replaced fp8 entries, add NVFP4 entries
    new_index = {"metadata": idx.get("metadata", {}), "weight_map": {}}
    for k, v in idx["weight_map"].items():
        if k in replaced:
            continue
        new_index["weight_map"][k] = v
    new_index["weight_map"].update(new_map)
    json.dump(new_index, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    print(f"\n# wrote overlay -> {out}  ({len(new_map)} new tensors, "
          f"{len(replaced)} fp8 tensors dropped)")
    print("# NOTE: also add the nvfp4 prefix list to config.json + wire the "
          "Fp8Config loader hook (see spark/NVFP4-DENSE.md) before serving.")


if __name__ == "__main__":
    main()
