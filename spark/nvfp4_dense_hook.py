"""NVFP4 weight-only (W4A16) hook for GLM-5.2 DENSE/attention linears.

Drop-in module for site-packages/vllm/model_executor/layers/quantization/utils/.
Env-gated, prefix-keyed — mirrors the moe_w2 expert-plane hook so the two are
orthogonal: experts keep the 2-bit plane path (Fp8MoEMethod), attention + shared
expert get weight-only NVFP4 (MarlinNvFp4LinearKernel), the ~1.47x decode lever
(see spark/NVFP4-DENSE.md, spark/GOAL.md).

Integration (one insertion in Fp8Config.get_quant_method, LinearBase branch,
right AFTER the is_layer_skipped/UnquantizedLinearMethod check):

    from vllm.model_executor.layers.quantization.utils import nvfp4_dense_hook
    _m = nvfp4_dense_hook.maybe_linear_method(prefix)
    if _m is not None:
        return _m

Enable at serve time:  VLLM_NVFP4_DENSE=1  and serve from the overlay dir
(nvfp4_dense_overlay) whose config.json still carries the fp8 quantization_config
(experts) — the linears' NVFP4 weights come from the overlay shards.
"""
import os
import re

_CFG = None          # cached ModelOptNvFp4Config (W4A16)


def enabled() -> bool:
    return os.getenv("VLLM_NVFP4_DENSE", "0") == "1"


# Module-name suffixes to route to NVFP4, controlled by VLLM_NVFP4_TARGETS
# (comma list). Default = "big-3" non-merged attention linears only: o_proj,
# q_b_proj, kv_b_proj (~85% of the attention win, and they avoid vLLM's
# merged-column load path that fused linears gate_up_proj/fused_qkv_a_proj hit).
# To include the fused linears once their stacked NVFP4 load is fixed, add e.g.
# gate_up_proj,fused_qkv_a_proj,down_proj to VLLM_NVFP4_TARGETS.
_DEFAULT_TARGETS = "o_proj,q_b_proj,kv_b_proj"
_MERGED = {"gate_up_proj", "fused_qkv_a_proj", "qkv_proj", "gate_proj", "up_proj"}


def _targets():
    raw = os.getenv("VLLM_NVFP4_TARGETS", _DEFAULT_TARGETS)
    return [t.strip() for t in raw.split(",") if t.strip()]


# always exclude: indexer.* (tiny), any *_layernorm, router gate.
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def is_nvfp4_linear(prefix: str, num_hidden_layers: int | None = None) -> bool:
    if ".indexer." in prefix or "layernorm" in prefix:
        return False
    tgt = _targets()
    # match either self_attn.<name> or shared_experts.<name>
    if not any(prefix.endswith(".self_attn." + t)
               or prefix.endswith(".mlp.shared_experts." + t)
               for t in tgt):
        return False
    if num_hidden_layers is not None:
        m = _LAYER.search(prefix)
        if m and int(m.group(1)) >= num_hidden_layers:
            return False   # MTP drafter stays full precision
    # never quantize the indexer or any norm (defensive; regexes already exclude)
    if ".indexer." in prefix or "layernorm" in prefix:
        return False
    return True


def _get_cfg():
    global _CFG
    if _CFG is None:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
        _CFG = ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
    return _CFG


def maybe_linear_method(prefix: str):
    """Return a weight-only NVFP4 LinearMethod for a targeted prefix, else None.

    Safe to call for every linear: returns None unless enabled AND the prefix is
    an attention/shared-expert projection, so all other linears fall through to
    the normal FP8 path.
    """
    if not enabled() or not is_nvfp4_linear(prefix):
        return None
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4W4A16LinearMethod,
    )
    return ModelOptNvFp4W4A16LinearMethod(_get_cfg())
