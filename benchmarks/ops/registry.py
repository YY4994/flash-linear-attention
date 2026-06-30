# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Op registry, input factory, and shape configs for the unified benchmark system.

See ``benchmarks/ops/run.py`` docstring for full usage and how to register new ops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shape helpers: reusable callables  (B, T, H, D, **kw) -> tuple
# ---------------------------------------------------------------------------


def shape_BTHD(B, T, H, D, **kw):
    return (B, T, H, D)


def shape_BTH(B, T, H, D, **kw):
    return (B, T, H)


def shape_BTD(B, T, H, D, **kw):
    return (B, T, H * D)


def shape_H(B, T, H, D, **kw):
    return (H,)


def shape_HD(B, T, H, D, **kw):
    return (H, D)


def shape_D(B, T, H, D, **kw):
    return (D,)


def shape_LBTD(B, T, H, D, L=None, **kw):
    """AttnRes-style residuals stack: [L, B, T, D] where L is the number of residual sources."""
    if L is None:
        raise ValueError("shape_LBTD requires the 'L' shape config key")
    return (L, B, T, D)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

logsigmoid = F.logsigmoid


def sigmoid_transform(t):
    return t.sigmoid()


def logsigmoid_clamp(t):
    return F.logsigmoid(t).clamp_min(-5)


RWKV7_W_MIN = -0.6065306597126334


def rwkv7_w_transform(t):
    w = RWKV7_W_MIN * t.sigmoid()
    return w.clamp(min=RWKV7_W_MIN, max=-1e-6)


# ---------------------------------------------------------------------------
# TensorSpec: describes how to create one input tensor
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    """Specification for generating a single benchmark input tensor.

    Args:
        shape_fn:       (B, T, H, D, **kw) -> tuple of ints
        requires_grad:  whether the tensor needs gradients
        dtype:          'default' inherits from the benchmark, or 'float32'/'long'
        transform:      applied after randn, e.g. F.logsigmoid
    """
    shape_fn: Callable
    requires_grad: bool = True
    dtype: str = 'default'
    transform: Callable | None = None


# ---------------------------------------------------------------------------
# OpConfig: registry entry for one op
# ---------------------------------------------------------------------------


@dataclass
class OpConfig:
    """Registry entry describing how to benchmark a single op.

    Args:
        name (str):
            Display and registry name, such as `chunk_gla`.
        import_path (str):
            Python module path, such as `fla.ops.gla`.
        inputs (dict[str, TensorSpec]):
            Mapping from function argument names to tensor specs.
        func_name (str, Optional):
            Function attribute name to import when it differs from `name`.
            Default: None.
        extra_kwargs (dict[str, Any], Optional):
            Constant keyword arguments passed to the op. Default: `{}`.
        output_is_tuple (bool):
            Whether the op returns a tuple whose first item is the tensor used
            for `.backward()`. Default: `True`.
        skip_backward (bool):
            Whether to skip forward-backward benchmark mode. Default: `False`.
        post_init (Callable, Optional):
            Callback invoked as `post_init(inputs, B=B, T=T, H=H, D=D, **kw)`
            for custom input mutation. Default: None.
        category (str):
            Grouping label used in reports. Default: `''`.
        dim_constraints (dict, Optional):
            Shape constraints, such as `{'D': [64, 128]}`. Shapes that do not
            match are skipped. Default: None.
        default_shapes (dict[str, dict[str, int]], Optional):
            Per-op shape configs used instead of the global `SHAPE_CONFIGS`.
            This is useful when the op's shape semantics differ from `B/T/H/D`,
            for example when AttnRes uses an extra `L` residual-source axis.
            Default: None.
    """
    name: str
    import_path: str
    inputs: dict[str, TensorSpec]
    func_name: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    output_is_tuple: bool = True
    skip_backward: bool = False
    post_init: Callable | None = None
    category: str = ''
    dim_constraints: dict | None = None
    default_shapes: dict[str, dict[str, int]] | None = None


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, OpConfig] = {}


def register_op(config: OpConfig) -> None:
    _REGISTRY[config.name] = config


def get_op(name: str) -> OpConfig:
    if name not in _REGISTRY:
        raise KeyError(f"Op '{name}' not registered. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_ops() -> list[str]:
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Shape configs
# ---------------------------------------------------------------------------

SHAPE_CONFIGS = {
    'B1_T8192_H96_D128':  {'B': 1,  'T': 8192,  'H': 96, 'D': 128},
    'B2_T16384_H16_D128': {'B': 2,  'T': 16384, 'H': 16, 'D': 128},
    'B4_T2048_H16_D128':  {'B': 4,  'T': 2048,  'H': 16, 'D': 128},
    'B4_T4096_H64_D128':  {'B': 4,  'T': 4096,  'H': 64, 'D': 128},
    'B8_T2048_H32_D256':  {'B': 8,  'T': 2048,  'H': 32, 'D': 256},
    'B8_T1024_H8_D64':    {'B': 8,  'T': 1024,  'H': 8,  'D': 64},
}


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def generate_inputs(
    config: OpConfig,
    B: int, T: int, H: int, D: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = 'cuda',
    **extra_shape_kw,
) -> dict[str, torch.Tensor]:
    """Create input tensors for *config* at the given shape.

    Returns a dict mapping parameter names to tensors.
    Raises ValueError if dim_constraints are not satisfied (caller should skip).
    """
    # Check dim constraints
    if config.dim_constraints:
        shape_vals = {'B': B, 'T': T, 'H': H, 'D': D, **extra_shape_kw}
        for dim_name, allowed in config.dim_constraints.items():
            val = shape_vals.get(dim_name)
            if val is not None and val not in allowed:
                raise ValueError(
                    f"Op '{config.name}' requires {dim_name} in {allowed}, got {val}"
                )

    inputs = {}
    for param_name, spec in config.inputs.items():
        shape = spec.shape_fn(B, T, H, D, **extra_shape_kw)

        # Determine dtype
        if spec.dtype == 'default':
            tensor_dtype = dtype
        elif spec.dtype == 'float32':
            tensor_dtype = torch.float32
        elif spec.dtype == 'long':
            tensor_dtype = torch.long
        else:
            tensor_dtype = dtype

        if tensor_dtype == torch.long:
            tensor = torch.randint(0, 10, shape, dtype=tensor_dtype, device=device)
        else:
            tensor = torch.randn(shape, dtype=tensor_dtype, device=device)

        if spec.transform is not None:
            tensor = spec.transform(tensor)

        if spec.requires_grad and tensor.is_floating_point():
            tensor = tensor.requires_grad_(True)

        inputs[param_name] = tensor

    # Custom post-init mutation
    if config.post_init is not None:
        config.post_init(inputs, B=B, T=T, H=H, D=D, **extra_shape_kw)

    return inputs


# ===========================================================================
# Op registrations
# ===========================================================================

# --- A: Simple qkv (no extra inputs) ---

_simple_qkv = {
    'q': TensorSpec(shape_BTHD),
    'k': TensorSpec(shape_BTHD),
    'v': TensorSpec(shape_BTHD),
}

register_op(OpConfig(
    name='chunk_retention',
    import_path='fla.ops.retention',
    inputs={**_simple_qkv},
    category='simple_qkv',
))

register_op(OpConfig(
    name='chunk_linear_attn',
    import_path='fla.ops.linear_attn',
    inputs={**_simple_qkv},
    category='simple_qkv',
))

# --- B: +elem gate (g=[B,T,H,D] with logsigmoid_clamp) ---

register_op(OpConfig(
    name='chunk_gla',
    import_path='fla.ops.gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTHD, transform=logsigmoid_clamp),
    },
    category='elem_gate',
))

# --- C: +beta (beta=[B,T,H] with sigmoid) ---

register_op(OpConfig(
    name='chunk_delta_rule',
    import_path='fla.ops.delta_rule',
    inputs={
        **_simple_qkv,
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    category='beta',
))

# --- D: +gate + beta ---

register_op(OpConfig(
    name='chunk_gdn',
    import_path='fla.ops.gated_delta_rule',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    func_name='chunk_gated_delta_rule',
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='gate_beta',
))

register_op(OpConfig(
    name='chunk_kda',
    import_path='fla.ops.kda',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTHD, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True, 'safe_gate': True, 'lower_bound': -5},
    category='gate_beta',
))

# --- E: +head gate (g=[B,T,H] with logsigmoid) ---

register_op(OpConfig(
    name='chunk_simple_gla',
    import_path='fla.ops.simple_gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
    },
    category='head_gate',
))

# --- F: RWKV ---


def _rwkv7_post_init(inputs, B, T, H, D, **kw):
    """RWKV7 needs a/b to be initialized as small positive values."""
    with torch.no_grad():
        inputs['a'] = (torch.randn_like(inputs['a']) * 0.1).requires_grad_(True)
        inputs['b'] = (torch.randn_like(inputs['b']) * 0.1).requires_grad_(True)


register_op(OpConfig(
    name='chunk_rwkv6',
    import_path='fla.ops.rwkv6',
    inputs={
        'r': TensorSpec(shape_BTHD),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
        'w': TensorSpec(shape_BTHD, transform=logsigmoid),
        'u': TensorSpec(shape_HD, requires_grad=False),
    },
    category='rwkv',
))

register_op(OpConfig(
    name='chunk_rwkv7',
    import_path='fla.ops.rwkv7',
    inputs={
        'r': TensorSpec(shape_BTHD),
        'w': TensorSpec(shape_BTHD, transform=rwkv7_w_transform),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
        'a': TensorSpec(shape_BTHD),
        'b': TensorSpec(shape_BTHD),
    },
    extra_kwargs={'safe_gate': True, 'chunk_size': 64},
    post_init=_rwkv7_post_init,
    category='rwkv',
))

# --- H: Comba ---

register_op(OpConfig(
    name='chunk_comba',
    import_path='fla.ops.comba',
    inputs={
        **_simple_qkv,
        'p': TensorSpec(shape_BTHD),
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='comba',
))

# --- I: HGRN (x, g only, no qkv) ---

register_op(OpConfig(
    name='fused_recurrent_hgrn',
    import_path='fla.ops.hgrn',
    inputs={
        'x': TensorSpec(shape_BTD),
        'g': TensorSpec(shape_BTD, transform=logsigmoid),
    },
    category='hgrn',
))

# --- J: Generalized delta rule (DPLR) ---

register_op(OpConfig(
    name='chunk_dplr_delta_rule',
    import_path='fla.ops.generalized_delta_rule',
    inputs={
        **_simple_qkv,
        'a': TensorSpec(shape_BTHD),
        'b': TensorSpec(shape_BTHD),
        'gk': TensorSpec(shape_BTHD, transform=logsigmoid),
    },
    category='gen_delta',
))

# --- K: Lightning attention (needs layer_idx, num_layers) ---

register_op(OpConfig(
    name='chunk_lightning_attn',
    import_path='fla.ops.lightning_attn',
    inputs={**_simple_qkv},
    extra_kwargs={'layer_idx': 0, 'num_layers': 12},
    category='lightning',
))

# --- L: Attention baselines ---

register_op(OpConfig(
    name='parallel_attn',
    import_path='fla.ops.attn',
    inputs={**_simple_qkv},
    output_is_tuple=False,
    category='attn',
))


register_op(OpConfig(
    name='flash_attn',
    import_path='flash_attn',
    inputs={**_simple_qkv},
    func_name='flash_attn_func',
    extra_kwargs={'causal': True},
    output_is_tuple=False,
    category='flash_attn',
))

# --- M: layer-axis residual aggregation (AttnRes, mHC, ...) ---
# These ops attend / aggregate over an `L` axis of stacked residual sources.
# Inputs and shape sweeps are shared so future ops (mHC etc.) can reuse them.

_layer_default_shapes = {
    'L8_B1_T8K_D2K':   {'L': 8,  'B': 1, 'T': 8192,  'H': 1, 'D': 2048},
    'L8_B1_T32K_D2K':  {'L': 8,  'B': 1, 'T': 32768, 'H': 1, 'D': 2048},
    'L10_B1_T8K_D4K':  {'L': 10, 'B': 1, 'T': 8192,  'H': 1, 'D': 4096},
    'L10_B1_T32K_D4K': {'L': 10, 'B': 1, 'T': 32768, 'H': 1, 'D': 4096},
    'L32_B1_T8K_D2K':  {'L': 64, 'B': 1, 'T': 8192,  'H': 1, 'D': 8192},
}


_attnres_inputs = {
    'query': TensorSpec(shape_D),
    'residuals': TensorSpec(shape_LBTD),
    'rms_weight': TensorSpec(shape_D),
}

register_op(OpConfig(
    name='fused_attnres',
    import_path='fla.ops.attnres',
    inputs=_attnres_inputs,
    output_is_tuple=False,
    default_shapes=_layer_default_shapes,
    category='fused_attnres',
))

register_op(OpConfig(
    name='naive_attnres',
    import_path='fla.ops.attnres',
    inputs=_attnres_inputs,
    output_is_tuple=False,
    default_shapes=_layer_default_shapes,
    category='naive_attnres',
))

# --- N: NSA (Native Sparse Attention) ---
# NSA uses GQA with H kv heads and HQ = H * G query heads (G >= 16).
# The shape configs below use `H` for kv heads and `HQ` for query heads.


def shape_BTHD_HQ(B, T, H, D, HQ=None, **kw):
    """Shape for q/g tensors with query heads HQ potentially != kv heads H."""
    if HQ is not None:
        return (B, T, HQ, D)
    return (B, T, H, D)


def shape_BTH_HQ(B, T, H, D, HQ=None, **kw):
    """Shape for per-head scalars with query heads HQ."""
    if HQ is not None:
        return (B, T, HQ)
    return (B, T, H)

# _nsa_default_shapes = {
#     # ====================================================================
#     # Group 1: H=4 sequence length sweep — 观察瓶颈随 S 的变化
#     # (分析 Group 2: 保持 H=4, 改变长度, 看 latency 曲线趋势)
#     # ====================================================================
#     'H4_S4K':   {'B': 1, 'T': 4096,  'H': 4, 'D': 64,  'HQ': 64},
#     'H4_S8K':   {'B': 1, 'T': 8192,  'H': 4, 'D': 64,  'HQ': 64},
#     'H4_S16K':  {'B': 1, 'T': 16384, 'H': 4, 'D': 64,  'HQ': 64},  # ← 当前 NCU shape (Case E)
#     'H4_S32K':  {'B': 1, 'T': 32768, 'H': 4, 'D': 64,  'HQ': 64},

#     # ====================================================================
#     # Group 2: H=16 sequence length sweep — 论文主场景 (NSA 优势区域)
#     # Cases A-D: 短/中/长/极长上下文
#     # ====================================================================
#     'H16_S4K':  {'B': 1, 'T': 4096,  'H': 16, 'D': 64,  'HQ': 256},  # Case A: 短上下文  (G=16)
#     'H16_S8K':  {'B': 1, 'T': 8192,  'H': 16, 'D': 64,  'HQ': 256},  #              (G=16)
#     'H16_S16K': {'B': 1, 'T': 16384, 'H': 16, 'D': 64,  'HQ': 256},  # Case B: 主力  (G=16)
#     'H16_S32K': {'B': 1, 'T': 32768, 'H': 16, 'D': 64,  'HQ': 256},  # Case C: 长上下文 (G=16)
#     'H16_S64K': {'B': 1, 'T': 65536, 'H': 16, 'D': 64,  'HQ': 256},  # Case D: NSA 优势 (G=16)

#     # ====================================================================
#     # Group 3: H=32 (真实模型 head 数) — 长上下文拓展
#     # ====================================================================
#     'H32_S16K': {'B': 1, 'T': 16384, 'H': 32, 'D': 64,  'HQ': 512},  # Case F: 真实模型 (G=16)
#     'H32_S32K': {'B': 1, 'T': 32768, 'H': 32, 'D': 64,  'HQ': 512},  #             (G=16)
#     'H32_S64K': {'B': 1, 'T': 65536, 'H': 32, 'D': 64,  'HQ': 512},  #             (G=16)

#     # ====================================================================
#     # Group 4: Head dimension 对比 (S=16K, H=16)
#     # ====================================================================
#     'H16_D128': {'B': 1, 'T': 16384, 'H': 16, 'D': 128, 'HQ': 256},  # (G=16)

#     # ====================================================================
#     # Group 5: 多序列训练场景
#     # ====================================================================
#     'B2_H16':   {'B': 2, 'T': 16384, 'H': 16, 'D': 64,  'HQ': 256},  # (G=16)
# }

_nsa_default_shapes = {

    # ================================================================
    # Group 1: Small H regime
    # 目的:
    #   - 验证 FSA 思路
    #   - 低并行度
    #   - occupancy / warp scheduling 瓶颈
    #
    # G = HQ/H
    # ================================================================

    'H2_S16K': {
        'B':1,
        'T':16384,
        'H':2,
        'D':64,
        'HQ':32,
    },

    'H4_S16K': {
        'B':1,
        'T':16384,
        'H':4,
        'D':64,
        'HQ':64,
    },

    'H8_S16K': {
        'B':1,
        'T':16384,
        'H':8,
        'D':64,
        'HQ':128,
    },


    # ================================================================
    # Group 2: Main NSA workload
    # 目的:
    #   - NSA 主优势区
    #   - 长上下文
    #   - memory throughput
    #
    # G=16
    # ================================================================

    'H16_S8K': {
        'B': 1, 'T': 8192,
        'H': 16, 'D': 64,
        'HQ': 256,
    },

    'H16_S16K': {
        'B': 1, 'T': 16384,
        'H': 16, 'D': 64,
        'HQ': 256,
    },

    'H16_S32K': {
        'B': 1, 'T': 32768,
        'H': 16, 'D': 64,
        'HQ': 256,
    },

    'H16_S64K': {
        'B': 1, 'T': 65536,
        'H': 16, 'D': 64,
        'HQ': 256,
    },


    # ================================================================
    # Group 3: Large H
    # 目的:
    #   - 接近真实模型
    #   - 高并行度
    #   - 测优化是否通用
    #
    # ================================================================

    'H32_S16K': {
        'B': 1, 'T': 16384,
        'H': 32, 'D': 64,
        'HQ': 512,
    },

    'H32_S32K': {
        'B': 1, 'T': 32768,
        'H': 32, 'D': 64,
        'HQ': 512,
    },

    'H32_S64K': {
        'B': 1, 'T': 65536,
        'H': 32, 'D': 64,
        'HQ': 512,
    },


    # ================================================================
    # Group 4: Head dimension sensitivity
    # 目的:
    #   - register pressure
    #   - tile size
    #   - tensor core utilization
    #
    # ================================================================

    'H16_D128': {
        'B': 1,
        'T': 16384,
        'H': 16,
        'D': 128,
        'HQ': 256,
    },

    'H16_D192': {
        'B': 1,
        'T': 16384,
        'H': 16,
        'D': 192,
        'HQ': 256,
    },


    # ================================================================
    # Group 5: GQA sensitivity
    # 目的:
    #   - 测不同 KV sharing
    # ================================================================

    'H16_G16': {
        'B':1,
        'T':16384,
        'H':16,
        'D':64,
        'HQ':256,
    },

    'H16_G32': {
        'B':1,
        'T':16384,
        'H':16,
        'D':64,
        'HQ':512,
    },


    # ================================================================
    # Group 6: Batch scaling
    # 目的:
    #   - 非纯 decode 场景
    # ================================================================

    'B4_H16_S8K': {
        'B': 4,
        'T': 8192,
        'H': 16,
        'D': 64,
        'HQ': 256,
    },
}


def _nsa_post_init(inputs, B, T, H, D, HQ=None, **kw):
    """Generate valid block_indices for naive_nsa benchmark."""
    S = 16
    n_blocks = (T + 63) // 64
    with torch.no_grad():
        idx = torch.randint(0, max(1, n_blocks), (B, T, H, S), device=inputs['q'].device, dtype=torch.long)
        inputs['block_indices'] = idx


register_op(OpConfig(
    name='parallel_nsa',
    import_path='fla.ops.nsa',
    inputs={
        'q': TensorSpec(shape_BTHD_HQ),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
        'g_cmp': TensorSpec(shape_BTH_HQ, transform=sigmoid_transform),
        'g_slc': TensorSpec(shape_BTH_HQ, transform=sigmoid_transform),
    },
    extra_kwargs={'block_counts': 16, 'block_size': 64, 'window_size': 0},
    output_is_tuple=False,
    default_shapes=_nsa_default_shapes,
    category='nsa',
))

register_op(OpConfig(
    name='naive_nsa',
    import_path='fla.ops.nsa',
    inputs={
        'q': TensorSpec(shape_BTHD_HQ),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
    },
    func_name='naive_nsa',
    extra_kwargs={'block_size': 64},
    output_is_tuple=False,
    default_shapes=_nsa_default_shapes,
    post_init=_nsa_post_init,
    category='nsa',
))

register_op(OpConfig(
    name='parallel_nsa_compression',
    import_path='fla.ops.nsa.compression',
    inputs={
        'q': TensorSpec(shape_BTHD_HQ),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
    },
    func_name='parallel_nsa_compression',
    extra_kwargs={'block_size': 64},
    output_is_tuple=True,
    default_shapes=_nsa_default_shapes,
    category='nsa',
))
