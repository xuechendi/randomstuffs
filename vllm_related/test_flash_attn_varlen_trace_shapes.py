# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for flash_attn_varlen_func shapes captured from a real profile.

Source: Torch profiler trace on DeepSeek-V2-Lite prefill (FA2 path), e.g.
``dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz`` — operator
``_vllm_fa2_C::varlen_fwd`` with ``Input Dims`` / strides / scalars below.

**Backend selection** matches ``vllm.v1.attention.backends.fa_utils``:
``current_platform.is_cuda()`` → ``vllm.vllm_flash_attn``;
``is_xpu()`` → ``vllm._xpu_ops.xpu_ops.flash_attn_varlen_func``;
``is_rocm()`` → upstream ``flash_attn`` if import succeeds.
If the import for the active platform fails, the module skips at collection time.

This test uses **non-paged** varlen layout: ``cu_seqlens_q`` / ``cu_seqlens_k`` for
one packed sequence (same tensor ranks as the trace; no block table).

Examples::

    pytest randomstuffs/vllm_related/test_flash_attn_varlen_trace_shapes.py
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    try:
        from vllm.vllm_flash_attn import (
            fa_version_unsupported_reason,
            flash_attn_varlen_func,
            is_fa_version_supported,
        )
    except ImportError as exc:
        pytest.skip(
            f"vllm.vllm_flash_attn not importable on CUDA platform: {exc}",
            allow_module_level=True,
        )
    _BACKEND: str = "cuda"
elif current_platform.is_xpu():
    try:
        from vllm._xpu_ops import xpu_ops
    except ImportError as exc:
        pytest.skip(
            f"vllm._xpu_ops not importable on XPU platform: {exc}",
            allow_module_level=True,
        )
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func
    _BACKEND = "xpu"
else:
    pytest.skip(
        "This test only runs on CUDA, XPU, or ROCm vLLM platforms.",
        allow_module_level=True,
    )


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and hasattr(xpu, "manual_seed"):
        xpu.manual_seed(seed)


def _resolve_device() -> torch.device:
    if _BACKEND in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available.")
        return torch.device("cuda")
    if _BACKEND == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            pytest.skip("XPU device not available.")
        return torch.device("xpu")
    pytest.fail(f"Unexpected backend {_BACKEND!r}")


def _filter_varlen_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the bound ``flash_attn_varlen_func`` does not accept (ROCm API)."""
    params = inspect.signature(flash_attn_varlen_func).parameters
    return {k: v for k, v in kwargs.items() if k in params}


# --- Values taken from profiler trace (single _vllm_fa2_C::varlen_fwd signature) ---
TRACE_TOTAL_Q: int = 2049
TRACE_NUM_HEADS: int = 16
TRACE_HEAD_DIM: int = 192
TRACE_DTYPE = torch.bfloat16
# Contiguous (total_q, nheads, headdim) -> strides from trace
TRACE_STRIDE_0: int = 3072  # 16 * 192
TRACE_STRIDE_1: int = 192
TRACE_STRIDE_2: int = 1
TRACE_SOFTMAX_SCALE: float = 0.11472138679292609
TRACE_MAX_SEQLEN_Q: int = 2049
TRACE_MAX_SEQLEN_K: int = 2049
TRACE_CAUSAL: bool = True
TRACE_WINDOW: tuple[int, int] = (-1, -1)
# cu_seqlens length batch_size + 1 = 2 for one sequence [0, total_q]
TRACE_CU_SEQLENS_LEN: int = 2


def _ref_causal_attention_single_seq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Full causal attention for one packed sequence, shapes (T, H, D)."""
    t, _h, _d = q.shape
    qf = q.float() * softmax_scale
    kf = k.float()
    vf = v.float()
    attn = torch.einsum("qhd,khd->hqk", qf, kf)
    mask = torch.triu(
        torch.ones(t, t, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    attn = attn.masked_fill(mask, float("-inf"))
    weights = torch.softmax(attn, dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, vf).to(q.dtype)


@pytest.mark.parametrize("use_preallocated_out", [False, True])
@torch.inference_mode()
def test_flash_attn_varlen_deepseek_v2_lite_trace_shapes(
    use_preallocated_out: bool,
) -> None:
    """Match flash_attn_varlen_func I/O shapes from a captured FA2 trace."""
    device = _resolve_device()

    if _BACKEND == "cuda":
        if not is_fa_version_supported(2):
            pytest.skip(
                "FA2 not supported on this CUDA device: "
                f"{fa_version_unsupported_reason(2)}"
            )

    _set_seed(0)

    t, h, d = TRACE_TOTAL_Q, TRACE_NUM_HEADS, TRACE_HEAD_DIM
    q = torch.randn(t, h, d, dtype=TRACE_DTYPE, device=device)
    k = torch.randn(t, h, d, dtype=TRACE_DTYPE, device=device)
    v = torch.randn(t, h, d, dtype=TRACE_DTYPE, device=device)

    assert q.is_contiguous()
    assert q.stride() == (TRACE_STRIDE_0, TRACE_STRIDE_1, TRACE_STRIDE_2)
    assert k.stride() == (TRACE_STRIDE_0, TRACE_STRIDE_1, TRACE_STRIDE_2)
    assert v.stride() == (TRACE_STRIDE_0, TRACE_STRIDE_1, TRACE_STRIDE_2)

    cu_seqlens_q = torch.tensor([0, t], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, t], dtype=torch.int32, device=device)
    assert cu_seqlens_q.shape == (TRACE_CU_SEQLENS_LEN,)
    assert cu_seqlens_k.shape == (TRACE_CU_SEQLENS_LEN,)

    out_buf: torch.Tensor | None
    if use_preallocated_out:
        out_buf = torch.empty(t, h, d, dtype=TRACE_DTYPE, device=device)
    else:
        out_buf = None

    call_kwargs: dict[str, Any] = {
        "q": q,
        "k": k,
        "v": v,
        "out": out_buf,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": TRACE_MAX_SEQLEN_Q,
        "max_seqlen_k": TRACE_MAX_SEQLEN_K,
        "softmax_scale": TRACE_SOFTMAX_SCALE,
        "causal": TRACE_CAUSAL,
        "window_size": list(TRACE_WINDOW),
        "dropout_p": 0.0,
        "softcap": 0.0,
    }
    if _BACKEND == "cuda":
        call_kwargs["fa_version"] = 2

    fn: Callable[..., Any] = flash_attn_varlen_func
    output = fn(**_filter_varlen_kwargs(call_kwargs))
    if use_preallocated_out and "out" in inspect.signature(fn).parameters:
        assert output is out_buf
    assert output is not None
    assert output.shape == (t, h, d)
    assert output.dtype == TRACE_DTYPE

    ref = _ref_causal_attention_single_seq(q, k, v, TRACE_SOFTMAX_SCALE)
    torch.testing.assert_close(output, ref, atol=1.5e-2, rtol=1.0e-2)
