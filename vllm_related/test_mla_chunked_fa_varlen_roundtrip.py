# SPDX-License-Identifier: Apache-2.0
"""Replay MLA ``flash_attn_varlen_func`` dumps from ``mla_chunked_fa``.

Default directory: ``~/randomstuffs/vllm_related/mla_chunked_fa`` (MLA LSE tensor dumps).
Override with ``VLLM_MLA_CHUNKED_FA_DIR`` if dumps live elsewhere.

Expected files (from ``VLLM_MLA_FA_VARLEN_DUMP_DIR`` during inference):

- ``mla_varlen_lse_inputs_{idx:06d}.pt`` — ``{"args": (), "kwargs": {...}}``
- ``mla_varlen_lse_output_{idx:06d}.pt`` — ``{"output": (attn, lse)}`` or a single tensor

Run::

    pytest vllm_related_hooks/test_mla_chunked_fa_varlen_roundtrip.py -v
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    try:
        from vllm.vllm_flash_attn import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    except ImportError as exc:
        pytest.skip(
            f"vllm flash_attn / fa_utils not importable: {exc}",
            allow_module_level=True,
        )
    _BACKEND = "cuda"
elif current_platform.is_xpu():
    try:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    except ImportError as exc:
        pytest.skip(
            f"vllm fa_utils not importable: {exc}",
            allow_module_level=True,
        )
    _BACKEND = "xpu"
else:
    pytest.skip(
        "This test expects CUDA or XPU (same as MLA FA varlen dumps).",
        allow_module_level=True,
    )

_HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MLA_CHUNKED_FA_DIR = os.path.expanduser(
    "~/randomstuffs/vllm_related/mla_chunked_fa"
)


def _dump_dir() -> str:
    return os.environ.get(
        "VLLM_MLA_CHUNKED_FA_DIR", _DEFAULT_MLA_CHUNKED_FA_DIR
    ).strip()


def _torch_load(path: str, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _device() -> torch.device:
    if _BACKEND == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available.")
        return torch.device("cuda")
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        pytest.skip("XPU not available.")
    return torch.device("xpu")


def _unwrap_for_signature(fn: Any) -> Any:
    f = fn
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    if isinstance(f, (classmethod, staticmethod)):
        f = f.__func__
    return f


def _call_target(fn: Any) -> Any:
    return getattr(fn, "__wrapped__", fn)


def _move_kwarg_tensors(kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device)
        elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(x) for x in v):
            out[k] = type(v)(x.to(device=device) for x in v)
        else:
            out[k] = v
    return out


def _move_arg_tensors(args: tuple[Any, ...], device: torch.device) -> tuple[Any, ...]:
    out: list[Any] = []
    for a in args:
        if torch.is_tensor(a):
            out.append(a.to(device=device))
        elif isinstance(a, (list, tuple)) and a and all(torch.is_tensor(x) for x in a):
            out.append(type(a)(x.to(device=device) for x in a))
        else:
            out.append(a)
    return tuple(out)


_INPUT_RE = re.compile(r"^mla_varlen_lse_inputs_(\d+)\.pt$")


def _input_index(path: str) -> int | None:
    m = _INPUT_RE.match(os.path.basename(path))
    return int(m.group(1)) if m else None


def test_mla_chunked_fa_flash_attn_varlen_matches_dump() -> None:
    d = _dump_dir()
    inp_paths = sorted(
        glob.glob(os.path.join(d, "mla_varlen_lse_inputs_*.pt")),
        key=lambda p: (_input_index(p) or -1, p),
    )
    if not inp_paths:
        pytest.skip(
            f"No mla_varlen_lse_inputs_*.pt under {d!r}. "
            "Set VLLM_MLA_CHUNKED_FA_DIR or place dumps in the default path."
        )

    device = _device()

    if _BACKEND == "cuda" and not is_fa_version_supported(2):
        pytest.skip("FA2 not supported: " + fa_version_unsupported_reason(2))

    import inspect

    call_fn = _call_target(flash_attn_varlen_func)
    sig_fn = _unwrap_for_signature(call_fn)
    try:
        params = set(inspect.signature(sig_fn).parameters.keys())
    except (ValueError, TypeError):
        params = set()

    for inp_path in inp_paths:
        idx = _input_index(inp_path)
        if idx is None:
            continue
        out_path = os.path.join(d, f"mla_varlen_lse_output_{idx:06d}.pt")
        if not os.path.isfile(out_path):
            pytest.fail(f"Missing output dump {out_path!r} for {inp_path!r}")

        payload = _torch_load(inp_path, map_location=device)
        raw_args = payload.get("args")
        if raw_args is None:
            pytest.fail(
                f"{inp_path!r} has no 'args' key (expected "
                '{"args": (), "kwargs": {...}} from MLA dump).'
            )
        args = _move_arg_tensors(tuple(raw_args), device)
        kwargs = payload.get("kwargs") or {}
        kwargs = _move_kwarg_tensors(kwargs, device)
        if _BACKEND == "cuda" and "fa_version" in params and "fa_version" not in kwargs:
            kwargs["fa_version"] = 2
        if params:
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        ref_blob = _torch_load(out_path, map_location=device)
        expected = ref_blob["output"]

        got = call_fn(*args, **kwargs)

        if isinstance(expected, tuple):
            assert isinstance(got, tuple), (
                f"call {idx}: expected tuple output, got {type(got)}"
            )
            assert len(got) == len(expected), (
                f"call {idx}: tuple len {len(got)} vs {len(expected)}"
            )
            for j, (a, b) in enumerate(zip(got, expected)):
                torch.testing.assert_close(
                    a,
                    b,
                    atol=1.5e-2,
                    rtol=1.0e-2,
                    msg=lambda msg, j=j: f"call {idx} output[{j}]: {msg}",
                )
        else:
            torch.testing.assert_close(
                got,
                expected,
                atol=1.5e-2,
                rtol=1.0e-2,
                msg=lambda msg: f"call {idx}: {msg}",
            )
