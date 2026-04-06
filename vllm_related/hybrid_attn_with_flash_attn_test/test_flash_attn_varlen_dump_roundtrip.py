# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay ``flash_attn_varlen_func`` dumps (from ``fa_utils`` tensor dump or legacy hook).

Set ``VLLM_FA_DUMP_DIR`` to the directory containing ``inputs_*.pt`` / ``output_*.pt``.
Default: ``<this_dir>/tensor_dumps/latest``.

Uses ``flash_attn_varlen_func`` from ``vllm.v1.attention.backends.fa_utils`` (same as
runtime). The actual call uses ``__wrapped__`` when present so the optional dump
wrapper does not run again during the test. ``inspect.signature`` follows ``__wrapped__``
so ``q``/``k``/``v`` are not dropped when filtering kwargs.

Examples::

    VLLM_FA_DUMP_DIR=/path/to/dumps pytest vllm_related_hooks/test_flash_attn_varlen_dump_roundtrip.py -v
"""

from __future__ import annotations

import glob
import os
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
        "Roundtrip test expects CUDA or XPU (same as dump hook target).",
        allow_module_level=True,
    )

ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DUMP_DIR = os.path.join(ROOT, "tensor_dumps", "latest")


def _dump_dir() -> str:
    return os.environ.get("VLLM_FA_DUMP_DIR", _DEFAULT_DUMP_DIR).strip()


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
    """Follow ``__wrapped__`` and unwrap staticmethod/classmethod for ``inspect.signature``."""
    f = fn
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    if isinstance(f, (classmethod, staticmethod)):
        f = f.__func__
    return f


def _call_target(fn: Any) -> Any:
    """Call underlying op: use ``__wrapped__`` when present so fa_utils dump wrapper is skipped."""
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
    """Move tensor positional args (e.g. q, k, v) to ``device``."""
    out: list[Any] = []
    for a in args:
        if torch.is_tensor(a):
            out.append(a.to(device=device))
        elif isinstance(a, (list, tuple)) and a and all(torch.is_tensor(x) for x in a):
            out.append(type(a)(x.to(device=device) for x in a))
        else:
            out.append(a)
    return tuple(out)


def test_flash_attn_varlen_dump_roundtrip_all_calls() -> None:
    d = _dump_dir()
    inp_paths = sorted(glob.glob(os.path.join(d, "inputs_*.pt")))
    if not inp_paths:
        pytest.skip(f"No inputs_*.pt in dump dir: {d!r}")

    device = _device()

    if _BACKEND == "cuda" and not is_fa_version_supported(2):
        pytest.skip("FA2 not supported: " f"{fa_version_unsupported_reason(2)}")

    import inspect

    call_fn = _call_target(flash_attn_varlen_func)
    sig_fn = _unwrap_for_signature(call_fn)
    try:
        params = set(inspect.signature(sig_fn).parameters.keys())
    except (ValueError, TypeError):
        params = set()

    for inp_path in inp_paths:
        stem = os.path.basename(inp_path).replace("inputs_", "").replace(".pt", "")
        try:
            index = int(stem)
        except ValueError:
            continue
        out_path = os.path.join(d, f"output_{index:06d}.pt")
        if not os.path.isfile(out_path):
            pytest.fail(f"Missing {out_path!r} for {inp_path!r}")

        payload = _torch_load(inp_path, map_location=device)
        # Positional q/k/v are used by some paths (e.g. vit_attn_wrappers); older dumps
        # only stored kwargs — re-dump with an updated fa_utils if args is missing.
        raw_args = payload.get("args")
        if raw_args is None:
            pytest.fail(
                f"{inp_path!r} has no 'args' key. Older dumps omitted positional q,k,v; "
                "re-run inference with VLLM_FA_DUMP_DIR and an updated fa_utils.py."
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

        print(f"args: {[i.shape for i in args]}")
        print(f"kwargs: {kwargs.keys()}")

        got = call_fn(*args, **kwargs)

        if isinstance(expected, tuple):
            assert isinstance(got, tuple)
            assert len(got) == len(expected)
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b, atol=1.5e-2, rtol=1.0e-2)
        else:
            torch.testing.assert_close(got, expected, atol=1.5e-2, rtol=1.0e-2)
