# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrap ``vllm.vllm_flash_attn.flash_attn_varlen_func`` to dump inputs/outputs to disk.

Call :func:`install` **before** importing code that binds
``from vllm.vllm_flash_attn import flash_attn_varlen_func`` — use
:mod:`run_generate_with_fa_dump` for offline inference.

Environment:

* ``VLLM_FA_DUMP_DIR`` — directory for ``inputs_XXXXXX.pt`` / ``output_XXXXXX.pt`` / ``run_meta_XXXXXX.json``.
* ``VLLM_FA_DUMP_MAX_CALLS`` — max calls to dump (default ``8``). Use ``0`` for no limit (very large runs).
* ``VLLM_FA_DUMP_CPU`` — ``1`` (default) saves tensors on CPU for portable ``.pt`` files.

See also: ``test_flash_attn_varlen_dump_roundtrip.py`` to replay dumps.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

import torch

_INSTALLED = False


def _to_cpu(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(y) for y in x)
    return x


def _json_safe_meta(kwargs: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            meta[k] = None
        elif isinstance(v, (bool, int, float, str)):
            meta[k] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (int, float)) for x in v
        ):
            meta[k] = list(v)
    return meta


def install() -> None:
    """Monkey-patch ``flash_attn_varlen_func`` on the interface and package."""
    global _INSTALLED
    if _INSTALLED:
        return

    dump_dir = os.environ.get("VLLM_FA_DUMP_DIR", "").strip()
    if not dump_dir:
        raise RuntimeError("Set VLLM_FA_DUMP_DIR to a writable directory before install().")

    os.makedirs(dump_dir, exist_ok=True)

    max_calls = int(os.environ.get("VLLM_FA_DUMP_MAX_CALLS", "8"))
    to_cpu = os.environ.get("VLLM_FA_DUMP_CPU", "1") not in ("0", "false", "False")

    import vllm.vllm_flash_attn as pkg
    import vllm.vllm_flash_attn.flash_attn_interface as fi

    orig = fi.flash_attn_varlen_func
    call_idx = [0]

    @functools.wraps(orig)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        idx = call_idx[0]
        should_dump = (max_calls == 0) or (idx < max_calls)
        print(f"should_dump: {should_dump}")

        if should_dump:
            payload_in: dict[str, Any] = {
                "args_meta": {"len": len(args)},
                "kwargs": {},
            }
            if args:
                payload_in["args_meta"]["note"] = (
                    "positional args are unusual; only count stored"
                )
            for k, v in kwargs.items():
                payload_in["kwargs"][k] = _to_cpu(v) if to_cpu else v
            payload_in["meta"] = _json_safe_meta(kwargs)
            in_path = os.path.join(dump_dir, f"inputs_{idx:06d}.pt")
            torch.save(payload_in, in_path)

        out = orig(*args, **kwargs)

        if should_dump:
            if isinstance(out, tuple):
                out_save = tuple(_to_cpu(t) if to_cpu else t for t in out)
            else:
                out_save = _to_cpu(out) if to_cpu else out
            out_path = os.path.join(dump_dir, f"output_{idx:06d}.pt")
            torch.save({"output": out_save}, out_path)
            meta_path = os.path.join(dump_dir, f"run_meta_{idx:06d}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"call_index": idx, "dump_dir": dump_dir, "meta": _json_safe_meta(kwargs)},
                    f,
                    indent=2,
                )

        call_idx[0] += 1
        return out

    fi.flash_attn_varlen_func = wrapped
    pkg.flash_attn_varlen_func = wrapped
    _INSTALLED = True
