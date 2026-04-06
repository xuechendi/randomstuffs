# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import os
from typing import Any, Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _to_cpu_for_dump(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu_for_dump(y) for y in x)
    return x


def _json_safe_meta_for_dump(kwargs: dict[str, Any]) -> dict[str, Any]:
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


def _maybe_wrap_flash_attn_varlen_for_dump(
    orig: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap ``flash_attn_varlen_func`` to dump I/O when ``VLLM_FA_DUMP_DIR`` is set.

    Writes ``inputs_NNNNNN.pt``, ``output_NNNNNN.pt``, ``run_meta_NNNNNN.json`` under
    the dump directory.

    Environment:
        VLLM_FA_DUMP_DIR: directory path (required to enable dumping).
        VLLM_FA_DUMP_MAX_CALLS: max calls to record (default 8); 0 = unlimited.
        VLLM_FA_DUMP_CPU: if 1 (default), tensors are moved to CPU before torch.save.
    """
    dump_dir = os.environ.get("VLLM_FA_DUMP_DIR", "").strip()
    if not dump_dir:
        return orig

    try:
        os.makedirs(dump_dir, exist_ok=True)
    except OSError as e:
        logger.warning(
            "VLLM_FA_DUMP_DIR=%r is not usable (%s); skipping flash_attn varlen dumps.",
            dump_dir,
            e,
        )
        return orig

    max_calls = int(os.environ.get("VLLM_FA_DUMP_MAX_CALLS", "8"))
    to_cpu = os.environ.get("VLLM_FA_DUMP_CPU", "1") not in ("0", "false", "False")
    call_idx = [0]

    @functools.wraps(orig)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        idx = call_idx[0]
        should_dump = (max_calls == 0) or (idx < max_calls)

        if should_dump:
            # Some call sites pass q, k, v positionally (e.g. vit_attn_wrappers); others
            # use keywords. Persist both *args and **kwargs so roundtrip tests can replay.
            payload_in: dict[str, Any] = {
                "args": tuple(
                    _to_cpu_for_dump(a) if to_cpu else a for a in args
                ),
                "kwargs": {},
            }
            for k, v in kwargs.items():
                payload_in["kwargs"][k] = _to_cpu_for_dump(v) if to_cpu else v
            payload_in["meta"] = _json_safe_meta_for_dump(kwargs)
            in_path = os.path.join(dump_dir, f"inputs_{idx:06d}.pt")
            torch.save(payload_in, in_path)

        out = orig(*args, **kwargs)

        if should_dump:
            if isinstance(out, tuple):
                out_save = tuple(_to_cpu_for_dump(t) if to_cpu else t for t in out)
            else:
                out_save = _to_cpu_for_dump(out) if to_cpu else out
            out_path = os.path.join(dump_dir, f"output_{idx:06d}.pt")
            torch.save({"output": out_save}, out_path)
            meta_path = os.path.join(dump_dir, f"run_meta_{idx:06d}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "call_index": idx,
                        "dump_dir": dump_dir,
                        "meta": _json_safe_meta_for_dump(kwargs),
                    },
                    f,
                    indent=2,
                )

        call_idx[0] += 1
        return out

    logger.info_once(
        "flash_attn_varlen_func tensor dumps enabled (VLLM_FA_DUMP_DIR=%r, "
        "max_calls=%s)",
        dump_dir,
        "unlimited" if max_calls == 0 else str(max_calls),
    )
    return wrapped


# Track whether upstream flash-attn is available on ROCm.
# Set during module initialization and never modified afterwards.
# This module-level flag avoids repeated import attempts and ensures
# consistent behavior (similar to IS_AITER_FOUND in _aiter_ops.py).
_ROCM_FLASH_ATTN_AVAILABLE = False

if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        flash_attn_varlen_func as _flash_attn_varlen_func_impl,
        get_scheduler_metadata,
    )

elif current_platform.is_xpu():
    from vllm import _custom_ops as ops
    from vllm._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    _flash_attn_varlen_func_impl = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func_impl  # type: ignore[no-redef]

        # Mark that upstream flash-attn is available on ROCm
        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError:

        def _flash_attn_varlen_func_impl(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef,misc]
            raise ImportError(
                "ROCm platform requires upstream flash-attn "
                "to be installed. Please install flash-attn first."
            )

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash

flash_attn_varlen_func = _maybe_wrap_flash_attn_varlen_for_dump(
    _flash_attn_varlen_func_impl
)


def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None:
    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        else:
            # Fallback to FA2
            fa_version = 2

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        # FA4 currently uses batch-shape-dependent scheduling
        # heuristics on SM100+, which breaks batch invariance.
        if vllm_is_batch_invariant() and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, "
                "defaulting to FA version 2.",
                scope="local",
            )
            fa_version = 2

        # FA4 on SM100 (Blackwell) has TMEM capacity limits that restrict
        # supported head dimensions.
        # See: https://github.com/Dao-AILab/flash-attention/issues/1959
        # Exception: hdim 192 is supported for MLA's diff-headdim case
        # (qk=192, v=128), added upstream in commits 1a15733e/1b36ab19.
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and head_size != 192
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    return (
        get_flash_attn_version() == 3
        and current_platform.is_device_capability_family(90)
    )


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)

            # NOTE(Lucas): FA4 CuteDSL does NOT currently support MLA's non-standard
            # head dimensions (576 for qk, 512 for v) due to TMEM capacity limits.

        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func (if available)

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
