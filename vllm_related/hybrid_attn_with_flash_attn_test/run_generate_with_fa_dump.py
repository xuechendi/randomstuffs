#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a vLLM script with tensor dumps enabled via ``fa_utils`` (not a separate hook).

Set ``VLLM_FA_DUMP_DIR`` before starting the engine. vLLM loads
``vllm.v1.attention.backends.fa_utils``, which wraps ``flash_attn_varlen_func`` when
that env var is set (see ``fa_utils._maybe_wrap_flash_attn_varlen_for_dump``).

Example (from vLLM repo root)::

    export VLLM_FA_DUMP_DIR="/path/to/tensor_dumps/qwen35"
    export VLLM_FA_DUMP_MAX_CALLS=8
    python run_generate_with_fa_dump.py \\
        examples/basic/offline_inference/generate.py \\
        --model Qwen/Qwen3.5-9B --enforce-eager

Default dump dir if unset: ``<this_dir>/tensor_dumps/latest``.
"""

from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

dump_dir = os.environ.get(
    "VLLM_FA_DUMP_DIR", os.path.join(ROOT, "tensor_dumps", "latest")
)
os.makedirs(dump_dir, exist_ok=True)
os.environ["VLLM_FA_DUMP_DIR"] = dump_dir

if len(sys.argv) < 2:
    print(
        "Usage: run_generate_with_fa_dump.py /path/to/script.py [script args...]\n"
        "Tensor dumps: set VLLM_FA_DUMP_DIR (default: tensor_dumps/latest under this dir).\n"
        "Dumps are written by vllm.v1.attention.backends.fa_utils when the engine loads.",
        file=sys.stderr,
    )
    sys.exit(2)

script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
