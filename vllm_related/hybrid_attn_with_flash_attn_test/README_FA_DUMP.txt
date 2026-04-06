Flash-attention varlen tensor dumps (vLLM)
==========================================

**Preferred:** dumps are implemented in ``vllm/v1/attention/backends/fa_utils.py``
(wrap ``flash_attn_varlen_func`` when ``VLLM_FA_DUMP_DIR`` is set). A copy of the
updated file lives here as ``fa_utils.py`` — install with::

  sudo cp /home/chendixu/vllm_related_hooks/fa_utils.py \\
      /path/to/vllm/vllm/v1/attention/backends/fa_utils.py

The separate ``flash_attn_varlen_dump_hook.py`` monkey-patch is optional / legacy
once ``fa_utils`` is patched.

Run Qwen offline inference with dumps (from **vLLM repo root**):

  export VLLM_FA_DUMP_DIR="/home/chendixu/vllm_related_hooks/tensor_dumps/qwen35"
  export VLLM_FA_DUMP_MAX_CALLS=8

  python /home/chendixu/vllm_related_hooks/run_generate_with_fa_dump.py \\
      examples/basic/offline_inference/generate.py \\
      --model Qwen/Qwen3.5-9B --enforce-eager

Replay dumps as a unit test:

  VLLM_FA_DUMP_DIR="${VLLM_FA_DUMP_DIR}" \\
  pytest /home/chendixu/vllm_related_hooks/test_flash_attn_varlen_dump_roundtrip.py -v
