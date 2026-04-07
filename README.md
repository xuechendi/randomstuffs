# randomstuffs


## Install Git LFS to get saved input/output tensors

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y git-lfs

# Initialize Git LFS once per machine
git lfs install

git pull origin main
```


## Run MLA chunked FA varlen roundtrip test
UT for testing flash_attn_varlen_func softmax_lfs support

```bash
VLLM_MLA_CHUNKED_FA_DIR=`pwd`/mla_chunked_fa pytest -v test_mla_chunked_fa_varlen_roundtrip.py
```
Status: Expected to fail on XPU


## Run Hybrid Attn for FA

```bash
VLLM_FA_DUMP_DIR=`pwd`/hybrid_attn_with_flash_attn_test/qwen35 pytest -v test_flash_attn_varlen_dump_roundtrip.py
```
Status: Expected to pass





