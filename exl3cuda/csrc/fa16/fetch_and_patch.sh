#!/bin/bash
# Fetch the FlashAttention 2 sources the FA16 / I8FA kernels build against and apply srcmod.diff:
#   vllm-project/flash-attention @ caaa4eb5 (the FA pin of vLLM 0.26.0) + its csrc/cutlass submodule (62750a2b),
#   csrc/flash_attn/src copied to ./srcmod and patched (fp16-accumulate MMA variants FA16_PV / FA16_QK, 64 x 64 tile).
# Result: ./fa (clone, for the CUTLASS headers) and ./srcmod (patched FA2 sources). Network access needed once.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
FA_URL=${FA_URL:-https://github.com/vllm-project/flash-attention}
FA_COMMIT=caaa4eb59845388a20b1f435ecaafb4bd9517ad8
if [ ! -d fa/.git ]; then
  git clone --filter=blob:none "$FA_URL" fa
fi
git -C fa checkout --quiet "$FA_COMMIT"
git -C fa submodule update --init --depth 1 csrc/cutlass 2>/dev/null || git -C fa submodule update --init csrc/cutlass
rm -rf srcmod
cp -r fa/csrc/flash_attn/src srcmod
(cd srcmod && patch -p1 --no-backup-if-mismatch < ../srcmod.diff)
echo "FA2 $FA_COMMIT + cutlass $(git -C fa/csrc/cutlass rev-parse --short HEAD); patched sources in $(pwd)/srcmod"
