#!/bin/bash
# Build several FA16 variants one after another (one nvcc at a time: each needs ~12 GB RAM), at idle priority.
#   ./build-seq.sh fa_f:1:2 [fa_b:1:0 ...]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
for v in "$@"; do IFS=: read -r name pv qk <<< "$v"
  echo "$(date +%T) build $name FA16_PV=$pv FA16_QK=$qk"
  MAX_JOBS=1 chrt -i 0 nice -n 19 ionice -c 3 python3 build.py "$name" "$pv" "$qk" > "build-$name.log" 2>&1 \
    && echo "  $(tail -1 "build-$name.log")" || { echo "  failed, see build-$name.log"; exit 1; }
done
