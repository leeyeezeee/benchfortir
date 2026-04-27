#!/usr/bin/env bash
set -euo pipefail

# Run inference against an already-running vLLM service pool.
# infer.py resolves endpoints from llm_config (including remote=false fallback rules).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# infer.py 只接受配置文件名（stem），对应 src/config/<llm_config|dataset_config>/
LLM_CONFIG="${LLM_CONFIG:-Qwen3_4B}"
: "${DATASET_NAMES:=interaction math500}"
: "${DATASET_NAMES_NOTOOL:=}"

echo "[run_infer_4b] Using LLM_CONFIG=$LLM_CONFIG"
echo "[run_infer_4b] DATASET_NAMES=$DATASET_NAMES"
echo "[run_infer_4b] Output path is inferred by infer.py (use_tool/model/dataset)."

echo "[run_infer_4b] Waiting for vLLM to be ready..."
sleep "${WAIT_FOR_VLLM_SECONDS:-10}"

for name in $DATASET_NAMES; do
  echo "========== Infer (use_tool=true): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    with use_tool=true
done

for name in $DATASET_NAMES_NOTOOL; do
  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    with use_tool=false \
    2>/dev/null
done