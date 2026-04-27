#!/usr/bin/env bash
set -euo pipefail

# Run inference against an already-running vLLM service pool.
#
# It builds endpoints from GPUS/BASE_PORT and loops over DATASET_NAMES (dataset yaml stems).
# For each dataset, it runs infer twice:
# - use_tool=true
# - use_tool=false
#
# infer.py is invoked in the recommended way:
#   python infer.py --llm_config ... --dataset_config ... with ...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# infer.py 只接受配置文件名（stem），对应 src/config/<llm_config|dataset_config>/
LLM_CONFIG="${LLM_CONFIG:-Qwen3_4B}"
: "${DATASET_NAMES:=math500}"
: "${DATASET_NAMES_NOTOOL:=}"

export VLLM_TOTAL_GPUS="${VLLM_TOTAL_GPUS:-4}"

# vLLM 多实例端口从 BASE_PORT 递增（4B：tensor_parallel_size=1 → 4 个实例）
# 注意：这里 endpoints 写死为 base..base+3；确保你部署时的端口起点一致。
if [ -n "${VLLM_BASE_PORT:-}" ]; then
  BASE_PORT="${VLLM_BASE_PORT}"
fi
ENDPOINTS=(
  "http://127.0.0.1:8001/v1"
  "http://127.0.0.1:8002/v1"
  "http://127.0.0.1:8003/v1"
  "http://127.0.0.1:8004/v1"
)

# Build Sacred-compatible Python list literal, e.g. ['http://...','http://...'].
ENDPOINTS_EXPR="["
for ep in "${ENDPOINTS[@]}"; do
  ENDPOINTS_EXPR="${ENDPOINTS_EXPR}'${ep}',"
done
ENDPOINTS_EXPR="${ENDPOINTS_EXPR%?}]"

echo "[run_infer_4b] Using LLM_CONFIG=$LLM_CONFIG"
echo "[run_infer_4b] DATASET_NAMES=$DATASET_NAMES"
echo "[run_infer_4b] Output path is inferred by infer.py (use_tool/model/dataset)."
echo "[run_infer_4b] Endpoints: ${ENDPOINTS[*]}"

echo "[run_infer_4b] Waiting for vLLM to be ready..."
sleep "${WAIT_FOR_VLLM_SECONDS:-10}"

for name in $DATASET_NAMES; do
  echo "========== Infer (use_tool=true): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    with use_tool=true endpoints="$ENDPOINTS_EXPR"
done

for name in $DATASET_NAMES_NOTOOL; do
  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    with use_tool=false endpoints="$ENDPOINTS_EXPR" \
    2>/dev/null
done