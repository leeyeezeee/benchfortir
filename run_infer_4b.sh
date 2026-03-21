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
#   python infer.py --llm_config ... --dataset_config ... --tool_config ...

cd "$(dirname "$0")"

# infer.py 只接受配置文件名（stem），对应 src/config/<llm_config|dataset_config|tool_config>/
LLM_CONFIG="${LLM_CONFIG:-Qwen3_4B}"
TOOL_CONFIG="${TOOL_CONFIG:-example}"
: "${DATASET_NAMES:=interaction expodesign math500 gsm8k500 omini500 hotpotqa simpleqa}"
: "${DATASET_NAMES_NOTOOL:=math500 gsm8k500 omini500 hotpotqa simpleqa}"
OUTPUT_DIR_TOOL="${OUTPUT_DIR_TOOL:-results/tool/qwen3_4b}"
OUTPUT_DIR_NOTOOL="${OUTPUT_DIR_NOTOOL:-results/notool/qwen3_4b}"

export VLLM_TOTAL_GPUS="${VLLM_TOTAL_GPUS:-4}"

mkdir -p "$OUTPUT_DIR_TOOL" "$OUTPUT_DIR_NOTOOL"

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

echo "[run_infer_4b] Using LLM_CONFIG=$LLM_CONFIG"
echo "[run_infer_4b] Using TOOL_CONFIG=$TOOL_CONFIG"
echo "[run_infer_4b] DATASET_NAMES=$DATASET_NAMES"
echo "[run_infer_4b] Output(use_tool=true)  -> $OUTPUT_DIR_TOOL"
echo "[run_infer_4b] Output(use_tool=false) -> $OUTPUT_DIR_NOTOOL"
echo "[run_infer_4b] Endpoints: ${ENDPOINTS[*]}"

echo "[run_infer_4b] Waiting for vLLM to be ready..."
sleep "${WAIT_FOR_VLLM_SECONDS:-10}"

for name in $DATASET_NAMES; do
  echo "========== Infer (use_tool=true): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool true \
    --output_path "$OUTPUT_DIR_TOOL" \
    --endpoints "${ENDPOINTS[@]}" \
done

for name in $DATASET_NAMES_NOTOOL; do
  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool false \
    --output_path "$OUTPUT_DIR_NOTOOL" \
    --endpoints "${ENDPOINTS[@]}" \
    2>/dev/null
done
