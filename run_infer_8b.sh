#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B：与 deploy.py 一致，按 VLLM_TOTAL_GPUS 与 yaml 中 tensor_parallel_size 起多实例并轮询 endpoints。

cd "$(dirname "$0")"

LLM_CONFIG="${LLM_CONFIG:-src/config/llm_config/Qwen3_8B.yaml}"
TOOL_CONFIG="${TOOL_CONFIG:-src/config/tool_config/example.yaml}"
DATASET_CONFIG_DIR="${DATASET_CONFIG_DIR:-src/config/dataset_config}"
OUTPUT_DIR_TOOL="${OUTPUT_DIR_TOOL:-results_tool_8b}"
OUTPUT_DIR_NOTOOL="${OUTPUT_DIR_NOTOOL:-results_notool_8b}"

export VLLM_TOTAL_GPUS="${VLLM_TOTAL_GPUS:-4}"
mkdir -p "$OUTPUT_DIR_TOOL" "$OUTPUT_DIR_NOTOOL" logs

# vLLM 多实例端口从 BASE_PORT 递增（8B：tensor_parallel_size=2 → 2 个实例）
if [ -n "${VLLM_BASE_PORT:-}" ]; then
  BASE_PORT="${VLLM_BASE_PORT}"
fi
ENDPOINTS=(
  "http://127.0.0.1:8001/v1"
  "http://127.0.0.1:8002/v1"
)

echo "[run_infer_8b] Endpoints: ${ENDPOINTS[*]}"

echo "[run_infer_8b] Using already-deployed vLLM (LLM_CONFIG=$LLM_CONFIG, VLLM_TOTAL_GPUS=$VLLM_TOTAL_GPUS)."

echo "[run_infer_8b] Waiting for vLLM to be ready..."
sleep "${WAIT_FOR_VLLM_SECONDS:-10}"

for dataset_config in "$DATASET_CONFIG_DIR"/*.yaml; do
  name=$(basename "$dataset_config" .yaml)
  echo "========== Infer (use_tool=true): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$dataset_config" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool true \
    --output_path "$OUTPUT_DIR_TOOL" \
    --endpoints "${ENDPOINTS[@]}" \
    2>/dev/null

  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$dataset_config" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool false \
    --output_path "$OUTPUT_DIR_NOTOOL" \
    --endpoints "${ENDPOINTS[@]}" \
    2>/dev/null
done

