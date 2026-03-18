#!/usr/bin/env bash
set -euo pipefail

# Run inference against an already-running vLLM service pool.
#
# It builds endpoints from GPUS/BASE_PORT and loops over dataset configs in
# src/config/dataset_config/*.yaml. For each dataset yaml, it runs infer twice:
# - use_tool=true
# - use_tool=false
#
# infer.py is invoked in the recommended way:
#   python infer.py --llm_config ... --dataset_config ... --tool_config ...

cd "$(dirname "$0")"

LLM_CONFIG="${LLM_CONFIG:-src/config/llm_config/Qwen3_4B.yaml}"
TOOL_CONFIG="${TOOL_CONFIG:-src/config/tool_config/example.yaml}"
DATASET_CONFIG_DIR="${DATASET_CONFIG_DIR:-src/config/dataset_config}"
OUTPUT_DIR_TOOL="${OUTPUT_DIR_TOOL:-results_tool}"
OUTPUT_DIR_NOTOOL="${OUTPUT_DIR_NOTOOL:-results_notool}"

GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8001}"

mkdir -p "$OUTPUT_DIR_TOOL" "$OUTPUT_DIR_NOTOOL"

# Build endpoints list from GPU count / ports
IFS=',' read -r -a GPU_ARR <<<"$GPUS"
ENDPOINTS=()
for i in "${!GPU_ARR[@]}"; do
  port=$((BASE_PORT + i))
  ENDPOINTS+=("http://127.0.0.1:${port}/v1")
done

echo "[run_infer_all] Using LLM_CONFIG=$LLM_CONFIG"
echo "[run_infer_all] Using TOOL_CONFIG=$TOOL_CONFIG"
echo "[run_infer_all] Using DATASET_CONFIG_DIR=$DATASET_CONFIG_DIR"
echo "[run_infer_all] Output(use_tool=true)  -> $OUTPUT_DIR_TOOL"
echo "[run_infer_all] Output(use_tool=false) -> $OUTPUT_DIR_NOTOOL"
echo "[run_infer_all] Endpoints: ${ENDPOINTS[*]}"

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

  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$dataset_config" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool false \
    --output_path "$OUTPUT_DIR_NOTOOL" \
    --endpoints "${ENDPOINTS[@]}"
done

