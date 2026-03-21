#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B：与 deploy.py 一致，按 VLLM_TOTAL_GPUS 与 yaml 中 tensor_parallel_size 起多实例并轮询 endpoints。

cd "$(dirname "$0")"

# infer.py 只接受配置文件名（stem），对应 src/config/<llm_config|dataset_config|tool_config>/
LLM_CONFIG="${LLM_CONFIG:-Qwen3_30b}"
TOOL_CONFIG="${TOOL_CONFIG:-example}"
# 空格分隔的数据集 yaml stem；覆盖示例: DATASET_NAMES="expodesign math500" ./run_infer_30b.sh
: "${DATASET_NAMES:=interaction expodesign math500 gsm8k500 omini500 hotpotqa}"
: "${DATASET_NAMES_NOTOOL:=math500 gsm8k500 omini500 hotpotqa simpleqa}"
OUTPUT_DIR_TOOL="${OUTPUT_DIR_TOOL:-results/tool/qwen3_30b}"
OUTPUT_DIR_NOTOOL="${OUTPUT_DIR_NOTOOL:-results/notool/qwen3_30b}"

export VLLM_TOTAL_GPUS="${VLLM_TOTAL_GPUS:-4}"
mkdir -p "$OUTPUT_DIR_TOOL" "$OUTPUT_DIR_NOTOOL" logs

# vLLM 多实例端口从 BASE_PORT 递增（30B：tensor_parallel_size=4 → 1 个实例）
if [ -n "${VLLM_BASE_PORT:-}" ]; then
  BASE_PORT="${VLLM_BASE_PORT}"
fi
ENDPOINTS=(
  "http://127.0.0.1:8001/v1"
)

echo "[run_infer_30b] Endpoints: ${ENDPOINTS[*]}"

echo "[run_infer_30b] Using already-deployed vLLM (LLM_CONFIG=$LLM_CONFIG, VLLM_TOTAL_GPUS=$VLLM_TOTAL_GPUS)."
echo "[run_infer_30b] DATASET_NAMES: $DATASET_NAMES"

echo "[run_infer_30b] Waiting for vLLM to be ready..."
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
    2>/dev/null
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
