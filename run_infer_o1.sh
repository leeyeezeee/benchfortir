#!/usr/bin/env bash
set -euo pipefail

# 使用 OpenAI 远程 API（o1 系列）跑推理，用法对齐 run_infer_4b.sh。
#
# 必填：
#   export OPENAI_API_KEY="sk-..."
# URL：
#   官方 OpenAI：   https://api.openai.com/v1
#
# 可选：
#   OPENAI_BASE_URL   默认 https://api.openai.com/v1
#   DEFAULT_MODEL     默认 o1-mini（可改为 o1 / o1-preview 等）
#   MODEL_PATH        tokenizer 路径
#   LLM_CONFIG        默认 gpt_o1

cd "$(dirname "$0")"

LLM_CONFIG="${LLM_CONFIG:-gpt_o1}"
TOOL_CONFIG="${TOOL_CONFIG:-example}"
: "${DATASET_NAMES:=interaction expodesign math500 gsm8k500 omini500 hotpotqa simpleqa}"
: "${DATASET_NAMES_NOTOOL:=math500 gsm8k500 omini500 hotpotqa simpleqa}"
OUTPUT_DIR_TOOL="${OUTPUT_DIR_TOOL:-results/tool/gpt_o1}"
OUTPUT_DIR_NOTOOL="${OUTPUT_DIR_NOTOOL:-results/notool/gpt_o1}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-32B}"


mkdir -p "$OUTPUT_DIR_TOOL" "$OUTPUT_DIR_NOTOOL"

echo "[run_infer_o1] LLM_CONFIG=$LLM_CONFIG"
echo "[run_infer_o1] MODEL_PATH=$MODEL_PATH"
echo "[run_infer_o1] DATASET_NAMES=$DATASET_NAMES"
echo "[run_infer_o1] Output(use_tool=true)  -> $OUTPUT_DIR_TOOL"
echo "[run_infer_o1] Output(use_tool=false) -> $OUTPUT_DIR_NOTOOL"

for name in $DATASET_NAMES; do
  echo "========== Infer (use_tool=true): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool true \
    --output_path "$OUTPUT_DIR_TOOL" \
    --model_path "$MODEL_PATH" \
done

for name in $DATASET_NAMES_NOTOOL; do
  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool false \
    --output_path "$OUTPUT_DIR_NOTOOL" \
    --model_path "$MODEL_PATH" \
    2>/dev/null
done