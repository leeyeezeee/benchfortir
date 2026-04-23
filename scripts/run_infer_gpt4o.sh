#!/usr/bin/env bash
set -euo pipefail

# 使用 OpenAI 远程 API（gpt-4o）跑推理，用法对齐 run_infer_4b.sh。
#
# 必填：
#   export OPENAI_API_KEY="sk-..."
# URL（base_url）：
#   官方 OpenAI：   https://api.openai.com/v1
#
# 可选：
#   OPENAI_BASE_URL   覆盖默认（默认 https://api.openai.com/v1）
#   MODEL_PATH        tokenizer 路径（默认 Qwen/Qwen2.5-7B-Instruct）
#   DEFAULT_MODEL     默认 gpt-4o
#   LLM_CONFIG        默认 gpt4o

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LLM_CONFIG="${LLM_CONFIG:-gpt_5}"
: "${DATASET_NAMES:=hotpotqa squadv2}"
: "${DATASET_NAMES_NOTOOL:=}"
MODEL_PATH="${MODEL_PATH:-/data/lyz/models/Qwen3-32B}"


echo "[run_infer_gpt4o] LLM_CONFIG=$LLM_CONFIG"
echo "[run_infer_gpt4o] MODEL_PATH=$MODEL_PATH"
echo "[run_infer_gpt4o] DATASET_NAMES=$DATASET_NAMES"
echo "[run_infer_gpt4o] Output path is inferred by infer.py (use_tool/model/dataset)."

for name in $DATASET_NAMES; do
  echo "========== Infer (use_tool=true): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    with use_tool=true model_path="$MODEL_PATH"
done

for name in $DATASET_NAMES_NOTOOL; do
  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$name" \
    with use_tool=false model_path="$MODEL_PATH" \
    2>/dev/null
done