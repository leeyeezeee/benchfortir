#!/usr/bin/env bash
# 按 task：qa / math / expodesign / interaction 顺序评测（需先跑完 infer）
# 评测输入路径由 evaluate.py 自动推导：
#   results/tool|notool/<llm_name>/<dataset_name>_output.json
# 若推理目录不遵循该约定，请在命令中手动覆盖 --output_path
# Token 统计：与推理模型 tokenizer 一致（可与 infer 的 llm_config model_path 对齐），例如：
#   TOKENIZER_PATH=/root/autodl-tmp/models/Qwen3-30B ./evaluate.sh
# 不需要 token 统计时：TOKENIZER_PATH= ./evaluate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LLM_CONFIG="${LLM_CONFIG:-Qwen3_4B}"
USE_TOOL="${USE_TOOL:-true}"
# 可手动指定；未指定时 evaluate.py 会回退到 llm_config.model_path
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
TOK_ARGS=()
if [[ -n "${TOKENIZER_PATH}" ]]; then
  TOK_ARGS=(--tokenizer_path "${TOKENIZER_PATH}")
fi

# echo "========== QA =========="
# echo "--- hotpotqa ---"
# python evaluate.py \
#   --llm_config "${LLM_CONFIG}" \
#   --dataset_config hotpotqa \
#   with use_tool="${USE_TOOL}" \
#   "${TOK_ARGS[@]}"

# echo "--- squadv2 ---"
# python evaluate.py \
#   --llm_config "${LLM_CONFIG}" \
#   --dataset_config squadv2 \
#   with use_tool="${USE_TOOL}" \
#   "${TOK_ARGS[@]}"

# echo "--- SimpleQA ---"
# python evaluate.py \
#   --llm_config "${LLM_CONFIG}" \
#   --dataset_config simpleqa \
#   with use_tool="${USE_TOOL}" \
#   "${TOK_ARGS[@]}"

echo "========== MATH =========="
echo "--- math (math500) ---"
python evaluate.py \
  --llm_config "${LLM_CONFIG}" \
  --dataset_config math500 \
  with use_tool="${USE_TOOL}" \
  "${TOK_ARGS[@]}"

# echo "--- gsm8k (gsm8k500) ---"
# python evaluate.py \
#   --llm_config "${LLM_CONFIG}" \
#   --dataset_config gsm8k500 \
#   with use_tool="${USE_TOOL}" \
#   "${TOK_ARGS[@]}"

# echo "--- omini (omini500) ---"
# python evaluate.py \
#   --llm_config "${LLM_CONFIG}" \
#   --dataset_config omini500 \
#   with use_tool="${USE_TOOL}" \
#   "${TOK_ARGS[@]}"

# echo "========== EXPODESIGN =========="
# python evaluate.py \
#   --llm_config "${LLM_CONFIG}" \
#   --dataset_config expodesign \
#   with use_tool="${USE_TOOL}" \
#   "${TOK_ARGS[@]}"

# echo "========== INTERACTION =========="
# python evaluate.py \
#   --llm_config "${LLM_CONFIG}" \
#   --dataset_config interaction \
#   with use_tool="${USE_TOOL}" \

echo "All evaluation jobs finished."
