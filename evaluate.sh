#!/usr/bin/env bash
# 按 task：qa / math / expodesign / interaction 顺序评测（需先跑完 infer）
# 推理结果目录：默认 results；可覆盖，例如：
#   OUT_ROOT=autodl-tmp/benchfortir/results/tool/qwen3_4b ./evaluate.sh
# Token 统计：与推理模型 tokenizer 一致（可与 infer 的 llm_config model_path 对齐），例如：
#   TOKENIZER_PATH=/root/autodl-tmp/models/Qwen3-30B ./evaluate.sh
# 不需要 token 统计时：TOKENIZER_PATH= ./evaluate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-/home/lyz/benchfortir/results/tool/gpt_5}"
# 默认与 src/config/llm_config/Qwen3_30b.yaml 的 model_path 一致；本地无该目录时请覆盖或置空跳过 token 统计
TOKENIZER_PATH="${TOKENIZER_PATH:-/data/lyz/models/Qwen3-32B}"
TOK_ARGS=()
if [[ -n "${TOKENIZER_PATH}" ]]; then
  TOK_ARGS=(--tokenizer_path "${TOKENIZER_PATH}")
fi

EVAL_QA="src/config/eval_config/eval_qa.yaml"
EVAL_MATH="src/config/eval_config/eval_math.yaml"
EVAL_EXPO="src/config/eval_config/eval_expo.yaml"
EVAL_INTER="src/config/eval_config/eval_inter.yaml"

echo "========== QA =========="
echo "--- hotpotqa ---"
python evaluate.py --eval_config "$EVAL_QA" \
  --dataset_config src/config/dataset_config/hotpotqa.yaml \
  --output_path "${OUT_ROOT}/hotpotqa_output.json" \
  "${TOK_ARGS[@]}"

echo "--- squadv2 ---"
python evaluate.py --eval_config "$EVAL_QA" \
  --dataset_config src/config/dataset_config/squadv2.yaml \
  --output_path "${OUT_ROOT}/squadv2_output.json" \
  "${TOK_ARGS[@]}"

# echo "--- SimpleQA ---"
# python evaluate.py --eval_config "$EVAL_QA" \
#   --dataset_config src/config/dataset_config/simpleqa.yaml \
#   --output_path "${OUT_ROOT}/SimpleQA_output.json" \
#   "${TOK_ARGS[@]}"

# echo "========== MATH =========="
# echo "--- math (math500) ---"
# python evaluate.py --eval_config "$EVAL_MATH" \
#   --dataset_config src/config/dataset_config/math500.yaml \
#   --output_path "${OUT_ROOT}/math_output.json" \
#   "${TOK_ARGS[@]}"

# echo "--- gsm8k (gsm8k500) ---"
# python evaluate.py --eval_config "$EVAL_MATH" \
#   --dataset_config src/config/dataset_config/gsm8k500.yaml \
#   --output_path "${OUT_ROOT}/gsm8k_output.json" \
#   "${TOK_ARGS[@]}"

# echo "--- omini (omini500) ---"
# python evaluate.py --eval_config "$EVAL_MATH" \
#   --dataset_config src/config/dataset_config/omini500.yaml \
#   --output_path "${OUT_ROOT}/omini_output.json" \
#   "${TOK_ARGS[@]}"

# echo "========== EXPODESIGN =========="
# python evaluate.py --eval_config "$EVAL_EXPO" \
#   --dataset_config src/config/dataset_config/expodesign.yaml \
#   --output_path "${OUT_ROOT}/expodesign_output.json" \
#   "${TOK_ARGS[@]}"

# echo "========== INTERACTION =========="
# python evaluate.py --eval_config "$EVAL_INTER" \
#   --dataset_config src/config/dataset_config/interaction.yaml \
#   --output_path "${OUT_ROOT}/interaction_output.json" \

echo "All evaluation jobs finished."
