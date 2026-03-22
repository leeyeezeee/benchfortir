#!/usr/bin/env bash
# 按 task：qa / math / expodesign / interaction 顺序评测（需先跑完 infer）
# 推理结果目录：默认 results；可覆盖，例如：
#   OUT_ROOT=autodl-tmp/benchfortir/results/tool/qwen3_4b ./evaluate_4b.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/benchfortir/results/tool/qwen3_4b}"

EVAL_QA="src/config/eval_config/eval_qa.yaml"
EVAL_MATH="src/config/eval_config/eval_math.yaml"
EVAL_EXPO="src/config/eval_config/eval_expo.yaml"
EVAL_INTER="src/config/eval_config/eval_inter.yaml"

echo "========== QA =========="
echo "--- hotpotqa ---"
python evaluate.py --eval_config "$EVAL_QA" \
  --dataset_config src/config/dataset_config/hotpotqa.yaml \
  --output_path "${OUT_ROOT}/hotpotqa_output.json"

echo "--- squadv2 ---"
python evaluate.py --eval_config "$EVAL_QA" \
  --dataset_config src/config/dataset_config/squadv2.yaml \
  --output_path "${OUT_ROOT}/squadv2_output.json"

echo "--- SimpleQA ---"
python evaluate.py --eval_config "$EVAL_QA" \
  --dataset_config src/config/dataset_config/simpleqa.yaml \
  --output_path "${OUT_ROOT}/SimpleQA_output.json"

echo "========== MATH =========="
echo "--- math (math500) ---"
python evaluate.py --eval_config "$EVAL_MATH" \
  --dataset_config src/config/dataset_config/math500.yaml \
  --output_path "${OUT_ROOT}/math_output.json"

echo "--- gsm8k (gsm8k500) ---"
python evaluate.py --eval_config "$EVAL_MATH" \
  --dataset_config src/config/dataset_config/gsm8k500.yaml \
  --output_path "${OUT_ROOT}/gsm8k_output.json"

echo "--- omini (omini500) ---"
python evaluate.py --eval_config "$EVAL_MATH" \
  --dataset_config src/config/dataset_config/omini500.yaml \
  --output_path "${OUT_ROOT}/omini_output.json"

echo "========== EXPODESIGN =========="
python evaluate.py --eval_config "$EVAL_EXPO" \
  --dataset_config src/config/dataset_config/expodesign.yaml \
  --output_path "${OUT_ROOT}/expodesign_output.json"

echo "========== INTERACTION =========="
python evaluate.py --eval_config "$EVAL_INTER" \
  --dataset_config src/config/dataset_config/interaction.yaml \
  --output_path "${OUT_ROOT}/interaction_output.json"

echo "All evaluation jobs finished."