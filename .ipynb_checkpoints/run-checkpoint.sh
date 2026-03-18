#!/usr/bin/env bash
set -e

# 进入脚本所在目录（evaluation/），保证相对路径一致
cd "$(dirname "$0")"

LLM_CONFIG="src/config/llm_config/Qwen3_4B.yaml"
TOOL_CONFIG="src/config/tool_config/example.yaml"
DATASET_CONFIG_DIR="src/config/dataset_config"
OUTPUT_DIR_TOOL="results_tool"
OUTPUT_DIR_NOTOOL="results_notool"

# 1. 启动 Qwen3-4B 的 vLLM 服务（本地）
python deploy.py --config "$LLM_CONFIG" &
DEPLOY_PID=$!
echo "Started vLLM (PID=$DEPLOY_PID), waiting 20s for it to be ready..."
sleep 20

# 2. 对 config 中所有数据集使用同一 LLM 配置进行推理，分别保留“使用工具”和“不使用工具”的结果
mkdir -p "$OUTPUT_DIR_TOOL" "$OUTPUT_DIR_NOTOOL"
for dataset_config in "$DATASET_CONFIG_DIR"/*.yaml; do
  name=$(basename "$dataset_config" .yaml)

  echo "========== Infer (use_tool=true): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$dataset_config" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool true \
    --output_path "$OUTPUT_DIR_TOOL"

  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$dataset_config" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool false \
    --output_path "$OUTPUT_DIR_NOTOOL"
done

# 3. 可选：结束 vLLM 进程
kill $DEPLOY_PID 2>/dev/null || true
echo "Done."
