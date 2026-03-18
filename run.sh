#!/usr/bin/env bash
set -euo pipefail

# 进入脚本所在目录（evaluation/），保证相对路径一致
cd "$(dirname "$0")"

LLM_CONFIG="src/config/llm_config/Qwen3_4B.yaml"
TOOL_CONFIG="src/config/tool_config/example.yaml"
DATASET_CONFIG_DIR="src/config/dataset_config"
OUTPUT_DIR_TOOL="results_tool"
OUTPUT_DIR_NOTOOL="results_notool"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 1. 启动 4 个 vLLM 实例（4*4090 数据并行）
GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
STARTUP_SLEEP="${STARTUP_SLEEP:-0}"

DEPLOY_PID=""
cleanup() {
  echo "[run] interrupt received, stopping vLLM..."
  if [[ -n "${DEPLOY_PID}" ]] && kill -0 "${DEPLOY_PID}" 2>/dev/null; then
    # send SIGINT so deploy.py can stop all child vLLM instances
    kill -INT "${DEPLOY_PID}" 2>/dev/null || true
    wait "${DEPLOY_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup INT TERM

python deploy.py \
  --config "$LLM_CONFIG" \
  --gpus "$GPUS" \
  --base_port "$BASE_PORT" \
  --host "$HOST" \
  --error_only \
  --log_dir "$LOG_DIR" \
  --startup_sleep "$STARTUP_SLEEP" \
  1>/dev/null 2>>"$LOG_DIR/deploy.err.log" &
DEPLOY_PID=$!

echo "[run] Started vLLM deployer (PID=$DEPLOY_PID), waiting 20s..."
sleep 20

# infer.py 多 endpoint
IFS=',' read -r -a GPU_ARR <<<"$GPUS"
ENDPOINTS=()
for i in "${!GPU_ARR[@]}"; do
  port=$((BASE_PORT + i))
  ENDPOINTS+=("http://127.0.0.1:${port}/v1")
done

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
    --output_path "$OUTPUT_DIR_TOOL" \
    --endpoints "${ENDPOINTS[@]}" \
    1>/dev/null 2>>"$LOG_DIR/infer.err.log"

  echo "========== Infer (use_tool=false): $name =========="
  python infer.py \
    --llm_config "$LLM_CONFIG" \
    --dataset_config "$dataset_config" \
    --tool_config "$TOOL_CONFIG" \
    --use_tool false \
    --output_path "$OUTPUT_DIR_NOTOOL" \
    --endpoints "${ENDPOINTS[@]}" \
    1>/dev/null 2>>"$LOG_DIR/infer.err.log"
done

# 3. 结束 vLLM 进程（通过 deployer 清理所有实例）
cleanup
