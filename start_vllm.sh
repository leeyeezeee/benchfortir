#!/usr/bin/env bash
set -euo pipefail

# Start vLLM services via deploy.py (detached).
# This script is intentionally separated from inference runs so that:
# - vLLM can be long-lived and reused across multiple infer runs
# - infer failures/restarts do not kill the model service

cd "$(dirname "$0")"

LLM_CONFIG="${LLM_CONFIG:-src/config/llm_config/llm_for_test.yaml}"
GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
STARTUP_SLEEP="${STARTUP_SLEEP:-0}"

PID_FILE="${PID_FILE:-.vllm_deploy.pid}"

echo "[start_vllm] Using config: ${LLM_CONFIG}"
echo "[start_vllm] GPUS=${GPUS} BASE_PORT=${BASE_PORT} HOST=${HOST}"

python deploy.py \
  --config "$LLM_CONFIG" \
  --gpus "$GPUS" \
  --base_port "$BASE_PORT" \
  --host "$HOST" \
  --error_only \
  --log_dir "" \
  --startup_sleep "$STARTUP_SLEEP" \
  1>/dev/null &

DEPLOY_PID=$!
echo "$DEPLOY_PID" > "$PID_FILE"

echo "[start_vllm] Started vLLM deployer (PID=$DEPLOY_PID)."
echo "[start_vllm] PID saved to $PID_FILE"
echo "[start_vllm] Wait for readiness before running inference."

