#!/usr/bin/env bash
set -euo pipefail

# 通过 deploy.py 启动 vLLM（可多实例，逻辑见 deploy.py）。
# 设置 VLLM_TOTAL_GPUS（或 DEPLOY_TOTAL_GPUS）与 LLM_CONFIG 中 vllm.tensor_parallel_size 决定实例数。

cd "$(dirname "$0")"

LLM_CONFIG="${LLM_CONFIG:-src/config/llm_config/Qwen3_30B.yaml}"
export VLLM_TOTAL_GPUS="${VLLM_TOTAL_GPUS:-4}"

echo "[start_vllm] Using config: ${LLM_CONFIG} VLLM_TOTAL_GPUS=${VLLM_TOTAL_GPUS}"

python deploy.py \
  --config "$LLM_CONFIG" \
  &

DEPLOY_PID=$!

disown "$DEPLOY_PID" 2>/dev/null || true

echo "[start_vllm] Started deploy.py (PID=$DEPLOY_PID); vLLM stderr goes to this terminal"
echo "[start_vllm] Wait for readiness on ports from vllm.port upward."
