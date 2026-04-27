#!/usr/bin/env bash
set -euo pipefail

# Start local vLLM instances via deploy.py.
# Instance count / ports are determined by llm_config (vllm.gpu_ids / vllm.port, etc.).

# 统一在仓库根目录执行，确保 deploy.py 与相对配置路径可解析
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LLM_CONFIG="${LLM_CONFIG:-Qwen3_4B}"

echo "[start_vllm] Using config: ${LLM_CONFIG}"

python deploy.py \
  --config "$LLM_CONFIG" \
  &

DEPLOY_PID=$!

disown "$DEPLOY_PID" 2>/dev/null || true

echo "[start_vllm] Started deploy.py (PID=$DEPLOY_PID); vLLM stderr goes to this terminal."
echo "[start_vllm] Wait for readiness on ports configured by llm_config."
