#!/usr/bin/env bash
set -euo pipefail

PID_FILE="run/vllm_pids.txt"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[clean] PID 文件不存在: $PID_FILE"
  exit 0
fi

echo "[clean] 从 $PID_FILE 读取 PID 并清理..."

while IFS= read -r pid; do
  [[ -z "${pid:-}" ]] && continue
  [[ "$pid" =~ ^[0-9]+$ ]] || continue

  if ps -p "$pid" > /dev/null 2>&1; then
    cmd="$(ps -p "$pid" -o args= || true)"
    if [[ "$cmd" == *"vllm serve"* ]]; then
      echo "[clean] kill PID=$pid"
      kill "$pid" || true
    else
      echo "[clean] 跳过 PID=$pid（非 vllm serve）"
    fi
  else
    echo "[clean] PID=$pid 不存在，跳过"
  fi
done < "$PID_FILE"

sleep 1

while IFS= read -r pid; do
  [[ -z "${pid:-}" ]] && continue
  [[ "$pid" =~ ^[0-9]+$ ]] || continue

  if ps -p "$pid" > /dev/null 2>&1; then
    cmd="$(ps -p "$pid" -o args= || true)"
    if [[ "$cmd" == *"vllm serve"* ]]; then
      echo "[clean] force kill -9 PID=$pid"
      kill -9 "$pid" || true
    fi
  fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "[clean] 清理完成，已删除 PID 文件。"