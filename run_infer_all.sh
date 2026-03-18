#!/usr/bin/env bash
set -euo pipefail

# Run inference against an already-running vLLM service pool.
#
# It builds endpoints from GPUS/BASE_PORT and loops over dataset configs in
# src/config/dataset_config/*.yaml. For each dataset yaml, it extracts dataset
# parameters and calls infer.py with CLI overrides (config defaults still come
# from --config src/config).

cd "$(dirname "$0")"

CONFIG_DIR="${CONFIG_DIR:-src/config}"
DATASET_CONFIG_DIR="${DATASET_CONFIG_DIR:-src/config/dataset_config}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8001}"

mkdir -p "$OUTPUT_DIR"

# Build endpoints list from GPU count / ports
IFS=',' read -r -a GPU_ARR <<<"$GPUS"
ENDPOINTS=()
for i in "${!GPU_ARR[@]}"; do
  port=$((BASE_PORT + i))
  ENDPOINTS+=("http://127.0.0.1:${port}/v1")
done

echo "[run_infer_all] Using CONFIG_DIR=$CONFIG_DIR"
echo "[run_infer_all] Using DATASET_CONFIG_DIR=$DATASET_CONFIG_DIR"
echo "[run_infer_all] Output -> $OUTPUT_DIR"
echo "[run_infer_all] Endpoints: ${ENDPOINTS[*]}"

for dataset_config in "$DATASET_CONFIG_DIR"/*.yaml; do
  name=$(basename "$dataset_config" .yaml)
  echo "========== Infer: $name =========="

  # Extract dataset fields via python (no external yq dependency)
  read -r DATASET_NAME DATA_PATH COUNTS PROMPT_TYPE SAMPLE_TIMEOUT < <(
    python - <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    d = yaml.safe_load(f) or {}
print(d.get("dataset_name",""), d.get("data_path",""), d.get("counts",""), d.get("prompt_type",""), d.get("sample_timeout",""))
PY
    "$dataset_config"
  )

  python infer.py \
    --config "$CONFIG_DIR" \
    --dataset_name "$DATASET_NAME" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_DIR" \
    ${COUNTS:+--counts "$COUNTS"} \
    ${PROMPT_TYPE:+--prompt_type "$PROMPT_TYPE"} \
    ${SAMPLE_TIMEOUT:+--sample_timeout "$SAMPLE_TIMEOUT"} \
    --endpoints "${ENDPOINTS[@]}"
done

