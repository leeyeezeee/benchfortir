CUDA_VISIBLE_DEVICES=0 python app.py \
  --docs /path/to/wiki18_100w.jsonl \
  --index /path/to/indexes/e5_Flat.index \
  --encoder /path/to/e5-base-v2 \
  --host 0.0.0.0 \
  --port 6006