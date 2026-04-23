CUDA_VISIBLE_DEVICES=0 python app.py \
  --docs /data/lyz/LocalSearchData/wiki18_100w.jsonl \
  --index /data/lyz/LocalSearchData/indexes/e5_Flat.index \
  --encoder /data/lyz/models/e5-base-v2 \
  --host 0.0.0.0 \
  --port 6006