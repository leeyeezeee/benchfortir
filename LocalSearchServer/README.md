<h1 id="local-search-service-deployment" style="color: black;">Local Search Service Deployment</h1>

1. For tasks such as HotpotQA, a local Wikipedia retrieval service is required. We deploy the local retrieval service with FlashRAG and FastAPI. First, install the [FlashRAG environment](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#wrench-installation). Then, install the additional dependencies required by this service in the same environment:

```bash
pip install -r requirements.txt
```

2. Enter the corresponding environment, then download the [Wikipedia corpus](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus) and [corresponding retriever models](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary). The index construction method can be found [here](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start), or you can directly download the [pre-indexed Wikipedia](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

More details can be found in the [FlashRAG documentation](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

After the resources are prepared, replace the paths in the command below with your local paths, or edit and run `start_search.sh`.

```bash
python app.py \
  --docs /path/to/wiki18_100w.jsonl \
  --index /path/to/indexes/e5_Flat.index \
  --encoder /path/to/e5-base-v2 \
  --host 0.0.0.0 \
  --port 6006
```

In our experiments, we use `e5-base-v2` as the retriever model.
