#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

import faiss
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import uvicorn


def load_docs(path: str) -> List[Dict[str, Any]]:
    """
    支持：
    1) JSONL
    2) JSON array
    要求至少有 contents
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)

        if head == "[":
            arr = json.load(f)
            if not isinstance(arr, list):
                raise ValueError("JSON root must be a list")
            for i, obj in enumerate(arr):
                if not isinstance(obj, dict):
                    continue
                c = str(obj.get("contents", "")).strip()
                if not c:
                    continue
                docs.append({
                    "id": str(obj.get("id", i)),
                    "title": str(obj.get("title", "")),
                    "contents": c,
                    "source": str(obj.get("source", "")),
                })
        else:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                c = str(obj.get("contents", "")).strip()
                if not c:
                    continue
                docs.append({
                    "id": str(obj.get("id", i)),
                    "title": str(obj.get("title", "")),
                    "contents": c,
                    "source": str(obj.get("source", "")),
                })

    if not docs:
        raise ValueError("No valid docs loaded.")
    return docs


class SearchReq(BaseModel):
    query: str
    top_n: int = 10


class BatchReq(BaseModel):
    queries: List[str]
    top_n: int = 10


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class E5Encoder:
    def __init__(self, model_name_or_path: str, device: str = None, max_length: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)
        embeddings = mean_pooling(outputs.last_hidden_state, batch["attention_mask"])

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy().astype("float32")


def build_app(
    docs: List[Dict[str, Any]],
    index,
    encoder: E5Encoder,
    instruction: str = "query: "
) -> FastAPI:
    app = FastAPI(title="FullWiki Dense Search (E5 + FAISS)")

    @app.get("/health")
    def health():
        return {"ok": True, "num_docs": len(docs)}

    @app.post("/search")
    def search(req: SearchReq):
        q = (req.query or "").strip()
        if not q:
            return []

        k = max(1, int(req.top_n))

        query_text = instruction + q
        q_emb = encoder.encode([query_text])

        scores, indices = index.search(q_emb, k)

        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(docs):
                continue
            d = docs[idx]
            out.append({
                "id": d["id"],
                "title": d["title"],
                "contents": d["contents"],
                "source": d["source"],
                "score": float(score),
            })
        return out

    @app.post("/batch_search")
    def batch_search(req: BatchReq):
        clean_queries = [(instruction + q.strip()) for q in req.queries if q and q.strip()]
        if not clean_queries:
            return [[] for _ in req.queries]

        k = max(1, int(req.top_n))
        q_embs = encoder.encode(clean_queries)
        scores_all, indices_all = index.search(q_embs, k)

        results = []
        valid_ptr = 0
        for raw_q in req.queries:
            q = (raw_q or "").strip()
            if not q:
                results.append([])
                continue

            out = []
            scores = scores_all[valid_ptr]
            indices = indices_all[valid_ptr]
            valid_ptr += 1

            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(docs):
                    continue
                d = docs[idx]
                out.append({
                    "id": d["id"],
                    "title": d["title"],
                    "contents": d["contents"],
                    "source": d["source"],
                    "score": float(score),
                })
            results.append(out)

        return results

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Path to docs JSONL/JSON")
    parser.add_argument("--index", required=True, help="Path to FAISS index file")
    parser.add_argument("--encoder", default="intfloat/e5-base-v2", help="Embedding model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动判断")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    docs = load_docs(args.docs)
    index = faiss.read_index(args.index)
    encoder = E5Encoder(
        model_name_or_path=args.encoder,
        device=args.device,
        max_length=args.max_length,
    )

    app = build_app(
        docs=docs,
        index=index,
        encoder=encoder,
        instruction=args.query_prefix
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()