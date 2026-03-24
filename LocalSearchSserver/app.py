# file: fullwiki_bm25_server.py
#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import uvicorn


def simple_tokenize(text: str) -> List[str]:
    # 简单英文 token，够用；后续可替换更强 tokenizer
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def load_docs(path: str) -> List[Dict[str, Any]]:
    """
    支持两种格式：
    1) JSONL: 每行一个对象，至少包含 contents
    2) JSON array: [ {...}, {...} ]
    可选字段: id/title/source
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
        raise ValueError("No valid docs loaded; make sure each record has non-empty 'contents'.")
    return docs


class SearchReq(BaseModel):
    query: str
    top_n: int = 10


class BatchReq(BaseModel):
    queries: List[str]
    top_n: int = 10


def build_app(docs: List[Dict[str, Any]]) -> FastAPI:
    app = FastAPI(title="FullWiki BM25 Local Search")

    corpus_tokens = [simple_tokenize(d["contents"]) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)

    @app.get("/health")
    def health():
        return {"ok": True, "num_docs": len(docs)}

    @app.post("/search")
    def search(req: SearchReq):
        q = (req.query or "").strip()
        if not q:
            return []

        q_tokens = simple_tokenize(q)
        if not q_tokens:
            return []

        scores = bm25.get_scores(q_tokens)
        k = max(1, int(req.top_n))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        out = []
        for idx in top_idx:
            d = docs[idx]
            out.append({
                "id": d["id"],
                "title": d["title"],
                "contents": d["contents"],   # local_search_tool 依赖这个字段
                "source": d["source"],
                "score": float(scores[idx]),
            })
        return out

    @app.post("/batch_search")
    def batch_search(req: BatchReq):
        out_all = []
        for q in req.queries:
            single = search(SearchReq(query=q, top_n=req.top_n))
            out_all.append(single)
        return out_all

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Path to wiki docs JSONL/JSON; each record needs 'contents'.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    docs = load_docs(args.docs)
    app = build_app(docs)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()