#!/usr/bin/env python
"""benchfortir/evaluate.py

Evaluate *already generated* model outputs and compute TIR benchmark metrics.

This script DOES NOT run inference.
You must first run `infer.py` to produce an output file, then run this script
against that output file.

=================
Input: --output_path
=================
`--output_path` must point to a file produced by this repository's inference
pipeline (see `src/inference_engine.py` and `src/sample_processor.py`).

Supported file types:
- .json   : a single JSON array, i.e. `List[Dict[str, Any]]`
- .jsonl  : JSON Lines, i.e. one `Dict[str, Any]` per line
- .txt    : treated the same as .jsonl

Each sample record is a JSON object (Python: `Dict[str, Any]`).

The *typical* schema produced by this repo looks like:

Required fields (for correctness metrics)
----------------------------------------
- input (str): the question / problem text shown to the model.
- answer (str | List[str] | bool): the gold answer(s).
    * math benchmarks usually provide a string answer
    * QA benchmarks may provide a string or a list of acceptable answers
- prediction (str): the extracted final answer produced by `src/utils.extract_answer`.

Optional fields (for process/tool/cost metrics)
----------------------------------------------
- output (str): the full interaction trace (assistant output + tool tags/results),
  usually containing tags emitted by the prompt template:
    <think>...</think>   intermediate reasoning (not always present)
    <search>...</search> search tool call (may appear multiple times)
    <python>...</python> python tool call (may appear multiple times)
    <result>...</result> tool result blocks inserted by the system
    <answer>...</answer> final answer block

- instruction (str): the system prompt.
- timing (dict): timing info written by `SampleProcessor.log_timing()`, e.g.
    {
      "llm_time": float,
      "python_time": float,
      "search_time": float,
      "total_time": float
    }

=================
Outputs
=================
Two files are written next to the input file:

1) Per-sample metrics
   <output_path_without_ext>_metrics.json
   Type: `List[Dict[str, Any]]`
   Each element is the original sample record plus an extra key:
      - metrics (dict): per-sample metrics

2) Aggregated metrics
   <output_path_without_ext>_metrics_overall.json
   Type: `Dict[str, Any]`
   Contains overall averages/sums for correctness, tool usage, and cost metrics.

=================
Token accounting
=================
The inference pipeline does not store token usage by default.

If you provide `--tokenizer_path`, the evaluator will *estimate* token usage by
re-tokenizing:
- the prompt reconstructed from (instruction, input)
- plus the stored full trace (output)

This is an approximation. It is useful for comparing relative space cost across
runs, but may not match the exact token counts used internally by vLLM.

Return value
============
`main()` returns the aggregated metrics dictionary (same content as the overall
metrics JSON written to disk). On failure, it returns:
    {"status": "error", "message": "..."}

"""

import sys
import os
sys.path.append(os.getcwd())

import argparse
import asyncio
import json
import time
from typing import Any, List

from src.evaluator import Evaluator


def _load_output_file(path: str) -> List[Any]:
    """Load a JSON/JSONL output file.

    Returns
    -------
    List[Any]
        - For .json: the parsed JSON array
        - For .jsonl/.txt: list of parsed JSON objects

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is unsupported or JSON is invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Output file does not exist: {path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON array in {path}, got: {type(data)}")
            return data
        if ext in {".jsonl", ".txt"}:
            out: List[Any] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
            return out

    raise ValueError(f"Unsupported file format: {ext} (supported: .json, .jsonl, .txt)")


async def main() -> dict:
    parser = argparse.ArgumentParser(description="Evaluation Tool (TIR metrics)")

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the model output file produced by infer.py (.json or .jsonl).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["math", "qa"],
        help="Task type. qa => EM/F1; math => correctness with math equivalence.",
    )

    # Optional: LLM-based judging
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for equivalence evaluation")
    parser.add_argument("--api_base_url", type=str, default=None, help="Base URL of the LLM API")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the LLM model used for evaluation")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for LLM judge (if needed)")

    parser.add_argument("--concurrent_limit", type=int, default=50, help="Max concurrent evaluations")

    # IMPORTANT: this is a per-sample timeout (used inside Evaluator.evaluate_batch)
    parser.add_argument("--timeout", type=int, default=1800, help="Per-sample evaluation timeout (seconds)")

    # Optional: tokenizer for token counting
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer name/path for estimating token usage. Usually the same as infer.py --model_path.",
    )

    args = parser.parse_args()

    try:
        print(f"Model output file path: {args.output_path}")

        print("Loading data...")
        t0 = time.time()
        data: List[Any] = _load_output_file(args.output_path)
        print(f"Data loading completed. Total {len(data)} samples. Time: {time.time() - t0:.2f}s")

        evaluator = Evaluator(
            task_type=args.task,
            output_path=args.output_path,
            use_llm=args.use_llm,
            api_base_url=args.api_base_url,
            model_name=args.model_name,
            api_key=args.api_key,
            concurrent_limit=args.concurrent_limit,
            tokenizer_name_or_path=args.tokenizer_path,
        )

        print(f"Detailed metrics path: {evaluator.output_metrics_path}")
        print(f"Overall metrics path: {evaluator.output_metrics_overall_path}")

        overall_metrics = await evaluator.run(data, timeout=args.timeout)
        return overall_metrics

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    results = asyncio.run(main())

    if results.get("status") in {"error", "timeout"}:
        print(f"\n===== Evaluation Not Completed: {results.get('status')} =====")
        if results.get("message"):
            print(f"Reason: {results.get('message')}")
        sys.exit(1)

    print("\n===== Evaluation Summary =====")
    for key, value in results.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    sys.exit(0)
