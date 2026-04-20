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
      "read_time": float,
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

import asyncio
import json
import time
from typing import Any, List, Optional, Sequence

from src.evaluator import Evaluator
from src.sacred_config import (
    build_experiment,
    dict_to_namespace,
    load_yaml,
    parse_bootstrap_args,
    resolve_named_yaml,
)


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


def _evaluate_base_config() -> dict:
    return {
        "default_config": "default",
        "dataset_config": None,
        "output_path": None,
        "task": None,
        "use_llm": False,
        "api_base_url": None,
        "model_name": None,
        "api_key": "EMPTY",
        "concurrent_limit": 50,
        "timeout": 1800,
        "tokenizer_path": None,
    }


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _build_runtime_config(argv: Sequence[str]) -> tuple[dict, list[str]]:
    bootstrap, sacred_argv = parse_bootstrap_args(
        argv,
        description="Evaluation bootstrap options",
        options=(
            ("--default_config", {"type": str, "default": "default"}),
            ("--dataset_config", {"type": str, "default": None}),
        ),
    )

    default_name = bootstrap.default_config.strip() if bootstrap.default_config else ""
    if default_name.lower().endswith(".yaml"):
        default_file = default_name
    elif default_name.lower().endswith(".yml"):
        default_file = default_name[:-4] + ".yaml"
    else:
        default_file = default_name + ".yaml"
    default_path = os.path.join(os.getcwd(), "src", "config", default_file)
    dataset_path = resolve_named_yaml(
        bootstrap.dataset_config,
        "dataset_config",
        root_dir=os.getcwd(),
    )

    for label, raw, resolved in (
        ("default_config", bootstrap.default_config, default_path),
        ("dataset_config", bootstrap.dataset_config, dataset_path),
    ):
        if raw and resolved and not os.path.isfile(resolved):
            print(f"[evaluate] Warning: {label} not found: {resolved} (from {raw!r})")

    default_defaults = load_yaml(default_path)
    dataset_defaults = load_yaml(dataset_path)

    runtime_config = _evaluate_base_config()
    runtime_config.update(
        {
            "default_config": bootstrap.default_config,
            "dataset_config": bootstrap.dataset_config,
        }
    )
    runtime_config.update(default_defaults)
    runtime_config.update(dataset_defaults)

    ds_name = dataset_defaults.get("dataset_name")
    ds_out_root = dataset_defaults.get("output_path")
    if ds_name and ds_out_root:
        runtime_config["output_path"] = os.path.join(ds_out_root, f"{ds_name}_output.json")

    loaded_from = []
    if default_path and os.path.isfile(default_path):
        loaded_from.append(f"default={default_path}")
    if dataset_path and os.path.isfile(dataset_path):
        loaded_from.append(f"dataset={dataset_path}")
    if loaded_from:
        print(f"[evaluate] Loaded config defaults from: {', '.join(loaded_from)}")

    return runtime_config, sacred_argv


def _normalize_eval_config(config: dict):
    config["use_llm"] = _coerce_bool(config.get("use_llm"))
    if config["use_llm"] is None:
        config["use_llm"] = False

    if not config.get("task"):
        raise ValueError(
            "task is required. Provide it in dataset_config or override with Sacred, "
            "for example: with task='math'"
        )
    if not config.get("output_path"):
        raise ValueError(
            "output_path is required when it cannot be derived from dataset_config. "
            "Override with Sacred, for example: with output_path='results/math_output.json'"
        )
    if config["use_llm"] and (not config.get("api_base_url") or not config.get("model_name")):
        raise ValueError(
            "use_llm requires api_base_url and model_name. "
            "Set them in default.yaml or override with Sacred."
        )
    return dict_to_namespace(config)


async def main(config: dict) -> dict:
    args = _normalize_eval_config(config)

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

        return await evaluator.run(data, timeout=args.timeout)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def run_from_cli(argv: Optional[Sequence[str]] = None):
    cli_args = list(argv if argv is not None else sys.argv[1:])
    bootstrap, _ = parse_bootstrap_args(
        cli_args,
        description="Evaluation bootstrap options",
        options=(
            ("--default_config", {"type": str, "default": "default"}),
            ("--dataset_config", {"type": str, "default": None}),
        ),
    )
    base_config, sacred_argv = _build_runtime_config(cli_args)

    experiment = build_experiment("evaluation_metrics", base_config)

    @experiment.main
    def evaluate_entry(_config):
        return asyncio.run(main(dict(_config)))

    if getattr(bootstrap, "_show_help", False):
        print("Bootstrap options: --default_config <name> --dataset_config <name>")
        print(
            "Sacred overrides: python evaluate.py --dataset_config math500 "
            "with output_path='results/math_output.json'"
        )

    results = experiment.run_commandline(sacred_argv)
    return results.result


if __name__ == "__main__":
    results = run_from_cli()

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
