#!/usr/bin/env python
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
    derive_output_path,
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
        "llm_config": None,
        "dataset_config": None,
        "use_tool": True,
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
            ("--llm_config", {"type": str, "default": None}),
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
    llm_path = resolve_named_yaml(
        bootstrap.llm_config,
        "llm_config",
        root_dir=os.getcwd(),
    )

    for label, raw, resolved in (
        ("default_config", bootstrap.default_config, default_path),
        ("llm_config", bootstrap.llm_config, llm_path),
        ("dataset_config", bootstrap.dataset_config, dataset_path),
    ):
        if raw and resolved and not os.path.isfile(resolved):
            print(f"[evaluate] Warning: {label} not found: {resolved} (from {raw!r})")

    default_defaults = load_yaml(default_path)
    llm_defaults = load_yaml(llm_path)
    dataset_defaults = load_yaml(dataset_path)

    runtime_config = _evaluate_base_config()
    runtime_config.update(
        {
            "default_config": bootstrap.default_config,
            "llm_config": bootstrap.llm_config,
            "dataset_config": bootstrap.dataset_config,
        }
    )
    runtime_config.update(default_defaults)
    runtime_config.update(llm_defaults)
    runtime_config.update(dataset_defaults)

    # tokenizer_path: only infer from llm_config.model_path when not explicitly provided
    if not runtime_config.get("tokenizer_path"):
        llm_model_path = llm_defaults.get("model_path")
        if llm_model_path:
            runtime_config["tokenizer_path"] = llm_model_path

    loaded_from = []
    if default_path and os.path.isfile(default_path):
        loaded_from.append(f"default={default_path}")
    if llm_path and os.path.isfile(llm_path):
        loaded_from.append(f"llm={llm_path}")
    if dataset_path and os.path.isfile(dataset_path):
        loaded_from.append(f"dataset={dataset_path}")
    if loaded_from:
        print(f"[evaluate] Loaded config defaults from: {', '.join(loaded_from)}")

    return runtime_config, sacred_argv


def _normalize_eval_config(config: dict):
    config["use_llm"] = _coerce_bool(config.get("use_llm"))
    if config["use_llm"] is None:
        config["use_llm"] = False

    model_name = config.get("default_model")
    if not model_name:
        raise ValueError(
            "default_model is required in llm_config for output_path derivation. "
            "Set it in llm_config or override with Sacred, for example: with default_model='Qwen3-4B'"
        )
    config["output_path"] = derive_output_path(
        current_output=config.get("output_path"),
        dataset_name=config.get("dataset_name"),
        use_tool=config.get("use_tool"),
        model_name=model_name,
    )

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
            ("--llm_config", {"type": str, "default": None}),
            ("--dataset_config", {"type": str, "default": None}),
        ),
    )
    base_config, sacred_argv = _build_runtime_config(cli_args)

    experiment = build_experiment("evaluation_metrics", base_config)

    @experiment.main
    def evaluate_entry(_config):
        return asyncio.run(main(dict(_config)))

    if getattr(bootstrap, "_show_help", False):
        print(
            "Bootstrap options: --default_config <name> --llm_config <name> "
            "--dataset_config <name>"
        )
        print(
            "Sacred overrides: python evaluate.py --dataset_config math500 "
            "--llm_config Qwen3_4B with use_tool=true"
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
