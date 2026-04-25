import sys
import os
sys.path.append(os.getcwd())
import asyncio
import nltk
from typing import Optional, Sequence
# print(nltk.data.path)
# nltk.download('punkt')
# nltk.download('punkt_tab')

from src.sacred_config import (
    build_experiment,
    derive_output_path,
    dict_to_namespace,
    load_yaml,
    parse_bootstrap_args,
    resolve_named_yaml,
)


def resolve_config_yaml_path(raw: Optional[str], config_subdir: str) -> Optional[str]:
    """Resolve ``<name>.yaml`` from ``src/config/<config_subdir>``."""
    return resolve_named_yaml(raw, config_subdir, root_dir=os.getcwd())


def load_config_defaults(
    default_config_path: str = None,
    llm_config_path: str = None,
    dataset_config_path: str = None,
) -> dict:
    """
    从共享默认值、模型配置、数据集配置加载并合并为 infer 参数字典。

    使用方式（示例）：只传配置文件名（stem），从固定目录加载::

        python infer.py \\
          --default_config default \\
          --llm_config Qwen3_8B \\
          --dataset_config math500

    对应文件为 ``src/config/default.yaml``、
    ``src/config/llm_config/Qwen3_8B.yaml``、
    ``src/config/dataset_config/math500.yaml``；不接受路径，只接受文件名。

    合并规则：
    - 依次按 default_config → llm_config → dataset_config 的顺序加载；
    - 后加载的同名字段会覆盖先前的值；
    - 缺失的文件会被忽略。
    """
    merged: dict = {}
    for p in (default_config_path, llm_config_path, dataset_config_path):
        if not p:
            continue
        # 允许相对路径；相对于当前工作目录解析
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(os.getcwd(), p))
        if os.path.isfile(p):
            merged.update(load_yaml(p))
    return _config_to_infer_defaults(merged)


def _config_to_infer_defaults(c: dict) -> dict:
    """将合并后的 YAML 字典映射为 infer 命令行参数字典（用于 defaults）。"""
    vllm = c.get("vllm") or {}
    remote = vllm.get("remote", False)
    if remote:
        endpoints = vllm.get("endpoints") or []
        api_keys = vllm.get("api_keys") or []
    else:
        host = vllm.get("host", "0.0.0.0")
        port = vllm.get("port", 8001)
        endpoints = [f"http://{host}:{port}/v1"] if host and port else []
        api_keys = vllm.get("api_keys") or []

    def _list_or_none(x):
        if x is None:
            return None
        return x if isinstance(x, list) else [x]

    defaults = {
        "remote": bool(remote),
        "endpoints": endpoints or None,
        "model_path": c.get("model_path"),
        "api_keys": _list_or_none(api_keys) if api_keys else None,
        "default_model": c.get("default_model") or c.get("llm_name") or "Qwen2.5-7B-Instruct",
        "temperature": c.get("temperature", 0),
        "max_tokens": c.get("max_tokens", 4096),
        "top_p": c.get("top_p", 0.8),
        "top_k": c.get("top_k", 20),
        "min_p": c.get("min_p", 0.0),
        "repetition_penalty": c.get("repetition_penalty", 1.1),
        "include_stop_str_in_output": c.get("include_stop_str_in_output", True),
        "max_concurrent_requests": c.get("max_concurrent_requests", 50),
        "dataset_name": c.get("dataset_name", "math"),
        "output_path": c.get("output_path"),
        "prompt_type": c.get("prompt_type", "code_search"),
        "counts": c.get("counts", 500),
        "data_path": c.get("data_path"),
        "max_python_times": c.get("max_python_times", 5),
        "max_search_times": c.get("max_search_times", 3),
        "max_read_times": c.get("max_read_times", 3),
        "sample_timeout": c.get("sample_timeout", 120),
        "use_sds": c.get("use_sds", False),
        "conda_path": c.get("conda_path"),
        "conda_env": c.get("conda_env"),
        "python_max_concurrent": c.get("python_max_concurrent", 32),
        "read_allowed_roots": c.get("read_allowed_roots"),
        "read_max_chars": c.get("read_max_chars", 8000),
        "read_timeout": c.get("read_timeout", 30),
        "read_enable_ocr": c.get("read_enable_ocr", False),
        "bing_api_key": c.get("bing_api_key", ""),
        "bing_zone": c.get("bing_zone", "serp_api1"),
        "search_max_results": c.get("search_max_results", 10),
        "search_result_length": c.get("search_result_length", 1000),
        "bing_requests_per_second": c.get("bing_requests_per_second", 2.0),
        "bing_max_retries": c.get("bing_max_retries", 3),
        "bing_retry_delay": c.get("bing_retry_delay", 1.0),
        "summ_model_urls": c.get("summ_model_urls") or ["http://localhost:8004/v1"],
        "summ_model_name": c.get("summ_model_name", "Qwen2.5-72B-Instruct"),
        "summ_model_path": c.get("summ_model_path"),
        "search_cache_file": c.get("search_cache_file", "search_cache.db"),
        "url_cache_file": c.get("url_cache_file", "search_url_cache.db"),
        "use_local_search": c.get("use_local_search", False),
        "local_search_url": c.get("local_search_url"),
        "max_sequence_length": c.get("max_sequence_length", 20000),
        "compatible_search": c.get("compatible_search", False),
        # Whether the model is allowed to call tools (tag-based protocols)
        "use_tool": c.get("use_tool", True),
        # Local search post-process summarization (requires model_path for tokenizer when True)
        "use_summarize": c.get("use_summarize", False),
        # Interaction (customer side) configuration
        "customer_api_key": c.get("customer_api_key"),
        "customer_base_url": c.get("customer_base_url"),
        "customer_model": c.get("customer_model", "kimi-k2-0905-preview"),
        "customer_temperature": c.get("customer_temperature", 0.2),
        "customer_max_tokens": c.get("customer_max_tokens", 2048),
        "max_turns": c.get("max_turns", 10),
        # Interaction scenario JSONL（见 data_loader_inter._resolve_interaction_jsonl_path）
        "inter_filename": c.get("inter_filename"),
        "inter_data_path": c.get("inter_data_path"),
    }
    if isinstance(c.get("summ_model_urls"), list):
        defaults["summ_model_urls"] = c["summ_model_urls"]
    return defaults


def _infer_base_config() -> dict:
    """Default infer config before YAML files or Sacred overrides are applied."""
    return {
        "default_config": "default",
        "llm_config": None,
        "dataset_config": None,
        "print_keys": False,
        "remote": False,
        "endpoints": None,
        "model_path": None,
        "api_keys": None,
        "default_model": "Qwen2.5-7B-Instruct",
        "temperature": 0,
        "max_tokens": 4096,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "include_stop_str_in_output": True,
        "max_concurrent_requests": 50,
        "dataset_name": "math",
        "output_path": None,
        "prompt_type": "code_search",
        "counts": 500,
        "data_path": None,
        "max_python_times": 5,
        "max_search_times": 3,
        "max_read_times": 3,
        "sample_timeout": 120,
        "use_sds": False,
        "use_tool": True,
        "use_summarize": False,
        "conda_path": None,
        "conda_env": None,
        "python_max_concurrent": 32,
        "read_allowed_roots": None,
        "read_max_chars": 8000,
        "read_timeout": 30,
        "read_enable_ocr": False,
        "bing_api_key": "",
        "bing_zone": "serp_api1",
        "search_max_results": 10,
        "search_result_length": 1000,
        "bing_requests_per_second": 2.0,
        "bing_max_retries": 3,
        "bing_retry_delay": 1.0,
        "summ_model_urls": ["http://localhost:8004/v1"],
        "summ_model_name": "Qwen2.5-72B-Instruct",
        "summ_model_path": None,
        "search_cache_file": "search_cache.db",
        "url_cache_file": "search_url_cache.db",
        "use_local_search": False,
        "local_search_url": None,
        "max_sequence_length": 20000,
        "compatible_search": False,
        "customer_api_key": None,
        "customer_base_url": None,
        "customer_model": "kimi-k2-0905-preview",
        "customer_temperature": 0.2,
        "customer_max_tokens": 2048,
        "max_turns": 10,
        "inter_filename": None,
        "inter_data_path": None,
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
    """Load YAML defaults first, then leave remaining CLI args to Sacred."""
    bootstrap, sacred_argv = parse_bootstrap_args(
        argv,
        description="Asynchronous inference engine bootstrap options",
        options=(
            ("--default_config", {"type": str, "default": "default"}),
            ("--llm_config", {"type": str, "default": None}),
            ("--dataset_config", {"type": str, "default": None}),
            ("--print_keys", {"action": "store_true", "default": False}),
        ),
    )

    default_resolved = resolve_named_yaml(
        bootstrap.default_config,
        "",
        root_dir=os.getcwd(),
        option_name="default_config",
    )
    llm_resolved = resolve_config_yaml_path(bootstrap.llm_config, "llm_config")
    dataset_resolved = resolve_config_yaml_path(bootstrap.dataset_config, "dataset_config")

    for label, raw, resolved in (
        ("default_config", bootstrap.default_config, default_resolved),
        ("llm_config", bootstrap.llm_config, llm_resolved),
        ("dataset_config", bootstrap.dataset_config, dataset_resolved),
    ):
        if raw and resolved and not os.path.isfile(resolved):
            print(f"[infer] Warning: {label} not found: {resolved} (from {raw!r})")

    runtime_config = _infer_base_config()
    runtime_config.update(
        {
            "default_config": bootstrap.default_config,
            "llm_config": bootstrap.llm_config,
            "dataset_config": bootstrap.dataset_config,
            "print_keys": bootstrap.print_keys,
        }
    )

    defaults = load_config_defaults(
        default_config_path=default_resolved,
        llm_config_path=llm_resolved,
        dataset_config_path=dataset_resolved,
    )
    runtime_config.update(defaults)

    loaded_from = []
    if default_resolved and os.path.isfile(default_resolved):
        loaded_from.append(f"default={default_resolved}")
    if llm_resolved and os.path.isfile(llm_resolved):
        loaded_from.append(f"llm={llm_resolved}")
    if dataset_resolved and os.path.isfile(dataset_resolved):
        loaded_from.append(f"dataset={dataset_resolved}")
    if loaded_from:
        print(f"[infer] Loaded config defaults from: {', '.join(loaded_from)}")

    return runtime_config, sacred_argv


def _normalize_infer_config(config: dict):
    config["use_tool"] = _coerce_bool(config.get("use_tool"))
    if config["use_tool"] is None:
        config["use_tool"] = True
    config["remote"] = _coerce_bool(config.get("remote"))
    if config["remote"] is None:
        config["remote"] = False
    config["use_summarize"] = _coerce_bool(config.get("use_summarize"))
    if config["use_summarize"] is None:
        config["use_summarize"]=False
    if config.get("endpoints") is not None and not isinstance(config["endpoints"], list):
        config["endpoints"] = [config["endpoints"]]
    if config.get("api_keys") is not None and not isinstance(config["api_keys"], list):
        config["api_keys"] = [config["api_keys"]]
    if config.get("summ_model_urls") is not None and not isinstance(config["summ_model_urls"], list):
        config["summ_model_urls"] = [config["summ_model_urls"]]
    if config.get("read_allowed_roots") is not None and not isinstance(config["read_allowed_roots"], list):
        config["read_allowed_roots"] = [config["read_allowed_roots"]]

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

    if not config.get("endpoints"):
        raise ValueError(
            "endpoints is required when not set in llm_config. "
            "Use YAML defaults or Sacred override: with endpoints=['http://host:8001/v1']"
        )
    if not config.get("model_path"):
        if not (config.get("remote") and not config.get("use_summarize")):
            raise ValueError(
                "model_path is required when not set in llm_config, unless "
                "remote=True and use_summarize=False. "
                "Use YAML or Sacred: with model_path='/path/to/model' "
                "or with remote=True use_summarize=False"
            )

    return dict_to_namespace(config)


def get_inference_instance(config: dict):
    args = _normalize_infer_config(config)
    # Print args for debugging. WARNING: --print_keys will print sensitive values.
    if getattr(args, "print_keys", False):
        print("[infer] --- SENSITIVE DEBUG (print_keys enabled) ---")
        print(f"[infer] bing_api_key={args.bing_api_key}")
        print(f"[infer] api_keys={args.api_keys}")
        print("[infer] --- END SENSITIVE DEBUG ---")
    print(vars(args))

    # When running interaction task, switch to interaction inference engine
    if args.prompt_type == "interaction":
        try:
            from src.infer_engines.inference_engine_inter import (
                AsyncInteractionInference as AsyncInfer,
            )
            return AsyncInfer(args)
        except ImportError:
            print("ImportError: No module named 'src.infer_engines.inference_engine_inter'")
        

    # Default math/QA inference engines
    try:
        if args.use_sds:
            from src.infer_engines.inference_engine import AsyncInferenceCompletionSDS as AsyncInfer
        else:
            from src.infer_engines.inference_engine import AsyncInference as AsyncInfer
        return AsyncInfer(args)
    except ImportError:
        print("ImportError: No module named 'src.infer_engines.inference_engine'")

    
async def main(config: dict):
    inference = get_inference_instance(config)
    await inference.run()


def run_from_cli(argv: Optional[Sequence[str]] = None):
    cli_args = list(argv if argv is not None else sys.argv[1:])
    base_config, sacred_argv = _build_runtime_config(cli_args)

    experiment = build_experiment("evaluation_infer", base_config)

    @experiment.main
    def infer_entry(_config):
        return asyncio.run(main(dict(_config)))

    if getattr(parse_bootstrap_args(
        cli_args,
        description="Asynchronous inference engine bootstrap options",
        options=(),
    )[0], "_show_help", False):
        print(
            "Bootstrap options: --default_config <name> --llm_config <name> "
            "--dataset_config <name> --print_keys"
        )
        print(
            "Sacred overrides: python infer.py --llm_config Qwen3_8B "
            "--dataset_config math500 with counts=10 temperature=0.2"
        )
    return experiment.run_commandline(sacred_argv)


if __name__ == "__main__":
    run_from_cli()