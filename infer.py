import sys
import os
sys.path.append(os.getcwd())
import asyncio
import argparse
import nltk
# print(nltk.data.path)
# nltk.download('punkt')
# nltk.download('punkt_tab')

try:
    import yaml
except ImportError:
    yaml = None

infer_mode_help = """Inference mode selection
[default]       :    Basic behavior similar to the original search tool, uses summarization and continuously appends to the assistant content.
[completion]    :    Builds on [default] by adding feedback for exceeding the Python or search call limits, and for repeated search queries.
[completion_sds]:    Builds on [completion] by using a simple deep search engine.
"""


def _load_yaml(path: str) -> dict:
    """Load a YAML file; return empty dict if file missing or invalid."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            out = yaml.safe_load(f)
            return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def load_config_defaults(
    llm_config_path: str = None,
    dataset_config_path: str = None,
    tool_config_path: str = None,
) -> dict:
    """
    从三个独立的 YAML 配置文件（llm / dataset / tool）加载并合并为 infer 参数字典。

    使用方式（示例）::

        python infer.py \\
          --llm_config src/config/llm_config/llm_for_test.yaml \\
          --dataset_config src/config/dataset_config/example.yaml \\
          --tool_config src/config/tool_config/example.yaml

    合并规则：
    - 依次按 llm_config → dataset_config → tool_config 的顺序加载；
    - 后加载的同名字段会覆盖先前的值；
    - 缺失的文件会被忽略。
    """
    if not yaml:
        return {}
    merged: dict = {}
    for p in (llm_config_path, dataset_config_path, tool_config_path):
        if not p:
            continue
        # 允许相对路径；相对于当前工作目录解析
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(os.getcwd(), p))
        if os.path.isfile(p):
            merged.update(_load_yaml(p))
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
        "counts": c.get("counts", 100),
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
        # Interaction (customer side) configuration
        "customer_api_key": c.get("customer_api_key"),
        "customer_base_url": c.get("customer_base_url"),
        "customer_model": c.get("customer_model", "kimi-k2-0905-preview"),
        "customer_temperature": c.get("customer_temperature", 0.2),
        "customer_max_tokens": c.get("customer_max_tokens", 2048),
        "max_turns": c.get("max_turns", 10),
    }
    if isinstance(c.get("summ_model_urls"), list):
        defaults["summ_model_urls"] = c["summ_model_urls"]
    return defaults


def parse_arguments():
    """
    解析推理参数。

    配置只通过三个显式 YAML 文件提供：
      --llm_config     : LLM & vLLM 相关配置
      --dataset_config : 数据集相关配置
      --tool_config    : 各类工具（python/read/search 等）配置

    这三个文件会被依次加载并合并为默认参数；命令行显式传入的参数仍然可以覆盖它们。
    不再支持旧版的 --config 入口/目录模式。
    """
    # 第一轮：只解析三个配置文件路径，以便构建默认参数
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--llm_config",
        type=str,
        default=None,
        help="Path to LLM config YAML (required unless all key LLM args are provided via CLI).",
    )
    config_parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Path to dataset config YAML (required unless dataset args are provided via CLI).",
    )
    config_parser.add_argument(
        "--tool_config",
        type=str,
        default=None,
        help="Path to tool config YAML (optional; python/read/search defaults can also be set via CLI).",
    )
    config_args, _ = config_parser.parse_known_args()

    defaults = {}
    if yaml:
        defaults = load_config_defaults(
            llm_config_path=config_args.llm_config,
            dataset_config_path=config_args.dataset_config,
            tool_config_path=config_args.tool_config,
        )
        if defaults:
            loaded_from = []
            if config_args.llm_config:
                loaded_from.append(f"llm={config_args.llm_config}")
            if config_args.dataset_config:
                loaded_from.append(f"dataset={config_args.dataset_config}")
            if config_args.tool_config:
                loaded_from.append(f"tool={config_args.tool_config}")
            print(f"[infer] Loaded config defaults from: {', '.join(loaded_from) if loaded_from else 'CLI only'}")

    # 第二轮：完整 parser
    parser = argparse.ArgumentParser(description="Asynchronous inference engine")
    # 为了兼容帮助信息，这里仍然暴露三个配置参数，但它们只用于显示；真正的默认值已在第一轮解析时应用。
    parser.add_argument(
        "--llm_config",
        type=str,
        default=None,
        help="Path to LLM config YAML (already applied as defaults if provided).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Path to dataset config YAML (already applied as defaults if provided).",
    )
    parser.add_argument(
        "--tool_config",
        type=str,
        default=None,
        help="Path to tool config YAML (already applied as defaults if provided).",
    )
    parser.add_argument(
        "--print_keys",
        action="store_true",
        default=False,
        help="Print full API keys at startup (DANGEROUS; for debugging only).",
    )

    vllm_group = parser.add_argument_group("VLLM Configuration")
    vllm_group.add_argument(
        "--endpoints",
        type=str,
        nargs="+",
        default=defaults.get("endpoints"),
        help="List of VLLM endpoints",
    )
    vllm_group.add_argument(
        "--model_path",
        type=str,
        default=defaults.get("model_path"),
        help="Model path for tokenizer loading",
    )
    vllm_group.add_argument(
        "--api_keys",
        type=str,
        nargs="+",
        default=defaults.get("api_keys"),
        help="List of API keys corresponding to endpoints",
    )
    vllm_group.add_argument(
        "--default_model",
        type=str,
        default=defaults.get("default_model", "Qwen2.5-7B-Instruct"),
        help="Default model name to use",
    )

    generation_group = parser.add_argument_group("Generation Parameters")
    generation_group.add_argument(
        "--temperature",
        type=float,
        default=defaults.get("temperature", 0),
        help="Temperature for generation",
    )
    generation_group.add_argument(
        "--max_tokens",
        type=int,
        default=defaults.get("max_tokens", 4096),
        help="Maximum number of new tokens to generate",
    )
    generation_group.add_argument(
        "--top_p",
        type=float,
        default=defaults.get("top_p", 0.8),
        help="Top-p sampling cutoff",
    )
    generation_group.add_argument(
        "--top_k",
        type=int,
        default=defaults.get("top_k", 20),
        help="Top-k sampling cutoff",
    )
    generation_group.add_argument(
        "--min_p",
        type=float,
        default=defaults.get("min_p", 0.0),
        help="Minimum probability threshold",
    )
    generation_group.add_argument(
        "--repetition_penalty",
        type=float,
        default=defaults.get("repetition_penalty", 1.1),
        help="Repetition penalty factor",
    )
    generation_group.add_argument(
        "--include_stop_str_in_output",
        type=bool,
        default=defaults.get("include_stop_str_in_output", True),
        help="Whether to include stop strings in output",
    )

    inference_group = parser.add_argument_group("Inference Configuration")
    inference_group.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=defaults.get("max_concurrent_requests", 50),
        help="Maximum number of concurrent samples to process",
    )
    inference_group.add_argument(
        "--dataset_name",
        type=str,
        default=defaults.get("dataset_name", "math"),
        help="Dataset name",
    )
    inference_group.add_argument(
        "--output_path",
        type=str,
        default=defaults.get("output_path"),
        help="Root directory for saving results",
    )
    inference_group.add_argument(
        "--prompt_type",
        type=str,
        default=defaults.get("prompt_type", "code_search"),
        help="Prompt type (code_search, search, math, base)",
    )
    inference_group.add_argument(
        "--counts",
        type=int,
        default=defaults.get("counts", 100),
        help="Number of samples to process",
    )
    inference_group.add_argument(
        "--data_path",
        type=str,
        default=defaults.get("data_path"),
        help="Custom data path. Datasets are expected at /root_path/dataset_name/test.jsonl",
    )
    inference_group.add_argument(
        "--max_python_times",
        type=int,
        default=defaults.get("max_python_times", 5),
        help="Maximum number of Python tool invocations",
    )
    inference_group.add_argument(
        "--max_search_times",
        type=int,
        default=defaults.get("max_search_times", 3),
        help="Maximum number of search tool invocations",
    )
    inference_group.add_argument(
        "--max_read_times",
        type=int,
        default=defaults.get("max_read_times", 3),
        help="Maximum number of read tool invocations",
    )
    inference_group.add_argument(
        "--sample_timeout",
        type=int,
        default=defaults.get("sample_timeout", 120),
        help="Timeout in seconds for processing a single sample",
    )
    inference_group.add_argument(
        "--use_sds",
        action="store_true",
        default=defaults.get("use_sds", False),
        help="Whether to use SDS",
    )
    inference_group.add_argument(
        "--use_tool",
        type=str,
        default=None,
        choices=["true", "false", "1", "0"],
        metavar="true|false",
        help=(
            "Whether the model is allowed to call tools (true/false). "
            "If not set, value is read from config. "
            "When disabled, PromptManager uses no-tool sub-templates."
        ),
    )

    tools_group = parser.add_argument_group("Tool Configuration")
    tools_group.add_argument(
        "--conda_path",
        type=str,
        default=defaults.get("conda_path"),
        help="Path to Conda installation",
    )
    tools_group.add_argument(
        "--conda_env",
        type=str,
        default=defaults.get("conda_env"),
        help="Conda environment name",
    )
    tools_group.add_argument(
        "--python_max_concurrent",
        type=int,
        default=defaults.get("python_max_concurrent", 32),
        help="Maximum concurrency for Python executor",
    )
    tools_group.add_argument(
        "--read_allowed_roots",
        type=str,
        nargs="+",
        default=defaults.get("read_allowed_roots"),
        help="Allowed roots for the read tool. Defaults to cwd plus data/output roots when omitted.",
    )
    tools_group.add_argument(
        "--read_max_chars",
        type=int,
        default=defaults.get("read_max_chars", 8000),
        help="Maximum characters returned by one read call",
    )
    tools_group.add_argument(
        "--read_timeout",
        type=int,
        default=defaults.get("read_timeout", 30),
        help="Timeout in seconds for remote read requests",
    )
    tools_group.add_argument(
        "--read_enable_ocr",
        action="store_true",
        default=defaults.get("read_enable_ocr", False),
        help="Enable optional OCR for image reads when pytesseract is installed",
    )
    tools_group.add_argument(
        "--bing_api_key",
        type=str,
        default=defaults.get("bing_api_key", ""),
        help="Bing Search API key",
    )
    tools_group.add_argument(
        "--bing_zone",
        type=str,
        default=defaults.get("bing_zone", "serp_api1"),
        help="Bing search region",
    )
    tools_group.add_argument(
        "--search_max_results",
        type=int,
        default=defaults.get("search_max_results", 10),
        help="Maximum number of search results",
    )
    tools_group.add_argument(
        "--search_result_length",
        type=int,
        default=defaults.get("search_result_length", 1000),
        help="Maximum length of each search result",
    )
    tools_group.add_argument(
        "--bing_requests_per_second",
        type=float,
        default=defaults.get("bing_requests_per_second", 2.0),
        help="Maximum Bing requests per second",
    )
    tools_group.add_argument(
        "--bing_max_retries",
        type=int,
        default=defaults.get("bing_max_retries", 3),
        help="Maximum number of Bing retries",
    )
    tools_group.add_argument(
        "--bing_retry_delay",
        type=float,
        default=defaults.get("bing_retry_delay", 1.0),
        help="Delay between Bing retries (in seconds)",
    )
    tools_group.add_argument(
        "--summ_model_urls",
        type=str,
        nargs="+",
        default=defaults.get("summ_model_urls", ["http://localhost:8004/v1"]),
        help="Local summarization LLM API endpoints",
    )
    tools_group.add_argument(
        "--summ_model_name",
        type=str,
        default=defaults.get("summ_model_name", "Qwen2.5-72B-Instruct"),
        help="Name of local summarization LLM",
    )
    tools_group.add_argument(
        "--summ_model_path",
        type=str,
        default=defaults.get("summ_model_path"),
        help="Path to local summarization LLM for tokenizer",
    )
    tools_group.add_argument(
        "--search_cache_file",
        type=str,
        default=defaults.get("search_cache_file", "search_cache.db"),
        help="Cache file for search results",
    )
    tools_group.add_argument(
        "--url_cache_file",
        type=str,
        default=defaults.get("url_cache_file", "search_url_cache.db"),
        help="Cache file for web pages",
    )
    tools_group.add_argument(
        "--use_local_search",
        action="store_true",
        default=defaults.get("use_local_search", False),
        help="Whether to use local search",
    )
    tools_group.add_argument(
        "--local_search_url",
        type=str,
        default=defaults.get("local_search_url"),
        help="URL of the local search tool",
    )
    tools_group.add_argument(
        "--max_sequence_length",
        type=int,
        default=defaults.get("max_sequence_length", 20000),
        help="Maximum sequence length for summarization",
    )
    tools_group.add_argument(
        "--compatible_search",
        action="store_true",
        default=defaults.get("compatible_search", False),
        help="Whether to use compatible search",
    )

    # Interaction customer configuration
    interaction_group = parser.add_argument_group("Interaction Customer Configuration")
    interaction_group.add_argument(
        "--customer_api_key",
        type=str,
        default=defaults.get("customer_api_key"),
        help="API key for the customer model (e.g., Kimi)",
    )
    interaction_group.add_argument(
        "--customer_base_url",
        type=str,
        default=defaults.get("customer_base_url"),
        help="Base URL for the customer model API",
    )
    interaction_group.add_argument(
        "--customer_model",
        type=str,
        default=defaults.get("customer_model", "kimi-k2-0905-preview"),
        help="Customer model name (used in interaction task)",
    )
    interaction_group.add_argument(
        "--customer_temperature",
        type=float,
        default=defaults.get("customer_temperature", 0.2),
        help="Temperature for the customer model",
    )
    interaction_group.add_argument(
        "--customer_max_tokens",
        type=int,
        default=defaults.get("customer_max_tokens", 2048),
        help="Maximum tokens for the customer model",
    )
    interaction_group.add_argument(
        "--max_turns",
        type=int,
        default=defaults.get("max_turns", 10),
        help="Maximum dialogue turns per interaction scenario",
    )

    args = parser.parse_args()

    # --use_tool: 命令行可传 true/false；未传则用配置
    if args.use_tool is not None:
        args.use_tool = str(args.use_tool).strip().lower() in ("true", "1")
    else:
        args.use_tool = bool(defaults.get("use_tool", True))

    # 若未从配置中获得必需字段，则要求命令行提供
    if not args.endpoints:
        parser.error(
            "--endpoints is required when not set in config (e.g. under src/config/llm_config and vllm)"
        )
    if not args.model_path:
        parser.error("--model_path is required when not set in config")

    return args


def get_inference_instance():
    args = parse_arguments()
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
        except ImportError:
            from src.inference_engine_inter import AsyncInteractionInference as AsyncInfer
        inference = AsyncInfer(args)
        return inference

    # Default math/QA inference engines
    try:
        if args.use_sds:
            from src.infer_engines.inference_engine import AsyncInferenceCompletionSDS as AsyncInfer
        else:
            from src.infer_engines.inference_engine import AsyncInference as AsyncInfer
    except ImportError:
        if args.use_sds:
            from src.inference_engine import AsyncInferenceCompletionSDS as AsyncInfer
        else:
            from src.inference_engine import AsyncInference as AsyncInfer

    inference = AsyncInfer(args)
    return inference


async def main():
    inference = get_inference_instance()
    await inference.run()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
