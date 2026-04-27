import sys
import os

sys.path.append(os.getcwd())

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI  # type: ignore
from vllm import SamplingParams  # type: ignore
from tqdm.asyncio import tqdm as async_tqdm

from ..vllm_client_pool import VLLMClientPool
from ..prompt_manager import PromptManager
from ..tools.tool_executor import ToolExecutor
from ..tools.python_tool import PythonTool
from ..tools.read_tool import ReadTool
from ..tools.interaction_tools import (
    ProductSearchTool,
    InventoryCheckTool,
    PolicySearchTool,
    OrderLookupTool,
    PricingCalcTool,
)
from ..data_loaders.data_loader_inter import InteractionDataLoader
from ..processors.sample_processor_inter import SampleProcessorInter

def _resolve_customer_credentials(args: Any) -> Tuple[str, str]:
    """Resolve customer credentials from merged config args."""
    raw_key = getattr(args, "customer_api_key", None)
    raw_url = getattr(args, "customer_base_url", None)
    if isinstance(raw_key, str):
        raw_key = raw_key.strip() or None
    if isinstance(raw_url, str):
        raw_url = raw_url.strip() or None

    # API key must come from config (YAML/Sacred/CLI merged into args).
    api_key = raw_key
    # Base URL must come from config (YAML/Sacred/CLI merged into args).
    base_url = raw_url
    return api_key, base_url


class AsyncInteractionInference:
    """
    Interaction (Task1) inference engine.

    When running via infer.py with remote=True and use_summarize=False (the default),
    model_path may be omitted; this engine does not load a local HF tokenizer.

    Pipeline:
      1) Use InteractionDataLoader to load scenario templates from JSONL.
      2) For each scenario, use SampleProcessorInter to run a full agent↔customer dialogue:
         - Agent: evaluated model via VLLMClientPool + tag-based tools.
         - Customer: fixed Kimi/OpenAI model via interaction.run_customer_turn_kimi.
      3) Save each scenario's dialogue trace to JSONL, to be consumed by interaction/llm_as_judge.py.
    """

    def __init__(self, args):
        self.args = args

        # Agent side: vLLM/OpenAI-compatible endpoints managed by VLLMClientPool
        self.vllm_pool = VLLMClientPool(
            endpoints=args.endpoints,
            api_keys=args.api_keys,
            default_model=args.default_model,
            remote=args.remote,
        )

        # Prompt manager for agent (interaction prompt)
        # prompt_type 由 args.prompt_type 控制，建议配置为 "interaction"
        self.prompt_manager = PromptManager(args.prompt_type, args.use_tool)

        # Sampling parameters for agent model
        self.args.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            n=1,
            include_stop_str_in_output=getattr(args, "include_stop_str_in_output", True),
        )

        # Tools: Python + Read + interaction tools
        self.tool_executor = self._create_tool_executor()

        # Scenario-level data loader
        self.data_loader = InteractionDataLoader(self.args)

        # Customer side: fixed OpenAI-compatible client (e.g., Kimi)
        customer_api_key, customer_base_url = _resolve_customer_credentials(args)
        if not customer_api_key or not customer_base_url:
            raise RuntimeError(
                "customer_api_key and customer_base_url are empty after resolving "
                "(set both in config/YAML or pass via Sacred overrides)."
            )
        self.customer_client = OpenAI(api_key=customer_api_key, base_url=customer_base_url)
        self.customer_model = getattr(args, "customer_model", "kimi-k2-0905-preview")
        self.customer_temperature = getattr(args, "customer_temperature", 0.2)
        self.customer_max_tokens = getattr(args, "customer_max_tokens", 2048)

        # 最终 JSONL 路径在 run() 中由 _resolve_interaction_output_jsonl 解析（与 AsyncInference 目录 + 文件名一致）
        self.output_path: str = ""

        # Concurrency
        self.max_concurrent = getattr(args, "max_concurrent_requests", 10)
        print(f"Initialized {self.__class__.__name__} for interaction task.")

    def _build_read_allowed_roots(self) -> List[str]:
        configured = list(getattr(self.args, "read_allowed_roots", None) or [])
        roots: List[str] = []

        def _add(path: Optional[str]) -> None:
            if not path:
                return
            abs_path = os.path.abspath(path)
            if abs_path not in roots:
                roots.append(abs_path)

        if configured:
            for path in configured:
                _add(path)
        else:
            _add(os.getcwd())
            _add(getattr(self.args, "data_path", None))
            output_path = getattr(self.args, "output_path", None)
            if output_path:
                abs_out = os.path.abspath(
                    output_path
                    if os.path.isabs(output_path)
                    else os.path.join(os.getcwd(), output_path)
                )
                if os.path.isdir(abs_out):
                    _add(abs_out)
                elif str(output_path).endswith(".jsonl") or abs_out.endswith(".jsonl"):
                    _add(os.path.dirname(abs_out) or ".")
                else:
                    _add(abs_out)

        return roots or [os.getcwd()]

    def _create_tool_executor(self) -> ToolExecutor:
        executor = ToolExecutor()

        # Python tool (optional, for agent-side computations)
        python_tool = PythonTool(
            conda_path=self.args.conda_path,
            conda_env=self.args.conda_env,
            max_concurrent=self.args.python_max_concurrent,
        )
        executor.register_tool(python_tool)

        # Generic file/document reader for interaction scenarios
        executor.register_tool(
            ReadTool(
                allowed_roots=self._build_read_allowed_roots(),
                timeout=int(getattr(self.args, "read_timeout", 30)),
                max_chars=int(getattr(self.args, "read_max_chars", 8000)),
                enable_image_ocr=bool(getattr(self.args, "read_enable_ocr", False)),
            )
        )

        # Interaction-specific tools
        executor.register_tool(ProductSearchTool())
        executor.register_tool(InventoryCheckTool())
        executor.register_tool(PolicySearchTool())
        executor.register_tool(OrderLookupTool())
        executor.register_tool(PricingCalcTool())
        return executor

    async def _process_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        processor = SampleProcessorInter(
            vllm_pool=self.vllm_pool,
            tool_executor=self.tool_executor,
            prompt_manager=self.prompt_manager,
            args=self.args,
            scenario=scenario,
            agent_model=self.args.default_model,
            customer_client=self.customer_client,
            customer_model=self.customer_model,
            customer_temperature=self.customer_temperature,
            customer_max_tokens=self.customer_max_tokens,
        )
        return await processor.run()

    async def run(self) -> None:
        """Run interaction inference for all scenarios and save traces as a JSON array."""
        scenarios = self.data_loader.load_data()
        total = len(scenarios)
        print(f"Total interaction scenarios: {total}")

        results: List[Optional[Dict[str, Any]]] = [None] * total

        async def worker(idx: int):
            scenario = scenarios[idx]
            try:
                results[idx] = await self._process_scenario(scenario)
            except Exception as e:
                import traceback

                traceback.print_exc()
                results[idx] = {"error": str(e), "scenario_id": scenario.get("id")}

        start_time = time.time()
        tasks: List[asyncio.Task[Any]] = []
        sem = asyncio.Semaphore(self.max_concurrent)

        async def sem_wrapper(i: int):
            async with sem:
                await worker(i)

        for i in range(total):
            tasks.append(asyncio.create_task(sem_wrapper(i)))

        await async_tqdm.gather(*tasks, desc="Running interaction scenarios")
        elapsed = time.time() - start_time
        print(f"Interaction inference finished in {elapsed/60:.2f} min")

        # Save as a single JSON array (same convention as AsyncInference) for evaluate.py json.load
        output_file = os.path.join(
            self.args.output_path,
            f"{self.args.dataset_name}_output.json",
        )
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)

        to_save = [r for r in results if r is not None]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=4)

        print(f"Saved interaction traces to {output_file}")


__all__ = ["AsyncInteractionInference"]
