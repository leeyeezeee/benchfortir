"""benchfortir/src/evaluator.py

TIR Benchmark Evaluator (process + accuracy + cost metrics).

This module evaluates model outputs produced by this repository's inference pipeline.

The evaluator computes the metrics shown in your TIR table:
- Process evaluation
  * reasoning-answer consistency
  * tool contribution/support
- Correctness / Accuracy (dataset-appropriate)
  * QA: EM/F1 (max over multiple gold answers)
  * Math: math equivalence (0/1)
- Tool efficiency
  * average tool calls
  * tool efficiency: mean(score / tool_calls) over samples that used tools
- Cost
  * total time (sec) from inference logs
  * total tokens (estimated locally via tokenizer)

=================
Input to Evaluator
=================
`Evaluator.run(data)` expects:
    data: List[Dict[str, Any]]

Each element in the list is a single sample record.

The *typical* schema produced by this repo (see `src/inference_engine.py` and
`src/sample_processor.py`) looks like:

Required fields (for correctness metrics)
----------------------------------------
- input (str): the question/problem text.
- answer (str | List[str] | bool): gold answer(s).
- prediction (str): extracted final answer (written by `SampleProcessor` using
  `src/utils.extract_answer`).

Optional fields (for process/tool/cost metrics)
----------------------------------------------
- output (str): full trace string, usually containing:
    <think>...</think>   intermediate reasoning text (model)
    <search>...</search> search calls (model)
    <python>...</python> python calls (model)
    <result>...</result> tool results (system)
    <answer>...</answer> final answer block (model)

- instruction (str): system prompt.
- timing (dict): written by `SampleProcessor.log_timing()`, e.g.
    {"total_time": float, "llm_time": float, "python_time": float, "search_time": float}

Notes on missing fields
-----------------------
- If "prediction" is missing/empty but "output" is present, the evaluator will
  attempt to extract the final answer from "output" using `src/utils.extract_answer`.

=================
Evaluator outputs
=================
Given an `output_path` like "/path/to/gsm8k_output.json", the evaluator writes:

1) Per-sample metrics
   "{base}_metrics.json"
   Type: List[Dict[str, Any]]
   Each element is the original sample record plus:
      - metrics: Dict[str, Any]

2) Aggregated metrics
   "{base}_metrics_overall.json"
   Type: Dict[str, Any]

Per-sample metrics dict keys (record["metrics"])
------------------------------------------------
- is_valid_answer: bool
- em: int (0/1)
- f1: float (0~1)
- acc: int (0/1)             # legacy substring metric (kept for compatibility)
- math_equal: int (0/1)
- llm_equal: int (0/1)       # if LLM judge enabled
- python_calls: int
- search_calls: int
- tools_used: str            # "none" | "python" | "search" | "both"
- tool_counts: int           # python_calls + search_calls
- output_length: int         # char length of output after removing <result> blocks

New TIR per-sample metrics:
- reasoning_answer_consistency: Optional[int]  # 1/0, None if no <think> blocks
- tool_supported: Optional[int]                # 1/0, None if no <result> blocks
- time_total_sec: Optional[float]              # item["timing"]["total_time"]
- token_count: Optional[int]                   # estimated if tokenizer enabled

Overall metrics dict keys ({base}_metrics_overall.json)
------------------------------------------------------
Includes existing legacy keys plus new TIR keys.

Legacy keys retained:
- em, acc, f1, math_equal
- accuracy                      # QA: mean F1; Math: mean math_equal; may be replaced by llm_equal mean
- num_valid_answer
- tool_productivity, m1m2
- average_datas_used_tool_number
- tool_call, average_python_calls, average_search_calls
- llm_equal

New TIR keys:
- reasoning_answer_consistency: Optional[float]
- tool_supported_rate: Optional[float]
- tool_efficiency: float
- total_time_sec / avg_time_sec: Optional[float]
- total_tokens / avg_tokens: Optional[float]

=================
Token accounting
=================
The inference pipeline does not store token usage by default.

If `tokenizer_name_or_path` is provided to Evaluator, this module loads a
HuggingFace tokenizer and estimates token usage by re-tokenizing:
- the reconstructed prompt (system instruction + user input)
- plus the stored trace (output)

This is an approximation and is mainly useful for relative comparisons.

"""

import sys
import os
sys.path.append(os.getcwd())

import json
import asyncio
import datetime
import numpy as np
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm.asyncio import tqdm as async_tqdm

from .metrics import (
    evaluate_math_prediction,
    evaluate_qa_prediction,
    normalize_answer,
    compute_token_overlap,
    compute_f1_score,
)
from .math_equivalence import is_equiv
from .llm_evaluator_sds import LLMEvaluator
from .utils import extract_answer

try:
    # Optional dependency. transformers is already required by inference, but we keep it optional here.
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore


def _extract_tag_contents(text: str, tag: str) -> List[str]:
    """Extract all contents inside paired <tag>...</tag> blocks."""
    if not text:
        return []
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    return re.findall(pattern, text, flags=re.DOTALL)


def _safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


class Evaluator:
    def __init__(
        self,
        task_type: str,
        output_path: str,
        use_llm: bool = False,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "EMPTY",
        concurrent_limit: int = 50,
        sigma: float = 0.1,
        tokenizer_name_or_path: Optional[str] = None,
    ):
        """Create an evaluator.

        Parameters
        ----------
        task_type:
            "qa" or "math".
        output_path:
            Path to the model output file being evaluated.
        use_llm, api_base_url, model_name, api_key:
            Optional LLM-judge configuration.
        concurrent_limit:
            Maximum concurrent per-sample evaluation tasks.
        sigma:
            Smoothing constant used in legacy tool_productivity.
        tokenizer_name_or_path:
            If provided, load this tokenizer and estimate token counts.
        """
        self.task_type = task_type
        self.output_path = output_path
        self.use_llm = use_llm
        self.concurrent_limit = concurrent_limit
        self.sigma = sigma

        base_path, _ = os.path.splitext(output_path)
        self.output_metrics_path = f"{base_path}_metrics.json"
        self.output_metrics_overall_path = f"{base_path}_metrics_overall.json"

        # Optional tokenizer for token counting
        self.tokenizer = None
        if tokenizer_name_or_path and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path, trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: failed to load tokenizer from {tokenizer_name_or_path}: {e}")
                self.tokenizer = None

        # Optional LLM evaluator
        self.llm_evaluator = None
        if self.use_llm:
            self.llm_evaluator = LLMEvaluator(
                api_base_url=api_base_url,
                model_name=model_name,
                api_key=api_key,
                concurrent_limit=concurrent_limit,
            )

    # ------------------------------
    # Token / cost helpers
    # ------------------------------
    def _count_tokens(self, instruction: str, user_input: str, output: str) -> Optional[int]:
        """Estimate token count for one sample.

        Returns
        -------
        Optional[int]
            Total tokens (prompt + output trace), or None if tokenizer unavailable.
        """
        if self.tokenizer is None:
            return None

        instruction = instruction or ""
        user_input = user_input or ""
        output = output or ""

        try:
            # Prefer chat-template tokenization if available (closer to inference).
            prompt_len: int
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_input},
                ]
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                if isinstance(prompt_ids, dict) and "input_ids" in prompt_ids:
                    prompt_len = len(prompt_ids["input_ids"])  # type: ignore[index]
                else:
                    prompt_len = len(prompt_ids)  # type: ignore[arg-type]
            else:
                prompt_text = instruction + "\n" + user_input
                prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=True))

            out_len = len(self.tokenizer.encode(output, add_special_tokens=False))
            return int(prompt_len + out_len)
        except Exception:
            # Best-effort fallback
            try:
                prompt_text = instruction + "\n" + user_input
                return int(
                    len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
                    + len(self.tokenizer.encode(output, add_special_tokens=False))
                )
            except Exception:
                return None

    # ------------------------------
    # Process metrics
    # ------------------------------
    def _reasoning_consistency(self, output: str, prediction: str) -> Optional[int]:
        """Heuristic reasoning-answer consistency.

        - Extract all <think>...</think> blocks.
        - If none exist, return None.
        - Otherwise, return 1 if the final prediction is found/consistent, else 0.

        For math tasks, we also try a weak equivalence check by extracting boxed/number
        candidates from reasoning.
        """
        thinks = _extract_tag_contents(output or "", "think")
        if not thinks:
            return None

        reasoning = "\n".join(thinks)
        if not prediction:
            return 0

        if self.task_type == "math":
            pred_norm = normalize_answer(prediction)
            rea_norm = normalize_answer(reasoning)

            # Direct containment
            if pred_norm and pred_norm in rea_norm:
                return 1

            # Try equivalence against boxed or last numeric token
            boxed = re.findall(r"\\boxed\{([^}]*)\}", reasoning)
            cands: List[str] = boxed[:]
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", reasoning)
            if nums:
                cands.append(nums[-1])

            for c in cands:
                try:
                    if is_equiv(normalize_answer(prediction), normalize_answer(c)):
                        return 1
                except Exception:
                    continue
            return 0

        # QA
        pred_norm = normalize_answer(prediction, remove_articles=True, remove_punctuations=True)
        rea_norm = normalize_answer(reasoning, remove_articles=True, remove_punctuations=True)
        return 1 if pred_norm and pred_norm in rea_norm else 0

    def _tool_supported(self, output: str, prediction: str) -> Optional[int]:
        """Whether tool results (<result> blocks) support the final answer.

        Returns
        -------
        Optional[int]
            1/0 if <result> blocks exist, else None.
        """
        results = _extract_tag_contents(output or "", "result")
        if not results:
            return None
        if not prediction:
            return 0

        if self.task_type == "math":
            pred_norm = normalize_answer(prediction)
            pred_comp = re.sub(r"\s+", "", pred_norm)

            for r in results:
                r_norm = normalize_answer(r)
                r_comp = re.sub(r"\s+", "", r_norm)
                if pred_comp and pred_comp in r_comp:
                    return 1

                # Try numeric equivalence against any number-like fragments in tool output
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", r)
                for n in nums:
                    try:
                        if is_equiv(pred_norm, normalize_answer(n)):
                            return 1
                    except Exception:
                        continue
            return 0

        # QA
        pred_norm = normalize_answer(prediction, remove_articles=True, remove_punctuations=True)
        best = 0.0
        for r in results:
            r_norm = normalize_answer(r, remove_articles=True, remove_punctuations=True)
            if pred_norm and pred_norm in r_norm:
                return 1
            num_same, pred_len, ref_len = compute_token_overlap(pred_norm, r_norm)
            best = max(best, compute_f1_score(num_same, pred_len, ref_len))
        return 1 if best >= 0.5 else 0

    # ------------------------------
    # Main evaluation
    # ------------------------------
    async def evaluate_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample record and return the per-sample metrics dict."""
        question = item.get("input", "")
        answer = item.get("answer", "")
        prediction = item.get("prediction", "")
        output = item.get("output", "")
        instruction = item.get("instruction", "")

        # Fallback: extract final answer from output if prediction is missing
        if (not prediction) and output:
            try:
                prediction = extract_answer(output)
            except Exception:
                # last-resort heuristic
                prediction = "\n".join(output.replace("\n\n", "\n").strip().split("\n")[-5:])

        if not prediction:
            metrics: Dict[str, Any] = {
                "is_valid_answer": False,
                "em": 0,
                "acc": 0,
                "f1": 0,
                "math_equal": 0,
                "llm_equal": 0,
                "python_calls": 0,
                "search_calls": 0,
                "output_length": 0,
                # New TIR fields
                "reasoning_answer_consistency": None,
                "tool_supported": None,
                "token_count": self._count_tokens(instruction, question, output),
                "time_total_sec": (item.get("timing", {}) or {}).get("total_time", None),
            }
            metrics["tools_used"] = "none"
            metrics["tool_counts"] = 0
            return metrics

        metrics: Dict[str, Any] = {"is_valid_answer": True}

        # Tool usage counts
        python_calls = count_valid_tags(output or "", "python")
        search_calls = count_valid_tags(output or "", "search")
        tool_counts = python_calls + search_calls

        metrics.update(
            {
                "python_calls": python_calls,
                "search_calls": search_calls,
                "tools_used": (
                    "both"
                    if python_calls and search_calls
                    else "python"
                    if python_calls
                    else "search"
                    if search_calls
                    else "none"
                ),
                "tool_counts": tool_counts,
            }
        )

        # Output length excluding tool result blocks (legacy)
        metrics["output_length"] = len(remove_result_tags(output or ""))

        # Correctness metrics
        if self.task_type == "math":
            metrics.update(evaluate_math_prediction(prediction, answer))
            score_for_eff = metrics.get("math_equal", 0)
        elif self.task_type == "qa":
            metrics.update(evaluate_qa_prediction(prediction, answer))
            score_for_eff = metrics.get("f1", 0.0)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # New TIR per-sample metrics
        metrics["reasoning_answer_consistency"] = self._reasoning_consistency(output or "", prediction)
        metrics["tool_supported"] = self._tool_supported(output or "", prediction)
        metrics["token_count"] = self._count_tokens(instruction, question, output)
        metrics["time_total_sec"] = (item.get("timing", {}) or {}).get("total_time", None)
        metrics["score_for_efficiency"] = float(score_for_eff)  # internal helper

        # Optional LLM judge
        if self.use_llm and self.llm_evaluator:
            semaphore = asyncio.Semaphore(self.concurrent_limit)
            is_correct, llm_reason_answer = await self.llm_evaluator.evaluate(
                question=question,
                labeled_answer=answer,
                pred_answer=prediction,
                semaphore=semaphore,
            )
            metrics["llm_equal"] = int(is_correct)
            metrics["llm_response"] = llm_reason_answer

        return metrics

    async def evaluate_batch(self, data: List[Dict[str, Any]], timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """Evaluate a list of sample records (concurrently) and attach per-sample metrics."""
        semaphore = asyncio.Semaphore(self.concurrent_limit)

        async def _evaluate_with_semaphore(item: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    metrics = await asyncio.wait_for(self.evaluate_sample(item), timeout=timeout)
                except asyncio.TimeoutError:
                    print(f"Warning: Evaluation timed out ({timeout} seconds)")
                    metrics = {"status": "timeout"}
                item_copy = item.copy()
                item_copy["metrics"] = metrics
                return item_copy

        tasks = [_evaluate_with_semaphore(item) for item in data]
        results = await async_tqdm.gather(*tasks, desc="Evaluating samples")
        return results

    async def run(self, data: List[Dict[str, Any]], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run evaluation, write metrics files, and return overall metrics."""
        print(f"Starting evaluation of {len(data)} samples, task type: {self.task_type}")
        updated_data = await self.evaluate_batch(data, timeout)
        self.overall_metrics = self.calculate_overall_metrics(updated_data)
        self.overall_metrics["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_results(updated_data)
        return self.overall_metrics

    def calculate_overall_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Union[float, str, int, None]]:
        """Aggregate per-sample metrics into overall metrics."""
        num_valid_answer = sum(bool(item.get("metrics", {}).get("is_valid_answer", False)) for item in data)

        avg_em = [item["metrics"].get("em", 0) for item in data if "em" in item.get("metrics", {})]
        avg_acc = [item["metrics"].get("acc", 0) for item in data if "acc" in item.get("metrics", {})]
        avg_f1 = [item["metrics"].get("f1", 0) for item in data if "f1" in item.get("metrics", {})]
        avg_math = [item["metrics"].get("math_equal", 0) for item in data if "math_equal" in item.get("metrics", {})]
        avg_llm = [item["metrics"].get("llm_equal", 0) for item in data if "llm_equal" in item.get("metrics", {})]

        tool_counts = [item.get("metrics", {}).get("tool_counts", 0) for item in data]
        python_calls = [item.get("metrics", {}).get("python_calls", 0) for item in data]
        search_calls = [item.get("metrics", {}).get("search_calls", 0) for item in data]

        # Time / tokens (may be None)
        times = [item.get("metrics", {}).get("time_total_sec", None) for item in data]
        times_num = [t for t in times if isinstance(t, (int, float))]
        tokens = [item.get("metrics", {}).get("token_count", None) for item in data]
        tokens_num = [t for t in tokens if isinstance(t, int)]

        # Process metrics (0/1 or None)
        cons = [item.get("metrics", {}).get("reasoning_answer_consistency", None) for item in data]
        cons_num = [c for c in cons if isinstance(c, int)]
        tool_sup = [item.get("metrics", {}).get("tool_supported", None) for item in data]
        tool_sup_num = [t for t in tool_sup if isinstance(t, int)]

        # Tool usage rate: fraction of samples with at least one tool call
        tool_usage_rate = sum(1 for c in tool_counts if (c or 0) > 0) / len(data) if data else 0.0
        avg_tool_count = float(np.mean(tool_counts)) if tool_counts else 0.0

        # Accuracy definition
        if self.task_type == "math":
            accuracy = _safe_mean([float(x) for x in avg_math])
        elif self.task_type == "qa":
            accuracy = _safe_mean([float(x) for x in avg_f1])
        else:
            accuracy = 0.0

        # If LLM judge enabled, allow it to override accuracy
        final_accuracy = float(np.mean(avg_llm)) if (self.use_llm and avg_llm) else float(accuracy)

        # Legacy tool productivity
        final_tool_productivity = final_accuracy * avg_tool_count / (avg_tool_count + self.sigma) * 100.0

        # Tool efficiency: mean(score/tool_calls) for tool_calls>0
        eff_terms: List[float] = []
        for item in data:
            m = item.get("metrics", {})
            tc = m.get("tool_counts", 0) or 0
            score = m.get("score_for_efficiency", None)
            if tc > 0 and isinstance(score, (int, float)):
                eff_terms.append(float(score) / float(tc))
        tool_efficiency = _safe_mean(eff_terms)

        overall_metrics: Dict[str, Union[float, str, int, None]] = {
            # Legacy metrics
            "em": _safe_mean([float(x) for x in avg_em]),
            "acc": _safe_mean([float(x) for x in avg_acc]),
            "f1": _safe_mean([float(x) for x in avg_f1]),
            "math_equal": _safe_mean([float(x) for x in avg_math]),
            "accuracy": float(final_accuracy),
            "num_valid_answer": f"{num_valid_answer} of {len(data)}",
            "tool_productivity": float(final_tool_productivity),
            "average_datas_used_tool_number": float(tool_usage_rate),
            "tool_call": float(avg_tool_count),
            "average_python_calls": _safe_mean([float(x) for x in python_calls]),
            "average_search_calls": _safe_mean([float(x) for x in search_calls]),
            "llm_equal": _safe_mean([float(x) for x in avg_llm]),
            "m1m2": float(final_tool_productivity),

            # New TIR aggregates
            "reasoning_answer_consistency": _safe_mean([float(x) for x in cons_num]) if cons_num else None,
            "tool_supported_rate": _safe_mean([float(x) for x in tool_sup_num]) if tool_sup_num else None,
            "tool_efficiency": float(tool_efficiency),
            "total_time_sec": float(np.sum(times_num)) if times_num else None,
            "avg_time_sec": float(np.mean(times_num)) if times_num else None,
            "total_tokens": int(np.sum(tokens_num)) if tokens_num else None,
            "avg_tokens": float(np.mean(tokens_num)) if tokens_num else None,
        }
        return overall_metrics

    def save_results(self, data: List[Dict[str, Any]]):
        """Write per-sample and overall metrics files."""
        # Remove internal helper field before saving
        for item in data:
            m = item.get("metrics", {})
            if isinstance(m, dict) and "score_for_efficiency" in m:
                m.pop("score_for_efficiency", None)

        with open(self.output_metrics_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        with open(self.output_metrics_overall_path, mode="w", encoding="utf-8") as f:
            json.dump(self.overall_metrics, f, indent=4, ensure_ascii=False)

        print("Evaluation complete. Results saved.")
        print(f"Detailed metrics: {self.output_metrics_path}")
        print(f"Overall metrics: {self.output_metrics_overall_path}")

        # Console summary (keep it short)
        print(f"Accuracy: {self.overall_metrics.get('accuracy', 0):.4f}")
        print(f"Avg tool calls: {self.overall_metrics.get('tool_call', 0):.2f}")
        if self.overall_metrics.get("total_time_sec") is not None:
            print(f"Total time (sec): {self.overall_metrics['total_time_sec']:.2f}")
        if self.overall_metrics.get("total_tokens") is not None:
            print(f"Total tokens: {self.overall_metrics['total_tokens']}")


def count_valid_tags(text: str, tag: str) -> int:
    """Count valid paired tags like <tag> ... </tag>."""
    if not text:
        return 0

    count = 0
    current_pos = 0
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"

    while True:
        start_pos = text.find(start_tag, current_pos)
        if start_pos == -1:
            break
        end_pos = text.find(end_tag, start_pos + len(start_tag))
        if end_pos == -1:
            break
        count += 1
        current_pos = end_pos + len(end_tag)

    return count


def remove_result_tags(text: str) -> str:
    """Remove content inside <result>...</result> (and legacy <r>...</r>) tags."""
    if not text:
        return ""
    cleaned_text = re.sub(r"<r>.*?</r>", "", text, flags=re.DOTALL)
    cleaned_text = re.sub(r"<result>.*?</result>", "", cleaned_text, flags=re.DOTALL)
    return cleaned_text.strip()


# Compatibility alias for older versions
def remove_r_tags(text: str) -> str:
    return remove_result_tags(text)
