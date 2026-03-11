import sys
import os

sys.path.append(os.getcwd())

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI  # type: ignore


JUDGE_SYSTEM = (
    "You are a strict evaluator for a conversational e-commerce customer service system. "
    "You will be given a scenario summary and a full chat transcript. "
    "Score the agent on multiple dimensions using only evidence from the transcript. "
    "Be harsh on hallucinations, policy invention, and evasiveness. "
    "Reward effective tool usage, clear clarifying questions, and de-escalation."
)


JUDGE_INSTRUCTIONS = """
Return STRICT JSON only with this schema:
{
  "interaction_success": 0-5,
  "tool_efficiency": 0-5,
  "emotion_handling": 0-5,
  "truthfulness_and_policy_grounding": 0-5,
  "overall": 0-5,
  "pass": true/false,
  "short_justification": "2-4 sentences, cite concrete moments from the transcript.",
  "failure_modes": ["string", "..."],
  "best_moments": ["string", "..."]
}

Rubric (0-5 each):
- interaction_success: Did the customer end satisfied / task completed? If unclear, score <=2.
- tool_efficiency: Did the agent use tools when needed, avoid overuse, and use them correctly?
- emotion_handling: Did the agent remain calm, validate feelings, and guide toward resolution?
- truthfulness_and_policy_grounding: Did the agent avoid making up policies/order details and verify when uncertain?
- overall: Your holistic average-like judgment (not necessarily strict mean).

Pass criteria:
- pass=true only if interaction_success>=4 AND overall>=4 AND truthfulness_and_policy_grounding>=4.
No extra keys. No markdown. No code fences.
""".strip()


def _render_dialogue_text(dialogue: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """
    Render a readable transcript for the judge.
    Include tool calls/results briefly (but not huge dumps).
    Mirrors interaction/llm_as_judge.render_dialogue_text.
    """
    lines: List[str] = []
    for m in dialogue:
        role = m.get("role")
        content = (m.get("content") or "").strip()

        if role == "user":
            lines.append(f"Customer: {content}")
            meta = m.get("customer_meta")
            if meta:
                lines.append(
                    f"[CustomerMeta] satisfied={meta.get('satisfied')} "
                    f"score={meta.get('score')} reason={meta.get('reason')}"
                )
        elif role == "assistant":
            lines.append(f"Agent: {content}")
            tcalls = m.get("tool_calls") or []
            if tcalls:
                lines.append(f"[ToolCalls] {json.dumps(tcalls, ensure_ascii=False)}")
            ttrace = m.get("tool_trace") or []
            if ttrace:
                compact = []
                for t in ttrace:
                    compact.append(
                        {
                            "tool_name": t.get("tool_name"),
                            "tool_args": t.get("tool_args"),
                            "tool_result_keys": list((t.get("tool_result") or {}).keys())[:12],
                            "tool_result_preview": str(t.get("tool_result"))[:240],
                        }
                    )
                lines.append(f"[ToolResults] {json.dumps(compact, ensure_ascii=False)}")
        else:
            continue

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _build_judge_user_prompt(record: Dict[str, Any]) -> str:
    """
    Build the user prompt for judging one interaction trajectory.
    Mirrors interaction/llm_as_judge.build_judge_user_prompt.
    """
    scenario_summary = {
        "scenario_id": record.get("scenario_id"),
        "category": record.get("category"),
        "title": record.get("title"),
        "product_name": record.get("product_name"),
        "product_domain": record.get("product_domain"),
        "max_turns": record.get("max_turns"),
        "customer_satisfied_step2": record.get("customer_satisfied"),
        "customer_score_step2": record.get("customer_score"),
        "customer_reason_step2": record.get("customer_reason"),
        "agent_model": record.get("agent_model"),
    }

    dialogue = record.get("dialogue") or []
    transcript = _render_dialogue_text(dialogue)

    return f"""
SCENARIO SUMMARY (for context; do NOT trust it over transcript if conflicts):
{json.dumps(scenario_summary, ensure_ascii=False, indent=2)}

FULL TRANSCRIPT:
{transcript}

Now score the AGENT strictly using the rubric.
""".strip()


def _clamp_int(x: Any, lo: int = 0, hi: int = 5) -> int:
    try:
        v = int(x)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def _clamp_bool(x: Any) -> bool:
    return bool(x)


def _normalize_judge_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw judge JSON into a stable schema.
    Mirrors interaction/llm_as_judge.normalize_judge_obj.
    """
    norm: Dict[str, Any] = {}
    norm["interaction_success"] = _clamp_int(obj.get("interaction_success"))
    norm["tool_efficiency"] = _clamp_int(obj.get("tool_efficiency"))
    norm["emotion_handling"] = _clamp_int(obj.get("emotion_handling"))
    norm["truthfulness_and_policy_grounding"] = _clamp_int(
        obj.get("truthfulness_and_policy_grounding")
    )
    norm["overall"] = _clamp_int(obj.get("overall"))
    norm["pass"] = _clamp_bool(obj.get("pass"))

    sj = obj.get("short_justification", "")
    norm["short_justification"] = str(sj).strip()

    fm = obj.get("failure_modes", [])
    bm = obj.get("best_moments", [])
    if not isinstance(fm, list):
        fm = [str(fm)]
    if not isinstance(bm, list):
        bm = [str(bm)]
    norm["failure_modes"] = [str(x)[:240] for x in fm][:8]
    norm["best_moments"] = [str(x)[:240] for x in bm][:8]

    dims = [
        norm["interaction_success"],
        norm["tool_efficiency"],
        norm["emotion_handling"],
        norm["truthfulness_and_policy_grounding"],
    ]
    norm["overall_mean_4dims"] = round(sum(dims) / 4.0, 2)

    norm["pass_local"] = (
        norm["interaction_success"] >= 4
        and norm["overall"] >= 4
        and norm["truthfulness_and_policy_grounding"] >= 4
    )
    return norm


class LLMEvaluatorInter:
    """
    LLM-as-judge evaluator for interaction task.

    Interface:
        evaluate(record, semaphore) -> (is_pass, normalized_judge_json_str)

    - record: 单条 interaction 推理输出（包含 scenario_id / category / title / dialogue / customer_* 等）。
    - semaphore: 并发控制，由 evaluator.py 传入。
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "empty",
        concurrent_limit: int = 50,
        retry_limit: int = 3,
    ):
        if api_base_url is None:
            api_base_url = "http://localhost:8000/v1"
        if model_name is None:
            model_name = "<your_model_name>"

        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key
        self.concurrent_limit = concurrent_limit
        self.retry_limit = retry_limit

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )

    async def evaluate(
        self,
        record: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Tuple[bool, str]:
        """
        Evaluate a single interaction trajectory.

        Returns:
            (is_pass, response_str):
                - is_pass: 是否通过（来自 pass_local）
                - response_str: 归一化后的 judge JSON 字符串，用于 evaluator 聚合
        """
        user_prompt = _build_judge_user_prompt(record)

        for attempt in range(self.retry_limit):
            try:
                async with semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "system", "content": JUDGE_INSTRUCTIONS},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=2048,
                        response_format={"type": "json_object"},
                    )
                msg = completion.choices[0].message
                content = (msg.content or "").strip()
                if not content:
                    norm = {"error": "empty_judge_response"}
                    return False, json.dumps(norm, ensure_ascii=False)

                try:
                    raw_obj = json.loads(content)
                except Exception as e:
                    norm = {"parse_error": str(e), "raw": content}
                    return False, json.dumps(norm, ensure_ascii=False)

                norm = _normalize_judge_obj(raw_obj)
                is_pass = bool(norm.get("pass_local"))
                return is_pass, json.dumps(norm, ensure_ascii=False)

            except Exception as e:
                if attempt == self.retry_limit - 1:
                    norm = {"error": str(e)}
                    return False, json.dumps(norm, ensure_ascii=False)
                await asyncio.sleep(1 * (attempt + 1))

        norm = {"error": "max_retries_exceeded"}
        return False, json.dumps(norm, ensure_ascii=False)


__all__ = ["LLMEvaluatorInter"]