"""
LLM-as-judge evaluator for expodesign (experiment design) task.
Uses rubric-based scoring (scores 1-5, overall_score, rationales) aligned with
crawler/Scientific_Claim_Refutation/llm_as_judge.py. Expects <answer> content to be
parseable JSON (experiment card); builds judge prompt from tag-wrapped paper ``meta``
(<domain>, <title>, <url>, <abstract>), the task query string, and candidate answer JSON.
When ``meta`` is absent, falls back to legacy single-string question for paper tags.
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag> in question (tag-wrapped input)."""
    if not text:
        return ""
    m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _safe_clip(text: str, n: int = 3000) -> str:
    """Truncate and normalize whitespace for judge prompt."""
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    return text[:n] + " ..."


def _parse_prediction_as_json(pred_answer: str) -> Dict[str, Any]:
    """Parse prediction (content of <answer>) as JSON experiment card. Returns dict or fallback."""
    pred_answer = (pred_answer or "").strip()
    if not pred_answer:
        return {}
    if pred_answer.startswith("{"):
        try:
            return json.loads(pred_answer)
        except json.JSONDecodeError:
            pass
    return {"raw_or_invalid": _safe_clip(pred_answer, 2000)}


def _build_judge_messages(
    paper: Dict[str, Any],
    query: str,
    answer_payload: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Build system + user messages for expodesign judge (no code execution).
    Schema and rubric aligned with llm_as_judge.build_judge_messages.
    """
    title = _safe_clip(str(paper.get("title", "") or paper.get("paper_title", "")), 400)
    abstract = _safe_clip(str(paper.get("abstract", "") or paper.get("summary", "")), 1400)
    url = str(paper.get("url", "") or paper.get("link_abstract", "") or paper.get("id", "") or paper.get("paper_url", ""))

    system = (
        "You are a strict, evidence-based evaluator of LLM experiment-design answers. "
        "You must grade only based on the provided paper context, the query, and the answer JSON. "
        "Output MUST be valid JSON only."
    )

    user = f"""
Evaluate the candidate answer for the given paper and query.

Paper (ground truth reference):
- title: {title}
- url: {url}
- abstract: {abstract}

Query (task prompt shown to the evaluated model):
{_safe_clip(query, 800)}

Candidate answer (JSON):
{json.dumps(answer_payload, ensure_ascii=False)}

Optional execution evidence (none in this setting):
{{}}

Scoring dimensions (integer 1 to 5):
1) experimental_design (feasibility + value of MVP experiment; clear RQ, hypotheses, datasets, metrics, baselines, procedure)
2) tool_success (whether the design describes runnable code / procedure that could produce meaningful outputs; no execution evidence here)
3) result_conclusion_consistency (does the short_analysis logically follow from the described results_table / outputs; avoid hallucinated conclusions)

Rubric hints:
- Score 5: specific, runnable, minimal yet meaningful; baselines+metrics well-justified; conclusions trace back to results.
- Score 3: mostly reasonable but has gaps (unclear baselines/metrics/procedure, shaky analysis, weak reproducibility).
- Score 1: not runnable, vague, or conclusions not supported by results.

Output STRICT JSON only with exactly this schema:
{{
  "scores": {{
    "experimental_design": 1,
    "tool_success": 1,
    "result_conclusion_consistency": 1
  }},
  "overall_score": 0.0,
  "rationales": {{
    "experimental_design": "short reason",
    "tool_success": "short reason",
    "result_conclusion_consistency": "short reason"
  }},
  "flags": {{
    "missing_required_fields": ["string"],
    "suspected_hallucination": true,
    "code_not_runnable": true
  }}
}}

overall_score must be the mean of the three scores (as a float with one decimal).
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]


class LLMEvaluatorExpo:
    """
    LLM-as-judge evaluator for expodesign task.
    Interface compatible with LLMEvaluator: evaluate(question, labeled_answer, pred_answer, semaphore) -> (bool, str).
    question: task / user input string (e.g. tagged task). Use ``meta`` for paper tags when set.
    pred_answer: content inside <answer>, expected to be parseable JSON (experiment card).
    Returns (is_correct, response_str); response_str is JSON string of judge output (scores, overall_score, rationales, flags).
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "empty",
        concurrent_limit: int = 50,
        retry_limit: int = 3,
        overall_score_threshold: float = 3.0,
    ):
        if api_base_url is None:
            api_base_url = "https://api.moonshot.cn/v1"
        if model_name is None:
            model_name = "<your_model_name>"

        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key
        self.concurrent_limit = concurrent_limit
        self.retry_limit = retry_limit
        self.overall_score_threshold = overall_score_threshold

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )

    async def evaluate(
        self,
        question: str,
        labeled_answer: str,
        pred_answer: str,
        semaphore: asyncio.Semaphore,
        meta: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate one expodesign prediction.

        Args:
            question: When ``meta`` is set, the task / user input (e.g. tagged task prompt). Otherwise legacy combined input.
            labeled_answer: Unused for expo (no gold answer); kept for interface compatibility.
            pred_answer: Content of <answer>, expected to be parseable JSON (experiment card).
            semaphore: Concurrency limit.
            meta: Optional tag-wrapped paper info (<domain>, <title>, <url>, <abstract>); when provided, used for paper fields for the judge.

        Returns:
            (is_correct, response_str): is_correct from overall_score >= threshold;
            response_str is JSON string of judge output for downstream metrics.
        """
        paper = {
            "title": _extract_tag(meta, "title"),
            "abstract": _extract_tag(meta, "abstract"),
            "url": _extract_tag(meta, "url"),
            "domain": _extract_tag(meta, "domain"),
        }
        answer_payload = _parse_prediction_as_json(pred_answer)
        query = _safe_clip(question, 1200)

        messages = _build_judge_messages(paper=paper, query=query, answer_payload=answer_payload)

        for attempt in range(self.retry_limit):
            try:
                async with semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1200,
                        response_format={"type": "json_object"},
                    )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    return False, json.dumps({"error": "empty_judge_response"}, ensure_ascii=False)

                try:
                    judge_json = json.loads(content)
                except json.JSONDecodeError:
                    return False, json.dumps({"raw": content, "parse_error": True}, ensure_ascii=False)

                overall = float(judge_json.get("overall_score", 0.0))
                is_correct = overall >= self.overall_score_threshold
                return is_correct, json.dumps(judge_json, ensure_ascii=False)

            except Exception as e:
                if attempt == self.retry_limit - 1:
                    print(f"Error in LLMEvaluatorExpo: {e}")
                    err_payload = {"error": str(e), "pred_answer_preview": _safe_clip(pred_answer, 200)}
                    return False, json.dumps(err_payload, ensure_ascii=False)
                await asyncio.sleep(1 * (attempt + 1))

        return False, json.dumps({"error": "max_retries_exceeded"}, ensure_ascii=False)
