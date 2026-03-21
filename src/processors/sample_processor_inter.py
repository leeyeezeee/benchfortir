import sys
import os

sys.path.append(os.getcwd())

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI  # type: ignore


class SampleProcessorInter:
    """
    Orchestrate one full interaction scenario (agent ↔ customer) for the interaction task.

    - Agent side:
        Uses evaluation's VLLMClientPool + ToolExecutor, with tag-based tools
        (e.g., <product_search>{...}</product_search> + <result>...</result>).
    - Customer side:
        Reuses interaction.run_customer_turn_kimi with a fixed customer model (e.g., Kimi),
        no extra PromptManager required.
    """

    def __init__(
        self,
        vllm_pool,
        tool_executor,
        prompt_manager,
        args,
        scenario: Dict[str, Any],
        agent_model: Optional[str],
        customer_client: OpenAI,
        customer_model: str,
        customer_temperature: float,
        customer_max_tokens: int,
        session_id: Optional[str] = None,
    ):
        self.vllm_pool = vllm_pool
        self.tool_executor = tool_executor
        self.prompt_manager = prompt_manager
        self.args = args
        self.scenario = scenario
        self.agent_model = agent_model
        self.customer_client = customer_client
        self.customer_model = customer_model
        self.customer_temperature = customer_temperature
        self.customer_max_tokens = customer_max_tokens

        self.session_id = session_id or scenario.get("id") or "interaction_session"

        # Conversation context for agent model (chat messages)
        self.messages: List[Dict[str, Any]] = []
        # Full trace for downstream judge (dialogue field)
        self.full_trace: List[Dict[str, Any]] = []

        # Customer satisfaction status
        self.satisfied: bool = False
        self.customer_score: int = 0
        self.customer_reason: str = ""

        # Read tool stats / limits (other interaction tools still use max_tool_iters)
        self.read_rounds: int = 0
        self.read_time: float = 0.0
        self.max_read_times: int = int(getattr(self.args, "max_read_times", 3))

        self._init_conversation()

    def _init_conversation(self) -> None:
        """Initialize system prompt and first user turn."""
        system_prompt = self.prompt_manager.get_system_prompt()
        self.messages = [{"role": "system", "content": system_prompt}]

        first_user = (self.scenario.get("first_user_message") or "").strip()
        self.messages.append({"role": "user", "content": first_user})

        self.full_trace = [
            {"role": "user", "content": first_user, "source": "scenario_first_turn"}
        ]

    def _tool_timeout(self, tag: str) -> int:
        if tag == "read":
            return int(getattr(self.args, "read_timeout", 30))
        return 120

    def _build_tool_limit_result(self, tag: str) -> str:
        if tag == "read":
            return json.dumps(
                {
                    "error": "The maximum read tool call limit is exceeded. You are not allowed to use read anymore in this turn."
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "error": f"The maximum tool call limit is exceeded for tool: {tag}."
            },
            ensure_ascii=False,
        )

    async def _run_agent_turn(self, turn_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run one agent turn, possibly with multiple tag-based tool calls.

        The flow is:
        - Call chat.completions once to get assistant text.
        - If the text contains a tool tag (product_search / inventory_check / policy_search /
          order_lookup / pricing_calc / read), extract the tag content and execute the tool.
        - Append <result>...</result> as a new message, then call chat again.
        - Repeat until no new tool tag appears in the assistant message.
        """
        tool_trace: List[Dict[str, Any]] = []
        max_tool_iters = int(getattr(self.args, "max_tool_iters", 6))
        iters = 0

        while True:
            iters += 1
            if iters > max_tool_iters:
                # Give up further tool chaining; one last reply without tools
                resp = await self.vllm_pool.chat(
                    messages=self.messages,
                    sampling_params=self.args.sampling_params,
                    session_id=self.session_id,
                    model=self.agent_model,
                )
                if not resp or not getattr(resp, "choices", None):
                    final = {"role": "assistant", "content": ""}
                    self.messages.append(final)
                    return final, tool_trace
                msg = resp.choices[0].message
                final = {"role": "assistant", "content": msg.content or ""}
                self.messages.append(final)
                return final, tool_trace

            resp = await self.vllm_pool.chat(
                messages=self.messages,
                sampling_params=self.args.sampling_params,
                session_id=self.session_id,
                model=self.agent_model,
            )
            if not resp or not getattr(resp, "choices", None):
                final = {"role": "assistant", "content": ""}
                return final, tool_trace

            msg = resp.choices[0].message
            content = msg.content or ""
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            self.messages.append(assistant_entry)

            # Check for tag-based tool call
            tag = self.tool_executor.identify_tool(content)
            if not tag:
                # No further tools; this is the final reply for this turn
                return assistant_entry, tool_trace

            # Mark this assistant message as a tool-call turn so we can hide it from the simulated customer.
            assistant_entry["source"] = "tool_call"

            # Extract tag content and execute the corresponding tool
            tool_content = self.tool_executor.extract_content(content, tag)
            if tag == "read" and self.read_rounds >= self.max_read_times:
                raw_result = self._build_tool_limit_result(tag)
            else:
                tool_start = time.time()
                try:
                    raw_result = await self.tool_executor.execute(
                        tag,
                        tool_content,
                        timeout=self._tool_timeout(tag),
                    )
                except Exception as e:
                    raw_result = json.dumps({"error": str(e)}, ensure_ascii=False)
                if tag == "read":
                    self.read_rounds += 1
                    self.read_time += time.time() - tool_start

            try:
                parsed_result = json.loads(raw_result)
            except Exception:
                parsed_result = raw_result

            tool_trace.append(
                {
                    "tool_name": tag,
                    "tool_args": tool_content,
                    "tool_result": parsed_result,
                }
            )

            # Feed tool result back to the agent using <result> tag.
            # Mark it so the simulated customer does not see tool internals.
            self.messages.append(
                {
                    "role": "user",
                    "content": f"<result>{raw_result}</result>",
                    "source": "tool_result",
                    "tool_name": tag,
                }
            )

    async def run(self) -> Dict[str, Any]:
        """
        Run the full multi-turn interaction for one scenario.

        Returns a dict compatible with interaction.run_dialogue(...) output,
        so that interaction/llm_as_judge.py can consume it directly.
        """
        max_turns = int(self.scenario.get("max_turns", getattr(self.args, "max_turns", 10)))

        for turn_idx in range(1, max_turns + 1):
            # Agent turn (tag-based tools)
            assistant_msg, agent_tool_trace = await self._run_agent_turn(turn_idx)
            self.full_trace.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content", ""),
                    "tool_calls": assistant_msg.get("tool_calls", []),
                    "tool_trace": agent_tool_trace,
                    "turn": turn_idx,
                }
            )

            # Customer turn via Kimi — sync OpenAI client runs in thread pool to avoid blocking asyncio loop
            cust = await asyncio.to_thread(self._run_customer_turn_kimi)

            message = str(cust.get("message", "")).strip()
            self.satisfied = bool(cust.get("satisfied", False))
            try:
                self.customer_score = int(cust.get("score", 0))
            except Exception:
                self.customer_score = 0
            self.customer_reason = str(cust.get("reason", "")).strip()

            if message:
                self.messages.append({"role": "user", "content": message})
                self.full_trace.append(
                    {
                        "role": "user",
                        "content": message,
                        "turn": turn_idx,
                        "customer_meta": {
                            "satisfied": self.satisfied,
                            "score": self.customer_score,
                            "reason": self.customer_reason,
                        },
                    }
                )

            if self.satisfied:
                break

        return {
            "scenario_id": self.scenario.get("id"),
            "category": self.scenario.get("category"),
            "title": self.scenario.get("title"),
            "product_name": self.scenario.get("product_name"),
            "product_domain": self.scenario.get("product_domain"),
            "max_turns": max_turns,
            "customer_satisfied": self.satisfied,
            "customer_score": self.customer_score,
            "customer_reason": self.customer_reason,
            "dialogue": self.full_trace,
            "agent_model": self.agent_model,
            "ts": int(time.time()),
        }

    def _run_customer_turn_kimi(self) -> Dict[str, Any]:
        """
        Run one customer (Kimi) turn (sync HTTP). Invoked via asyncio.to_thread from run().

        Customer outputs strict JSON:
          {message, satisfied, score, reason}
        We feed scenario context + the chat transcript (agent/customer text only).
        """
        # Build transcript for customer model and avoid leaking tool internals.
        transcript_lines: List[str] = []
        for m in self.messages:
            role = m.get("role")
            if role == "system" or role == "tool":
                continue
            if m.get("source") in {"tool_call", "tool_result"}:
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                transcript_lines.append(f"Customer: {content}")
            elif role == "assistant":
                transcript_lines.append(f"Agent: {content}")

        transcript = "\n".join(transcript_lines[-30:])

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a strict, picky e-commerce customer in a simulated customer service chat. "
                    "You must be consistent with the provided scenario. "
                    "You should challenge vague answers, demand specifics, and get annoyed if the agent makes things up. "
                    "However, if the agent provides a correct, policy-consistent resolution and addresses your concerns, "
                    "you should become satisfied."
                ),
            },
            {
                "role": "system",
                "content": (
                    "SCENARIO (follow it strictly):\n"
                    f"- category: {self.scenario.get('category')}\n"
                    f"- title: {self.scenario.get('title')}\n"
                    f"- customer_profile: {self.scenario.get('customer_profile')}\n"
                    f"- customer_goal: {self.scenario.get('customer_goal')}\n"
                    f"- customer_tone: {self.scenario.get('customer_tone')}\n"
                    f"- constraints: {json.dumps(self.scenario.get('constraints', []))}\n"
                    f"- missing_info (not initially provided): {json.dumps(self.scenario.get('missing_info', []))}\n"
                    f"- potential_misunderstanding: {self.scenario.get('potential_misunderstanding')}\n"
                    f"- success_criteria: {json.dumps(self.scenario.get('success_criteria', []))}\n"
                    f"- product_domain: {self.scenario.get('product_domain')}\n"
                    f"- product_name: {self.scenario.get('product_name')}\n"
                    f"- order_context: {json.dumps(self.scenario.get('order_context', {}))}\n\n"
                    "You are the CUSTOMER. Start from the provided first_user_message and continue the conversation.\n"
                    "Be strict and picky, but if the agent satisfies success_criteria, mark satisfied=true."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Return STRICT JSON only with this schema:\n"
                    "{\n"
                    '  "message": "string, the next customer message to the agent",\n'
                    '  "satisfied": true/false,\n'
                    '  "score": 0-5,\n'
                    '  "reason": "brief reason for satisfied/unsatisfied"\n'
                    "}\n"
                    "No extra keys. No markdown. No code fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CHAT TRANSCRIPT SO FAR:\n{transcript}\n\n"
                    "Now produce the next customer JSON response."
                ),
            },
        ]

        completion = self.customer_client.chat.completions.create(
            model=self.customer_model,
            messages=messages,
            temperature=self.customer_temperature,
            max_tokens=self.customer_max_tokens,
            response_format={"type": "json_object"},
        )

        msg = completion.choices[0].message
        content = (msg.content or "").strip()
        try:
            obj = json.loads(content) if content else {}
        except Exception as e:
            obj = {
                "message": "I don't think you answered my question. Be specific.",
                "satisfied": False,
                "score": 0,
                "reason": f"parse_error: {e}",
            }

        message = str(obj.get("message", "")).strip()
        satisfied = bool(obj.get("satisfied", False))
        try:
            score = int(obj.get("score", 0))
        except Exception:
            score = 0
        reason = str(obj.get("reason", "")).strip()

        score = max(0, min(5, score))
        if not message and not satisfied:
            message = "This still doesn't solve my issue. Please clarify."

        return {
            "message": message,
            "satisfied": satisfied,
            "score": score,
            "reason": reason,
            "raw": obj,
        }


__all__ = ["SampleProcessorInter"]
