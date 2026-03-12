import sys
import os
sys.path.append(os.getcwd())


# ------------------------------
# Global prompt templates / blocks
# ------------------------------

# Expodesign：工具使用说明（标签模式）
EXPO_DESIGN_TOOL_PROTOCOL = """
You should reason step by step and you MAY call tools during your reasoning:
- To run code, wrap Python code inside <python> and </python>. The system will execute it and return the output inside <result> and </result>.
- To look up information, wrap a search query inside <search> and </search>. The system will return search results inside <result> and </result>.
- To read a file, wrap a file path or a JSON request inside <read> and </read>. The system will return extracted file content inside <result> and </result>.

You may interleave multiple rounds of thinking and tool calls. After using tools as needed, you MUST output a final experiment card as JSON inside <answer> and </answer>.

Hard constraints:
- The experiment must be CPU-friendly and runnable on a single machine.
- Prefer public or synthetic data; if public data is used, specify how to obtain it (do not assume private access).
- Include at least 2 baselines: one weak, one strong.
- Include at least 2 metrics: one primary, one secondary.
- Code used for checking or prototyping should appear in <python> blocks; do NOT embed runnable code fields in the final JSON.

""".strip()

READ_TOOL_PROTOCOL_EN = """
You also have access to a file reading tool.
- To read a local file or a downloadable URL, wrap the request inside <read> and </read>.
- The system will return the extracted content inside <result> and </result>.
- The read tool supports common file types such as txt, md, json, csv, html, pdf, docx, xlsx, and image metadata/OCR when available.
Examples:
<read>/absolute/path/to/report.pdf</read>
<read>{"path": "/absolute/path/to/report.pdf", "page": 2, "max_chars": 4000}</read>
""".strip()

READ_TOOL_PROTOCOL_ZH = """
你还可以使用文件读取工具。
- 需要读取本地文件或可下载 URL 时，请使用 <read> 和 </read>。
- 系统会把提取出的内容放在 <result> 和 </result> 中返回。
- 读取工具支持常见文件类型，如 txt、md、json、csv、html、pdf、docx、xlsx，以及在可用时返回图片 OCR/元数据。
示例：
<read>/absolute/path/to/report.pdf</read>
<read>{"path": "/absolute/path/to/report.pdf", "page": 2, "max_chars": 4000}</read>
""".strip()

# Expodesign：最终 JSON schema（不包含工具调用说明）
EXPO_DESIGN_JSON_SCHEMA = """
Final answer format: Inside <answer>...</answer> you MUST output ONLY a valid JSON object with exactly this schema (no runnable_code field and no extra keys):

{
  "research_question": "string",
  "hypotheses": ["string", "string"],
  "dataset_choice": {
    "type": "public|synthetic",
    "description": "string",
    "why_representative": "string",
    "access": "string (e.g., sklearn built-in / URL / generation procedure)"
  },
  "metrics": [
    {"name": "string", "type": "primary|secondary", "why": "string"}
  ],
  "baselines": [
    {"name": "string", "strength": "weak|strong", "why": "string"}
  ],
  "procedure": ["step 1", "step 2", "step 3"],
  "results_table": {
    "columns": ["string", "string"],
    "example_rows": [["string", "string"]],
    "how_to_interpret": "string"
  },
  "short_analysis": "string (must follow from results_table; be cautious)",
  "limitations_next_steps": {
    "limitations": ["string", "string"],
    "next_steps": ["string", "string"]
  }
}

Do NOT output any text outside of <answer>...</answer> except tool tags and their contents. The JSON inside <answer> must be strictly valid.
""".strip()



class PromptManager:
    """Manager for creating and formatting prompts."""

    def __init__(self, prompt_type: str, use_tool: bool = True):
        """
        Args
        ----
        prompt_type: str
            High-level prompt template type, e.g. code_search / math / interaction / expodesign.
        use_tool: bool
            Whether the model is allowed to call tools via tag-based protocol.
            When False, the template removes instructions about <search>/<python>/<result>/tool tags
            while keeping the rest of the wording as close as possible.
        """
        self.prompt_type = prompt_type
        self.use_tool = use_tool
        self.prompt_template = self._append_read_tool_notice(self._get_template())

    def _append_read_tool_notice(self, prompt: str) -> str:
        if not self.use_tool:
            return prompt
        if self.prompt_type == "interaction":
            return prompt
        if "<read>" in prompt or "read tool" in prompt.lower() or "文件读取工具" in prompt:
            return prompt
        notice = READ_TOOL_PROTOCOL_ZH if self.prompt_type == "code_search_cn" else READ_TOOL_PROTOCOL_EN
        return prompt.rstrip() + "\n\n" + notice

    def _get_template(self) -> str:
        """Get the prompt template based on prompt type."""
        # code + search
        if self.prompt_type == "code_search":
            if self.use_tool:
                return (
                    "You are a helpful assistant that can solve the given question step by step "
                    "with the help of the wikipedia search tool and python interpreter tool. "
                    "Given a question, you need to first think about the reasoning process in the mind "
                    "and then provide the answer. "
                    "During thinking, you can invoke the wikipedia search tool to search and python interpreter "
                    "tool to calculate the math problem for fact information about specific topics if needed. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, "
                    "and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. "
                    "For example, <think> This is the reasoning process. </think> "
                    "<search> search query here </search> <result> search result here </result> "
                    "<think> This is the reasoning process. </think> "
                    "<python> python code here </python> <result> python interpreter result here </result> "
                    "<think> This is the reasoning process. </think> "
                    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
                    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
                )
            else:
                # 无工具版：删除关于 wikipedia/python 工具和标签的说明，仅保留思维 + 答案结构
                return (
                    "You are a helpful assistant that can solve the given question step by step. "
                    "Given a question, you need to first think about the reasoning process in the mind "
                    "and then provide the answer. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. "
                    "For example, <think> This is the reasoning process. </think> "
                    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
                    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
                )

        # search only
        if self.prompt_type == "search":
            if self.use_tool:
                return (
                    "You are a helpful assistant that can solve the given question step by step with the help "
                    "of the wikipedia search tool. "
                    "Given a question, you need to first think about the reasoning process in the mind and then provide the answer. "
                    "During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, "
                    "and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. "
                    "For example, <think> This is the reasoning process. </think> "
                    "<search> search query here </search> <result> search result here </result> "
                    "<think> This is the reasoning process. </think> "
                    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
                    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
                )
            else:
                return (
                    "You are a helpful assistant that can solve the given question step by step. "
                    "Given a question, you need to first think about the reasoning process in the mind and then provide the answer. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. "
                    "For example, <think> This is the reasoning process. </think> "
                    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
                    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
                )

        # math (python only)
        if self.prompt_type == "math":
            if self.use_tool:
                return (
                    "You are a helpful assistant that can solve the given question step by step with the help of the python interpreter tool. "
                    "Given a question, you need to first think about the reasoning process in the mind and then provide the answer. "
                    "During thinking, you can invoke the python interpreter tool to calculate the math problem for fact information about specific topics if needed. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. "
                    "For example, <think> This is the reasoning process. </think> "
                    "<python> python code here </python> <result> python interpreter result here </result> "
                    "<think> This is the reasoning process. </think> "
                    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
                    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
                )
            else:
                return (
                    "You are a helpful assistant that can solve the given math question step by step. "
                    "Given a question, you need to first think about the reasoning process in the mind and then provide the answer. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. "
                    "For example, <think> This is the reasoning process. </think> "
                    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
                    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
                )

        # no tools
        if self.prompt_type == "base":
            return (
                "You are a helpful assistant that can solve the given question step by step. "
                "Given a question, you need to first think about the reasoning process in the mind and then provide the answer. "
                "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. "
                "For example, <think> This is the reasoning process. </think> "
                "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
                "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
            )

        # Chinese code + search
        if self.prompt_type == "code_search_cn":
            if self.use_tool:
                return (
                    "你是一个乐于助人的助手，能够借助 Wikipedia 搜索工具和 Python 解释器工具，逐步解决给定的问题。"
                    "给定一个问题后，你需要先在头脑中进行推理过程，然后再提供答案。"
                    "在思考过程中，你可以调用 Wikipedia 搜索工具来搜索特定主题的事实信息，也可以使用 Python 解释器工具来计算数学问题（如有需要）。"
                    "推理过程和答案分别用 <think> 和 </think>，以及 <answer> 和 </answer> 标签括起来；"
                    "搜索查询和结果分别用 <search> 和 </search>，以及 <result> 和 </result> 标签括起来。"
                    "例如："
                    "<think> 这是推理过程。 </think> <search> 这里是搜索查询 </search> <result> 这里是搜索结果 </result> "
                    "<think> 这是推理过程。 </think> <python> 这里是 Python 代码 </python> <result> 这里是 Python 解释器的结果 </result> "
                    "<think> 这是推理过程。 </think> <answer> 最终答案是 \\[ \\boxed{这里是答案} \\] </answer>"
                )
            else:
                return (
                    "你是一个乐于助人的助手，能够逐步解决给定的问题。"
                    "给定一个问题后，你需要先在头脑中进行推理过程，然后再提供答案。"
                    "推理过程和答案分别用 <think> 和 </think>，以及 <answer> 和 </answer> 标签括起来。"
                    "例如："
                    "<think> 这是推理过程。 </think> "
                    "<answer> 最终答案是 \\[ \\boxed{这里是答案} \\] </answer>"
                )

        # gemini / claude 风格：高级工具助手
        if self.prompt_type in ["gemini", "claude"]:
            if self.use_tool:
                return (
                    "You are an advanced problem-solving assistant with access to web search and Python interpreter tools. "
                    "Your task is to solve questions methodically through a structured approach that combines careful reasoning with appropriate tool usage.\n\n"
                    "## Response Structure\n"
                    "1. **Thinking Phase** - enclosed within `<think>` and `</think>` tags\n"
                    "2. **Tool Usage** - enclosed within `<search>` `</search>` or `<python>` `</python>` tags\n"
                    "3. **Tool Results** - enclosed within `<result>` and `</result>` tags\n"
                    "4. **Final Answer** - enclosed within `<answer>` and `</answer>` tags, with the exact answer in LaTeX format inside `\\boxed{}`\n\n"
                    "## Process Requirements\n"
                    "1. **Initial Analysis**: Begin by analyzing the problem, breaking it down into components, identifying the key information needed, "
                    "and outlining a solution strategy with specific tools to use.\n\n"
                    "2. **Iterative Reasoning**: After each tool use, evaluate the results and refine your approach. Continue this cycle until you reach the solution.\n\n"
                    "3. **Tool Utilization**:\n"
                    "   - Use web search for factual information, definitions, formulas, or domain knowledge\n"
                    "   - Use Python interpreter for calculations, data processing, and algorithm implementation\n"
                    "   - Prioritize these tools over relying on your internal knowledge for complex or specialized information\n\n"
                    "4. **Thinking Guidelines**:\n"
                    "   - Keep each thinking section concise (under 1000 words)\n"
                    "   - Focus on analysis and planning, not solving the entire problem within the thinking sections\n"
                    "   - Explicitly state what information you need to search for or what calculations to perform\n\n"
                    "5. **Final Answer Format**:\n"
                    "   - Present only the final answer without showing the solution process\n"
                    "   - Format the exact answer in LaTeX within `\\boxed{}`\n\n"
                    "Remember to delegate computational tasks to Python and knowledge-intensive tasks to web search "
                    "rather than attempting to compute or recall complex information yourself."
                )
            else:
                return (
                    "You are an advanced problem-solving assistant. "
                    "Your task is to solve questions methodically through a structured approach that emphasizes careful reasoning.\n\n"
                    "## Response Structure\n"
                    "1. **Thinking Phase** - enclosed within `<think>` and `</think>` tags\n"
                    "2. **Final Answer** - enclosed within `<answer>` and `</answer>` tags, with the exact answer in LaTeX format inside `\\boxed{}`\n\n"
                    "## Process Requirements\n"
                    "1. **Initial Analysis**: Begin by analyzing the problem, breaking it down into components, identifying the key information needed, "
                    "and outlining a solution strategy.\n\n"
                    "2. **Iterative Reasoning**: Refine your reasoning step by step based on logical deductions until you reach the solution.\n\n"
                    "3. **Thinking Guidelines**:\n"
                    "   - Keep each thinking section concise (under 1000 words)\n"
                    "   - Focus on analysis and planning, not solving the entire problem within the thinking sections\n\n"
                    "4. **Final Answer Format**:\n"
                    "   - Present only the final answer without showing the full solution process\n"
                    "   - Format the exact answer in LaTeX within `\\boxed{}`"
                )

        # react 风格：仅支持标签式工具调用
        if self.prompt_type == "react":
            if self.use_tool:
                return (
                    "You are a helpful assistant that answers complex questions step by step using web search.\n"
                    "Use the following structure:\n\n"
                    "- Action: <search> ... </search> → to issue a search query (Wikipedia entity)\n"
                    "- Observation: <result> ... </result> → to read the result returned from the system\n"
                    "- Final Answer: <answer> ... </answer> → when ready, output the final answer using LaTeX format \\[ \\boxed{{...}} \\]\n\n"
                    "You may use up to 10 search actions."
                )
            else:
                return (
                    "You are a helpful assistant that answers complex questions step by step. "
                    "Carefully reason about the problem and provide a final answer.\n\n"
                    "Use the following structure:\n\n"
                    "- Thinking: enclose your reasoning within <think> ... </think>\n"
                    "- Final Answer: enclose the final answer within <answer> ... </answer> using LaTeX format \\[ \\boxed{{...}} \\]"
                )

        # interaction_agent: 仿照 interaction/interaction_process.py 中的 AGENT_SYSTEM_PROMPT
        # 用于电商客服 Agent 角色，多轮对话任务，支持标签化工具调用。
        if self.prompt_type == "interaction":
            # 该 prompt 本身不依赖样本字段，直接作为 system prompt 使用。
            if self.use_tool:
                return (
                    "You are a helpful e-commerce customer service agent representing the seller. "
                    "Your goal is to solve the customer's request within 10 turns while staying truthful. "
                    "If information is missing, ask concise clarifying questions. "
                    "Use available tools when needed (product_search, inventory_check, policy_search, order_lookup, pricing_calc). "
                    "Do not invent order details or policies—verify using tools when uncertain. "
                    "Be calm and de-escalate conflicts.\n\n"
                    "Tool usage (tag-based protocol):\n"
                    "- To search products, wrap your request in <product_search> and </product_search>.\n"
                    "  Example: <product_search>{\"query\": \"noise cancelling headphones\", \"domain\": \"electronics\", \"k\": 5}</product_search>.\n"
                    "- To check inventory, wrap your request in <inventory_check> and </inventory_check>.\n"
                    "  Example: <inventory_check>{\"sku\": \"ELEC-1001\"}</inventory_check>.\n"
                    "- To look up store policies, wrap your request in <policy_search> and </policy_search>.\n"
                    "  Example: <policy_search>{\"topic\": \"return policy for opened electronics\"}</policy_search>.\n"
                    "- To look up an order, wrap your request in <order_lookup> and </order_lookup>.\n"
                    "  Example: <order_lookup>{\"order_id\": \"12345-AB\"}</order_lookup>.\n"
                    "- To calculate pricing, wrap your request in <pricing_calc> and </pricing_calc>.\n"
                    "  Example: <pricing_calc>{\"sku\": \"ELEC-1001\", \"quantity\": 2, \"coupon_code\": \"SAVE10\"}</pricing_calc>.\n\n"
                    "For each tool call, the system will execute the tool and return its JSON result inside <result> and </result> tags "
                    "immediately after your tool tag. You may interleave multiple rounds of tool calls and natural language. "
                    "Use the information from <result>...</result> to update your reasoning and provide a clear, truthful answer to the customer."
                )
            else:
                return (
                    "You are a helpful e-commerce customer service agent representing the seller. "
                    "Your goal is to solve the customer's request within 10 turns while staying truthful. "
                    "If information is missing, ask concise clarifying questions. "
                    "Do not invent order details or policies—base your answers only on the given scenario and conversation history. "
                    "Be calm and de-escalate conflicts. "
                    "You do not have access to external tools; rely only on the information provided in the dialogue."
                )

        # interaction_customer: 仿照 interaction/interaction_process.py 中的
        # CUSTOMER_JUDGE_SYSTEM_PROMPT + build_customer_context + CUSTOMER_RESPONSE_INSTRUCTIONS。
        # 这里允许使用 {category}/{title}/... 这种占位符，推理阶段可通过 str.format(**scenario_dict) 填充。

        # expodesign: 实验设计任务，支持标签式工具调用（<python>/<search>/<result>/<answer>）
        # 输入：用户消息中论文信息以标签包裹，见下方说明
        if self.prompt_type == "expodesign":
            if self.use_tool:
                # 标签模式：模型通过 <python>/<search> 请求工具，系统在 <result> 中返回结果；
                # 最终实验卡 JSON 必须放在 <answer>...</answer> 中，且不包含 runnable_code 字段。
                return (
                    "You are an expert research engineer. Given a paper and a research problem statement, "
                    "you must produce a minimal, runnable experiment design and implementation plan.\n\n"
                    "Input format: The user message contains the paper information in tagged form. "
                    "Use these fields when designing your experiment:\n"
                    "- The domain is between <domain> and </domain>.\n"
                    "- The paper title is between <title> and </title>.\n"
                    "- The URL is between <url> and </url>.\n"
                    "- The abstract is between <abstract> and </abstract>.\n"
                    "- If present, extra extracted text is between <extra_text> and </extra_text>.\n\n"
                    f"{EXPO_DESIGN_TOOL_PROTOCOL}\n\n"
                    f"{EXPO_DESIGN_JSON_SCHEMA}"
                )
            else:
                # 无工具版：保留 JSON schema 和 <answer> 要求，但不再允许 <python>/<search>/<result> 标签。
                return (
                    "You are an expert research engineer. Given a paper and a research problem statement, "
                    "you must produce a minimal, runnable experiment design and implementation plan.\n\n"
                    "Input format: The user message contains the paper information in tagged form. "
                    "Use these fields when designing your experiment:\n"
                    "- The domain is between <domain> and </domain>.\n"
                    "- The paper title is between <title> and </title>.\n"
                    "- The URL is between <url> and </url>.\n"
                    "- The abstract is between <abstract> and </abstract>.\n"
                    "- If present, extra extracted text is between <extra_text> and </extra_text>.\n\n"
                    "You must reason step by step and rely only on the information provided in the paper and your general knowledge; "
                    "do NOT call any external tools such as web search or code execution, and do NOT emit any <python>, <search>, or <result> tags. "
                    "After your reasoning, you MUST output a final experiment card as JSON inside <answer> and </answer>, "
                    "with exactly the following schema (no runnable_code field and no extra keys):\n\n"
                    f"{EXPO_DESIGN_JSON_SCHEMA}"
                )

        # 未识别的 prompt_type
        raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def get_csbench_prompt(self, format: str = None) -> str:
        """
        CSBench 专用系统提示词（标签式工具调用协议：<search>/<python>/<read>/<result>/<answer>）。
        """
        tool_line = (
            "When needed, call tools with <search>...</search>, <python>...</python>, or <read>...</read>. "
            "The system will return tool outputs inside <result>...</result>. "
        )
        final_line = (
            "Put your reasoning in <think>...</think> and your final answer in <answer>...</answer>. "
            "When the answer is exact, enclose it in LaTeX \\[ \\boxed{...} \\]."
        )

        if format == "Multiple-choice":
            if self.use_tool:
                return (
                    "You are a helpful assistant that answers multiple-choice questions step by step. "
                    "Analyze the question and the options carefully, then decide whether you need factual lookup, computation, or file reading. "
                    + tool_line
                    + final_line
                    + "The final boxed answer should be the option label or the exact option content."
                )
            return (
                "You are a helpful assistant that answers multiple-choice questions step by step. "
                + final_line
                + "The final boxed answer should be the option label or the exact option content."
            )

        if format == "Assertion":
            if self.use_tool:
                return (
                    "You are a helpful assistant that determines whether a statement is true or false. "
                    "First identify what facts, calculations, or file evidence are needed. "
                    + tool_line
                    + final_line
                    + "Output the final verdict as \\[ \\boxed{true} \\] or \\[ \\boxed{false} \\]."
                )
            return (
                "You are a helpful assistant that determines whether a statement is true or false. "
                + final_line
                + "Output the final verdict as \\[ \\boxed{true} \\] or \\[ \\boxed{false} \\]."
            )

        if format == "Fill-in-the-blank":
            if self.use_tool:
                return (
                    "You are a helpful assistant that solves fill-in-the-blank questions step by step. "
                    "Reason about what information is missing and whether it should come from search, computation, or file reading. "
                    + tool_line
                    + final_line
                )
            return (
                "You are a helpful assistant that solves fill-in-the-blank questions step by step. "
                + final_line
            )

        if format == "Open-ended":
            if self.use_tool:
                return (
                    "You are a helpful assistant that answers open-ended questions step by step. "
                    "Analyze the question, decide what outside information is needed, and use search, Python, or file reading when appropriate. "
                    + tool_line
                    + final_line
                )
            return (
                "You are a helpful assistant that answers open-ended questions step by step. "
                + final_line
            )

        raise ValueError(f"Unknown format: {format}")

    def get_system_prompt(self, format: str=None) -> str:
        """Get the system prompt."""
        if format is not None and format == "Multiple-choice":
            return self.get_csbench_prompt(format)
        else:
            return self.prompt_template
    
