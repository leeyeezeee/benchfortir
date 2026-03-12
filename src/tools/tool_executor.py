import sys
import os

sys.path.append(os.getcwd())
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool


class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tools_type: List[str] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=20)

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.trigger_tag] = tool
        if tool.name not in self.tools_type:
            self.tools_type.append(tool.name)

    def get_tool(self, tag: str) -> Optional[BaseTool]:
        return self.tools.get(tag)

    async def execute(self, tag: str, content: str, **kwargs) -> str:
        tool = self.get_tool(tag)
        if not tool:
            raise ValueError(f"Tool '{tag}' is not found")

        try:
            result = await tool.execute(content, **kwargs)
            return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            import sys

            sys.stderr.flush()
            print("Tool execute failed!")
            sys.stdout.flush()
            return f"Tool execute failed: {str(e)}"

    def extract_content(self, text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_pos = text.find(start_tag)
        if start_pos == -1:
            return ""

        start_pos += len(start_tag)
        end_pos = text.find(end_tag, start_pos)
        if end_pos == -1:
            return ""

        return text[start_pos:end_pos].strip()

    def identify_tool(self, text: str) -> Optional[str]:
        matches: List[Tuple[int, str]] = []
        for tag in self.tools_type:
            end_tag = f"</{tag}>"
            pos = text.find(end_tag)
            if pos != -1:
                matches.append((pos, tag))
        if not matches:
            return None
        matches.sort(key=lambda x: x[0])
        return matches[0][1]
