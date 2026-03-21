import sys
import os

sys.path.append(os.getcwd())

import json
from typing import Any, Dict, List, Optional



class InteractionDataLoader:
    """
    DataLoader for the interaction (Task1) dialogue task.

    - 输入：类似 interaction/task1_templates.jsonl 的 JSONL 场景模板文件
      每一行是一个场景 dict，字段包括：
        id, category, title, customer_profile, customer_goal, customer_tone,
        constraints, missing_info, potential_misunderstanding, success_criteria,
        product_domain, product_name, order_context, first_user_message, ...

    - 输出：List[scenario_dict]，每个元素可直接传给 interaction.run_dialogue(...)：
        run_dialogue(scenario=..., agent_client=..., agent_cfg=..., customer_client=..., customer_cfg=...)
    """

    def __init__(self, args):
        """
        Args 约定（与 DataLoader 对齐）：
            args.data_path:     数据根目录（如 ``data``）
            args.dataset_name:  子目录名（如 ``interaction``）
            args.counts:        最多加载多少条（<=0 表示全部）
            args.max_turns:     每个场景的最大对话轮数（默认为 10）
            args.inter_filename: 可选，场景 JSONL 文件名；未设则依次尝试
                ``test.jsonl``、``task1_templates.jsonl``
            args.inter_data_path: 可选，直接指定 JSONL 完整路径（优先级最高）
        """
        self.counts = getattr(args, "counts", -1)
        self.max_turns = getattr(args, "max_turns", 10)
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.data_path = os.path.join(self.data_path, self.dataset_name, 'test.jsonl')

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load interaction scenarios.

        Returns
        -------
        scenarios: List[Dict[str, Any]]
            每个元素是一个场景 dict，包含 run_dialogue 需要的字段：
            - id, category, title, product_name, product_domain, order_context,
              first_user_message, customer_profile, customer_goal, customer_tone,
              constraints, missing_info, potential_misunderstanding, success_criteria, ...
            - 以及 max_turns（从 args.max_turns 注入）
        """
        scenarios: List[Dict[str, Any]] = []

        print(f"Loading interaction scenarios from {self.data_path}")

        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Interaction templates file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if self.counts > 0 and idx >= self.counts:
                    break

                raw = json.loads(line)

                scenario: Dict[str, Any] = {
                    # 核心标识 / 元信息
                    "id": raw.get("id"),
                    "category": raw.get("category"),
                    "title": raw.get("title"),
                    "product_name": raw.get("product_name"),
                    "product_domain": raw.get("product_domain"),
                    "schema": raw.get("schema"),
                    "version": raw.get("version"),
                    "created_at": raw.get("created_at"),

                    # 客户画像与目标
                    "customer_profile": raw.get("customer_profile"),
                    "customer_goal": raw.get("customer_goal"),
                    "customer_tone": raw.get("customer_tone"),
                    "constraints": raw.get("constraints", []),
                    "missing_info": raw.get("missing_info", []),
                    "potential_misunderstanding": raw.get("potential_misunderstanding"),
                    "success_criteria": raw.get("success_criteria", []),

                    # 商品 / 订单上下文
                    "order_context": raw.get("order_context", {}),
                    "product_domain": raw.get("product_domain"),
                    "product_name": raw.get("product_name"),

                    # 起始用户消息（run_dialogue 会用这个做 first turn）
                    "first_user_message": raw.get("first_user_message", ""),

                    # 交互配置
                    "max_turns": self.max_turns,
                }

                scenarios.append(scenario)

        print(f"Loaded {len(scenarios)} interaction scenarios from {self.data_path}")
        return scenarios


__all__ = ["InteractionDataLoader"]

