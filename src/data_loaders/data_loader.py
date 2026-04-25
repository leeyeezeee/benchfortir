import sys
import os

sys.path.append(os.getcwd())

import json
from typing import List, Tuple, Union

from ..utils import extract_solution, last_boxed_only_string, remove_boxed


class DataLoader:
    """Loading Datasets"""

    def __init__(self, args):
        """
        Args:
            dataset_name
            data_path
        """
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.data_path = os.path.join(self.data_path, self.dataset_name, 'test.jsonl')
        self.counts = args.counts

    def load_data(
        self,
    ) -> Tuple[
        List[str],
        List[Union[str, List[str]]],
        List[List[str]],
        List[str],
        List[str],
    ]:
        """
        Load dataset

        Returns:
            (questions, answers, choices, formats, metas)
            answers 每项通常为 str；squadv2 可答题为 List[str]（多参考 span 去重后的文本，供 max EM/F1）。
            metas: expodesign 时与 test.jsonl 的 ``meta`` 字段一致（论文元信息）；其它数据集为与样本等长的空字符串。
        """
        questions = []
        answers = []
        choices = []
        formats = []
        metas: List[str] = []

        print(f"Loading dataset from {self.data_path}")

        if (
            "aime" in self.dataset_name
            or self.dataset_name == "amc23"
            or self.dataset_name == "gsm8k"
            or "tabmwp" == self.dataset_name
            or "gaokao2023en" == self.dataset_name
            or "college_math" == self.dataset_name
        ):
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data.get("question", data.get("problem")))
                    answer = data["answer"]
                    if "gsm8k" in self.data_path:
                        answer = extract_solution(answer)
                    answers.append(answer)
        elif "svamp" == self.dataset_name or "asdiv" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    body = data["body"] if "body" in data else data["Body"]
                    question = (
                        data["question"] if "question" in data else data["Question"]
                    )
                    answer = data["answer"] if "answer" in data else data["Answer"]
                    if "asdiv" in self.data_path:
                        answer = answer.split(" (")[0]
                    questions.append(body + " " + question)
                    answers.append(answer)
        elif "mawps" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["input"])
                    answers.append(data["target"])
        elif "carp_en" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["content"])
                    answers.append(data["answer"])
        elif "minerva_math" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    question = data["problem"]
                    answer = data["solution"]
                    try:
                        answer = remove_boxed(last_boxed_only_string(answer))
                    except:
                        pass
                    questions.append(question)
                    answers.append(answer)
        elif "olympiadbench" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["final_answer"][0])
        elif "math" == self.dataset_name or "aime25" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["problem"])
                    answers.append(data["answer"])
        elif "gaia" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["Question"])
                    answers.append(data["answer"])
        elif "csbench" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["Question"])
                    answers.append(data["Answer"])
                    formats.append(data['Format'])
                    if data['Format'] == "Multiple-choice":
                        choices.append(
                            [data['A'], data['B'], data['C'], data['D']]
                        )
                    else:
                        choices.append([])
        elif "mmlu-pro" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["answer"])
                    choices.append(data['options'])
                    formats.append("Multiple-choice")
        elif self.dataset_name == "expodesign":
            # 仅依赖 data/<dataset>/test.jsonl：推理输入为 question，评测用 meta；不在此拼接或从 task_prompt 构造。
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data.get("answer", ""))
                    metas.append(data.get("meta", "") or "")
        elif self.dataset_name.lower() == "simpleqa":
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    ans = data.get("answer")
                    if isinstance(ans, list):
                        answers.append(ans[0] if ans else "")
                    else:
                        answers.append(ans if ans is not None else "")
        elif self.dataset_name.lower() == "squadv2":
            # SQuAD 2.0：将 question 与 context 拼接作为模型输入。
            # 不可回答：金标为 ""；可答：多 span 去重后的文本列表，与官方 max EM/F1 一致。
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    question = (data.get("question") or "").strip()
                    context = (data.get("context") or "").strip()
                    title = (data.get("title") or "").strip()

                    # Provide passage context explicitly for extractive QA behavior.
                    if title:
                        model_input = (
                            f"Title: {title}\n"
                            f"Context: {context}\n"
                            f"Question: {question}"
                        )
                    else:
                        model_input = (
                            f"Context: {context}\n"
                            f"Question: {question}"
                        )
                    questions.append(model_input)
                    raw_answers = data.get("answers") or []
                    if data.get("is_impossible") or not raw_answers:
                        answers.append("")
                    else:
                        seen = set()
                        texts: List[str] = []
                        for a in raw_answers:
                            if isinstance(a, dict):
                                t = (a.get("text") or "").strip()
                            else:
                                t = str(a).strip()
                            if t and t not in seen:
                                seen.add(t)
                                texts.append(t)
                        if not texts:
                            answers.append("")
                        else:
                            # 统一为 list，便于评测侧对多参考取 max EM/F1（单条时为单元素列表）
                            answers.append(texts)
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["answer"])
        if len(choices) == 0:
            choices = [[]] * len(questions)
        if len(formats) == 0:
            formats = [None] * len(questions)
        if len(metas) != len(questions):
            metas = [""] * len(questions)

        print(f"Loading {len(questions)} samples from {self.data_path}...")
        questions = questions[:self.counts]
        answers = answers[:self.counts]
        choices = choices[:self.counts]
        formats = formats[:self.counts]
        metas = metas[: self.counts]
        return questions, answers, choices, formats, metas
