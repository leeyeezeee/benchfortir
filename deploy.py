#!/usr/bin/env python
"""
deploy.py

读取 llm_config YAML：
  - vllm.gpu_ids 指定实例与显卡绑定关系；
  - 支持字符串（单实例）或字符串列表（多实例）；
  - 每个实例端口从首端口递增（base_port + index）。

可选环境变量：
  VLLM_BASE_PORT — 第一个实例端口（默认用 YAML 中 vllm.port，否则 8001）

各 vLLM 子进程：stderr 输出到当前终端，stdout 丢弃（避免多实例混排）；不写日志文件。

用法示例：
  python deploy.py --config src/config/llm_config/Qwen3_8B.yaml
  # 若 vllm.gpu_ids: ["0,1", "2,3"]，则部署 2 个实例，端口 8001、8002
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("请先安装 pyyaml: pip install pyyaml")
    sys.exit(1)


def pid_file_path() -> str:
    """Return PID file path for deployed local vLLM instances."""
    run_dir = os.path.join(os.getcwd(), "run")
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "vllm_pids.txt")


def write_pid_file(pids: List[int]) -> None:
    path = pid_file_path()
    with open(path, "w", encoding="utf-8") as f:
        for pid in pids:
            f.write(f"{pid}\n")
    print(f"[deploy] PID 文件已写入: {path}")


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def gpus_per_instance(config: Dict[str, Any]) -> int:
    """每个实例需要的 GPU 数：以 vllm.tensor_parallel_size 为准。"""
    vllm_cfg = config.get("vllm", {}) or {}
    tp = int(vllm_cfg.get("tensor_parallel_size", 1))
    if tp < 1:
        raise ValueError("vllm.tensor_parallel_size 必须 >= 1")
    return tp


def base_port_from_env_and_config(config: Dict[str, Any]) -> int:
    env_p = os.environ.get("VLLM_BASE_PORT")
    if env_p is not None and str(env_p).strip() != "":
        return int(str(env_p).strip())
    vllm_cfg = config.get("vllm", {}) or {}
    return int(vllm_cfg.get("port", 8001))


def parse_gpu_groups(vllm_cfg: Dict[str, Any], gpus_per_inst: int) -> List[str]:
    """解析并校验 vllm.gpu_ids，返回每个实例的 CUDA_VISIBLE_DEVICES。"""
    raw = vllm_cfg.get("gpu_ids")
    if raw is None:
        raise ValueError("缺少 vllm.gpu_ids")

    if isinstance(raw, str):
        groups = [raw.strip()]
    elif isinstance(raw, list):
        groups = []
        for i, item in enumerate(raw):
            if not isinstance(item, str):
                raise ValueError(
                    f"vllm.gpu_ids[{i}] 必须是字符串，如 '0,1'；当前类型={type(item).__name__}"
                )
            groups.append(item.strip())
    else:
        raise ValueError("vllm.gpu_ids 必须是字符串或字符串列表")

    if not groups or any(not g for g in groups):
        raise ValueError("vllm.gpu_ids 不能为空；示例：'0,1' 或 ['0,1', '2,3']")

    seen_cards = set()
    for idx, group in enumerate(groups):
        tokens = [x.strip() for x in group.split(",")]
        if any(not t for t in tokens):
            raise ValueError(
                f"vllm.gpu_ids[{idx}] 格式非法：{group!r}（请使用逗号分隔，如 '0,1'）"
            )
        try:
            cards = [int(x) for x in tokens]
        except ValueError as e:
            raise ValueError(
                f"vllm.gpu_ids[{idx}] 包含非整数卡号：{group!r}"
            ) from e
        if any(c < 0 for c in cards):
            raise ValueError(
                f"vllm.gpu_ids[{idx}] 包含负数卡号：{group!r}"
            )
        if len(cards) != gpus_per_inst:
            raise ValueError(
                f"vllm.gpu_ids[{idx}] 包含 {len(cards)} 张卡，但 tensor_parallel_size={gpus_per_inst}"
            )
        for c in cards:
            if c in seen_cards:
                raise ValueError(f"GPU {c} 被重复分配，请检查 vllm.gpu_ids")
            seen_cards.add(c)
    return groups


def build_vllm_command(
    config: Dict[str, Any],
    *,
    gpu_ids: str,
    port: int,
) -> Tuple[Dict[str, str], list]:
    """构建 vllm serve；gpu_ids 与 port 按实例覆盖（YAML 中 vllm.gpu_ids / port 仅作模板时不再用于多实例）。"""
    llm_name = config.get("llm_name")
    model_path = config.get("model_path")
    default_model = config.get("default_model") or llm_name

    if not model_path:
        raise ValueError("配置文件中缺少必需字段: model_path")

    vllm_cfg = config.get("vllm", {}) or {}
    host = vllm_cfg.get("host", "0.0.0.0")
    tensor_parallel_size = int(vllm_cfg.get("tensor_parallel_size", 1))
    max_model_len = int(vllm_cfg.get("max_model_len", 32768))
    gpu_memory_utilization = float(vllm_cfg.get("gpu_memory_utilization", 0.9))
    max_logprobs = vllm_cfg.get("max_logprobs")
    extra_args = vllm_cfg.get("extra_args") or []

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        default_model,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--port",
        str(port),
        "--host",
        host,
    ]

    if max_logprobs is not None:
        cmd.extend(["--max-logprobs", str(max_logprobs)])

    if isinstance(extra_args, list):
        cmd.extend(str(x) for x in extra_args)
    elif isinstance(extra_args, str) and extra_args.strip():
        cmd.extend(extra_args.split())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    return env, cmd


def deploy_vllm_multi(config_path: str) -> None:
    config = load_config(config_path)
    vllm_cfg = config.get("vllm", {}) or {}

    if bool(vllm_cfg.get("remote", False)):
        endpoints = vllm_cfg.get("endpoints") or []
        api_keys = vllm_cfg.get("api_keys") or []
        default_model = config.get("default_model") or config.get("llm_name")
        print(f"endpoints: {endpoints}")
        print(f"api_keys: {['***' if k else '' for k in api_keys]}")
        print(f"default_model: {default_model}")
        print("本地不会启动 vllm 服务；推理脚本请直接使用这些 endpoint 和 default_model。")
        return

    per = gpus_per_instance(config)
    gpu_groups = parse_gpu_groups(vllm_cfg, gpus_per_inst=per)
    base_port = base_port_from_env_and_config(config)
    print(f"[deploy] 每实例 GPU 数（tensor_parallel_size）: {per}")
    print(f"[deploy] 按 vllm.gpu_ids 启动实例数: {len(gpu_groups)}")

    procs: List[subprocess.Popen] = []

    for i, gid in enumerate(gpu_groups):
        port = base_port + i
        env, cmd = build_vllm_command(config, gpu_ids=gid, port=port)

        print(
            f"[deploy] 实例 {i}: CUDA_VISIBLE_DEVICES={gid} port={port}"
        )
        print("  " + " ".join(cmd))

        # 仅 stderr 到终端；stdout 丢弃，避免多实例输出交错难以阅读
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
        procs.append(proc)
        print(f"[deploy] 实例 {i} 已启动，PID = {proc.pid}")

    print(f"[deploy] 共启动 {len(procs)} 个 vLLM 进程。")
    write_pid_file([proc.pid for proc in procs])


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 YAML 中 vllm.gpu_ids 指定的显卡列表部署 vLLM 实例。"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="路径：llm_config 配置文件",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    deploy_vllm_multi(args.config)


if __name__ == "__main__":
    main()
