#!/usr/bin/env python
"""
deploy.py

根据 llm_config 配置文件启动或描述大模型部署。
用法示例：
    python deploy.py --config src/config/llm_config/llm_for_test.yaml
"""

import argparse
import os
import subprocess
import sys
from typing import Any, Dict

try:
    import yaml
except ImportError:
    print("请先安装 pyyaml: pip install pyyaml")
    sys.exit(1)


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_vllm_command(config: Dict[str, Any]) -> (Dict[str, str], list):
    """根据配置构建 vllm serve 命令和环境变量。"""
    llm_name = config.get("llm_name") or config.get("llm_nmae")  # 兼容老字段
    model_path = config.get("model_path")
    default_model = config.get("default_model") or llm_name

    if not model_path:
        raise ValueError("配置文件中缺少必需字段: model_path")

    vllm_cfg = config.get("vllm", {}) or {}
    host = vllm_cfg.get("host", "0.0.0.0")
    port = int(vllm_cfg.get("port", 8001))
    gpu_ids = str(vllm_cfg.get("gpu_ids", "0"))
    tensor_parallel_size = int(vllm_cfg.get("tensor_parallel_size", 1))
    max_model_len = int(vllm_cfg.get("max_model_len", 32768))
    gpu_memory_utilization = float(vllm_cfg.get("gpu_memory_utilization", 0.9))
    max_logprobs = vllm_cfg.get("max_logprobs")
    extra_args = vllm_cfg.get("extra_args") or []

    # 构建命令
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

    # 额外透传参数
    if isinstance(extra_args, list):
        cmd.extend(str(x) for x in extra_args)
    elif isinstance(extra_args, str) and extra_args.strip():
        # 简单按空格拆分
        cmd.extend(extra_args.split())

    # 环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    return env, cmd


def deploy_local_vllm(config_path: str):
    config = load_config(config_path)
    vllm_cfg = config.get("vllm", {}) or {}

    remote = bool(vllm_cfg.get("remote", False))
    if remote:
        # 远程模式: 不在本地起服务，只打印配置提示
        endpoints = vllm_cfg.get("endpoints") or []
        api_keys = vllm_cfg.get("api_keys") or []
        default_model = config.get("default_model") or config.get("llm_name")

        print("检测到 vllm.remote = True，假定使用远程已部署的 LLM 服务。")
        print(f"endpoints: {endpoints}")
        print(f"api_keys: {['***' if k else '' for k in api_keys]}")
        print(f"default_model: {default_model}")
        print("本地不会启动 vllm 服务；推理脚本请直接使用这些 endpoint 和 default_model。")
        return

    # 本地模式: 构建并启动 vllm serve
    env, cmd = build_vllm_command(config)

    print("即将启动 vLLM 服务，命令为：")
    print("CUDA_VISIBLE_DEVICES={}".format(env["CUDA_VISIBLE_DEVICES"]))
    print(" ".join(cmd))

    # 这里直接用 Popen 后台起服务，你可以按需要改成 nohup / systemd 等
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(f"vLLM 服务已启动，PID = {proc.pid}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy LLM using vLLM based on config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="路径：llm_config 配置文件，例如 src/config/llm_config/llm_for_test.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    deploy_local_vllm(args.config)


if __name__ == "__main__":
    main()