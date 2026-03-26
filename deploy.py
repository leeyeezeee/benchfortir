#!/usr/bin/env python
"""
deploy.py

读取 llm_config YAML：
  - 用 vllm.tensor_parallel_size 表示「每个实例」需要的 GPU 数；
  - 用环境变量表示本机参与部署的 GPU 总数；
  - 可部署实例数 = 总 GPU 数 // 每实例 GPU 数；
  - 按实例依次分配连续物理 GPU（0 起），端口从首端口递增。

环境变量（二选一，优先 VLLM_TOTAL_GPUS）：
  VLLM_TOTAL_GPUS 或 DEPLOY_TOTAL_GPUS
    参与本次部署的 GPU 数量（默认 4），物理编号假设为连续 0..N-1。

可选环境变量：
  VLLM_GPU_OFFSET 或 DEPLOY_GPU_OFFSET
    物理 GPU 起始偏移（默认 0）。例如:
      offset=1, total_gpus=4 表示本次部署使用物理 GPU [1,2,3,4]。

可选环境变量：
  VLLM_BASE_PORT — 第一个实例端口（默认用 YAML 中 vllm.port，否则 8001）

各 vLLM 子进程：stderr 输出到当前终端，stdout 丢弃（避免多实例混排）；不写日志文件。

用法示例：
  export VLLM_TOTAL_GPUS=4
  python deploy.py --config src/config/llm_config/Qwen3_8B.yaml
  # 若 tensor_parallel_size=2，则部署 2 个实例：GPU 0,1 与 2,3，端口 8001、8002
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


def total_gpus_from_env() -> int:
    raw = os.environ.get("VLLM_TOTAL_GPUS") or os.environ.get("DEPLOY_TOTAL_GPUS") or "4"
    try:
        n = int(str(raw).strip())
    except ValueError as e:
        raise ValueError(
            f"VLLM_TOTAL_GPUS / DEPLOY_TOTAL_GPUS 必须是整数，当前为: {raw!r}"
        ) from e
    if n < 1:
        raise ValueError("环境变量中的 GPU 总数必须 >= 1")
    return n


def gpu_offset_from_env() -> int:
    raw = os.environ.get("VLLM_GPU_OFFSET") or os.environ.get("DEPLOY_GPU_OFFSET") or "0"
    try:
        off = int(str(raw).strip())
    except ValueError as e:
        raise ValueError(
            f"VLLM_GPU_OFFSET / DEPLOY_GPU_OFFSET 必须是整数，当前为: {raw!r}"
        ) from e
    if off < 0:
        raise ValueError("VLLM_GPU_OFFSET / DEPLOY_GPU_OFFSET 必须 >= 0")
    return off


def base_port_from_env_and_config(config: Dict[str, Any]) -> int:
    env_p = os.environ.get("VLLM_BASE_PORT")
    if env_p is not None and str(env_p).strip() != "":
        return int(str(env_p).strip())
    vllm_cfg = config.get("vllm", {}) or {}
    return int(vllm_cfg.get("port", 8001))


def gpu_ids_for_instance(
    instance_index: int,
    gpus_per_inst: int,
    total_gpus: int,
    gpu_offset: int,
) -> str:
    """第 instance_index 个实例的 CUDA_VISIBLE_DEVICES，如 '0,1'。

    这里的 GPU 编号是“物理 GPU 编号”（考虑 gpu_offset）。
    """
    start = gpu_offset + instance_index * gpus_per_inst
    end = start + gpus_per_inst
    # total_gpus 表示从 gpu_offset 开始的参与 GPU 数量
    if end > gpu_offset + total_gpus:
        raise RuntimeError(
            f"实例 {instance_index} 需要 GPU [{start},{end})，但参与池仅有 {total_gpus} 张 GPU "
            f"(gpu_offset={gpu_offset})"
        )
    return ",".join(str(i) for i in range(start, end))


def build_vllm_command(
    config: Dict[str, Any],
    *,
    gpu_ids: str,
    port: int,
) -> Tuple[Dict[str, str], list]:
    """构建 vllm serve；gpu_ids 与 port 按实例覆盖（YAML 中 vllm.gpu_ids / port 仅作模板时不再用于多实例）。"""
    llm_name = config.get("llm_name") or config.get("llm_nmae")
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
        print("检测到 vllm.remote = True，假定使用远程已部署的 LLM 服务。")
        print(f"endpoints: {endpoints}")
        print(f"api_keys: {['***' if k else '' for k in api_keys]}")
        print(f"default_model: {default_model}")
        print("本地不会启动 vllm 服务；推理脚本请直接使用这些 endpoint 和 default_model。")
        return

    per = gpus_per_instance(config)
    total = total_gpus_from_env()
    gpu_offset = gpu_offset_from_env()
    if per > total:
        raise ValueError(
            f"单实例需要 {per} 张 GPU（tensor_parallel_size），但环境预设总 GPU 数为 {total}，无法部署任何实例。"
        )

    num_instances = total // per
    remainder = total % per
    base_port = base_port_from_env_and_config(config)

    print(
        f"[deploy] 每实例 GPU 数（tensor_parallel_size）: {per}；"
        f"环境总 GPU 数: {total}；可部署实例数: {num_instances}"
    )
    if remainder > 0:
        print(
            f"[deploy] 提示: {total} 无法被 {per} 整除，剩余 {remainder} 张 GPU 未使用。"
        )

    procs: List[subprocess.Popen] = []

    for i in range(num_instances):
        gid = gpu_ids_for_instance(i, per, total, gpu_offset)
        port = base_port + i
        env, cmd = build_vllm_command(config, gpu_ids=gid, port=port)

        print(
            f"[deploy] 实例 {i}: CUDA_VISIBLE_DEVICES={gid} port={port} (gpu_offset={gpu_offset})"
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 YAML 与 VLLM_TOTAL_GPUS 计算并部署尽可能多的 vLLM 实例。"
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
