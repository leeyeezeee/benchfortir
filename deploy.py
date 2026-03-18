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
import signal
import time
from typing import Any, Dict, Optional, Tuple, List

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


def _parse_gpus(gpus: str) -> list:
    if not gpus:
        return []
    items = []
    for x in str(gpus).split(","):
        x = x.strip()
        if not x:
            continue
        items.append(x)
    return items


def _spawn_vllm(
    *,
    cmd: list,
    env: Dict[str, str],
    error_only: bool,
    stderr_path: Optional[str],
):
    if error_only:
        stdout = subprocess.DEVNULL
    else:
        stdout = sys.stdout

    if stderr_path:
        os.makedirs(os.path.dirname(stderr_path), exist_ok=True)
        stderr_f = open(stderr_path, "ab", buffering=0)
        stderr = stderr_f
    else:
        stderr_f = None
        stderr = sys.stderr

    proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
    return proc, stderr_f


def deploy_multi_vllm(
    config_path: str,
    gpus: str,
    base_port: int,
    host: str,
    error_only: bool,
    log_dir: Optional[str],
    startup_sleep: float,
):
    config = load_config(config_path)
    vllm_cfg = config.get("vllm", {}) or {}
    if bool(vllm_cfg.get("remote", False)):
        raise ValueError("multi-instance mode requires vllm.remote = false (local serve).")

    gpu_list = _parse_gpus(gpus)
    if not gpu_list:
        raise ValueError("--gpus is required in multi-instance mode, e.g. --gpus 0,1,2,3")

    procs: List[subprocess.Popen] = []
    opened_logs = []

    stopping = {"flag": False}

    def _stop_all(signum=None, frame=None):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        time.sleep(1.0)
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _stop_all)
    signal.signal(signal.SIGTERM, _stop_all)

    for i, gpu in enumerate(gpu_list):
        port = int(base_port) + i

        # Override per-instance fields.
        cfg_i = dict(config)
        vllm_i = dict(vllm_cfg)
        vllm_i["gpu_ids"] = str(gpu)
        vllm_i["port"] = int(port)
        vllm_i["host"] = host or vllm_i.get("host", "0.0.0.0")
        cfg_i["vllm"] = vllm_i

        env, cmd = build_vllm_command(cfg_i)

        stderr_path = None
        if log_dir:
            model_tag = str(cfg_i.get("default_model") or cfg_i.get("llm_name") or "model")
            safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_tag)
            stderr_path = os.path.join(log_dir, f"vllm_{safe_tag}_gpu{gpu}_port{port}.err.log")

        proc, stderr_f = _spawn_vllm(cmd=cmd, env=env, error_only=error_only, stderr_path=stderr_path)
        procs.append(proc)
        if stderr_f is not None:
            opened_logs.append(stderr_f)

        if not error_only:
            print(f"[deploy] started vLLM pid={proc.pid} gpu={gpu} port={port}")
        if startup_sleep and startup_sleep > 0:
            time.sleep(float(startup_sleep))

    # Wait for all; if any exits unexpectedly, stop the rest.
    try:
        while True:
            alive = 0
            for p in procs:
                if p.poll() is None:
                    alive += 1
            if alive == 0:
                break
            # If one died while others alive, stop all to avoid dangling.
            for p in procs:
                if p.poll() is not None and alive > 0:
                    _stop_all()
                    break
            time.sleep(0.5)
    finally:
        _stop_all()
        for f in opened_logs:
            try:
                f.close()
            except Exception:
                pass


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
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help='多实例模式：GPU 列表（逗号分隔），例如 "0,1,2,3"；为空则按配置启动单实例。',
    )
    parser.add_argument(
        "--base_port",
        type=int,
        default=8001,
        help="多实例模式：起始端口（后续实例按 +1 递增）。",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="多实例模式：绑定 host（默认 0.0.0.0）。",
    )
    parser.add_argument(
        "--error_only",
        action="store_true",
        help="仅输出错误日志：vLLM stdout 丢弃，stderr 记录到文件（需 --log_dir）或终端。",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="多实例模式：错误日志目录（默认 logs）。为空则不写文件。",
    )
    parser.add_argument(
        "--startup_sleep",
        type=float,
        default=0.0,
        help="多实例模式：每个实例启动后等待秒数（避免同时加载造成抖动）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpus:
        deploy_multi_vllm(
            config_path=args.config,
            gpus=args.gpus,
            base_port=args.base_port,
            host=args.host,
            error_only=bool(args.error_only),
            log_dir=(args.log_dir if str(args.log_dir).strip() else None),
            startup_sleep=float(args.startup_sleep or 0.0),
        )
    else:
        deploy_local_vllm(args.config)


if __name__ == "__main__":
    main()