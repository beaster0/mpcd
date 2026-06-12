# 让类型注解延迟求值，保持不同 Python 小版本兼容。
from __future__ import annotations

"""统一的 GPU 调度工具。

这个文件不作为命令入口使用。以后所有需要“按 GPU 分片运行任务”的脚本，
都应该调用这里的 `launch_gpu_workers()`，避免每个脚本各写一套 GPU 分配逻辑。
"""

# os 用来复制当前环境变量并设置 CUDA_VISIBLE_DEVICES。
import os
# subprocess 用来启动每张 GPU 对应的子进程。
import subprocess
# sys 用来拿当前 Python 解释器路径。
import sys
# Path 用来传递项目根目录。
from pathlib import Path


def env_for_gpu(gpu: int) -> dict[str, str]:
    """返回只暴露指定 GPU 的环境变量。"""
    # 复制当前环境，保留 conda、PATH 等必要变量。
    env = os.environ.copy()
    # 当前子进程只看到这一张物理 GPU。
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # 返回给 subprocess 使用。
    return env


def launch_gpu_workers(root: Path, script: str, gpus: list[int], extra_args: list[str]) -> int:
    """为多张 GPU 启动多个子进程。

    参数含义：
    - root：项目根目录。
    - script：要运行的入口脚本，例如 `scripts/run.py`。
    - gpus：物理 GPU 编号列表，例如 `[0, 1]`。
    - extra_args：要继续传给入口脚本的额外参数。

    每个子进程只看到一张 GPU，并通过 `--shard-index/--shard-count`
    运行自己负责的那部分任务。
    """
    # 至少要给一张 GPU。
    if not gpus:
        raise ValueError("至少需要指定一张 GPU")
    # 子进程列表。
    processes: list[subprocess.Popen[bytes]] = []
    # 分片总数等于参与运行的 GPU 数量。
    shard_count = len(gpus)
    # 每张 GPU 一个子进程。
    for shard_index, gpu in enumerate(gpus):
        # 子进程调用指定脚本，并带上内部 worker 和分片参数。
        cmd = [
            sys.executable,
            script,
            "--worker",
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
            *extra_args,
        ]
        # 打印实际调度关系。
        print(f"[启动] GPU {gpu} -> 分片 {shard_index}/{shard_count}", flush=True)
        # cwd 固定在项目根目录；env 限制当前子进程只看到指定 GPU。
        processes.append(subprocess.Popen(cmd, cwd=root, env=env_for_gpu(gpu)))
    # 汇总退出码。
    exit_code = 0
    # 等待所有 GPU 子进程结束。
    for process in processes:
        # 任意一个子进程失败，整体返回非零。
        code = process.wait()
        if code != 0:
            exit_code = code
    # 返回整体退出码。
    return exit_code
