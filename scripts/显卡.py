# 让类型注解延迟求值，保持不同 Python 小版本兼容。
from __future__ import annotations

"""GPU 选择工具。

这个文件只做一件事：让某个运行脚本只使用指定的一张 GPU。
它不是命令入口，平时不要直接运行它。

可以改什么：
- 一般不需要改。
- 如果以后服务器换了显卡选择方式，只改这里。

输入：
- 调用方传入一个 GPU 编号，例如 0。

输出：
- 设置 `CUDA_VISIBLE_DEVICES`，让当前 Python 进程只看到这一张卡。
"""

# os 用来设置环境变量。
import os


def env_for_gpu(gpu: int) -> dict[str, str]:
    """返回只暴露指定 GPU 的环境变量。"""
    # 复制当前环境，保留 conda、PATH 等必要变量。
    env = os.environ.copy()
    # 当前子进程只看到这一张物理 GPU。
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # 返回给 subprocess 使用。
    return env


def use(gpu: int | None) -> None:
    """让当前进程使用指定 GPU。

    `gpu=None` 表示不改环境变量，由系统或已有环境决定可见 GPU。
    """
    # 没有指定 GPU 时不做任何事，方便在 CPU 检查或已有 CUDA 设置下运行。
    if gpu is None:
        return
    # GPU 编号必须从 0 开始，负数没有意义。
    if gpu < 0:
        raise ValueError("GPU 编号不能为负数")
    # CUDA_VISIBLE_DEVICES 是 CUDA 识别可见显卡的标准环境变量。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # 打印一次选择结果，避免长任务跑错卡还不知道。
    print(f"[显卡] 使用 GPU {gpu}", flush=True)
