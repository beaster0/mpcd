# 让类型注解延迟求值。
from __future__ import annotations

"""用一个简单命令启动指定 GPU 的任务。

用法：
python scripts/gpu.py 0
python scripts/gpu.py 1

默认运行时间尺度任务。正式任务生成后可以加：
python scripts/gpu.py 0 --stage production
"""

# argparse 用来读取命令行里的 GPU 编号和阶段。
import argparse
# os 用来设置 CUDA_VISIBLE_DEVICES。
import os
# subprocess 用来调用 scripts/run.py。
import subprocess
# sys 用来拿到当前 Python 解释器路径。
import sys


def main() -> None:
    """命令行入口。"""
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser()
    # gpu 是必填位置参数，只允许 0 或 1。
    parser.add_argument("gpu", choices=["0", "1"], help="使用哪张 GPU，只允许 0 或 1")
    # stage 表示跑时间尺度任务还是正式任务。
    parser.add_argument("--stage", choices=["timescale", "production"], default="timescale", help="任务阶段")
    # 解析命令行参数。
    args = parser.parse_args()

    # 只让当前进程看到指定 GPU。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 根据阶段选择任务表文件。
    tasks = f"data/tasks_{args.stage}_gpu{args.gpu}.csv"
    # 组装真正要执行的 run.py 命令。
    cmd = [
        sys.executable,
        "scripts/run.py",
        "--config",
        "config/base.json",
        "--tasks",
        tasks,
    ]
    # 打印命令，方便你知道实际运行了什么。
    print("[运行]", " ".join(cmd))
    # 执行 run.py；check=True 表示 run.py 失败时 gpu.py 也失败。
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # 直接运行 python scripts/gpu.py 时执行 main。
    main()
