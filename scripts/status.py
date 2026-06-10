# 让类型注解延迟求值，保持脚本在不同 Python 小版本下更稳。
from __future__ import annotations

"""汇总任务运行状态。

这个脚本只读任务表和 results 目录，不启动模拟、不修改结果。

常用命令：
- 看零流时间尺度任务：python scripts/status.py
- 看正式任务：python scripts/status.py --tasks data/tasks_production.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_tasks(path: Path) -> list[dict[str, str]]:
    """读取任务表。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_status(path: Path) -> str:
    """读取单条任务状态。

    status.json 不存在时，说明这条任务还没有开始。
    status.json 格式坏掉时，说明任务可能在写文件时中断。
    """
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "broken_status_json"
    return str(data.get("status", "unknown"))


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks_timescale.csv"))
    args = parser.parse_args()

    root = Path.cwd()
    tasks = read_tasks(root / args.tasks)

    statuses = Counter(
        read_status(root / "results" / task["run_id"] / "status.json")
        for task in tasks
    )

    print("任务表:", args.tasks)
    print("任务总数:", len(tasks))
    for name in sorted(statuses):
        print(f"{name}: {statuses[name]}")

    running = [
        task["run_id"]
        for task in tasks
        if read_status(root / "results" / task["run_id"] / "status.json") == "running"
    ]
    if running:
        print("正在运行:")
        for run_id in running:
            print(run_id)


if __name__ == "__main__":
    main()
