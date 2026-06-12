# 让类型注解延迟求值，保持不同 Python 小版本兼容。
from __future__ import annotations

"""运行零流时间尺度数据。

这个文件就是“我要零流时间尺度数据”时运行的入口。
它直接按 config/base.json 里的结构、seed 和步数循环，不生成任务 CSV。

常用命令：
    python scripts/run_timescale.py 0 1
"""

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from gpu import launch_gpu_workers
from run import read_json, run_task, status_path, write_json


def iter_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """列出零流时间尺度数据要跑的所有情况。"""
    cases: list[dict[str, Any]] = []
    pipe_radius = float(config["pipe"]["radius"])
    pipe_tag = f"r{pipe_radius:g}"
    for n in config["structure"]["n_values"]:
        for flow_strength in config["timescale"]["flow_strength"]:
            for seed in config["timescale"]["seeds"]:
                cases.append({
                    "run_id": f"ts_{pipe_tag}_g{n}_s{seed}",
                    "kind": "gel",
                    "stage": "timescale",
                    "pipe_radius": pipe_radius,
                    "structure": f"g{n}",
                    "flow_strength": flow_strength,
                    "seed": seed,
                    "steps": config["timescale"]["steps"],
                })
    return cases


def select_shard(cases: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    """选择当前 GPU 负责的那部分情况。"""
    if shard_count <= 0:
        raise ValueError("GPU 数量必须大于 0")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("分片编号必须满足 0 <= shard_index < shard_count")
    return [case for index, case in enumerate(cases) if index % shard_count == shard_index]


def run_cases(root: Path, config: dict[str, Any], cases: list[dict[str, Any]], steps: int | None) -> None:
    """顺序运行当前 GPU 分到的情况。"""
    failed = 0
    for case in cases:
        try:
            print(f"[开始] {case['run_id']}", flush=True)
            result = run_task(root, config, case, steps)
            print(f"[完成] {case['run_id']} solvent={result['solvent_count']} tps={result['tps']}", flush=True)
        except Exception as exc:
            failed += 1
            payload = {
                "run_id": case.get("run_id"),
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(status_path(root, case["run_id"]), payload)
            print(f"[失败] {case['run_id']} {type(exc).__name__}: {exc}", flush=True)
    if failed:
        raise SystemExit(f"有 {failed} 条运行失败")


def read_case_status(path: Path) -> str:
    """读取单个结果目录的状态。"""
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "broken_status_json"
    return str(data.get("status", "unknown"))


def print_status(root: Path, cases: list[dict[str, Any]]) -> None:
    """打印零流时间尺度数据完成情况。"""
    statuses = Counter(
        read_case_status(root / "results" / case["run_id"] / "status.json")
        for case in cases
    )
    print("数据类型: timescale")
    print("数据总数:", len(cases))
    for name in sorted(statuses):
        print(f"{name}: {statuses[name]}")
    running = [
        case["run_id"]
        for case in cases
        if read_case_status(root / "results" / case["run_id"] / "status.json") == "running"
    ]
    if running:
        print("正在运行:")
        for run_id in running:
            print(run_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gpus", nargs="*", type=int, help="要使用的 GPU 编号，例如 0 1")
    parser.add_argument("--config", type=Path, default=Path("config/base.json"))
    parser.add_argument("--only", default=None, help="只运行指定 run_id")
    parser.add_argument("--max", type=int, default=None, help="最多运行多少条")
    parser.add_argument("--steps", type=int, default=None, help="临时覆盖步数")
    parser.add_argument("--status", action="store_true", help="只查看完成状态，不启动模拟")
    parser.add_argument("--worker", action="store_true", help="内部参数：单 GPU 子进程")
    parser.add_argument("--shard-index", type=int, default=0, help="当前 GPU 分片编号")
    parser.add_argument("--shard-count", type=int, default=1, help="GPU 分片总数")
    args = parser.parse_args()

    root = Path.cwd()
    if args.gpus and not args.worker:
        extra_args: list[str] = ["--config", str(args.config)]
        if args.only:
            extra_args += ["--only", args.only]
        if args.max is not None:
            extra_args += ["--max", str(args.max)]
        if args.steps is not None:
            extra_args += ["--steps", str(args.steps)]
        raise SystemExit(launch_gpu_workers(root, "scripts/run_timescale.py", args.gpus, extra_args))

    config = read_json(args.config)
    all_cases = iter_cases(config)
    if args.status:
        print_status(root, all_cases)
        return
    cases = select_shard(all_cases, args.shard_index, args.shard_count)
    if args.only:
        cases = [case for case in cases if case["run_id"] == args.only]
    if args.max is not None:
        cases = cases[: args.max]
    if not cases:
        print("当前 GPU 分片没有要运行的数据。", flush=True)
        return
    run_cases(root, config, cases, args.steps)


if __name__ == "__main__":
    main()
