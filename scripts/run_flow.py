# 让类型注解延迟求值，保持不同 Python 小版本兼容。
from __future__ import annotations

"""运行正式流动数据。

这个文件就是“我要正式流动数据”时运行的入口。
它直接读取 config/base.json 和 tables/analysis/timescales.csv，
现场计算步数和采样间隔，然后运行模拟，不生成任务 CSV。

常用命令：
    python scripts/run_flow.py 0 1
"""

import argparse
import csv
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from gpu import launch_gpu_workers
from run import read_json, run_task, status_path, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 表格。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_timescales(path: Path, expected_radius: float) -> dict[str, dict[str, float]]:
    """读取每个结构最终采用的时间尺度。"""
    table: dict[str, dict[str, float]] = {}
    for row in read_csv(path):
        structure = row["structure"]
        if row.get("pipe_radius"):
            pipe_radius = float(row["pipe_radius"])
            if abs(pipe_radius - expected_radius) > 1e-9:
                raise ValueError(f"{structure} 的时间尺度管半径是 {pipe_radius}，当前配置管半径是 {expected_radius}")
        if not row.get("tau_shape_used"):
            raise ValueError(f"{structure} 缺少 tau_shape_used，请先运行 scripts/timescales.py")
        tau_int_key = "tau_int_r_used" if row.get("tau_int_r_used") else "tau_r_used"
        if not row.get(tau_int_key):
            raise ValueError(f"{structure} 缺少 tau_int_r_used，请先运行 scripts/timescales.py")
        table[structure] = {
            "tau_shape": float(row["tau_shape_used"]),
            "tau_int_r": float(row[tau_int_key]),
        }
    return table


def design_steps(config: dict[str, Any], tau_shape: float, tau_int_r: float) -> tuple[int, int, float]:
    """由时间尺度计算正式凝胶运行步数和采样间隔。"""
    dt = float(config["mpcd"]["dt"])
    rule = config["production"]
    sample_time = float(rule["sample_dt_over_tau_shape"]) * tau_shape
    sample_interval = max(1, int(math.ceil(sample_time / dt)))
    total_time = max(
        float(rule["shape_tau_multiplier"]) * tau_shape,
        float(rule["radial_tau_multiplier"]) * tau_int_r,
        float(rule["min_time"]),
    )
    total_time = min(total_time, float(rule["max_time"]))
    steps = int(math.ceil(total_time / dt))
    return steps, sample_interval, total_time


def iter_cases(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """列出正式流动数据要跑的所有情况。"""
    cases: list[dict[str, Any]] = []
    pipe_radius = float(config["pipe"]["radius"])
    pipe_tag = f"r{pipe_radius:g}"
    tau_table = load_timescales(root / "tables" / "analysis" / "timescales.csv", pipe_radius)

    for flow_strength in config["production"]["fluid_flow_strength"]:
        for seed in config["production"]["fluid_seeds"]:
            fluid_steps = int(config["production"]["fluid_steps"])
            cases.append({
                "run_id": f"{pipe_tag}_fluid_f{flow_strength:g}_s{seed}",
                "kind": "fluid",
                "stage": "production",
                "pipe_radius": pipe_radius,
                "structure": "fluid",
                "flow_strength": flow_strength,
                "seed": seed,
                "steps": fluid_steps,
                "sample_interval": config["output"]["sample_interval"],
                "design_time": fluid_steps * float(config["mpcd"]["dt"]),
            })

    for n in config["structure"]["n_values"]:
        structure = f"g{n}"
        if structure not in tau_table:
            raise ValueError(f"缺少 {structure} 的时间尺度，请先运行 scripts/timescales.py")
        tau = tau_table[structure]
        steps, sample_interval, design_time = design_steps(config, tau["tau_shape"], tau["tau_int_r"])
        for flow_strength in config["production"]["gel_flow_strength"]:
            for seed in config["production"]["gel_seeds"]:
                cases.append({
                    "run_id": f"{pipe_tag}_g{n}_f{flow_strength:g}_s{seed}",
                    "kind": "gel",
                    "stage": "production",
                    "pipe_radius": pipe_radius,
                    "structure": structure,
                    "flow_strength": flow_strength,
                    "seed": seed,
                    "steps": steps,
                    "sample_interval": sample_interval,
                    "design_time": design_time,
                    "tau_shape_used": tau["tau_shape"],
                    "tau_int_r_used": tau["tau_int_r"],
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
    """打印正式流动数据完成情况。"""
    statuses = Counter(
        read_case_status(root / "results" / case["run_id"] / "status.json")
        for case in cases
    )
    print("数据类型: flow")
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
        raise SystemExit(launch_gpu_workers(root, "scripts/run_flow.py", args.gpus, extra_args))

    config = read_json(args.config)
    all_cases = iter_cases(config, root)
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
