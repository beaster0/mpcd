# 让类型注解延迟求值，避免运行时解析复杂类型。
from __future__ import annotations

"""根据零流时间尺度生成正式任务表。

输入：
- config/base.json：基础物理参数和正式任务规则。
- data/timescales.csv：每个结构的 tau_shape_used 和 tau_r_used。

输出：
- data/tasks_production.csv
- data/tasks_production_gpu0.csv
- data/tasks_production_gpu1.csv
"""

# csv 用来读取时间尺度表、写出正式任务表。
import csv
# json 用来读取基础配置。
import json
# math 用来向上取整。
import math
# Path 用来处理路径。
from pathlib import Path
# Any 表示任意类型，用于配置字典。
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 配置。"""
    # 把 JSON 文本转成 Python 字典。
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 表格。"""
    # utf-8-sig 兼容 Excel 和 Python 写出的 CSV。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # DictReader 把每行转成字典。
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV 表格。"""
    # 确保输出目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 没有行就不能确定列名，直接报错。
    if not rows:
        raise ValueError(f"没有可写入的行: {path}")
    # newline="" 避免 Windows 多空行；utf-8-sig 方便 Excel 打开。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        # 使用第一行的键作为列名。
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        # 写表头。
        writer.writeheader()
        # 写数据。
        writer.writerows(rows)


def split_gpu_tasks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把任务轮流分给 GPU0 和 GPU1。"""
    # 两张卡各一个任务列表。
    gpu0: list[dict[str, Any]] = []
    gpu1: list[dict[str, Any]] = []
    # 偶数行给 GPU0，奇数行给 GPU1。
    for index, row in enumerate(rows):
        (gpu0 if index % 2 == 0 else gpu1).append(row)
    # 返回两个分片。
    return gpu0, gpu1


def load_timescales(path: Path, expected_radius: float) -> dict[str, dict[str, float]]:
    """读取每个结构最终采用的时间尺度。"""
    # table 用 structure 作为键，例如 g1/g2/g3/g4。
    table: dict[str, dict[str, float]] = {}
    # 逐行读取 timescales.csv。
    for row in read_csv(path):
        # 结构名必须存在。
        structure = row["structure"]
        if row.get("pipe_radius"):
            pipe_radius = float(row["pipe_radius"])
            if abs(pipe_radius - expected_radius) > 1e-9:
                raise ValueError(f"{structure} 的时间尺度管半径是 {pipe_radius}，当前配置管半径是 {expected_radius}")
        if not row.get("tau_shape_used") or not row.get("tau_r_used"):
            raise ValueError(f"{structure} 缺少可用时间尺度，请先运行当前管半径的零流时间尺度任务")
        # tau_shape_used 是保守汇总后的形状弛豫时间。
        tau_shape = float(row["tau_shape_used"])
        # tau_r_used 是保守汇总后的径向相关时间。
        tau_r = float(row["tau_r_used"])
        # 保存到表里。
        table[structure] = {"tau_shape": tau_shape, "tau_r": tau_r}
    # 返回结构到时间尺度的映射。
    return table


def design_steps(config: dict[str, Any], tau_shape: float, tau_r: float) -> tuple[int, int, float]:
    """由时间尺度计算采样间隔和正式运行步数。"""
    # 基础时间步。
    dt = float(config["mpcd"]["dt"])
    # 正式任务规则。
    rule = config["production"]
    # 采样时间间隔 = 0.1 * tau_shape 这类相对时间。
    sample_time = float(rule["sample_dt_over_tau_shape"]) * tau_shape
    # sample_interval 是采样间隔对应的步数，至少为 1。
    sample_interval = max(1, int(math.ceil(sample_time / dt)))
    # 总时间由形状弛豫、径向弛豫和最低时间共同决定。
    total_time = max(
        float(rule["shape_tau_multiplier"]) * tau_shape,
        float(rule["radial_tau_multiplier"]) * tau_r,
        float(rule["min_time"]),
    )
    # max_time 防止某个时间尺度异常大时直接生成不可承受任务。
    total_time = min(total_time, float(rule["max_time"]))
    # 把总物理时间换算成步数。
    steps = int(math.ceil(total_time / dt))
    # 返回步数、采样间隔和采用的总物理时间。
    return steps, sample_interval, total_time


def production_rows(config: dict[str, Any], timescales: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """生成正式任务表。"""
    # rows 保存最终任务。
    rows: list[dict[str, Any]] = []
    # 管半径写进 run_id，避免不同管径的正式结果互相覆盖。
    pipe_radius = float(config["pipe"]["radius"])
    pipe_tag = f"r{pipe_radius:g}"
    # 纯流体任务仍使用固定步数，因为它用于流场标定，不依赖凝胶 tau。
    for wi in config["production"]["fluid_wi"]:
        for seed in config["production"]["fluid_seeds"]:
            rows.append({
                "run_id": f"{pipe_tag}_fluid_w{wi:g}_s{seed}",
                "kind": "fluid",
                "stage": "production",
                "pipe_radius": pipe_radius,
                "structure": "fluid",
                "wi": wi,
                "seed": seed,
                "steps": config["production"]["fluid_steps"],
                "sample_interval": config["output"]["sample_interval"],
                "design_time": config["production"]["fluid_steps"] * float(config["mpcd"]["dt"]),
                "tau_shape_used": "",
                "tau_r_used": "",
            })

    # 凝胶正式任务使用各结构自己的 tau_shape/tau_r 设计步数。
    for n in config["structure"]["n_values"]:
        # 结构名，例如 g3。
        structure = f"g{n}"
        # 没有时间尺度就不能生成正式任务。
        if structure not in timescales:
            raise ValueError(f"缺少 {structure} 的时间尺度，请先写入 data/timescales.csv")
        # 读取该结构时间尺度。
        tau = timescales[structure]
        # 算出正式步数和采样间隔。
        steps, sample_interval, design_time = design_steps(config, tau["tau_shape"], tau["tau_r"])
        # 结构 × Wi × seed 生成正式凝胶任务。
        for wi in config["production"]["gel_wi"]:
            for seed in config["production"]["gel_seeds"]:
                rows.append({
                    "run_id": f"{pipe_tag}_g{n}_w{wi:g}_s{seed}",
                    "kind": "gel",
                    "stage": "production",
                    "pipe_radius": pipe_radius,
                    "structure": structure,
                    "wi": wi,
                    "seed": seed,
                    "steps": steps,
                    "sample_interval": sample_interval,
                    "design_time": design_time,
                    "tau_shape_used": tau["tau_shape"],
                    "tau_r_used": tau["tau_r"],
                })
    # 返回正式任务表。
    return rows


def main() -> None:
    """命令行入口。"""
    # 当前目录应是项目根目录。
    root = Path.cwd()
    # 读取基础配置。
    config = read_json(root / "config" / "base.json")
    # 读取时间尺度表。
    timescales = load_timescales(root / "data" / "timescales.csv", float(config["pipe"]["radius"]))
    # 生成正式任务。
    rows = production_rows(config, timescales)
    # 拆成两张卡。
    gpu0, gpu1 = split_gpu_tasks(rows)
    # 写出正式任务表。
    write_csv(root / "data" / "tasks_production.csv", rows)
    # 写出 GPU0 分片。
    write_csv(root / "data" / "tasks_production_gpu0.csv", gpu0)
    # 写出 GPU1 分片。
    write_csv(root / "data" / "tasks_production_gpu1.csv", gpu1)
    # 打印摘要。
    print(f"生成完成: {len(rows)} 条正式任务")


if __name__ == "__main__":
    # 直接运行 python scripts/plan.py 时执行 main。
    main()
