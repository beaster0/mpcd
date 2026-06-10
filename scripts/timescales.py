# 让类型注解延迟求值，避免 Python 在导入时过早解析复杂类型。
from __future__ import annotations

"""从零流结果估计每个凝胶结构的时间尺度。

这个脚本只做一件事：读取 results/ts_*/timeseries.npz，生成 data/timescales.csv。

为什么不直接手写正式任务步数：
- 不同大小的网格凝胶弛豫速度不同。
- 同一个结构的不同 seed 会有波动。
- 正式任务长度应该由已经测到的时间尺度决定，而不是拍脑袋写死。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件并转成 Python 字典。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 任务表，每一行返回成一个字典。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV 表格。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"没有可写入的行: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def second_half(values: np.ndarray) -> np.ndarray:
    """取后半段数据，减少初始构型瞬态对统计量的影响。"""
    half = len(values) // 2
    return np.asarray(values[half:], dtype=float)


def autocorr_time(values: np.ndarray, sample_dt: float) -> float:
    """计算积分自相关时间。

    自相关时间越长，说明这个量记忆越久，独立样本越少。
    这里积分到自相关函数第一次变成负值为止，避免长尾噪声把时间尺度夸大。
    """
    x = second_half(values)
    x = x - x.mean()
    std = x.std()
    if std < 1e-12:
        return 0.0

    x = x / std
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1 :] / n
    acf = acf / acf[0]
    negative = np.where(acf < 0.0)[0]
    end = int(negative[0]) if len(negative) else min(len(acf) - 1, max(1, n // 4))
    return float(np.trapz(acf[: end + 1], dx=sample_dt))


def diffusion_coefficient(com: np.ndarray, time: np.ndarray) -> float:
    """由质心均方位移估计三维扩散系数 D_cm。

    三维 Einstein 关系是 MSD(t) = 6 D_cm t。
    脚本只用后半段轨迹，避免把初始松弛也算进扩散。
    """
    r = second_half(com)
    t = second_half(time)
    if len(r) < 20:
        return 0.0

    sample_dt = float(np.median(np.diff(t)))
    max_lag = min(len(r) // 4, 200)
    lag_times: list[float] = []
    msd_values: list[float] = []

    for lag in range(1, max_lag):
        displacement = r[lag:] - r[:-lag]
        msd = float(np.mean(np.sum(displacement * displacement, axis=1)))
        lag_times.append(lag * sample_dt)
        msd_values.append(msd)

    if len(lag_times) < 2 or msd_values[-1] < 1e-12:
        return 0.0

    slope = float(np.polyfit(lag_times, msd_values, 1)[0])
    return max(slope / 6.0, 0.0)


def analyze_run(path: Path, radius: float) -> dict[str, float]:
    """分析单条零流轨迹。"""
    data = np.load(path)
    time = np.asarray(data["time"], dtype=float)
    sample_dt = float(np.median(np.diff(time)))

    tau_shape = autocorr_time(np.asarray(data["asphericity"], dtype=float), sample_dt)
    tau_int_r = autocorr_time(np.asarray(data["r_cm"], dtype=float), sample_dt)
    d_cm = diffusion_coefficient(np.asarray(data["com"], dtype=float), time)

    # 径向扩散时间用 R^2 / D_cm 做保守估计。
    # 如果 D_cm 太小，说明当前轨迹不足以稳定估计径向平衡，后面会用 flag 标出来。
    tau_rad = (radius * radius) / d_cm if d_cm > 1e-12 else 0.0
    tau_r = max(tau_int_r, tau_rad)

    return {
        "tau_shape": tau_shape,
        "tau_int_r": tau_int_r,
        "d_cm": d_cm,
        "tau_rad": tau_rad,
        "tau_r": tau_r,
        "samples": float(len(time)),
    }


def robust_used(values: list[float]) -> float:
    """用 median + MAD 得到稳健保守值。

    median 是中位数，MAD 是各 seed 到中位数偏差的中位数。
    它比直接取最大值更不容易被单个异常 seed 支配，同时仍比普通平均值保守。
    """
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    q75 = float(np.quantile(arr, 0.75))
    return max(median + mad, q75)


def variation_flag(values: list[float]) -> str:
    """根据 seed 间离散程度给出简单标记。"""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if mean <= 1e-12:
        return "zero_mean"
    cv = float(arr.std(ddof=0) / mean)
    if cv > 0.8:
        return "unstable"
    if cv > 0.4:
        return "wide"
    return "ok"


def summarize_structure(values: list[dict[str, float]]) -> dict[str, Any]:
    """把同一结构的多个 seed 汇总成正式任务可用的时间尺度。"""
    tau_shape = [row["tau_shape"] for row in values]
    tau_r = [row["tau_r"] for row in values]
    d_cm = [row["d_cm"] for row in values]
    return {
        "tau_shape_used": robust_used(tau_shape),
        "tau_r_used": robust_used(tau_r),
        "tau_shape_median": float(np.median(tau_shape)),
        "tau_r_median": float(np.median(tau_r)),
        "d_cm_median": float(np.median(d_cm)),
        "tau_shape_flag": variation_flag(tau_shape),
        "tau_r_flag": variation_flag(tau_r),
        "n_seeds": len(values),
    }


def empty_timescale_row(structure: str, pipe_radius: float, reason: str) -> dict[str, Any]:
    """生成还不能用于正式任务的占位行。

    改变管半径后，旧结果目录会被隔离，新的零流结果还不存在。
    此时写出占位行比继续沿用旧时间尺度更安全。
    """
    return {
        "structure": structure,
        "pipe_radius": f"{pipe_radius:.6f}",
        "tau_shape_used": "",
        "tau_r_used": "",
        "tau_shape_median": "",
        "tau_r_median": "",
        "d_cm_median": "",
        "tau_shape_flag": reason,
        "tau_r_flag": reason,
        "n_seeds": 0,
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/base.json"))
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks_timescale.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/timescales.csv"))
    args = parser.parse_args()

    root = Path.cwd()
    config = read_json(root / args.config)
    radius = float(config["pipe"]["radius"])
    tasks = read_csv(root / args.tasks)

    by_structure: dict[str, list[dict[str, float]]] = {}
    missing_by_structure: dict[str, int] = {}
    for task in tasks:
        run_id = task["run_id"]
        structure = task["structure"]
        series_path = root / "results" / run_id / "timeseries.npz"
        if not series_path.exists():
            missing_by_structure[structure] = missing_by_structure.get(structure, 0) + 1
            continue

        metrics = analyze_run(series_path, radius)
        by_structure.setdefault(structure, []).append(metrics)
        print(
            f"{run_id}: "
            f"tau_shape={metrics['tau_shape']:.2f}, "
            f"tau_int_r={metrics['tau_int_r']:.2f}, "
            f"D_cm={metrics['d_cm']:.3e}, "
            f"tau_r={metrics['tau_r']:.2f}"
        )

    rows: list[dict[str, Any]] = []
    all_structures = sorted({task["structure"] for task in tasks})
    for structure in all_structures:
        if structure not in by_structure:
            rows.append(empty_timescale_row(structure, radius, "missing_results"))
            print(f"{structure}: 缺少新半径结果，写入 missing_results 占位行")
            continue

        summary = summarize_structure(by_structure[structure])
        row = {
            "structure": structure,
            "pipe_radius": f"{radius:.6f}",
            "tau_shape_used": f"{summary['tau_shape_used']:.6f}",
            "tau_r_used": f"{summary['tau_r_used']:.6f}",
            "tau_shape_median": f"{summary['tau_shape_median']:.6f}",
            "tau_r_median": f"{summary['tau_r_median']:.6f}",
            "d_cm_median": f"{summary['d_cm_median']:.8e}",
            "tau_shape_flag": summary["tau_shape_flag"],
            "tau_r_flag": summary["tau_r_flag"],
            "n_seeds": summary["n_seeds"],
        }
        rows.append(row)
        print(
            f"{structure}: "
            f"tau_shape_used={summary['tau_shape_used']:.2f}, "
            f"tau_r_used={summary['tau_r_used']:.2f}, "
            f"shape_flag={summary['tau_shape_flag']}, "
            f"r_flag={summary['tau_r_flag']}"
        )

    write_csv(root / args.output, rows)
    print(f"写入完成: {args.output}")


if __name__ == "__main__":
    main()
