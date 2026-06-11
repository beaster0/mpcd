from __future__ import annotations

"""从零流结果估计每个凝胶结构的时间尺度，并生成论文图。

默认用法：
    python scripts/analyze_timescales.py

默认输入：
    config/base.json
    data/tasks_timescale.csv

如果 data/tasks_timescale.csv 不存在，则自动读取：
    data/tasks_timescale_gpu*.csv

默认输出：
    data/timescales.csv
    data/timescale_runs.csv
    figures/timescales/

时间尺度定义：
    tau_shape_used:
        asphericity 自相关时间，用于形状弛豫判断。

    tau_int_r_used:
        r_cm 自相关时间，用于径向记忆时间判断。

    tau_r_used:
        保留兼容字段，当前等同于 tau_int_r_used。

    tau_rad_diagnostic:
        R^2 / D_perp，只作为横向扩散诊断，不直接参与正式步数设计。

默认假设：
    管道轴向为 z，因此横向平面为 x-y。
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MIN_T_OVER_TAU = 3.0
MARGINAL_T_OVER_TAU = 5.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"没有可写入的行: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_default_tasks(root: Path) -> list[Path]:
    """自动寻找 timescale 任务表，避免运行时写一堆参数。"""
    single = root / "data" / "tasks_timescale.csv"
    if single.exists():
        return [single]

    gpu_tasks = sorted((root / "data").glob("tasks_timescale_gpu*.csv"))
    if gpu_tasks:
        return gpu_tasks

    raise FileNotFoundError(
        "找不到 timescale 任务表。需要存在 data/tasks_timescale.csv "
        "或 data/tasks_timescale_gpu*.csv"
    )


def read_tasks(paths: list[Path]) -> list[dict[str, str]]:
    """读取一个或多个任务表，并按 run_id 去重。"""
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()

    for path in paths:
        for row in read_csv(path):
            run_id = row["run_id"]
            if run_id in seen:
                continue
            seen.add(run_id)
            tasks.append(row)

    if not tasks:
        raise ValueError("任务表为空")

    return tasks


def second_half(values: np.ndarray) -> np.ndarray:
    half = len(values) // 2
    return np.asarray(values[half:], dtype=float)


def analyzed_time(time: np.ndarray) -> float:
    """后半段轨迹的物理时间长度。"""
    t = second_half(time)
    if len(t) < 2:
        return 0.0
    return float(t[-1] - t[0])


def autocorr_curve(values: np.ndarray) -> np.ndarray:
    """用 FFT 计算归一化自相关曲线。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 4:
        return np.asarray([1.0], dtype=float)

    x = x - x.mean()
    std = x.std()
    if std < 1e-12:
        return np.ones(len(x), dtype=float)

    x = x / std
    n = len(x)

    fft_size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=fft_size)
    acf = np.fft.irfft(f * np.conjugate(f), n=fft_size)[:n]

    norm = np.arange(n, 0, -1, dtype=float)
    acf = acf / norm

    if abs(acf[0]) < 1e-14:
        return np.zeros(n, dtype=float)

    acf = acf / acf[0]
    acf[0] = 1.0
    return acf


def autocorr_time(values: np.ndarray, sample_dt: float) -> tuple[float, np.ndarray, np.ndarray, float]:
    """计算积分自相关时间，并返回曲线用于画图。

    积分到自相关函数第一次变负为止。
    如果没有变负，则最多积分到后半段轨迹长度的 1/4。
    """
    x = second_half(values)
    acf = autocorr_curve(x)
    lag_time = np.arange(len(acf), dtype=float) * sample_dt

    if len(acf) <= 1:
        return 0.0, lag_time, acf, 0.0

    negative = np.where(acf[1:] < 0.0)[0] + 1
    if len(negative):
        end = int(negative[0])
    else:
        end = min(len(acf) - 1, max(1, len(acf) // 4))

    tau = float(np.trapz(acf[: end + 1], dx=sample_dt))
    tau = max(tau, 0.0)
    cutoff_time = float(lag_time[end])

    return tau, lag_time, acf, cutoff_time


def select_coordinates(com: np.ndarray, dims: int, pipe_axis: str) -> np.ndarray:
    """选择三维或横向二维坐标。

    默认 pipe_axis='z'，因此横向平面为 x-y。
    """
    if dims == 3:
        return com

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_to_index[pipe_axis]
    keep = [i for i in range(3) if i != axis_index]
    return com[:, keep]


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 2:
        return 0.0, 0.0, 0.0

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0

    return float(slope), float(intercept), float(r2)


def msd_slope(
    com: np.ndarray,
    time: np.ndarray,
    dims: int,
    pipe_axis: str = "z",
    max_lag: int = 200,
) -> dict[str, Any]:
    """由质心 MSD 线性斜率估计扩散系数。

    dims=3:
        MSD = 6 D_cm t

    dims=2:
        MSD_perp = 4 D_perp t
    """
    r = second_half(com)
    t = second_half(time)

    if len(r) < 20:
        return empty_msd_result()

    r = select_coordinates(r, dims=dims, pipe_axis=pipe_axis)

    sample_dt = float(np.median(np.diff(t)))
    max_lag = min(len(r) // 4, max_lag)

    lag_times: list[float] = []
    msd_values: list[float] = []

    for lag in range(1, max_lag + 1):
        displacement = r[lag:] - r[:-lag]
        msd = float(np.mean(np.sum(displacement * displacement, axis=1)))
        lag_times.append(lag * sample_dt)
        msd_values.append(msd)

    lag_time = np.asarray(lag_times, dtype=float)
    msd = np.asarray(msd_values, dtype=float)

    if len(lag_time) < 3 or msd[-1] < 1e-12:
        return empty_msd_result(lag_time, msd)

    n = len(lag_time)
    i0 = max(0, int(0.2 * n))
    i1 = min(n, max(i0 + 2, int(0.8 * n)))

    slope, intercept, fit_r2 = linear_fit(lag_time[i0:i1], msd[i0:i1])
    factor = 4.0 if dims == 2 else 6.0
    diffusivity = max(slope / factor, 0.0)

    return {
        "diffusivity": diffusivity,
        "lag_time": lag_time,
        "msd": msd,
        "fit_slope": slope,
        "fit_intercept": intercept,
        "fit_r2": fit_r2,
        "fit_start_time": float(lag_time[i0]),
        "fit_end_time": float(lag_time[i1 - 1]),
    }


def empty_msd_result(
    lag_time: np.ndarray | None = None,
    msd: np.ndarray | None = None,
) -> dict[str, Any]:
    if lag_time is None:
        lag_time = np.asarray([], dtype=float)
    if msd is None:
        msd = np.asarray([], dtype=float)

    return {
        "diffusivity": 0.0,
        "lag_time": lag_time,
        "msd": msd,
        "fit_slope": 0.0,
        "fit_intercept": 0.0,
        "fit_r2": 0.0,
        "fit_start_time": 0.0,
        "fit_end_time": 0.0,
    }


def analyze_run(path: Path, radius: float, pipe_axis: str = "z") -> dict[str, Any]:
    """分析单条零流轨迹。"""
    data = np.load(path)

    time = np.asarray(data["time"], dtype=float)
    asphericity = np.asarray(data["asphericity"], dtype=float)
    r_cm = np.asarray(data["r_cm"], dtype=float)
    com = np.asarray(data["com"], dtype=float)

    sample_dt = float(np.median(np.diff(time)))

    tau_shape, shape_lag, shape_acf, shape_cutoff = autocorr_time(asphericity, sample_dt)
    tau_int_r, r_lag, r_acf, r_cutoff = autocorr_time(r_cm, sample_dt)

    msd_cm = msd_slope(com, time, dims=3, pipe_axis=pipe_axis)
    msd_perp = msd_slope(com, time, dims=2, pipe_axis=pipe_axis)

    d_cm = float(msd_cm["diffusivity"])
    d_perp = float(msd_perp["diffusivity"])

    tau_rad = (radius * radius) / d_perp if d_perp > 1e-12 else 0.0
    t_analyzed = analyzed_time(time)

    return {
        "tau_shape": tau_shape,
        "tau_shape_cutoff_time": shape_cutoff,
        "tau_int_r": tau_int_r,
        "tau_int_r_cutoff_time": r_cutoff,
        "d_cm": d_cm,
        "d_perp": d_perp,
        "tau_rad_diagnostic": tau_rad,
        "t_analyzed": t_analyzed,
        "samples": float(len(time)),
        "sample_dt": sample_dt,
        "shape_lag": shape_lag,
        "shape_acf": shape_acf,
        "r_lag": r_lag,
        "r_acf": r_acf,
        "msd_cm": msd_cm,
        "msd_perp": msd_perp,
    }


def robust_used(values: list[float]) -> float:
    """用 median + MAD 得到稳健保守值。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return 0.0

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    q75 = float(np.quantile(arr, 0.75))

    return max(median + mad, q75)


def variation_flag(values: list[float]) -> str:
    """根据 seed 间离散程度给出标记。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return "missing"
    if len(arr) == 1:
        return "single_seed"

    mean = float(arr.mean())
    if abs(mean) <= 1e-12:
        return "zero_mean"

    cv = float(arr.std(ddof=0) / abs(mean))
    if cv > 0.8:
        return "unstable"
    if cv > 0.4:
        return "wide"
    return "ok"


def coverage_flag(t_analyzed: float, tau: float) -> str:
    if tau <= 1e-12:
        return "zero_tau"

    ratio = t_analyzed / tau
    if ratio < MIN_T_OVER_TAU:
        return "short"
    if ratio < MARGINAL_T_OVER_TAU:
        return "marginal"
    return "ok"


def combine_flags(*flags: str) -> str:
    bad: list[str] = []
    for flag in flags:
        if not flag or flag == "ok":
            continue
        if flag not in bad:
            bad.append(flag)
    return "+".join(bad) if bad else "ok"


def median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.median(arr))


def fmt6(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def fmte(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.8e}"


def summarize_structure(values: list[dict[str, Any]], expected_seeds: int) -> dict[str, Any]:
    """把同一结构的多个 seed 汇总成正式任务可用的时间尺度。"""
    tau_shape = [row["tau_shape"] for row in values]
    tau_int_r = [row["tau_int_r"] for row in values]
    tau_rad = [row["tau_rad_diagnostic"] for row in values]
    d_cm = [row["d_cm"] for row in values]
    d_perp = [row["d_perp"] for row in values]
    t_analyzed = median([row["t_analyzed"] for row in values])

    tau_shape_used = robust_used(tau_shape)
    tau_int_r_used = robust_used(tau_int_r)
    tau_rad_diag = robust_used(tau_rad)

    shape_var = variation_flag(tau_shape)
    int_r_var = variation_flag(tau_int_r)

    shape_cov = coverage_flag(t_analyzed, tau_shape_used)
    int_r_cov = coverage_flag(t_analyzed, tau_int_r_used)

    n_seeds = len(values)
    n_missing = max(expected_seeds - n_seeds, 0)
    seed_flag = "incomplete_seeds" if n_missing > 0 else "ok"

    tau_shape_flag = combine_flags(shape_var, shape_cov, seed_flag)
    tau_int_r_flag = combine_flags(int_r_var, int_r_cov, seed_flag)

    if tau_rad_diag <= 1e-12:
        rad_note = "zero_d_perp"
    elif tau_rad_diag > max(tau_int_r_used, 1.0) * 10.0:
        rad_note = "extrapolated_rad"
    else:
        rad_note = "ok"

    return {
        "tau_shape_used": tau_shape_used,
        "tau_int_r_used": tau_int_r_used,
        "tau_r_used": tau_int_r_used,
        "tau_rad_diagnostic": tau_rad_diag,
        "tau_shape_median": median(tau_shape),
        "tau_int_r_median": median(tau_int_r),
        "tau_r_median": median(tau_int_r),
        "d_cm_median": median(d_cm),
        "d_perp_median": median(d_perp),
        "msd_cm_fit_r2_median": median([row["msd_cm"]["fit_r2"] for row in values]),
        "msd_perp_fit_r2_median": median([row["msd_perp"]["fit_r2"] for row in values]),
        "t_analyzed": t_analyzed,
        "t_over_tau_shape": t_analyzed / tau_shape_used if tau_shape_used > 0 else 0.0,
        "t_over_tau_int_r": t_analyzed / tau_int_r_used if tau_int_r_used > 0 else 0.0,
        "tau_shape_flag": tau_shape_flag,
        "tau_int_r_flag": tau_int_r_flag,
        "tau_r_flag": tau_int_r_flag,
        "tau_rad_note": rad_note,
        "n_seeds": n_seeds,
        "n_expected": expected_seeds,
        "n_missing": n_missing,
    }


def empty_timescale_row(structure: str, pipe_radius: float, reason: str, expected: int) -> dict[str, Any]:
    return {
        "structure": structure,
        "pipe_radius": f"{pipe_radius:.6f}",
        "tau_shape_used": "",
        "tau_int_r_used": "",
        "tau_r_used": "",
        "tau_rad_diagnostic": "",
        "tau_shape_median": "",
        "tau_int_r_median": "",
        "tau_r_median": "",
        "d_cm_median": "",
        "d_perp_median": "",
        "msd_cm_fit_r2_median": "",
        "msd_perp_fit_r2_median": "",
        "t_analyzed": "",
        "t_over_tau_shape": "",
        "t_over_tau_int_r": "",
        "tau_shape_flag": reason,
        "tau_int_r_flag": reason,
        "tau_r_flag": reason,
        "tau_rad_note": reason,
        "n_seeds": 0,
        "n_expected": expected,
        "n_missing": expected,
    }


def make_run_row(task: dict[str, str], metrics: dict[str, Any], pipe_radius: float) -> dict[str, Any]:
    return {
        "run_id": task["run_id"],
        "structure": task["structure"],
        "seed": task.get("seed", ""),
        "pipe_radius": fmt6(pipe_radius),
        "samples": int(metrics["samples"]),
        "sample_dt": fmt6(metrics["sample_dt"]),
        "t_analyzed": fmt6(metrics["t_analyzed"]),
        "tau_shape": fmt6(metrics["tau_shape"]),
        "tau_shape_cutoff_time": fmt6(metrics["tau_shape_cutoff_time"]),
        "tau_int_r": fmt6(metrics["tau_int_r"]),
        "tau_int_r_cutoff_time": fmt6(metrics["tau_int_r_cutoff_time"]),
        "d_cm": fmte(metrics["d_cm"]),
        "d_perp": fmte(metrics["d_perp"]),
        "tau_rad_diagnostic": fmt6(metrics["tau_rad_diagnostic"]),
        "msd_cm_fit_r2": fmt6(metrics["msd_cm"]["fit_r2"]),
        "msd_perp_fit_r2": fmt6(metrics["msd_perp"]["fit_r2"]),
        "msd_fit_start_time": fmt6(metrics["msd_perp"]["fit_start_time"]),
        "msd_fit_end_time": fmt6(metrics["msd_perp"]["fit_end_time"]),
    }


def setup_plot_style() -> None:
    """论文图风格。"""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def save_fig(fig: plt.Figure, figdir: Path, name: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def plot_summary(rows: list[dict[str, Any]], by_structure: dict[str, list[dict[str, Any]]], figdir: Path) -> None:
    valid = [row for row in rows if int(row["n_seeds"]) > 0]
    if not valid:
        return

    structures = [row["structure"] for row in valid]
    x = np.arange(len(structures), dtype=float)

    tau_shape = np.asarray([float(row["tau_shape_used"]) for row in valid])
    tau_int_r = np.asarray([float(row["tau_int_r_used"]) for row in valid])
    t_over_shape = np.asarray([float(row["t_over_tau_shape"]) for row in valid])
    t_over_int_r = np.asarray([float(row["t_over_tau_int_r"]) for row in valid])
    d_cm = np.asarray([float(row["d_cm_median"]) for row in valid])
    d_perp = np.asarray([float(row["d_perp_median"]) for row in valid])
    tau_rad = np.asarray([float(row["tau_rad_diagnostic"]) for row in valid])
    ratio_rad = tau_rad / np.maximum(tau_int_r, 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    ax = axes[0, 0]
    panel_label(ax, "A")
    for i, structure in enumerate(structures):
        for item in by_structure[structure]:
            ax.scatter(i - 0.07, item["tau_shape"], s=14, alpha=0.35, linewidths=0)
            ax.scatter(i + 0.07, item["tau_int_r"], s=14, alpha=0.35, linewidths=0)
    ax.plot(x - 0.07, tau_shape, marker="o", linewidth=1.2, label=r"$\tau_{\mathrm{shape}}$")
    ax.plot(x + 0.07, tau_int_r, marker="s", linewidth=1.2, label=r"$\tau_{r,\mathrm{int}}$")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel("Timescale")
    ax.set_title("Robust timescale estimates")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.plot(x - 0.07, t_over_shape, marker="o", linewidth=1.2, label=r"$T/\tau_{\mathrm{shape}}$")
    ax.plot(x + 0.07, t_over_int_r, marker="s", linewidth=1.2, label=r"$T/\tau_{r,\mathrm{int}}$")
    ax.axhline(MIN_T_OVER_TAU, linestyle="--", linewidth=0.8)
    ax.axhline(MARGINAL_T_OVER_TAU, linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel("Coverage ratio")
    ax.set_title("Analyzed trajectory coverage")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.plot(x - 0.07, d_cm, marker="o", linewidth=1.2, label=r"$D_{\mathrm{cm}}$")
    ax.plot(x + 0.07, d_perp, marker="s", linewidth=1.2, label=r"$D_{\perp}$")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel("Diffusivity")
    ax.set_title("Center-of-mass diffusion")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.plot(x, ratio_rad, marker="o", linewidth=1.2)
    ax.axhline(10.0, linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel(r"$\tau_{\mathrm{rad,diag}}/\tau_{r,\mathrm{int}}$")
    ax.set_title("Radial diffusion diagnostic")

    for ax in axes.ravel():
        ax.tick_params(which="both", top=True, right=True)

    fig.tight_layout()
    save_fig(fig, figdir, "timescale_summary")


def median_curve(items: list[dict[str, Any]], lag_key: str, value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    max_common = None

    for item in items:
        lag = np.asarray(item[lag_key], dtype=float)
        value = np.asarray(item[value_key], dtype=float)

        if len(lag) < 2:
            continue

        if max_common is None:
            max_common = float(lag[-1])
        else:
            max_common = min(max_common, float(lag[-1]))

    if max_common is None or max_common <= 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    grid = np.linspace(0.0, max_common, 400)

    for item in items:
        lag = np.asarray(item[lag_key], dtype=float)
        value = np.asarray(item[value_key], dtype=float)
        if len(lag) < 2:
            continue
        curves.append(np.interp(grid, lag, value))

    if not curves:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    arr = np.vstack(curves)
    q25 = np.quantile(arr, 0.25, axis=0)
    q50 = np.quantile(arr, 0.50, axis=0)
    q75 = np.quantile(arr, 0.75, axis=0)

    return grid, q25, q50, q75


def median_msd_curve(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    max_common = None

    for item in items:
        lag = np.asarray(item["msd_perp"]["lag_time"], dtype=float)
        if len(lag) < 2:
            continue
        max_common = float(lag[-1]) if max_common is None else min(max_common, float(lag[-1]))

    if max_common is None or max_common <= 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    grid = np.linspace(0.0, max_common, 400)

    for item in items:
        lag = np.asarray(item["msd_perp"]["lag_time"], dtype=float)
        y = np.asarray(item["msd_perp"]["msd"], dtype=float)
        if len(lag) < 2:
            continue
        curves.append(np.interp(grid, lag, y))

    if not curves:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    arr = np.vstack(curves)
    q25 = np.quantile(arr, 0.25, axis=0)
    q50 = np.quantile(arr, 0.50, axis=0)
    q75 = np.quantile(arr, 0.75, axis=0)
    return grid, q25, q50, q75


def plot_acf_grid(
    rows: list[dict[str, Any]],
    by_structure: dict[str, list[dict[str, Any]]],
    figdir: Path,
    kind: str,
) -> None:
    valid = [row for row in rows if int(row["n_seeds"]) > 0]
    if not valid:
        return

    if kind == "shape":
        lag_key = "shape_lag"
        acf_key = "shape_acf"
        tau_key = "tau_shape_used"
        ylabel = r"$C_{\mathrm{asphericity}}(\Delta t)$"
        title = "Asphericity autocorrelation"
        filename = "acf_asphericity_by_structure"
    else:
        lag_key = "r_lag"
        acf_key = "r_acf"
        tau_key = "tau_int_r_used"
        ylabel = r"$C_{r_{\mathrm{cm}}}(\Delta t)$"
        title = r"$r_{\mathrm{cm}}$ autocorrelation"
        filename = "acf_r_cm_by_structure"

    n = len(valid)
    ncols = 2 if n <= 4 else 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.6 * nrows), squeeze=False)

    for i, row in enumerate(valid):
        ax = axes[i // ncols][i % ncols]
        structure = row["structure"]
        items = by_structure[structure]

        for item in items:
            ax.plot(item[lag_key], item[acf_key], linewidth=0.6, alpha=0.25)

        grid, q25, q50, q75 = median_curve(items, lag_key, acf_key)
        if len(grid):
            ax.fill_between(grid, q25, q75, alpha=0.18, linewidth=0)
            ax.plot(grid, q50, linewidth=1.4, label="median")

        tau = float(row[tau_key])
        if tau > 0:
            ax.axvline(tau, linestyle="--", linewidth=0.8, label="used tau")

        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(structure)
        if kind == "shape":
            ax.set_xlim(0, 500)
        else:
            ax.set_xlim(0, 2000)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"Lag time $\Delta t$")
        ax.set_ylabel(ylabel)
        ax.tick_params(which="both", top=True, right=True)

        if i == 0:
            ax.legend(frameon=False)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle(title, y=1.01, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, filename)


def plot_msd_grid(rows: list[dict[str, Any]], by_structure: dict[str, list[dict[str, Any]]], figdir: Path) -> None:
    valid = [row for row in rows if int(row["n_seeds"]) > 0]
    if not valid:
        return

    n = len(valid)
    ncols = 2 if n <= 4 else 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.6 * nrows), squeeze=False)

    for i, row in enumerate(valid):
        ax = axes[i // ncols][i % ncols]
        structure = row["structure"]
        items = by_structure[structure]

        for item in items:
            msd = item["msd_perp"]
            lag = msd["lag_time"]
            y = msd["msd"]
            ax.plot(lag, y, linewidth=0.6, alpha=0.25)

            if msd["fit_end_time"] > msd["fit_start_time"]:
                x_fit = np.asarray([msd["fit_start_time"], msd["fit_end_time"]])
                y_fit = msd["fit_slope"] * x_fit + msd["fit_intercept"]
                ax.plot(x_fit, y_fit, linestyle="--", linewidth=0.7, alpha=0.35)

        grid, q25, q50, q75 = median_msd_curve(items)
        if len(grid):
            ax.fill_between(grid, q25, q75, alpha=0.18, linewidth=0)
            ax.plot(grid, q50, linewidth=1.4, label="median")

        d_perp = float(row["d_perp_median"])
        r2 = float(row["msd_perp_fit_r2_median"])

        ax.text(
            0.04,
            0.96,
            rf"$D_\perp={d_perp:.2e}$" + "\n" + rf"$R^2={r2:.3f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
        )

        ax.set_title(structure)
        ax.set_xlabel(r"Lag time $\Delta t$")
        ax.set_ylabel(r"$\mathrm{MSD}_{\perp}(\Delta t)$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.tick_params(which="both", top=True, right=True)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle("Transverse center-of-mass MSD", y=1.01, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, "msd_perp_by_structure")


def plot_per_structure(rows: list[dict[str, Any]], by_structure: dict[str, list[dict[str, Any]]], figdir: Path) -> None:
    outdir = figdir / "per_structure"
    valid = [row for row in rows if int(row["n_seeds"]) > 0]

    for row in valid:
        structure = row["structure"]
        items = by_structure[structure]

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

        ax = axes[0]
        panel_label(ax, "A")
        for item in items:
            ax.plot(item["shape_lag"], item["shape_acf"], linewidth=0.6, alpha=0.25)
        grid, q25, q50, q75 = median_curve(items, "shape_lag", "shape_acf")
        if len(grid):
            ax.fill_between(grid, q25, q75, alpha=0.18, linewidth=0)
            ax.plot(grid, q50, linewidth=1.4)
        ax.axvline(float(row["tau_shape_used"]), linestyle="--", linewidth=0.8)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlim(0, 500)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"Lag time $\Delta t$")
        ax.set_ylabel(r"$C_{\mathrm{asphericity}}$")
        ax.set_title("Shape ACF")

        ax = axes[1]
        panel_label(ax, "B")
        for item in items:
            ax.plot(item["r_lag"], item["r_acf"], linewidth=0.6, alpha=0.25)
        grid, q25, q50, q75 = median_curve(items, "r_lag", "r_acf")
        if len(grid):
            ax.fill_between(grid, q25, q75, alpha=0.18, linewidth=0)
            ax.plot(grid, q50, linewidth=1.4)
        ax.axvline(float(row["tau_int_r_used"]), linestyle="--", linewidth=0.8)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlim(0, 2000)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"Lag time $\Delta t$")
        ax.set_ylabel(r"$C_{r_{\mathrm{cm}}}$")
        ax.set_title(r"$r_{\mathrm{cm}}$ ACF")

        ax = axes[2]
        panel_label(ax, "C")
        for item in items:
            msd = item["msd_perp"]
            ax.plot(msd["lag_time"], msd["msd"], linewidth=0.6, alpha=0.25)
            if msd["fit_end_time"] > msd["fit_start_time"]:
                x_fit = np.asarray([msd["fit_start_time"], msd["fit_end_time"]])
                y_fit = msd["fit_slope"] * x_fit + msd["fit_intercept"]
                ax.plot(x_fit, y_fit, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_xlabel(r"Lag time $\Delta t$")
        ax.set_ylabel(r"$\mathrm{MSD}_{\perp}$")
        ax.set_title("Transverse MSD")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))

        for ax in axes:
            ax.tick_params(which="both", top=True, right=True)

        fig.suptitle(structure, y=1.04, fontsize=9)
        fig.tight_layout()
        save_fig(fig, outdir, f"{structure}_diagnostics")


def generate_figures(rows: list[dict[str, Any]], by_structure: dict[str, list[dict[str, Any]]], figdir: Path) -> None:
    setup_plot_style()
    plot_summary(rows, by_structure, figdir)
    plot_acf_grid(rows, by_structure, figdir, kind="shape")
    plot_acf_grid(rows, by_structure, figdir, kind="r_cm")
    plot_msd_grid(rows, by_structure, figdir)
    plot_per_structure(rows, by_structure, figdir)


def structure_sort_key(name: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits) if digits else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/base.json"))
    parser.add_argument("--tasks", type=Path, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/timescales.csv"))
    parser.add_argument("--run-output", type=Path, default=Path("data/timescale_runs.csv"))
    parser.add_argument("--figdir", type=Path, default=Path("figures/timescales"))
    parser.add_argument("--pipe-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()

    config = read_json(root / args.config)
    radius = float(config["pipe"]["radius"])

    task_paths = [root / path for path in args.tasks] if args.tasks else find_default_tasks(root)
    tasks = read_tasks(task_paths)

    expected_by_structure = Counter(task["structure"] for task in tasks)
    all_structures = sorted(expected_by_structure.keys(), key=structure_sort_key)

    by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_rows: list[dict[str, Any]] = []

    print(f"读取 config: {args.config}")
    print("读取 tasks:")
    for path in task_paths:
        print(f"  {path}")
    print(f"pipe_radius = {radius}")
    print(f"pipe_axis = {args.pipe_axis}")
    print(f"任务数 = {len(tasks)}")

    for task in tasks:
        run_id = task["run_id"]
        structure = task["structure"]
        series_path = root / "results" / run_id / "timeseries.npz"

        if not series_path.exists():
            print(f"[missing] {run_id}: {series_path}")
            continue

        metrics = analyze_run(series_path, radius, pipe_axis=args.pipe_axis)
        by_structure[structure].append(metrics)
        run_rows.append(make_run_row(task, metrics, radius))

        print(
            f"{run_id}: "
            f"tau_shape={metrics['tau_shape']:.2f}, "
            f"tau_int_r={metrics['tau_int_r']:.2f}, "
            f"D_perp={metrics['d_perp']:.3e}, "
            f"tau_rad_diag={metrics['tau_rad_diagnostic']:.0f}, "
            f"T_analyzed={metrics['t_analyzed']:.0f}, "
            f"MSD_R2={metrics['msd_perp']['fit_r2']:.3f}"
        )

    if not run_rows:
        raise RuntimeError("没有找到任何 results/<run_id>/timeseries.npz，无法分析")

    rows: list[dict[str, Any]] = []

    for structure in all_structures:
        values = by_structure.get(structure, [])
        expected = expected_by_structure[structure]

        if not values:
            rows.append(empty_timescale_row(structure, radius, "missing_results", expected))
            print(f"{structure}: 缺少结果，写入 missing_results 占位行")
            continue

        summary = summarize_structure(values, expected)

        row = {
            "structure": structure,
            "pipe_radius": f"{radius:.6f}",
            "tau_shape_used": fmt6(summary["tau_shape_used"]),
            "tau_int_r_used": fmt6(summary["tau_int_r_used"]),
            "tau_r_used": fmt6(summary["tau_r_used"]),
            "tau_rad_diagnostic": fmt6(summary["tau_rad_diagnostic"]),
            "tau_shape_median": fmt6(summary["tau_shape_median"]),
            "tau_int_r_median": fmt6(summary["tau_int_r_median"]),
            "tau_r_median": fmt6(summary["tau_r_median"]),
            "d_cm_median": fmte(summary["d_cm_median"]),
            "d_perp_median": fmte(summary["d_perp_median"]),
            "msd_cm_fit_r2_median": fmt6(summary["msd_cm_fit_r2_median"]),
            "msd_perp_fit_r2_median": fmt6(summary["msd_perp_fit_r2_median"]),
            "t_analyzed": fmt6(summary["t_analyzed"]),
            "t_over_tau_shape": fmt6(summary["t_over_tau_shape"]),
            "t_over_tau_int_r": fmt6(summary["t_over_tau_int_r"]),
            "tau_shape_flag": summary["tau_shape_flag"],
            "tau_int_r_flag": summary["tau_int_r_flag"],
            "tau_r_flag": summary["tau_r_flag"],
            "tau_rad_note": summary["tau_rad_note"],
            "n_seeds": summary["n_seeds"],
            "n_expected": summary["n_expected"],
            "n_missing": summary["n_missing"],
        }

        rows.append(row)

        print(
            f"{structure}: "
            f"n={row['n_seeds']}/{row['n_expected']}, "
            f"tau_shape_used={row['tau_shape_used']}, "
            f"tau_int_r_used={row['tau_int_r_used']}, "
            f"T/tau_int_r={row['t_over_tau_int_r']}, "
            f"shape_flag={row['tau_shape_flag']}, "
            f"r_flag={row['tau_int_r_flag']}, "
            f"rad_note={row['tau_rad_note']}"
        )

    write_csv(root / args.output, rows)
    write_csv(root / args.run_output, run_rows)

    print(f"写入结构汇总: {args.output}")
    print(f"写入单轨迹诊断: {args.run_output}")

    if not args.no_figures:
        generate_figures(rows, by_structure, root / args.figdir)
        print(f"图已写入: {args.figdir}")


if __name__ == "__main__":
    main()