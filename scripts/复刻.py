from __future__ import annotations

"""复刻参考论文图中当前项目可以直接画出的六张图。

运行位置：
    /home/zhangxh/students/sjl/mpcd_modular_v2

运行命令：
    /home/zhangxh/students/sjl/miniconda3/envs/A/bin/python scripts/复刻.py

输入：
    results/半径36_结构n*_体力*_种子*/timeseries.npz
    results/半径36_结构n*_体力*_种子*/state.npz
    data/structures/g*.json

输出：
    figures/复刻图/02_不同功能度径向分布.png
    figures/复刻图/03_径向分布宽度随流强变化.png
    figures/复刻图/05_回转张量分量随流强变化.png
    figures/复刻图/07_不同结构流动构型快照.png
    figures/复刻图/08_回转半径时间序列.png
    figures/复刻图/09_翻滚相关函数与特征时间.png
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


# 项目根目录。
ROOT = Path("/home/zhangxh/students/sjl/mpcd_modular_v2")
# 输出目录。
OUT = ROOT / "figures" / "复刻图"
# 当前项目的四个整体 n*n*n 网格凝胶结构。
STRUCTURES = ["g1", "g2", "g3", "g4"]
# 当前正式数据已有的真实体力。
FORCES = [0.0, 0.001, 0.003, 0.005, 0.01]
# 当前正式数据已有的独立种子。
SEEDS = [101, 102, 103, 104, 105]
# 管半径，用于无量纲化 r/R。
RADIUS = 36.0
# 管长，用于展开 z 方向周期边界。
PIPE_LENGTH = 100.0
# 凝胶正式统计窗口起点。前 5000 时间单位只用于稳定化，不进入论文统计。
ANALYSIS_START_TIME = 5000.0
# 凝胶正式统计窗口终点。400 万步、dt=0.005 对应 20000 时间单位。
ANALYSIS_END_TIME = 20000.0
# 快照默认使用最高体力和第一个 seed。
SNAP_FORCE = 0.01
SNAP_SEED = 101

# 体力颜色：参考 Liu 图中黑、红、绿、蓝的强弱序列，再补一个紫色。
FORCE_COLORS = {
    0.0: "#000000",
    0.001: "#D55E00",
    0.003: "#009E73",
    0.005: "#0072B2",
    0.01: "#CC79A7",
}
# 结构颜色：色盲友好，便于区分四个 n。
STRUCTURE_COLORS = {
    "g1": "#000000",
    "g2": "#D55E00",
    "g3": "#009E73",
    "g4": "#0072B2",
}
# 参考论文常用的空心标记。
MARKERS = {
    "g1": "o",
    "g2": "^",
    "g3": "v",
    "g4": "D",
}


def setup_style() -> None:
    """设置接近 Macromolecules 图的字体、线宽和坐标轴风格。"""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.linewidth": 1.25,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.2,
            "ytick.major.size": 4.2,
            "xtick.minor.size": 2.3,
            "ytick.minor.size": 2.3,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    """把图片保存到固定文件名，便于和参考图一一对应。"""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax: Axes, text: str) -> None:
    """添加论文图常见的 (a)、(b) 面板标签。"""
    ax.text(0.03, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=12)


def structure_label(structure: str) -> str:
    """把 g1 转成图中显示的 n=1。"""
    return rf"$n={structure[1:]}$"


def force_label(force: float) -> str:
    """把真实体力写成图例标签。"""
    return rf"$g={force:g}$"


def run_dir(structure: str, force: float, seed: int) -> Path:
    """返回一条结果目录。"""
    return ROOT / "results" / f"半径36_结构n{structure[1:]}_体力{force:g}_种子{seed}"


def series_path(structure: str, force: float, seed: int) -> Path:
    """返回时间序列文件路径。"""
    return run_dir(structure, force, seed) / "timeseries.npz"


def state_path(structure: str, force: float, seed: int) -> Path:
    """返回末态构型文件路径。"""
    return run_dir(structure, force, seed) / "state.npz"


def load_series(structure: str, force: float, seed: int) -> dict[str, np.ndarray] | None:
    """读取一条时间序列；没有文件时返回 None。"""
    path = series_path(structure, force, seed)
    if not path.exists():
        return None
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def analysis_window(data: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    """取凝胶正式统计窗口内的数据。

    正式凝胶轨迹使用 0 到 20000 的总时间。
    0 到 5000 是稳定化时间；5000 到 20000 才进入统计。
    旧文件如果缺少 time 字段，就退回后半段，保证脚本仍能读取历史数据。
    """
    values = np.asarray(values)
    time = data.get("time")
    if time is None or len(time) != len(values):
        return values[len(values) // 2 :]
    time = np.asarray(time, dtype=float)
    mask = (time >= ANALYSIS_START_TIME) & (time <= ANALYSIS_END_TIME)
    if not np.any(mask):
        return values[len(values) // 2 :]
    return values[mask]


def finite(values: np.ndarray) -> np.ndarray:
    """去掉 NaN 和 inf。"""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def mean_sem(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """计算多 seed 的均值和标准误。"""
    matrix = np.vstack(rows)
    mean = np.nanmean(matrix, axis=0)
    sem = np.nanstd(matrix, axis=0, ddof=1) / math.sqrt(matrix.shape[0]) if matrix.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem


def smooth(values: np.ndarray, passes: int = 2) -> np.ndarray:
    """对一维曲线做轻度平滑，让直方图更接近论文中的连续分布曲线。"""
    out = np.asarray(values, dtype=float).copy()
    kernel = np.asarray([1.0, 2.0, 1.0], dtype=float)
    kernel /= np.sum(kernel)
    for _ in range(passes):
        out = np.convolve(np.pad(out, 1, mode="edge"), kernel, mode="valid")
    return out


def collect_scalar(structure: str, force: float, field: str) -> list[float]:
    """收集某个结构和体力下多个 seed 的正式统计窗口均值。"""
    values: list[float] = []
    for seed in SEEDS:
        data = load_series(structure, force, seed)
        if data is None:
            continue
        values.append(float(np.mean(finite(analysis_window(data, data[field])))))
    return values


def read_structure(structure: str) -> dict:
    """读取结构 JSON。"""
    return json.loads((ROOT / "data" / "structures" / f"{structure}.json").read_text(encoding="utf-8"))


def unwrap_z(pos: np.ndarray, bonds: list[dict[str, int]]) -> np.ndarray:
    """沿 z 方向展开周期边界，让键不会跨过整个盒子显示。"""
    out = np.asarray(pos, dtype=float).copy()
    neighbors: list[list[int]] = [[] for _ in range(len(out))]
    for bond in bonds:
        i = int(bond["i"])
        j = int(bond["j"])
        neighbors[i].append(j)
        neighbors[j].append(i)
    seen = np.zeros(len(out), dtype=bool)
    queue = [0]
    seen[0] = True
    while queue:
        i = queue.pop(0)
        for j in neighbors[i]:
            if seen[j]:
                continue
            dz = out[j, 2] - out[i, 2]
            out[j, 2] -= round(dz / PIPE_LENGTH) * PIPE_LENGTH
            seen[j] = True
            queue.append(j)
    return out


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """用 FFT 计算归一化自相关函数。"""
    x = finite(x)
    if len(x) < 4:
        return np.asarray([])
    x = x - np.mean(x)
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    freq = np.fft.rfft(x, size)
    corr = np.fft.irfft(freq * np.conjugate(freq), size)[:n]
    norm = np.arange(n, 0, -1)
    corr = corr / norm
    if corr[0] == 0:
        return np.asarray([])
    return corr / corr[0]


def crosscorr_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算归一化互相关，用于翻滚行为类比图。"""
    x = finite(x)
    y = finite(y)
    n = min(len(x), len(y))
    if n < 4:
        return np.asarray([])
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    size = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, size)
    fy = np.fft.rfft(y, size)
    corr = np.fft.irfft(fx * np.conjugate(fy), size)[:n]
    corr = corr / np.arange(n, 0, -1)
    denom = np.std(x) * np.std(y)
    if denom == 0:
        return np.asarray([])
    return corr / denom


def figure_02() -> None:
    """复刻 02：不同结构在不同体力下的径向分布。"""
    bins = np.linspace(0.0, 0.65, 28)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True, sharey=True)
    for ax, structure, tag in zip(axes.ravel(), STRUCTURES, ["(a)", "(b)", "(c)", "(d)"]):
        for force in FORCES:
            rows: list[np.ndarray] = []
            for seed in SEEDS:
                data = load_series(structure, force, seed)
                if data is None:
                    continue
                r = finite(analysis_window(data, data["r_cm"]) / RADIUS)
                hist, _ = np.histogram(r, bins=bins, density=True)
                rows.append(hist)
            if not rows:
                continue
            mean, sem = mean_sem(rows)
            mean = smooth(mean, passes=2)
            sem = smooth(sem, passes=2)
            color = FORCE_COLORS[force]
            mask = mean > 1e-3
            ax.plot(centers[mask], mean[mask], color=color, lw=1.35, marker="o", ms=2.8, mfc="white", mec=color, label=force_label(force))
            ax.fill_between(centers[mask], np.maximum(mean[mask] - sem[mask], 1e-3), mean[mask] + sem[mask], color=color, alpha=0.10, lw=0)
        panel(ax, tag)
        ax.set_title(structure_label(structure), pad=2)
        ax.set_yscale("log")
        ax.set_xlim(0, 0.65)
        ax.set_ylim(1e-2, 2e2)
        ax.set_xlabel(r"$r_{\rm cm}/R$")
        ax.set_ylabel(r"$P_{\rm cm}(r)$")
        ax.minorticks_on()
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "02_不同功能度径向分布.png")


def figure_03() -> None:
    """复刻 03：径向分布宽度随体力变化。"""
    fig, ax = plt.subplots(figsize=(4.1, 3.4))
    for structure in STRUCTURES:
        base_values: list[float] = []
        for seed in SEEDS:
            data0 = load_series(structure, 0.0, seed)
            if data0 is None:
                continue
            base_values.append(float(np.std(finite(analysis_window(data0, data0["r_cm"]) / RADIUS), ddof=1)))
        base_width = float(np.mean(base_values)) if base_values else np.nan
        means: list[float] = []
        sems: list[float] = []
        for force in FORCES:
            widths: list[float] = []
            for seed in SEEDS:
                data = load_series(structure, force, seed)
                if data is None:
                    continue
                r = finite(analysis_window(data, data["r_cm"]) / RADIUS)
                if len(r) > 1:
                    widths.append(float(np.std(r, ddof=1) / base_width))
            means.append(float(np.mean(widths)))
            sems.append(float(np.std(widths, ddof=1) / math.sqrt(len(widths))) if len(widths) > 1 else 0.0)
        color = STRUCTURE_COLORS[structure]
        ax.errorbar(
            FORCES,
            means,
            yerr=sems,
            color=color,
            lw=1.35,
            marker=MARKERS[structure],
            ms=5.0,
            mfc="white",
            mec=color,
            capsize=2.0,
            label=structure_label(structure),
        )
    ax.set_xscale("symlog", linthresh=0.0008)
    ax.set_xlabel(r"Body force $g$")
    ax.set_ylabel(r"$\langle l_r\rangle/\langle l_{r,0}\rangle$")
    ax.set_ylim(bottom=0)
    ax.set_xticks([0.0, 0.001, 0.01])
    ax.set_xticklabels(["0", r"$10^{-3}$", r"$10^{-2}$"])
    ax.legend(frameon=False, loc="best")
    ax.minorticks_on()
    fig.tight_layout()
    save(fig, "03_径向分布宽度随流强变化.png")


def figure_05() -> None:
    """复刻 05：回转张量分量随体力变化。"""
    fig, axes = plt.subplots(2, 1, figsize=(4.3, 6.0), sharex=True)
    quantities = [("Gzz", r"$\langle G_{zz}\rangle/\langle G_{zz}\rangle_0$"), ("Gperp", r"$\langle G_\perp\rangle/\langle G_\perp\rangle_0$")]
    for ax, (field, ylabel), tag in zip(axes, quantities, ["(a)", "(b)"]):
        for structure in STRUCTURES:
            base = collect_scalar(structure, 0.0, field)
            base_mean = float(np.mean(base)) if base else np.nan
            means: list[float] = []
            sems: list[float] = []
            for force in FORCES:
                vals = np.asarray(collect_scalar(structure, force, field), dtype=float) / base_mean
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            color = STRUCTURE_COLORS[structure]
            ax.errorbar(
                FORCES,
                means,
                yerr=sems,
                color=color,
                lw=1.35,
                marker=MARKERS[structure],
                ms=5.0,
                mfc="white",
                mec=color,
                capsize=2.0,
                label=structure_label(structure),
            )
        panel(ax, tag)
        ax.set_xscale("symlog", linthresh=0.0008)
        ax.set_ylabel(ylabel)
        ax.minorticks_on()
    axes[-1].set_xlabel(r"Body force $g$")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "05_回转张量分量随流强变化.png")


def draw_snapshot(ax: Axes, structure: str, plane: str) -> None:
    """绘制单个凝胶末态快照。"""
    state = np.load(state_path(structure, SNAP_FORCE, SNAP_SEED))
    pos = np.asarray(state["bead_position"], dtype=float)
    kind = np.asarray(state["bead_typeid"], dtype=int)
    bonds = read_structure(structure)["bonds"]
    if plane == "xz":
        pos = unwrap_z(pos, bonds)
        pos[:, 2] -= np.mean(pos[:, 2])
        a, b = 2, 0
    else:
        a, b = 0, 1
    xy = pos[:, [a, b]] / RADIUS
    for bond in bonds:
        i = int(bond["i"])
        j = int(bond["j"])
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.62", lw=0.45, zorder=1)
    chain = kind == 1
    xlink = kind == 0
    ax.scatter(xy[chain, 0], xy[chain, 1], s=4.5, c="#56B4E9", edgecolors="none", alpha=0.85, zorder=2)
    ax.scatter(xy[xlink, 0], xy[xlink, 1], s=15, c="#D55E00", edgecolors="white", linewidths=0.25, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_07() -> None:
    """复刻 07：不同结构在流动后的构型快照。"""
    fig, axes = plt.subplots(4, 2, figsize=(5.7, 7.2))
    for row, structure in enumerate(STRUCTURES):
        draw_snapshot(axes[row, 0], structure, "xy")
        draw_snapshot(axes[row, 1], structure, "xz")
        axes[row, 0].text(-0.12, 0.5, f"({chr(97 + row)})", transform=axes[row, 0].transAxes, fontsize=12, va="center")
        axes[row, 0].text(0.50, -0.08, structure_label(structure), transform=axes[row, 0].transAxes, ha="center", va="top", fontsize=11)
        axes[row, 1].text(0.50, -0.08, structure_label(structure), transform=axes[row, 1].transAxes, ha="center", va="top", fontsize=11)
    axes[0, 0].text(0.5, 1.07, r"cross-section", transform=axes[0, 0].transAxes, ha="center", va="bottom")
    axes[0, 1].text(0.5, 1.07, r"flow direction", transform=axes[0, 1].transAxes, ha="center", va="bottom")
    fig.tight_layout(w_pad=0.2, h_pad=0.15)
    save(fig, "07_不同结构流动构型快照.png")


def figure_08() -> None:
    """复刻 08：回转半径时间序列。"""
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 5.5), sharex=True)
    picks = [("g1", "(a)"), ("g4", "(b)")]
    for ax, (structure, tag) in zip(axes, picks):
        for seed in SEEDS[:3]:
            data = load_series(structure, SNAP_FORCE, seed)
            if data is None:
                continue
            t = np.asarray(data["time"], dtype=float)
            rg = np.asarray(data["Rg"], dtype=float)
            ax.plot(t, rg, lw=0.75, alpha=0.45, color="0.45")
        data = load_series(structure, SNAP_FORCE, SEEDS[0])
        if data is not None:
            t = np.asarray(data["time"], dtype=float)
            rg = np.asarray(data["Rg"], dtype=float)
            window = 21
            kernel = np.ones(window) / window
            smooth_rg = np.convolve(np.pad(rg, window // 2, mode="edge"), kernel, mode="valid")
            ax.plot(t, smooth_rg, lw=1.35, color=STRUCTURE_COLORS[structure], label="smoothed")
        ax.set_ylabel(r"$R_g$")
        ax.set_title(structure_label(structure), pad=2)
        panel(ax, tag)
        ax.minorticks_on()
    axes[-1].set_xlabel(r"$t$")
    fig.tight_layout()
    save(fig, "08_回转半径时间序列.png")


def figure_09() -> None:
    """复刻 09：主轴翻滚互相关和特征时间。"""
    fig, axes = plt.subplots(2, 1, figsize=(4.3, 5.7))
    lag_max = 400
    for structure in STRUCTURES:
        rows: list[np.ndarray] = []
        dt = None
        for seed in SEEDS:
            data = load_series(structure, SNAP_FORCE, seed)
            if data is None:
                continue
            axis = np.asarray(data["axis"], dtype=float)
            corr = crosscorr_fft(analysis_window(data, axis[:, 0]), analysis_window(data, axis[:, 2]))
            if len(corr):
                rows.append(corr[:lag_max])
                time = np.asarray(data["time"], dtype=float)
                dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
        if rows:
            min_len = min(len(r) for r in rows)
            matrix = np.vstack([r[:min_len] for r in rows])
            tau = np.arange(min_len) * (dt if dt else 1.0)
            axes[0].plot(tau, np.mean(matrix, axis=0), lw=1.25, color=STRUCTURE_COLORS[structure], label=structure_label(structure))
    axes[0].axhline(0, color="0.25", lw=0.8)
    axes[0].set_xlabel(r"lag time")
    axes[0].set_ylabel(r"$C_{xz}(t)$")
    axes[0].legend(frameon=False, loc="upper right")
    panel(axes[0], "(a)")
    for structure in STRUCTURES:
        tau_values: list[float] = []
        for force in [0.001, 0.003, 0.005, 0.01]:
            seed_taus: list[float] = []
            for seed in SEEDS:
                data = load_series(structure, force, seed)
                if data is None:
                    continue
                axis = np.asarray(data["axis"], dtype=float)
                corr = autocorr_fft(analysis_window(data, axis[:, 2]))
                time = np.asarray(data["time"], dtype=float)
                if len(corr) < 4 or len(time) < 2:
                    continue
                dt = float(np.mean(np.diff(time)))
                below = np.where(corr < math.exp(-1))[0]
                if len(below):
                    seed_taus.append(float(below[0] * dt))
            tau_values.append(float(np.mean(seed_taus)) if seed_taus else np.nan)
        color = STRUCTURE_COLORS[structure]
        axes[1].plot([0.001, 0.003, 0.005, 0.01], tau_values, marker=MARKERS[structure], ms=5.5, mfc="white", mec=color, color=color, lw=1.25, label=structure_label(structure))
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Body force $g$")
    axes[1].set_ylabel(r"correlation time")
    axes[1].legend(frameon=False, loc="upper right")
    panel(axes[1], "(b)")
    for ax in axes:
        ax.minorticks_on()
    fig.tight_layout()
    save(fig, "09_翻滚相关函数与特征时间.png")


def main() -> None:
    """依次输出六张复刻图。"""
    setup_style()
    figure_02()
    figure_03()
    figure_05()
    figure_07()
    figure_08()
    figure_09()
    print(f"复刻图已输出到 {OUT}")


if __name__ == "__main__":
    main()
