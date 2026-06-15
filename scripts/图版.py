from __future__ import annotations

"""生成更接近真实论文排版的图版。

这个脚本只做画图，不跑模拟。
设计依据是管流聚合物/软颗粒论文常用的图式：
1. 先用空管速度剖面证明流场正确。
2. 再给出结构和流动后的构型快照。
3. 用径向概率分布展示质心占据，而不是只画均值折线。
4. 用“形变量随径向位置变化”的条件图解释形变和位置的关系。
5. 用二维热图总结结构和体力两个变量的响应。
6. 用 violin 图展示分布宽度和间歇性，而不是只报告平均数。

可以改什么：
- STRUCTURES：要比较的结构。
- FORCES：要画的真实体力。
- SEEDS：要读取的 seed。
- RADIUS：用于归一化的水动力管半径。
- SNAP_FORCE：构型快照使用的体力。

输入：
- data/structures/g<n>.json
- results/半径36_结构n<n>_体力<g>_种子<seed>/timeseries.npz
- results/半径36_结构n<n>_体力<g>_种子<seed>/state.npz
- results/半径36_空管_体力<g>_种子<seed>/profiles.npz

输出：
- figures/图版/MM-DD-HH-MM-*.png
"""

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from 命名 import 时间序列, 流场剖面, 结果目录, 数值


# 四个整体网格结构。
STRUCTURES = ["g1", "g2", "g3", "g4"]
# 当前探索数据已有的真实体力。
FORCES = [0.0, 0.001, 0.003, 0.005, 0.01]
# 当前正式数据已完成的五个独立 seed。
SEEDS = [101, 102, 103, 104, 105]
# 空管 seed。
FLUID_SEEDS = [301, 302, 303]
# 水动力管半径。
RADIUS = 36.0
# 构型快照和分布细节默认看最高体力。
SNAP_FORCE = 0.01
# 构型快照默认 seed。
SNAP_SEED = 101
# 管道周期长度；只用于把快照里的跨周期键展开成连续构型。
PIPE_LENGTH = 100.0
# 凝胶正式统计窗口。前 5000 时间单位用于流场建立和构型松弛。
ANALYSIS_START_TIME = 5000.0
# 凝胶正式统计窗口终点。400 万步、dt=0.005 时正好是 20000。
ANALYSIS_END_TIME = 20000.0
# 色盲友好配色。
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
# 体力配色，从低到高由灰到红紫，便于体现强弱。
FLOW_COLORS = ["#4D4D4D", "#0072B2", "#009E73", "#E69F00", "#D55E00"]


def 北京时间戳() -> str:
    """返回北京时间的月日时分字符串。

    服务器系统时间可能是 UTC。
    图片文件名用北京时间，和本地查看时间保持一致。
    """
    return datetime.now(timezone(timedelta(hours=8))).strftime("%m-%d-%H-%M")


def setup_style() -> None:
    """设置期刊式画图风格。"""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.05,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, root: Path, name: str) -> Path:
    """保存图片到 figures/图版。"""
    outdir = root / "figures" / "图版"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{北京时间戳()}-{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def label(ax: Axes, text: str) -> None:
    """给子图加粗体面板标签。"""
    ax.text(-0.14, 1.07, text, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")


def clean(ax: Axes) -> None:
    """去掉上边框和右边框。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


def title_structure(structure: str) -> str:
    """把 g1 转成 n=1。"""
    return rf"$n={structure[1:]}$"


def title_force(force: float) -> str:
    """把体力写成图例标签。"""
    return rf"$g={数值(force)}$"


def analysis_window(data: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    """取凝胶正式统计窗口内的数据。

    正式凝胶轨迹统一按 0 到 20000 的物理时间运行。
    0 到 5000 是稳定化时间，不进入论文统计；
    5000 到 20000 是正式统计时间。
    如果旧文件没有 time 字段，就退回到后半段，避免脚本直接崩溃。
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
    """过滤非法数值。"""
    return values[np.isfinite(values)]


def read_series(root: Path, structure: str, force: float, seed: int, radius: float) -> dict[str, np.ndarray] | None:
    """读取 timeseries.npz。"""
    path = 时间序列(root, radius, structure, force, seed)
    if not path.exists():
        return None
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def state_path(root: Path, structure: str, force: float, seed: int, radius: float) -> Path:
    """返回 state.npz 路径。"""
    return 结果目录(root, radius, structure, force, seed) / "state.npz"


def read_structure(root: Path, structure: str) -> dict:
    """读取初始结构文件，主要用里面的键连接。"""
    path = root / "data" / "structures" / f"{structure}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_z(pos: np.ndarray, bonds: list[dict[str, int]], box_length: float) -> np.ndarray:
    """沿 z 方向展开周期边界，避免快照中出现跨盒子的长假键。

    模拟轨迹为了周期边界会把珠子坐标重新放回盒子。
    直接画这些坐标时，一条真实很短的键可能被画成横跨整个盒子的长线。
    这里从第一个珠子出发，沿键连接逐个修正相邻珠子的 z 坐标，
    让每条键都选择最短的周期镜像。
    """
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
            out[j, 2] -= round(dz / box_length) * box_length
            seen[j] = True
            queue.append(j)
    return out


def metric(data: dict[str, np.ndarray], name: str, radius: float) -> np.ndarray:
    """把 timeseries 里的原始字段变成绘图用物理量。"""
    if name == "r":
        return np.asarray(data["r_cm"], dtype=float) / radius
    if name == "Rg":
        return np.asarray(data["Rg"], dtype=float) / radius
    if name == "Gzz":
        return np.asarray(data["Gzz"], dtype=float)
    if name == "Gperp":
        return np.asarray(data["Gperp"], dtype=float)
    if name == "stretch":
        return np.asarray(data["Gzz"], dtype=float) / np.maximum(np.asarray(data["Gperp"], dtype=float), 1e-12)
    if name == "theta":
        return np.degrees(np.asarray(data["theta"], dtype=float))
    if name == "clearance":
        return np.asarray(data["wall_clearance"], dtype=float) / radius
    if name == "R99":
        return np.asarray(data["R99"], dtype=float) / radius
    return np.asarray(data[name], dtype=float)


def mean_metric(root: Path, structure: str, force: float, seeds: list[int], radius: float, name: str) -> float:
    """计算一个结构、一个体力、多个 seed 的正式统计窗口平均值。"""
    values: list[float] = []
    for seed in seeds:
        data = read_series(root, structure, force, seed, radius)
        if data is None:
            continue
        raw = finite(analysis_window(data, metric(data, name, radius)))
        if len(raw):
            values.append(float(np.mean(raw)))
    if not values:
        return float("nan")
    return float(np.mean(values))


def collect_frames(root: Path, structure: str, force: float, seeds: list[int], radius: float, names: list[str]) -> dict[str, np.ndarray]:
    """收集多个物理量的所有正式统计窗口帧。"""
    out = {name: [] for name in names}
    for seed in seeds:
        data = read_series(root, structure, force, seed, radius)
        if data is None:
            continue
        for name in names:
            out[name].append(finite(analysis_window(data, metric(data, name, radius))))
    return {name: np.concatenate(parts) if parts else np.asarray([], dtype=float) for name, parts in out.items()}


def profile(root: Path, force: float, seeds: list[int], radius: float) -> tuple[np.ndarray, np.ndarray]:
    """读取空管速度剖面。"""
    curves: list[np.ndarray] = []
    grid: np.ndarray | None = None
    for seed in seeds:
        path = 流场剖面(root, radius, force, seed)
        if not path.exists():
            continue
        data = np.load(path)
        r = np.asarray(data["r"], dtype=float) / radius
        u = np.asarray(data["mean_vz"], dtype=float)
        if grid is None:
            grid = r
        if len(u) == len(grid):
            curves.append(u)
    if grid is None or not curves:
        return np.asarray([]), np.asarray([])
    return grid, np.mean(np.vstack(curves), axis=0)


def fit_flow(r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """拟合带常数偏置的泊肃叶剖面。"""
    x = np.vstack([1.0 - r**2, np.ones_like(r)]).T
    amp, slip = np.linalg.lstsq(x, u, rcond=None)[0]
    return amp * (1.0 - r**2) + slip


def draw_snapshot(ax: Axes, root: Path, structure: str, force: float, seed: int, radius: float, plane: str) -> None:
    """绘制流动后的凝胶快照。"""
    path = state_path(root, structure, force, seed, radius)
    if not path.exists():
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_title(title_structure(structure))
        ax.set_xticks([])
        ax.set_yticks([])
        clean(ax)
        return
    state = np.load(path)
    pos = np.asarray(state["bead_position"], dtype=float)
    kind = np.asarray(state["bead_typeid"], dtype=int)
    bonds = read_structure(root, structure)["bonds"]
    if plane == "xy":
        a, b = 0, 1
        ax.set_xlabel(r"$x/R$")
        ax.set_ylabel(r"$y/R$")
        circle = plt.Circle((0, 0), 1.0, fill=False, color="0.25", lw=1.0, ls="--")
        ax.add_patch(circle)
    else:
        pos = unwrap_z(pos, bonds, PIPE_LENGTH)
        a, b = 2, 0
        pos = pos.copy()
        pos[:, 2] -= np.mean(pos[:, 2])
        ax.set_xlabel(r"$(z-z_{\rm cm})/R$")
        ax.set_ylabel(r"$x/R$")
    xy = pos[:, [a, b]] / radius
    for bond in bonds:
        i = int(bond["i"])
        j = int(bond["j"])
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.72", lw=0.42, zorder=1)
    chain = kind == 1
    xlink = kind == 0
    ax.scatter(xy[chain, 0], xy[chain, 1], s=5.0, color="#5DA5DA", alpha=0.78, edgecolors="none", zorder=2)
    ax.scatter(xy[xlink, 0], xy[xlink, 1], s=16.0, color="#D55E00", alpha=0.95, edgecolors="white", linewidths=0.25, zorder=3)
    ax.set_aspect("equal")
    ax.set_title(title_structure(structure))
    clean(ax)


def figure_reference(root: Path, structures: list[str], forces: list[float], fluid_seeds: list[int], radius: float) -> Path:
    """图版 1：流场验证和结构快照。"""
    fig = plt.figure(figsize=(10.5, 5.8))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.22, 1, 1, 1], height_ratios=[1, 1], hspace=0.46, wspace=0.42)
    ax_flow = fig.add_subplot(gs[:, 0])
    for color, force in zip(FLOW_COLORS[1:], [f for f in forces if f > 0]):
        r, u = profile(root, force, fluid_seeds, radius)
        if not len(r):
            continue
        fit = fit_flow(r, u)
        ax_flow.plot(r, u / np.max(fit), "o", ms=3.8, color=color, label=title_force(force))
        ax_flow.plot(r, fit / np.max(fit), "-", lw=1.45, color=color)
    ax_flow.plot(np.linspace(0, 1, 200), 1 - np.linspace(0, 1, 200) ** 2, "--", color="0.15", lw=1.2, label="Poiseuille")
    ax_flow.set_xlabel(r"$r/R$")
    ax_flow.set_ylabel(r"$u_z/u_{\max}$")
    ax_flow.legend(frameon=False, loc="lower left")
    label(ax_flow, "A")
    clean(ax_flow)
    for i, structure in enumerate(structures[:3]):
        ax = fig.add_subplot(gs[0, i + 1])
        draw_snapshot(ax, root, structure, SNAP_FORCE, SNAP_SEED, radius, "xy")
        if i == 0:
            label(ax, "B")
    for i, structure in enumerate(structures[1:]):
        ax = fig.add_subplot(gs[1, i + 1])
        draw_snapshot(ax, root, structure, SNAP_FORCE, SNAP_SEED, radius, "xz")
        if i == 0:
            label(ax, "C")
    return save(fig, root, "流场与构型")


def figure_radial(root: Path, structures: list[str], forces: list[float], seeds: list[int], radius: float) -> Path:
    """图版 2：径向概率分布，按结构分面，按体力上色。"""
    bins = np.linspace(0, 1, 31)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True, sharey=True)
    for ax, structure in zip(axes.ravel(), structures):
        for color, force in zip(FLOW_COLORS, forces):
            rows: list[np.ndarray] = []
            for seed in seeds:
                data = read_series(root, structure, force, seed, radius)
                if data is None:
                    continue
                r = finite(analysis_window(data, metric(data, "r", radius)))
                if len(r):
                    hist, _ = np.histogram(r, bins=bins, density=True)
                    rows.append(hist)
            if rows:
                mean = np.mean(np.vstack(rows), axis=0)
                ax.fill_between(centers, mean, color=color, alpha=0.13, linewidth=0)
                ax.plot(centers, mean, color=color, lw=1.45, label=title_force(force))
        ax.set_title(title_structure(structure))
        ax.set_xlabel(r"$r_{\rm cm}/R$")
        ax.set_ylabel(r"$p(r_{\rm cm}/R)$")
        clean(ax)
    axes[0, 0].legend(frameon=False, ncol=1, loc="upper right")
    for tag, ax in zip("ABCD", axes.ravel()):
        label(ax, tag)
    fig.tight_layout()
    return save(fig, root, "径向分布")


def binned_xy(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 y 按 x 分箱取均值和标准误。"""
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full_like(centers, np.nan, dtype=float)
    err = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        values = y[mask]
        if len(values):
            mean[i] = float(np.mean(values))
            err[i] = 0.0 if len(values) == 1 else float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return centers, mean, err


def figure_conditioned(root: Path, structures: list[str], force: float, seeds: list[int], radius: float) -> Path:
    """图版 3：形变张量和取向随径向位置变化。"""
    bins = np.linspace(0, 0.45, 10)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharex=True)
    panels = [("Gperp", r"$G_\perp$"), ("Gzz", r"$G_{zz}$"), ("theta", r"$\theta$ (deg)")]
    for ax, (name, ylabel) in zip(axes, panels):
        for color, structure in zip(COLORS, structures):
            frames = collect_frames(root, structure, force, seeds, radius, ["r", name])
            x = frames["r"]
            y = frames[name]
            count = min(len(x), len(y))
            if not count:
                continue
            centers, mean, err = binned_xy(x[:count], y[:count], bins)
            ax.fill_between(centers, mean - err, mean + err, color=color, alpha=0.15, linewidth=0)
            ax.plot(centers, mean, "-", lw=1.65, color=color, label=title_structure(structure))
        ax.set_xlabel(r"$r_{\rm cm}/R$")
        ax.set_ylabel(ylabel)
        clean(ax)
    axes[0].legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.25))
    for tag, ax in zip("ABC", axes):
        label(ax, tag)
    fig.tight_layout()
    return save(fig, root, "条件形变")


def figure_maps(root: Path, structures: list[str], forces: list[float], seeds: list[int], radius: float) -> Path:
    """图版 4：结构-体力二维响应图。"""
    quantities = [
        ("r", r"$\langle r_{\rm cm}/R\rangle-\langle r_{\rm cm}/R\rangle_{g=0}$"),
        ("stretch", r"$\langle G_{zz}/G_\perp\rangle-\langle G_{zz}/G_\perp\rangle_{g=0}$"),
        ("asphericity", r"$\langle A\rangle-\langle A\rangle_{g=0}$"),
        ("clearance", r"$\langle h/R\rangle-\langle h/R\rangle_{g=0}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9), constrained_layout=True)
    for ax, (name, title) in zip(axes.ravel(), quantities):
        matrix = np.full((len(structures), len(forces)), np.nan)
        for i, structure in enumerate(structures):
            for j, force in enumerate(forces):
                matrix[i, j] = mean_metric(root, structure, force, seeds, radius, name)
        delta = matrix - matrix[:, [0]]
        vmax = np.nanmax(np.abs(delta))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if np.isfinite(vmax) and vmax > 0 else None
        image = ax.imshow(delta, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(forces)))
        ax.set_xticklabels([数值(f) for f in forces], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(structures)))
        ax.set_yticklabels([title_structure(s) for s in structures])
        ax.set_xlabel(r"Body force $g$")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                if np.isfinite(delta[i, j]):
                    ax.text(j, i, f"{delta[i, j]:.2g}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    for tag, ax in zip("ABCD", axes.ravel()):
        label(ax, tag)
    return save(fig, root, "二维响应")


def figure_distributions(root: Path, structures: list[str], force: float, seeds: list[int], radius: float) -> Path:
    """图版 5：高体力下的分布宽度和间歇性。"""
    quantities = [
        ("r", r"$r_{\rm cm}/R$"),
        ("stretch", r"$G_{zz}/G_\perp$"),
        ("theta", r"$\theta$ (deg)"),
        ("R99", r"$R_{99}/R$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.2))
    positions = np.arange(1, len(structures) + 1)
    for ax, (name, ylabel) in zip(axes, quantities):
        groups = []
        valid_positions = []
        valid_colors = []
        for index, structure in enumerate(structures):
            frames = collect_frames(root, structure, force, seeds, radius, [name])[name]
            frames = frames[np.isfinite(frames)]
            if len(frames) >= 2:
                groups.append(frames)
                valid_positions.append(positions[index])
                valid_colors.append(COLORS[index])
            elif len(frames) == 1:
                ax.plot(positions[index], frames[0], "o", color=COLORS[index], ms=5)
        if groups:
            violins = ax.violinplot(groups, positions=valid_positions, widths=0.7, showmeans=True, showextrema=False, showmedians=False)
            for body, color in zip(violins["bodies"], valid_colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.55)
            violins["cmeans"].set_color("black")
            violins["cmeans"].set_linewidth(1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels([title_structure(s) for s in structures])
        ax.set_ylabel(ylabel)
        clean(ax)
    for tag, ax in zip("ABCD", axes):
        label(ax, tag)
    fig.tight_layout()
    return save(fig, root, "分布宽度")


def figure_joint(root: Path, structures: list[str], force: float, seeds: list[int], radius: float) -> Path:
    """图版 6：径向位置和形变的二维关系。"""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True, sharey=True)
    for ax, structure in zip(axes.ravel(), structures):
        frames = collect_frames(root, structure, force, seeds, radius, ["r", "stretch"])
        x = frames["r"]
        y = frames["stretch"]
        count = min(len(x), len(y))
        if count:
            hist = ax.hist2d(x[:count], y[:count], bins=[12, 12], cmap="magma", cmin=1)
            fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.03, label="Counts")
        ax.set_title(title_structure(structure))
        ax.set_xlabel(r"$r_{\rm cm}/R$")
        ax.set_ylabel(r"$G_{zz}/G_\perp$")
        clean(ax)
    for tag, ax in zip("ABCD", axes.ravel()):
        label(ax, tag)
    fig.tight_layout()
    return save(fig, root, "二维分布")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=RADIUS)
    parser.add_argument("--structures", nargs="+", default=STRUCTURES)
    parser.add_argument("--forces", nargs="+", type=float, default=FORCES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--fluid-seeds", nargs="+", type=int, default=FLUID_SEEDS)
    parser.add_argument("--snap-force", type=float, default=SNAP_FORCE)
    parser.add_argument("--snap-seed", type=int, default=SNAP_SEED)
    args = parser.parse_args()

    root = Path.cwd()
    setup_style()
    outputs = [
        figure_reference(root, args.structures, args.forces, args.fluid_seeds, args.radius),
        figure_radial(root, args.structures, args.forces, args.seeds, args.radius),
        figure_conditioned(root, args.structures, args.snap_force, args.seeds, args.radius),
        figure_maps(root, args.structures, args.forces, args.seeds, args.radius),
        figure_distributions(root, args.structures, args.snap_force, args.seeds, args.radius),
        figure_joint(root, args.structures, args.snap_force, args.seeds, args.radius),
    ]
    for path in outputs:
        print(f"[图版] 输出 {path}")


if __name__ == "__main__":
    main()
