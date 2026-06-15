from __future__ import annotations

"""画空管泊肃叶流验证图。

这个脚本只做一件事：读取纯流体轨迹的径向速度剖面，
检查模拟得到的速度场是否符合泊肃叶流形状。

可以改什么：
- DEFAULT_FORCES：要画哪些真实体力。0 体力只作为基线，不参与拟合。
- DEFAULT_SEEDS：要合并哪些独立重复。
- DEFAULT_RADIUS：管半径，必须和结果目录一致。

输入：
- results/半径36_空管_体力<g>_种子<seed>/profiles.npz

输出：
- figures/流场/MM-DD-HH-MM.png
"""

# argparse 用来读取命令行参数。
import argparse
# datetime 用来生成月日时分格式图片名。
from datetime import datetime, timedelta, timezone
# Path 用来拼接文件路径。
from pathlib import Path

# matplotlib 负责画图。
import matplotlib.pyplot as plt
# numpy 负责数组计算和拟合。
import numpy as np


# 默认画旧项目同量级的真实体力；0 用来显示无流基线。
DEFAULT_FORCES = [0.0, 0.001, 0.003, 0.005, 0.01]
# 纯流体 seed。
DEFAULT_SEEDS = [301, 302, 303]
# 管半径。
DEFAULT_RADIUS = 36.0
# 色盲友好的 Okabe-Ito 配色。
COLORS = {0.0: "#555555", 0.001: "#0072B2", 0.003: "#D55E00", 0.005: "#009E73", 0.01: "#CC79A7"}


def 北京时间戳() -> str:
    """返回北京时间的月日时分字符串。

    服务器系统时间可能是 UTC。
    图片文件名用北京时间，和本地查看时间保持一致。
    """
    return datetime.now(timezone(timedelta(hours=8))).strftime("%m-%d-%H-%M")


def setup() -> None:
    """设置接近期刊论文的简洁画图风格。"""
    # 字体用 Times 风格；字号比普通默认值大，但不夸张。
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
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.05,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )


def tag(value: float) -> str:
    """把数值转成目录和图例里使用的短标签。"""
    return f"{value:g}"


def profile_path(root: Path, force: float, seed: int, radius: float) -> Path:
    """返回某个纯流体任务的剖面文件路径。"""
    # 中文目录名直接写出半径、真实体力和 seed。
    cn_path = root / "results" / f"半径{radius:g}_空管_体力{tag(force)}_种子{seed}" / "profiles.npz"
    if cn_path.exists():
        return cn_path
    # 兼容旧英文目录名。
    pipe = f"r{radius:g}"
    new_path = root / "results" / f"{pipe}_fluid_g{tag(force)}_s{seed}" / "profiles.npz"
    if new_path.exists():
        return new_path
    # 兼容旧目录名，避免已有短测试数据立刻失效。
    return root / "results" / f"{pipe}_fluid_f{tag(force)}_s{seed}" / "profiles.npz"


def load_profiles(root: Path, force: float, seeds: list[int], radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取同一真实体力下所有 seed 的速度剖面。

    返回值：
    - r：无量纲径向坐标 r/R。
    - mean：seed 平均轴向速度。
    - ci：seed 间 95% 置信区间半宽。
    """
    # curves 保存每个 seed 的速度曲线。
    curves: list[np.ndarray] = []
    # r_grid 保存径向坐标，只需要读取一次。
    r_grid: np.ndarray | None = None

    # 逐个 seed 读取 profiles.npz。
    for seed in seeds:
        path = profile_path(root, force, seed, radius)
        if not path.exists():
            continue
        data = np.load(path)
        r = np.asarray(data["r"], dtype=float) / radius
        vz = np.asarray(data["mean_vz"], dtype=float)
        if r_grid is None:
            r_grid = r
        if len(vz) == len(r_grid):
            curves.append(vz)

    # 如果没有读到数据，返回空数组。
    if r_grid is None or not curves:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    # 每行是一条 seed 曲线。
    arr = np.vstack(curves)
    # seed 平均曲线。
    mean = arr.mean(axis=0)
    # seed 间 95% 置信区间；只有一个 seed 时无法估计，记为 0。
    if arr.shape[0] > 1:
        ci = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        ci = np.zeros_like(mean)
    return r_grid, mean, ci


def fit_no_slip(r: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
    """拟合严格无滑移泊肃叶形状 u=A(1-r^2)。"""
    # shape 是圆管无滑移泊肃叶流的无量纲形状。
    shape = 1.0 - r**2
    # 只使用有限数值点。
    valid = np.isfinite(r) & np.isfinite(u) & np.isfinite(shape)
    # 单参数最小二乘拟合 A。
    amp = float(np.sum(u[valid] * shape[valid]) / np.sum(shape[valid] ** 2))
    return amp, amp * shape


def fit_with_slip(r: np.ndarray, u: np.ndarray) -> tuple[float, float, np.ndarray]:
    """拟合带近壁速度偏置的泊肃叶形状 u=A(1-r^2)+u_s。

    这里的 u_s 不是主动引入的物理假设，而是用来量化当前 MPCD 管壁
    在有限分辨率下表现出的近壁速度偏置。若 u_s 接近 0，结果退化为
    标准无滑移泊肃叶流。
    """
    # 第一列是泊肃叶形状，第二列是常数偏置。
    x = np.vstack([1.0 - r**2, np.ones_like(r)]).T
    # 最小二乘得到 A 和 u_s。
    amp, slip = np.linalg.lstsq(x, u, rcond=None)[0]
    return float(amp), float(slip), amp * (1.0 - r**2) + slip


def save(fig: plt.Figure, outdir: Path) -> Path:
    """保存月日时分命名的 PNG 图片。"""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{北京时间戳()}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--forces", nargs="+", type=float, default=DEFAULT_FORCES)
    parser.add_argument("--flows", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    root = Path.cwd()
    setup()

    # 两联图：左边看原始速度，右边看扣除近壁偏置后的归一化泊肃叶塌缩。
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax_raw, ax_norm = axes
    # 用更密的坐标画理论线。
    r_line = np.linspace(0.0, 1.0, 300)

    # 记录拟合摘要，方便终端检查。
    report: list[str] = []

    forces = args.flows if args.flows is not None else args.forces

    for force in forces:
        r, mean, ci = load_profiles(root, force, args.seeds, args.radius)
        if not len(r):
            continue
        color = COLORS.get(float(force), "#000000")

        # 零流只画基线，不参与泊肃叶拟合。
        if float(force) == 0.0:
            ax_raw.plot(r, mean, "o", ms=3.7, color=color, label=r"$g=0$")
            ax_raw.fill_between(r, mean - ci, mean + ci, color=color, alpha=0.12, linewidth=0)
            continue

        # 严格无滑移拟合，用来判断和理想泊肃叶曲线的偏差。
        amp0, fit0 = fit_no_slip(r, mean)
        # 带近壁速度偏置的拟合，用来量化 slip/偏置后是否仍是泊肃叶形状。
        amp, slip, fit = fit_with_slip(r, mean)
        # 等效 Navier slip length 的无量纲估计：u_s = 2 A b/R。
        slip_length = slip / (2.0 * amp) if amp != 0.0 else float("nan")
        # 记录拟合误差。
        rmse0 = float(np.sqrt(np.mean((mean - fit0) ** 2)))
        rmse = float(np.sqrt(np.mean((mean - fit) ** 2)))
        report.append(
            f"g={tag(force)}  no-slip RMSE={rmse0:.4g}  "
            f"slip-fit RMSE={rmse:.4g}  u_s={slip:.4g}  b/R={slip_length:.4g}"
        )

        # 左图：原始速度剖面，点为模拟，实线为带偏置泊肃叶拟合。
        label = rf"$g={tag(force)}$"
        ax_raw.plot(r, mean, "o", ms=4.0, color=color, label=label)
        ax_raw.fill_between(r, mean - ci, mean + ci, color=color, alpha=0.13, linewidth=0)
        ax_raw.plot(r, fit, "-", lw=1.55, color=color)
        # 只给一个流强画淡灰无滑移参考线，避免图面过乱。
        if float(force) == max([f for f in forces if f > 0.0]):
            ax_raw.plot(r, fit0, "--", lw=1.05, color="0.45", label="no-slip fit")

        # 右图：扣掉近壁偏置并归一化后，应塌缩到 1-(r/R)^2。
        y = (mean - slip) / amp
        yerr = ci / abs(amp)
        ax_norm.plot(r**2, y, "o", ms=4.0, color=color, label=label)
        ax_norm.fill_between(r**2, y - yerr, y + yerr, color=color, alpha=0.13, linewidth=0)

    # 右图理论直线，同行论文里常用这种 collapse 来证明泊肃叶形状。
    ax_norm.plot(r_line**2, 1.0 - r_line**2, "k--", lw=1.25, label=r"$1-(r/R)^2$")

    # 左图坐标和图例。
    ax_raw.set_xlabel(r"Radial position $r/R$")
    ax_raw.set_ylabel(r"Axial velocity $u_z$")
    ax_raw.legend(frameon=False, loc="upper right", handlelength=2.0)

    # 右图坐标和图例。
    ax_norm.set_xlabel(r"Squared radius $(r/R)^2$")
    ax_norm.set_ylabel(r"Normalized velocity $(u_z-u_s)/A$")
    ax_norm.legend(frameon=False, loc="upper right", handlelength=2.0)

    # 面板标签，期刊图常用 A/B 而不是大标题。
    for label, ax in zip(["A", "B"], axes):
        ax.text(
            0.03,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.tick_params(which="both", top=True, right=True)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout(w_pad=1.4)
    out = save(fig, root / "figures" / "流场")
    print(f"[流场] 输出 {out}")
    for line in report:
        print(f"[流场] {line}")


if __name__ == "__main__":
    main()
