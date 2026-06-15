from __future__ import annotations

"""根据空管泊肃叶流结果估计管壁半径修正。

这个脚本只做一件事：读取已经跑完的纯流体 profiles.npz，
拟合速度剖面，估计 HOOMD 几何半径应该比分析半径小多少。

可以改什么：
- DEFAULT_FORCES：用于校准的真实体力列表。
- DEFAULT_SEEDS：用于平均的 seed。
- DEFAULT_CONFIG：配置文件路径。

输入：
- results/半径36_空管_体力<g>_种子<seed>/profiles.npz

输出：
- tables/analysis/wall.json

运行例子：
- /home/zhangxh/students/sjl/miniconda3/envs/A/bin/python scripts/校准.py
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path("config/base.json")
DEFAULT_FORCES = [0.001, 0.003, 0.005, 0.01]
DEFAULT_SEEDS = [301, 302, 303]


def read_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def radius_from_config(config: dict) -> float:
    """读取分析半径 R。"""
    pipe = config["pipe"]
    return float(pipe.get("analysis_radius", pipe.get("radius")))


def tag(value: float) -> str:
    """把数值变成目录名里的短标签。"""
    return f"{value:g}"


def profile_path(root: Path, radius: float, force: float, seed: int) -> Path:
    """返回一条空管结果的 profiles.npz 路径。"""
    cn_path = root / "results" / f"半径{radius:g}_空管_体力{tag(force)}_种子{seed}" / "profiles.npz"
    if cn_path.exists():
        return cn_path
    return root / "results" / f"r{radius:g}_fluid_g{tag(force)}_s{seed}" / "profiles.npz"


def load_mean_profile(root: Path, radius: float, force: float, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """读取同一体力下多个 seed 的平均速度剖面。"""
    curves: list[np.ndarray] = []
    r_grid: np.ndarray | None = None
    for seed in seeds:
        path = profile_path(root, radius, force, seed)
        if not path.exists():
            continue
        data = np.load(path)
        r = np.asarray(data["r"], dtype=float)
        u = np.asarray(data["mean_vz"], dtype=float)
        if r_grid is None:
            r_grid = r
        if len(u) == len(r_grid):
            curves.append(u)
    if r_grid is None or not curves:
        return np.asarray([]), np.asarray([]), 0
    return r_grid, np.vstack(curves).mean(axis=0), len(curves)


def fit_effective_radius(r: np.ndarray, u: np.ndarray, analysis_radius: float) -> dict:
    """拟合 u = A(R_h^2-r^2)，得到有效水动力半径 R_h。"""
    valid = np.isfinite(r) & np.isfinite(u)
    x = r[valid] ** 2
    y = u[valid]
    slope, intercept = np.polyfit(x, y, 1)
    amp = float(-slope)
    hydro_radius = float(math.sqrt(intercept / amp)) if amp > 0 and intercept > 0 else float("nan")
    slip = (hydro_radius * hydro_radius - analysis_radius * analysis_radius) / (2.0 * analysis_radius)
    fit = amp * (hydro_radius * hydro_radius - r**2)
    rmse = float(np.sqrt(np.mean((u - fit) ** 2)))
    ss = float(np.sum((u - np.mean(u)) ** 2))
    r2 = 1.0 - float(np.sum((u - fit) ** 2)) / ss if ss > 0 else float("nan")
    return {
        "amplitude": amp,
        "hydrodynamic_radius": hydro_radius,
        "slip_length": float(slip),
        "recommended_geometry_radius_offset": float(analysis_radius - hydro_radius),
        "rmse": rmse,
        "r2": r2,
    }


def save_json(path: Path, data: dict) -> None:
    """写出便于人工阅读的 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--forces", nargs="+", type=float, default=DEFAULT_FORCES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    root = Path.cwd()
    config = read_json(args.config)
    radius = radius_from_config(config)
    rows = []

    for force in args.forces:
        r, u, n_seed = load_mean_profile(root, radius, force, args.seeds)
        if n_seed == 0:
            print(f"[校准] 缺少 g={force:g} 的空管结果")
            continue
        fit = fit_effective_radius(r, u, radius)
        rows.append({"force": force, "n_seed": n_seed, **fit})
        print(
            f"[校准] g={force:g} R_h={fit['hydrodynamic_radius']:.5f} "
            f"offset={fit['recommended_geometry_radius_offset']:+.5f} R2={fit['r2']:.6f}"
        )

    if not rows:
        raise SystemExit("没有可用于校准的空管结果")

    corrections = np.asarray([row["recommended_geometry_radius_offset"] for row in rows], dtype=float)
    valid = np.isfinite(corrections)
    correction = float(np.median(corrections[valid]))
    current = float(config["pipe"].get("geometry_radius_offset", 0.0))
    final = current + correction
    out = {
        "analysis_radius": radius,
        "current_geometry_radius_offset": current,
        "remaining_geometry_radius_correction": correction,
        "recommended_geometry_radius_offset": final,
        "recommended_geometry_radius": radius + final,
        "forces": rows,
        "说明": "把 recommended_geometry_radius_offset 写入 config/base.json 后，重新跑空管验证。",
    }
    save_json(root / "tables" / "analysis" / "wall.json", out)
    print(f"[校准] 剩余修正 = {correction:+.6f}")
    print(f"[校准] 建议最终 geometry_radius_offset = {final:+.6f}")


if __name__ == "__main__":
    main()
