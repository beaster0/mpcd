from __future__ import annotations

"""统一结果目录命名。

这个文件只做一件事：把结构、体力、半径和 seed 转成清楚的中文目录名。
所有运行和画图脚本都从这里取路径，避免同一批数据出现多套命名。
"""

from pathlib import Path


def 数值(value: float) -> str:
    """把数值转成简短字符串。"""
    return f"{value:g}"


def 结构名(structure: str) -> str:
    """把内部结构名 g1/g2/g3/g4 转成论文和目录里的 n1/n2/n3/n4。"""
    if structure.startswith("g") and structure[1:].isdigit():
        return "n" + structure[1:]
    return structure


def 目录名(radius: float, structure: str, force: float, seed: int, stage: str = "flow") -> str:
    """生成中文结果目录名。"""
    半径 = f"半径{数值(radius)}"
    种子 = f"种子{seed}"
    if stage == "time":
        return f"{半径}_时间尺度_结构{结构名(structure)}_{种子}"
    if structure == "fluid":
        return f"{半径}_空管_体力{数值(force)}_{种子}"
    return f"{半径}_结构{结构名(structure)}_体力{数值(force)}_{种子}"


def 结果目录(root: Path, radius: float, structure: str, force: float, seed: int, stage: str = "flow") -> Path:
    """返回一条结果目录路径。"""
    return root / "results" / 目录名(radius, structure, force, seed, stage)


def 时间序列(root: Path, radius: float, structure: str, force: float, seed: int) -> Path:
    """返回凝胶 timeseries.npz 路径。"""
    正式 = 结果目录(root, radius, structure, force, seed) / "timeseries.npz"
    if 正式.exists():
        return 正式
    零流 = 结果目录(root, radius, structure, 0.0, seed, "time") / "timeseries.npz"
    if force == 0.0 and 零流.exists():
        return 零流
    return 正式


def 流场剖面(root: Path, radius: float, force: float, seed: int) -> Path:
    """返回空管 profiles.npz 路径。"""
    return 结果目录(root, radius, "fluid", force, seed) / "profiles.npz"
