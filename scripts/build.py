# 让类型注解在运行时不立即求值，减少低版本 Python 的类型兼容问题。
from __future__ import annotations

"""生成整体三维网格凝胶结构。

这个脚本只做一件事：按 n x n x n 网格生成凝胶结构。
相邻网格单元共享同一个交联点。

输入：config/base.json
输出：
- data/structures/g<n>.json
- tables/structures/metrics.csv
"""

# csv 用来写结构指标表。
import csv
# json 用来读写结构化配置和结构文件，例如 config/base.json。
import json
# math 提供 sqrt、pi 等数学函数。
import math
# Path 比普通字符串路径更安全，能跨平台拼接路径。
from pathlib import Path
# Any 表示“任意类型”，用于配置字典这种混合数据。
from typing import Any

# Vec3 是三维向量类型别名，等价于 tuple[float, float, float]。
Vec3 = tuple[float, float, float]


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 配置。"""
    # read_text 读取整个文本文件；encoding="utf-8" 保证中文不乱码。
    # json.loads 把 JSON 字符串转换成 Python 字典。
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写出 JSON。"""
    # parent 是父目录；mkdir(..., exist_ok=True) 表示目录不存在就创建，存在也不报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 表示中文直接写成中文，不转义成 \uXXXX。
    # indent=2 表示缩进 2 个空格，方便人工检查。
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV。"""
    # 先保证输出目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # CSV 必须至少有一行，才能从第一行拿到列名。
    if not rows:
        raise ValueError(f"没有可写入的行: {path}")
    # newline="" 是 csv 模块推荐写法，避免 Windows 下多出空行。
    # utf-8-sig 会写入 BOM，Excel 打开中文表头更稳。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        # fieldnames 决定 CSV 列顺序，这里直接用第一行字典的键。
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        # 先写表头，再写所有数据行。
        writer.writeheader()
        writer.writerows(rows)


def add(a: Vec3, b: Vec3) -> Vec3:
    """向量相加。"""
    # 三维向量逐分量相加。
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: Vec3, s: float) -> Vec3:
    """向量乘以常数。"""
    # 三维向量每个分量都乘以同一个数 s。
    return (a[0] * s, a[1] * s, a[2] * s)


def norm(a: Vec3) -> float:
    """向量长度。"""
    # 欧几里得长度 sqrt(x^2 + y^2 + z^2)。
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def center_positions(beads: list[dict[str, Any]]) -> None:
    """把结构质心移到原点。"""
    # 珠子数，用来计算平均坐标。
    n = len(beads)
    # com 是 center of mass，这里所有珠子质量相同，所以就是坐标平均值。
    com = (
        sum(bead["r"][0] for bead in beads) / n,
        sum(bead["r"][1] for bead in beads) / n,
        sum(bead["r"][2] for bead in beads) / n,
    )
    # 每个珠子的坐标都减去质心坐标，使整个凝胶以原点为中心。
    for bead in beads:
        # round(..., 8) 保留 8 位小数，减少 JSON 文件里的浮点噪声。
        bead["r"] = [round(bead["r"][0] - com[0], 8), round(bead["r"][1] - com[1], 8), round(bead["r"][2] - com[2], 8)]


def add_bead(beads: list[dict[str, Any]], r: Vec3, bead_type: str) -> int:
    """添加一个珠子并返回编号。"""
    # 新珠子的编号就是当前列表长度；第一个珠子编号为 0。
    bead_id = len(beads)
    # 每个珠子记录 id、类型和三维坐标 r。
    beads.append({"id": bead_id, "type": bead_type, "r": [round(r[0], 8), round(r[1], 8), round(r[2], 8)]})
    # 返回编号，后面建键时要用。
    return bead_id


def add_chain(
    beads: list[dict[str, Any]],
    bonds: list[dict[str, int]],
    left_id: int,
    right_id: int,
    left_r: Vec3,
    right_r: Vec3,
    segment_count: int,
) -> None:
    """在两个交联点之间插入链珠并连接成一条链。

    segment_count 是两个交联点之间的键段数；例如 segment_count=5 表示中间插入 4 个链珠。
    """
    # segment_count 是“键段数”，至少为 1；为 1 时两个交联点直接相连。
    if segment_count < 1:
        raise ValueError("segment_count 必须至少为 1")
    # previous 始终表示当前链条最后一个珠子的编号。
    previous = left_id
    # range(1, segment_count) 会生成 1 到 segment_count-1，用来插入中间链珠。
    for index in range(1, segment_count):
        # t 是从左端到右端的插值比例。
        t = index / segment_count
        # 线性插值：r = (1-t)*left_r + t*right_r。
        r = add(scale(left_r, 1.0 - t), scale(right_r, t))
        # 添加一个普通链珠。
        bead_id = add_bead(beads, r, "chain")
        # 把前一个珠子和当前新珠子连成键。
        bonds.append({"i": previous, "j": bead_id})
        # 更新链条末端，准备连接下一个珠子。
        previous = bead_id
    # 最后把链条末端连接到右交联点。
    bonds.append({"i": previous, "j": right_id})


def build_grid(n: int, config: dict[str, Any]) -> dict[str, Any]:
    """生成整体 n x n x n 三维网格凝胶。

    网格交联点是全局共享的，所以相邻单元天然共用交联点，不会产生小块拼接时的重复节点。
    """
    # 取出结构配置块。
    geom = config["structure"]
    # 单个键的初始几何长度。
    # FENE-WCA 的真实键势由 run.py 设置；这里的 bond_equilibrium 只决定初始网格间距。
    bond_length = float(config["bead"]["bond_equilibrium"])
    # 相邻交联点之间有多少个键段。
    segment_count = int(geom["segments_per_edge"])
    # 一个网格单元的边长 = 键段数 * 单键长度。
    cell_edge = segment_count * bond_length

    # beads 保存所有珠子；bonds 保存所有键。
    beads: list[dict[str, Any]] = []
    bonds: list[dict[str, int]] = []
    # xlink_ids 把网格整数坐标映射到交联点珠子编号。
    xlink_ids: dict[tuple[int, int, int], int] = {}
    # xlink_pos 把网格整数坐标映射到真实三维坐标。
    xlink_pos: dict[tuple[int, int, int], Vec3] = {}

    # n 个网格单元在每个方向有 n+1 个节点。
    for ix in range(n + 1):
        for iy in range(n + 1):
            for iz in range(n + 1):
                # key 是交联点的整数网格坐标。
                key = (ix, iy, iz)
                # r 是交联点的真实空间坐标。
                r = (ix * cell_edge, iy * cell_edge, iz * cell_edge)
                # 记录交联点位置。
                xlink_pos[key] = r
                # 添加交联点珠子，并保存它的编号。
                xlink_ids[key] = add_bead(beads, r, "xlink")

    # 遍历所有交联点，沿 x/y/z 正方向连接相邻节点。
    for ix in range(n + 1):
        for iy in range(n + 1):
            for iz in range(n + 1):
                # left 是当前交联点。
                left = (ix, iy, iz)
                # 三个 delta 分别代表 x、y、z 正方向的邻居。
                for delta in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    # right 是当前方向上的相邻交联点。
                    right = (ix + delta[0], iy + delta[1], iz + delta[2])
                    # 如果 right 存在，说明没有越过网格边界，就连接一条链。
                    if right in xlink_ids:
                        add_chain(
                            beads,
                            bonds,
                            xlink_ids[left],
                            xlink_ids[right],
                            xlink_pos[left],
                            xlink_pos[right],
                            segment_count,
                        )

    # 把整个结构平移到质心在原点，方便放进管道中心。
    center_positions(beads)
    # 计算结构的基本几何指标。
    metrics = measure(beads, bonds, float(config["pipe"]["radius"]))
    # 返回一个完整结构字典，后面会写成 JSON。
    return {
        "name": f"g{n}",
        "n": n,
        "cell_count": n**3,
        "segments_per_edge": segment_count,
        "cell_edge": cell_edge,
        "beads": beads,
        "bonds": bonds,
        "metrics": metrics,
    }


def measure(beads: list[dict[str, Any]], bonds: list[dict[str, int]], pipe_radius: float) -> dict[str, Any]:
    """计算结构基本指标。"""
    # 把 JSON 里的坐标列表转成三维元组，方便计算。
    rs: list[Vec3] = [(float(b["r"][0]), float(b["r"][1]), float(b["r"][2])) for b in beads]
    # 回转半径平方 Rg^2 = 所有珠子到质心距离平方的平均值；此时质心已在原点。
    rg2 = sum(norm(r) ** 2 for r in rs) / len(rs)
    # radial 是每个珠子到管中心轴的径向距离 sqrt(x^2+y^2)。
    radial = sorted(math.sqrt(r[0] * r[0] + r[1] * r[1]) for r in rs)
    # 分位半径比最大半径更稳；最大值容易被一个角点支配。
    r95 = radial[min(len(radial) - 1, int(round(0.95 * (len(radial) - 1))))]
    r99 = radial[min(len(radial) - 1, int(round(0.99 * (len(radial) - 1))))]
    # 汇总结构指标。
    return {
        "pipe_radius": pipe_radius,
        "N_bead": len(beads),
        "N_bond": len(bonds),
        "N_xlink": sum(1 for bead in beads if bead["type"] == "xlink"),
        "N_chain_bead": sum(1 for bead in beads if bead["type"] == "chain"),
        "Rg": math.sqrt(rg2),
        "Rg_over_R": math.sqrt(rg2) / pipe_radius,
        "R95_over_R": r95 / pipe_radius,
        "R99_over_R": r99 / pipe_radius,
        "clearance_99_over_R": 1.0 - (r99 / pipe_radius),
    }


def main() -> None:
    """生成结构和结构指标表。"""
    # Path.cwd() 是当前运行命令所在目录；本项目要求在项目根目录运行。
    root = Path.cwd()
    # 读取总配置。
    config = read_json(root / "config" / "base.json")

    # metric_rows 用来汇总四个结构的几何指标。
    metric_rows: list[dict[str, Any]] = []
    # 按配置里的 n_values 逐个生成 G1--G4。
    for n in config["structure"]["n_values"]:
        # 生成一个整体 n x n x n 网格结构。
        structure = build_grid(int(n), config)
        # 写出结构文件，例如 data/structures/g3.json。
        write_json(root / "data" / "structures" / f"g{n}.json", structure)
        # 把结构名和指标合并成一行，等待写入 metrics.csv。
        metric_rows.append({"structure": f"g{n}", **structure["metrics"]})

    # 写出结构指标表。
    write_csv(root / "tables" / "structures" / "metrics.csv", metric_rows)
    # 给终端一个明确提示，说明生成完成。
    print("生成完成: data/structures/*.json, tables/structures/metrics.csv")


if __name__ == "__main__":
    # 只有直接运行 python scripts/build.py 时才执行 main。
    # 如果这个文件被其他脚本 import，则不会自动生成文件。
    main()
