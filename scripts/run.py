# 让类型注解延迟求值；这样 Python 不会在导入时立即解析所有类型。
from __future__ import annotations

"""运行单批网格凝胶 MPCD 任务。

输入：
- config/base.json：物理参数、结构列表、流强、随机种子和服务器设置。
- data/tasks_timescale.csv 或 data/tasks_production.csv：任务表。
- data/structures/*.json：由 scripts/build.py 生成的整体三维网格凝胶结构。

输出：
- results/<run_id>/status.json：任务状态。
- results/<run_id>/summary.json：任务摘要。
- results/<run_id>/state.npz：末态坐标和速度。
- results/<run_id>/timeseries.npz：凝胶质心、回转张量、形状、取向和壁面间隙时间序列。

这个文件只负责运行任务。结构怎么生成、任务怎么列、状态怎么汇总，分别放在 build.py
和 status.py 里，避免一个脚本承担太多职责。
"""

# argparse 用来解析命令行参数，例如 --only 和 --steps。
import argparse
# csv 用来读取任务表。
import csv
# json 用来读写配置、状态和摘要文件。
import json
# math 提供 sqrt、acos、pi 等数学函数。
import math
# traceback 用来在任务失败时保存完整报错堆栈，方便定位问题。
import traceback
# Path 用来处理文件路径。
from pathlib import Path
# Any 表示任意类型，用于 HOOMD 对象和混合配置字典。
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    # path.read_text 读取文本；json.loads 把 JSON 字符串变成 Python 字典。
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写出缩进 JSON，便于直接人工检查。"""
    # 写文件前先保证父目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 保留中文；indent=2 让 JSON 更容易阅读。
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_tasks(path: Path) -> list[dict[str, str]]:
    """读取任务表。"""
    # utf-8-sig 能正确读取 build.py 写出的带 BOM CSV。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # DictReader 把每一行任务变成字典。
        return list(csv.DictReader(handle))


def pipe_volume(radius: float, length: float) -> float:
    """圆管体积 V = pi R^2 L。"""
    # 圆柱体积公式：底面积 pi R^2 乘以长度 L。
    return math.pi * radius * radius * length


def solvent_count(config: dict[str, Any]) -> int:
    """由数密度和圆管体积计算 MPCD 溶剂粒子数。"""
    # 取出管道参数。
    pipe = config["pipe"]
    # MPCD 数密度，单位是每单位体积的溶剂粒子数。
    density = float(config["mpcd"]["number_density"])
    # 粒子数 = 数密度 * 管道体积；round 四舍五入成整数。
    count = int(round(density * pipe_volume(float(pipe["radius"]), float(pipe["length"]))))
    # 粒子数不能为 0 或负数，否则模拟没有物理意义。
    if count <= 0:
        raise ValueError("MPCD 溶剂粒子数必须为正数")
    # 返回最终溶剂粒子数。
    return count


def mpcd_time_scales(config: dict[str, Any]) -> dict[str, float]:
    """计算 MPCD 关键时间尺度。

    这些量写入 summary.json，方便确认配置是否处在成熟 MPCD 常用范围。
    """
    # 基础 MD 时间步。
    dt = float(config["mpcd"]["dt"])
    # MPCD streaming 间隔对应的物理时间。
    stream_time = dt * int(config["mpcd"]["stream_period"])
    # MPCD collision 间隔对应的物理时间。
    collision_time = dt * int(config["mpcd"]["collision_period"])
    # 热速度尺度 sqrt(kT/m)，这里 kB=1。
    thermal_speed = math.sqrt(float(config["mpcd"]["temperature"]) / float(config["mpcd"]["mass"]))
    # 平均自由程估计 lambda = thermal_speed * collision_time。
    mean_free_path = thermal_speed * collision_time
    return {
        "dt": dt,
        "stream_time": stream_time,
        "collision_time": collision_time,
        "thermal_speed": thermal_speed,
        "mean_free_path": mean_free_path,
    }


def run_dir(root: Path, run_id: str) -> Path:
    """单条任务的输出目录。"""
    # 每条任务一个独立结果目录，避免互相覆盖。
    return root / "results" / run_id


def status_path(root: Path, run_id: str) -> Path:
    """单条任务状态文件路径。"""
    # status.json 是监控脚本最先读取的文件。
    return run_dir(root, run_id) / "status.json"


def force_from_wi(config: dict[str, Any], wi: float) -> tuple[float, float, float]:
    """把计划流强标签换成体力。

    现在先用配置文件里的线性比例。真正完成空管标定后，只需要改 config/base.json
    里的 force_per_wi，不必改代码。
    """
    # 零流强时不施加体力。
    if wi <= 0.0:
        return (0.0, 0.0, 0.0)
    # 体力沿 z 方向施加；x/y 方向为 0。
    return (0.0, 0.0, float(config["flow"]["force_per_wi"]) * wi)


def load_structure(root: Path, structure: str) -> dict[str, Any]:
    """读取整体网格凝胶结构。"""
    # structure 形如 g1/g2/g3/g4，对应 data/structures/g1.json 等文件。
    return read_json(root / "data" / "structures" / f"{structure}.json")


def structure_arrays(structure: dict[str, Any]) -> dict[str, Any]:
    """把结构 JSON 转成 HOOMD 需要的数组。"""
    # numpy 用来高效处理坐标、类型和键数组。
    import numpy as np  # type: ignore

    # beads 是珠子列表，bonds 是键列表。
    beads = structure["beads"]
    bonds = structure["bonds"]
    # HOOMD 粒子类型需要整数 typeid；这里 xlink=0，chain=1。
    type_map = {"xlink": 0, "chain": 1}
    # 返回 HOOMD Snapshot 可以直接使用的 numpy 数组。
    return {
        # 所有珠子的三维坐标。
        "position": np.asarray([bead["r"] for bead in beads], dtype=np.float32),
        # 每个珠子的类型编号。
        "typeid": np.asarray([type_map[bead["type"]] for bead in beads], dtype=np.uint32),
        # 每个键连接的两个珠子编号。
        "bond": np.asarray([[bond["i"], bond["j"]] for bond in bonds], dtype=np.uint32),
        # 珠子总数。
        "n_bead": len(beads),
        # 键总数。
        "n_bond": len(bonds),
    }


def solvent_arrays(config: dict[str, Any], seed: int, count: int) -> dict[str, Any]:
    """生成圆管内均匀分布的 MPCD 溶剂。"""
    # numpy 用于生成随机数和数组。
    import numpy as np  # type: ignore

    # 管道半径和长度决定溶剂所在空间。
    pipe = config["pipe"]
    radius = float(pipe["radius"])
    length = float(pipe["length"])
    # 温度决定初始 Maxwell 速度分布的宽度。
    temperature = float(config["mpcd"]["temperature"])
    # 使用 seed 固定随机数，保证同一任务可重复。
    rng = np.random.default_rng(seed)

    # 在圆截面内均匀撒点：半径要用 sqrt(random)，否则中心和边缘密度不均匀。
    rho = radius * np.sqrt(rng.random(count)) * 0.98
    # theta 是圆截面内的极角，范围 0 到 2pi。
    theta = 2.0 * math.pi * rng.random(count)
    # z 在管长范围内均匀分布，中心在 0。
    z = (rng.random(count) - 0.5) * length
    # 把极坐标 rho/theta 转成 x/y，再和 z 拼成三维坐标。
    position = np.column_stack([rho * np.cos(theta), rho * np.sin(theta), z])
    # 速度按正态分布初始化，方差由温度决定。
    velocity = rng.normal(0.0, math.sqrt(temperature), size=(count, 3))
    # 去掉整体平移速度，避免初始溶剂整体漂移。
    velocity -= velocity.mean(axis=0)
    # 返回位置和速度数组。
    return {"position": position, "velocity": velocity}


def make_snapshot(root: Path, config: dict[str, Any], task: dict[str, str], n_solvent: int) -> Any:
    """创建 HOOMD 初态；纯流体任务没有凝胶珠子。"""
    # hoomd 是实际模拟引擎。
    import hoomd  # type: ignore
    # numpy 用于构造 HOOMD 需要的数组。
    import numpy as np  # type: ignore

    # 读取管道参数。
    pipe = config["pipe"]
    radius = float(pipe["radius"])
    length = float(pipe["length"])
    # 每条任务有自己的随机种子。
    seed = int(task["seed"])

    # Snapshot 是 HOOMD 的初始状态容器。
    snapshot = hoomd.Snapshot()
    # HOOMD 使用长方体模拟盒；x/y 比管径略大，z 为管长。
    snapshot.configuration.box = [2.0 * (radius + 1.0), 2.0 * (radius + 1.0), length, 0.0, 0.0, 0.0]

    # 纯流体任务不放凝胶珠子，只保留粒子类型名称。
    if task["structure"] == "fluid":
        snapshot.particles.N = 0
        snapshot.particles.types = ["xlink", "chain"]
    else:
        # 凝胶任务读取对应结构文件并转成数组。
        arrays = structure_arrays(load_structure(root, task["structure"]))
        # 设置凝胶珠子数。
        snapshot.particles.N = arrays["n_bead"]
        # 声明两类珠子：交联点和链珠。
        snapshot.particles.types = ["xlink", "chain"]
        # 写入每个珠子的类型编号。
        snapshot.particles.typeid[:] = arrays["typeid"]
        # 写入每个珠子的初始坐标。
        snapshot.particles.position[:] = arrays["position"]
        # 凝胶珠子质量 = mass_ratio * 溶剂粒子质量。
        # mass_ratio 写在 config/base.json 里，方便统一调整，不要在代码里写死。
        bead_mass = float(config["bead"].get("mass_ratio", 1.0)) * float(config["mpcd"]["mass"])
        snapshot.particles.mass[:] = bead_mass * np.ones(arrays["n_bead"], dtype=np.float32)
        # 直径设为 1，主要用于 HOOMD 粒子属性完整性。
        snapshot.particles.diameter[:] = np.ones(arrays["n_bead"], dtype=np.float32)
        # 初始凝胶速度设为 0，后续由 MPCD/MD 积分发展。
        snapshot.particles.velocity[:] = np.zeros((arrays["n_bead"], 3), dtype=np.float32)
        # 设置键数量。
        snapshot.bonds.N = arrays["n_bond"]
        # 所有键使用同一种 gel 键类型。
        snapshot.bonds.types = ["gel"]
        # 所有键的类型编号都是 0。
        snapshot.bonds.typeid[:] = np.zeros(arrays["n_bond"], dtype=np.uint32)
        # 写入每条键连接的两个珠子编号。
        snapshot.bonds.group[:] = arrays["bond"]

    # 生成 MPCD 溶剂初态。
    solvent = solvent_arrays(config, seed, n_solvent)
    # 设置 MPCD 溶剂粒子数。
    snapshot.mpcd.N = n_solvent
    # MPCD 溶剂只有一种类型。
    snapshot.mpcd.types = ["solvent"]
    # 写入溶剂位置。
    snapshot.mpcd.position[:] = solvent["position"]
    # 写入溶剂速度。
    snapshot.mpcd.velocity[:] = solvent["velocity"]
    # 所有溶剂粒子类型编号为 0。
    snapshot.mpcd.typeid[:] = np.zeros(n_solvent, dtype=np.uint32)
    # 设置溶剂粒子质量。
    snapshot.mpcd.mass = float(config["mpcd"]["mass"])
    # 返回完整初态。
    return snapshot


def gel_forces(config: dict[str, Any], hoomd_module: Any) -> list[Any]:
    """凝胶键势和短程排斥势。"""
    # 取出珠子相互作用参数。
    bead = config["bead"]
    # Cell 邻居表用于快速找短程相互作用对；exclusions=("bond",) 表示有键连接的邻居不算 LJ。
    nlist = hoomd_module.md.nlist.Cell(buffer=float(bead["neighbor_buffer"]), exclusions=("bond",))
    # LJ 势用 shift 模式，并把截断设在 WCA 截断处，得到纯排斥势。
    lj = hoomd_module.md.pair.LJ(nlist=nlist, default_r_cut=0.0, mode="shift")
    # xlink 和 chain 两类珠子两两之间都用同一套排斥参数。
    for left in ["xlink", "chain"]:
        for right in ["xlink", "chain"]:
            # epsilon 是能量尺度，sigma 是长度尺度。
            lj.params[(left, right)] = {"epsilon": float(bead["wca_epsilon"]), "sigma": float(bead["wca_sigma"])}
            # 2^(1/6)*sigma 是 LJ 势最低点，截断在这里就是 WCA 排斥。
            lj.r_cut[(left, right)] = (2.0 ** (1.0 / 6.0)) * float(bead["wca_sigma"])

    # FENE-WCA 是当前项目唯一使用的键模型。
    # 它限制键不能无限拉长，比普通谐和键更适合 bead-spring 聚合物网络。
    if bead["model"] != "fene":
        raise ValueError("当前项目只支持 bead.model = 'fene'")
    bond = hoomd_module.md.bond.FENEWCA()
    bond.params["gel"] = {
        "k": float(bead["fene_k"]),
        "r0": float(bead["fene_r0"]),
        "epsilon": float(bead["wca_epsilon"]),
        "sigma": float(bead["wca_sigma"]),
        "delta": float(bead["fene_delta"]),
    }
    # 返回 HOOMD 积分器需要的力列表。
    return [bond, lj]


def save_state(path: Path, snapshot: Any) -> None:
    """保存凝胶末态。

    MPCD 溶剂粒子数由管半径、管长和数密度决定，默认不保存全量溶剂末态。
    流体信息由 profiles.npz 里的径向剖面记录。
    """
    # numpy 用来把 HOOMD 数组转成可保存数组。
    import numpy as np  # type: ignore

    # 确保输出目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed 写压缩 npz 文件，减少磁盘占用。
    np.savez_compressed(
        path,
        # 保存凝胶末态坐标。
        bead_position=np.asarray(snapshot.particles.position),
        # 保存凝胶末态速度。
        bead_velocity=np.asarray(snapshot.particles.velocity),
        # 保存凝胶珠子类型。
        bead_typeid=np.asarray(snapshot.particles.typeid),
    )


def unwrap_positions(snapshot: Any) -> Any:
    """用 HOOMD 的周期 image 得到展开坐标。

    凝胶沿 z 方向跨过周期边界时，wrapped 坐标会把同一个凝胶拆到盒子两端。
    回转张量、主轴和翻滚都必须用展开坐标计算。
    """
    # numpy 用来做数组广播计算。
    import numpy as np  # type: ignore

    # wrapped position 是 HOOMD 盒子内坐标。
    position = np.asarray(snapshot.particles.position, dtype=float)
    # image 记录粒子跨过周期边界的次数。
    image = np.asarray(snapshot.particles.image, dtype=float)
    # box[:3] 是 x/y/z 三个方向的盒子长度。
    box = np.asarray(snapshot.configuration.box[:3], dtype=float)
    # 展开坐标 = 盒内坐标 + 跨边界次数 * 盒长。
    return position + image * box


def gel_observation(snapshot: Any, radius: float) -> dict[str, Any]:
    """从当前凝胶构型计算一帧核心观测量。

    这些量直接对应后续论文图中的径向占据、形变、取向和近壁状态。
    只读取凝胶珠子，不保存溶剂全量坐标。
    """
    # numpy 用于矩阵、本征值和分位数计算。
    import numpy as np  # type: ignore

    # 用展开坐标计算形状，避免周期边界把凝胶拆开。
    position = unwrap_positions(snapshot)
    # position.size 为 0 表示没有凝胶珠子，即纯流体任务。
    if position.size == 0:
        raise ValueError("纯流体任务没有凝胶观测量")

    # 质心坐标，所有珠子等质量，所以直接对坐标取平均。
    com = position.mean(axis=0)
    # rel 是每个珠子相对质心的位置。
    rel = position - com
    # 回转张量 G = rel^T rel / N，描述凝胶在三个方向上的空间展开。
    gyr = (rel.T @ rel) / len(position)
    # eigh 求对称矩阵本征值和本征向量；回转张量是对称矩阵。
    eigvals, eigvecs = np.linalg.eigh(gyr)
    # np.linalg.eigh 返回的本征值通常从小到大，但这里显式排序更稳。
    order = np.argsort(eigvals)
    # 排序后的本征值，后面最大本征值对应主轴。
    eigvals = eigvals[order]
    # 最大本征值对应的本征向量就是凝胶最长方向。
    main_axis = eigvecs[:, order[-1]]

    # Rg^2 等于回转张量三个本征值之和。
    rg2 = float(eigvals.sum())
    # max(rg2, 0) 防止极小负浮点误差导致 sqrt 报错。
    rg = math.sqrt(max(rg2, 0.0))
    # Gxx/Gyy/Gzz 是回转张量在实验室坐标系里的对角分量。
    gxx = float(gyr[0, 0])
    gyy = float(gyr[1, 1])
    gzz = float(gyr[2, 2])
    # 横向形变用 x 和 y 两个方向平均。
    gperp = 0.5 * (gxx + gyy)

    # 常用非球形度，球形时接近 0，越大表示越偏离球形。
    # lam1 >= lam2 >= lam3，分别表示最长、中间、最短主轴方向的形状方差。
    lam1, lam2, lam3 = (float(eigvals[2]), float(eigvals[1]), float(eigvals[0]))
    # rg2 > 0 时才能归一化；如果凝胶退化成一个点，就把非球形度记为 0。
    if rg2 > 0.0:
        # 非球形度公式：1.5*sum(lambda_i^2)/(sum lambda_i)^2 - 0.5。
        asphericity = 1.5 * (lam1 * lam1 + lam2 * lam2 + lam3 * lam3) / (rg2 * rg2) - 0.5
    else:
        asphericity = 0.0

    # 质心径向位置，只看 x-y 平面到管轴的距离。
    r_cm = math.sqrt(float(com[0] * com[0] + com[1] * com[1]))
    # 每个珠子到管轴的径向距离，用来计算外层分位数和壁面间隙。
    bead_r = np.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
    # 主轴与 z 方向夹角；abs 表示主轴正负方向等价。
    theta = math.acos(min(1.0, abs(float(main_axis[2]))))
    # 管壁间隙 = 管半径 - 最外层珠子径向距离。
    clearance = radius - float(bead_r.max())

    # 返回一帧观测量，后面会被写入 timeseries.npz。
    return {
        "com": com,
        "r_cm": r_cm,
        "Rg": rg,
        "Gxx": gxx,
        "Gyy": gyy,
        "Gzz": gzz,
        "Gperp": gperp,
        "lambda1": lam1,
        "lambda2": lam2,
        "lambda3": lam3,
        "lambda_gap": lam1 - lam2,
        "asphericity": float(asphericity),
        "axis": main_axis,
        "theta": theta,
        "R95": float(np.quantile(bead_r, 0.95)),
        "R99": float(np.quantile(bead_r, 0.99)),
        "max_bead_r": float(bead_r.max()),
        "wall_clearance": clearance,
    }


def save_timeseries(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """保存凝胶时间序列。

    一个时间点只写少量标量和主轴向量，便于后处理直接计算径向分布、形变和翻滚。
    """
    # numpy 用来把 Python 列表转成数组，并写 npz 文件。
    import numpy as np  # type: ignore

    # 写文件前确保目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 纯流体任务没有凝胶，写一个只有空 step 的 npz，保持文件结构统一。
    if not rows:
        np.savez_compressed(path, step=np.asarray([], dtype=np.int64), time=np.asarray([], dtype=float))
        return {"timeseries_npz": str(path), "samples": 0}

    # keys 是所有标量时间序列字段。
    keys = [
        "step",
        "time",
        "r_cm",
        "Rg",
        "Gxx",
        "Gyy",
        "Gzz",
        "Gperp",
        "lambda1",
        "lambda2",
        "lambda3",
        "lambda_gap",
        "asphericity",
        "theta",
        "R95",
        "R99",
        "max_bead_r",
        "wall_clearance",
    ]
    # 对每个标量字段，把每一帧的值收集成一维数组。
    arrays = {key: np.asarray([row[key] for row in rows]) for key in keys}
    # com 是三维向量，所以保存成 samples x 3 数组。
    arrays["com"] = np.asarray([row["com"] for row in rows])
    # axis 也是三维向量，表示每一帧的主轴方向。
    arrays["axis"] = np.asarray([row["axis"] for row in rows])
    # 写压缩 npz 文件。
    np.savez_compressed(path, **arrays)
    # 返回摘要信息，写入 summary.json。
    return {"timeseries_npz": str(path), "samples": len(rows)}


def new_profile_accumulator(config: dict[str, Any]) -> dict[str, Any]:
    """创建溶剂径向剖面累加器。"""
    # numpy 用于创建 bin 边界和累加数组。
    import numpy as np  # type: ignore

    # 管半径决定径向 bin 的最大值。
    radius = float(config["pipe"]["radius"])
    # fluid_radial_bins 是流体速度/密度剖面的径向分箱数量。
    bins = int(config["output"].get("fluid_radial_bins", config["output"].get("radial_bins", 24)))
    # edges 是 bin 边界，从 0 到 R，一共 bins+1 个点。
    edges = np.linspace(0.0, radius, bins + 1)
    # centers 是每个 bin 的中心位置，用于画图。
    centers = 0.5 * (edges[:-1] + edges[1:])
    # 返回累加器字典；count/vz/speed2 会随着采样不断累加。
    return {
        "edges": edges,
        "centers": centers,
        "count": np.zeros(bins, dtype=float),
        "vz": np.zeros(bins, dtype=float),
        "speed2": np.zeros(bins, dtype=float),
        "frames": 0,
    }


def add_profile_sample(acc: dict[str, Any], snapshot: Any) -> None:
    """把当前溶剂状态加入径向剖面时间平均。"""
    # numpy 用于计算径向距离和直方图。
    import numpy as np  # type: ignore

    # 取出所有 MPCD 溶剂粒子的位置。
    position = np.asarray(snapshot.mpcd.position)
    # 取出所有 MPCD 溶剂粒子的速度。
    velocity = np.asarray(snapshot.mpcd.velocity)
    # r 是溶剂粒子到管轴的径向距离。
    r = np.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
    # count 统计每个径向壳层里有多少溶剂粒子。
    count, _ = np.histogram(r, bins=acc["edges"])
    # vz_sum 统计每个壳层里 z 方向速度总和。
    vz_sum, _ = np.histogram(r, bins=acc["edges"], weights=velocity[:, 2])
    # speed2_sum 统计每个壳层里速度平方总和，只用于温控和异常速度检查。
    speed2_sum, _ = np.histogram(r, bins=acc["edges"], weights=np.sum(velocity * velocity, axis=1))
    # 把当前帧结果累加到总量里。
    acc["count"] += count
    acc["vz"] += vz_sum
    acc["speed2"] += speed2_sum
    # 记录已经累计了多少帧。
    acc["frames"] += 1


def save_profile_average(path: Path, acc: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """保存时间平均后的溶剂径向剖面。"""
    # numpy 用于安全除法和写 npz。
    import numpy as np  # type: ignore

    # count 是每个径向壳层累计粒子数。
    count = acc["count"]
    # mean_vz = 累计 vz / 累计粒子数；where=count>0 避免除以 0。
    mean_vz = np.divide(acc["vz"], count, out=np.zeros_like(acc["vz"], dtype=float), where=count > 0)
    # mean_speed2 = 累计速度平方 / 累计粒子数。
    # 它不是论文主结果，只用于检查温度和异常速度。
    mean_speed2 = np.divide(acc["speed2"], count, out=np.zeros_like(acc["speed2"], dtype=float), where=count > 0)
    # 径向壳层体积 = pi*(r_outer^2-r_inner^2)*L。
    # count 除以累计帧数和壳层体积，得到每单位体积的溶剂数密度。
    length = float(config["pipe"]["length"])
    shell_volume = math.pi * (acc["edges"][1:] ** 2 - acc["edges"][:-1] ** 2) * length
    number_density = np.divide(count, acc["frames"] * shell_volume, out=np.zeros_like(count, dtype=float), where=shell_volume > 0)
    # 确保输出目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 保存径向中心、粒子数、数密度、平均轴向速度、平均速度平方和采样帧数。
    np.savez_compressed(
        path,
        r=acc["centers"],
        count=count,
        number_density=number_density,
        mean_vz=mean_vz,
        mean_speed2=mean_speed2,
        frames=acc["frames"],
    )
    # 返回剖面摘要，写入 summary.json。
    return {
        "profile_npz": str(path),
        "radial_bins": len(count),
        "profile_frames": int(acc["frames"]),
        "min_bin_count": float(count.min()),
        "max_bin_count": float(count.max()),
        "min_number_density": float(number_density.min()),
        "max_number_density": float(number_density.max()),
    }


def run_with_sampling(sim: Any, n_steps: int, sample_interval: int, sample_gel: bool, radius: float, profile_path: Path, series_path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """分段推进模拟，并按固定间隔记录凝胶和流体剖面。

    每次采样后都写一次临时 `timeseries.npz`，长任务中途失败时也能保留已完成采样。
    """
    # rows 保存凝胶每个采样时刻的观测量。
    rows: list[dict[str, Any]] = []
    # profile_acc 保存溶剂剖面的时间累加量。
    profile_acc = new_profile_accumulator(config)
    # 采样间隔必须为正，否则 while 循环无法推进。
    if sample_interval <= 0:
        raise ValueError("sample_interval 必须为正数")

    # done 是已经完成的模拟步数。
    done = 0
    # 只要还没跑满 n_steps，就继续分段推进。
    while done < n_steps:
        # 最后一段可能不足 sample_interval，所以取 min。
        chunk = min(sample_interval, n_steps - done)
        # 推进 HOOMD 模拟 chunk 步。
        sim.run(chunk)
        # 更新已完成步数。
        done += chunk
        # 获取当前快照；这一步会把当前状态取到 Python 侧。
        snapshot = sim.state.get_snapshot()
        # 每个采样点都加入流体剖面平均。
        add_profile_sample(profile_acc, snapshot)
        # 只有凝胶任务才计算凝胶形状和取向。
        if sample_gel:
            # 计算一帧凝胶观测量，并把当前步数和真实时间也放进去。
            row = {"step": done, "time": done * float(config["mpcd"]["dt"]), **gel_observation(snapshot, radius)}
            # 追加到时间序列列表。
            rows.append(row)
            # 每次采样后立即落盘，长任务中断时也能保留已采到的数据。
            save_timeseries(series_path, rows)
    # 模拟结束后保存时间平均流体剖面。
    profile_summary = save_profile_average(profile_path, profile_acc, config)
    # 纯流体任务没有凝胶 rows，也写一个空 timeseries.npz。
    if not sample_gel:
        save_timeseries(series_path, rows)
    # 返回凝胶时间序列和流体剖面摘要。
    return rows, profile_summary


def close_sim(sim: Any) -> None:
    """显式清理 HOOMD 对象，避免解释器退出时打印无关析构异常。"""
    try:
        # 先断开 integrator，减少 HOOMD 退出时的析构噪声。
        sim.operations.integrator = None
    except Exception:
        # 清理失败不影响结果文件，所以忽略。
        pass
    try:
        # 某些 HOOMD 版本退出时会在 _operations 析构上打印异常；这里主动删除。
        delattr(sim, "_operations")
    except Exception:
        # 如果属性不存在，也直接忽略。
        pass


def run_task(root: Path, config: dict[str, Any], task: dict[str, str], steps: int | None) -> dict[str, Any]:
    """运行一条任务。"""
    # hoomd 是模拟引擎。
    import hoomd  # type: ignore
    # numpy 用于最后计算径向最大值等摘要。
    import numpy as np  # type: ignore

    # run_id 是任务唯一编号，例如 g4_w3_s105。
    run_id = task["run_id"]
    # out 是这条任务的结果目录。
    out = run_dir(root, run_id)
    # 创建结果目录。
    out.mkdir(parents=True, exist_ok=True)

    # 计算 MPCD 溶剂粒子数。
    n_solvent = solvent_count(config)
    # 计算并记录 MPCD 时间尺度。
    scales = mpcd_time_scales(config)
    # 如果命令行给了 --steps，就用临时步数；否则用任务表里的正式步数。
    n_steps = int(steps if steps is not None else task["steps"])
    # 管半径。
    radius = float(config["pipe"]["radius"])
    # MPCD 温度。
    temperature = float(config["mpcd"]["temperature"])
    # 把 Wi 标签换成 z 方向体力。
    body_force = force_from_wi(config, float(task["wi"]))

    # 任务开始前先写 running 状态，方便外部监控。
    write_json(status_path(root, run_id), {"run_id": run_id, "status": "running", "steps": n_steps})

    # 使用 GPU 设备；具体哪张卡由 CUDA_VISIBLE_DEVICES 控制。
    device = hoomd.device.GPU(notice_level=2)
    # 圆柱管道几何，outer_radius 是管半径，no_slip=True 表示无滑移边界。
    geometry = hoomd.mpcd.geometry.ConcentricCylinders(inner_radius=0.0, outer_radius=radius, no_slip=True)
    # BounceBack 是 MPCD 流动步骤，同时给溶剂施加恒定体力。
    stream = hoomd.mpcd.stream.BounceBack(
        period=int(config["mpcd"]["stream_period"]),
        geometry=geometry,
        mpcd_particle_force=hoomd.mpcd.force.ConstantForce(body_force),
    )
    # collision_method 的公共参数。
    collide_kwargs = {
        # 每隔多少步做一次 MPCD 碰撞。
        "period": int(config["mpcd"]["collision_period"]),
        # 随机旋转角，单位是度。
        "angle": float(config["mpcd"]["collision_angle_deg"]),
        # 温度。
        "kT": temperature,
    }
    # 凝胶任务需要把凝胶珠子作为 embedded particles 参与 MPCD 碰撞。
    if task["structure"] != "fluid":
        collide_kwargs["embedded_particles"] = hoomd.filter.All()
    # SRD 碰撞方法，是 MPCD 的核心碰撞步骤。
    collide = hoomd.mpcd.collide.StochasticRotationDynamics(**collide_kwargs)

    # methods 保存凝胶粒子和壁面的耦合方法。
    methods = []
    # forces 保存凝胶内部的键势和排斥势。
    forces = []
    # 只有凝胶任务才需要 MD 粒子方法和凝胶内部力。
    if task["structure"] != "fluid":
        # BounceBack 方法让凝胶粒子也遵守圆柱边界。
        methods.append(hoomd.mpcd.methods.BounceBack(filter=hoomd.filter.All(), geometry=geometry))
        # 构建凝胶键势和短程排斥势。
        forces = gel_forces(config, hoomd)

    # MPCD Integrator 把 MD 粒子方法、内部力、流动步骤和碰撞步骤组合起来。
    integrator = hoomd.mpcd.Integrator(
        dt=float(config["mpcd"]["dt"]),
        methods=methods,
        forces=forces,
        streaming_method=stream,
        collision_method=collide,
    )

    # 创建模拟对象；seed 控制 HOOMD 内部随机过程。
    sim = hoomd.Simulation(device=device, seed=int(task["seed"]))
    # 从初态 snapshot 创建模拟状态。
    sim.create_state_from_snapshot(make_snapshot(root, config, task, n_solvent))
    # 把积分器挂到模拟对象上。
    sim.operations.integrator = integrator
    # 采样间隔优先来自任务表；没有该列时用配置里的默认值。
    sample_interval = int(float(task.get("sample_interval", config["output"].get("sample_interval", 2000))))
    # 分段运行并采样凝胶时间序列和流体剖面。
    gel_rows, profile_summary = run_with_sampling(
        sim,
        n_steps,
        sample_interval,
        task["structure"] != "fluid",
        radius,
        out / "profiles.npz",
        out / "timeseries.npz",
        config,
    )

    # 运行结束后取末态快照。
    snapshot = sim.state.get_snapshot()
    # 保存凝胶末态。
    save_state(out / "state.npz", snapshot)
    # 再写一次最终版 timeseries.npz，确保文件完整。
    series_summary = save_timeseries(out / "timeseries.npz", gel_rows)

    # 计算溶剂最大径向位置，用于检查是否有粒子越界。
    solvent_r = np.sqrt(snapshot.mpcd.position[:, 0] ** 2 + snapshot.mpcd.position[:, 1] ** 2)
    # 凝胶最大径向位置；纯流体任务没有凝胶，所以默认为 None。
    bead_r_max = None
    # snapshot.particles.N 非零表示有凝胶珠子。
    if snapshot.particles.N:
        # 计算凝胶珠子到管轴的径向距离。
        bead_r = np.sqrt(snapshot.particles.position[:, 0] ** 2 + snapshot.particles.position[:, 1] ** 2)
        # 记录最大值。
        bead_r_max = float(bead_r.max())

    # summary 是这条任务的核心摘要，会写入 summary.json 和 status.json。
    summary = {
        "run_id": run_id,
        "structure": task["structure"],
        "wi": float(task["wi"]),
        "seed": int(task["seed"]),
        "steps": n_steps,
        "solvent_count": n_solvent,
        "solvent_density": float(config["mpcd"]["number_density"]),
        "mpcd_cell_size": float(config["mpcd"].get("cell_size", 1.0)),
        "body_force": body_force,
        "sample_interval": sample_interval,
        "sample_time_interval": sample_interval * float(config["mpcd"]["dt"]),
        **scales,
        "max_solvent_r": float(solvent_r.max()),
        "max_bead_r": bead_r_max,
        "tps": float(sim.tps) if sim.tps is not None else None,
        "status": "done",
        **profile_summary,
        **series_summary,
    }
    # 写摘要文件。
    write_json(out / "summary.json", summary)
    # 把 status.json 也更新成 done 状态和完整摘要。
    write_json(status_path(root, run_id), summary)
    # 清理 HOOMD 对象。
    close_sim(sim)
    # 返回摘要给 main 打印。
    return summary


def main() -> None:
    """命令行入口。"""
    # 创建命令行解析器。
    parser = argparse.ArgumentParser()
    # --config 指定配置文件路径。
    parser.add_argument("--config", type=Path, default=Path("config/base.json"))
    # --tasks 指定任务表路径。
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks_timescale.csv"))
    # --only 只运行某一个 run_id，常用于短验证。
    parser.add_argument("--only", default=None, help="只运行指定 run_id")
    # --max 最多运行多少条任务，常用于调试。
    parser.add_argument("--max", type=int, default=None, help="最多运行多少条")
    # --steps 临时覆盖步数，不改任务表。
    parser.add_argument("--steps", type=int, default=None, help="临时覆盖步数，用于短验证")
    # 解析命令行参数。
    args = parser.parse_args()

    # 当前目录应是项目根目录。
    root = Path.cwd()
    # 读取配置。
    config = read_json(args.config)
    # 读取任务表。
    tasks = read_tasks(args.tasks)
    # 如果指定 --only，就筛选出对应任务。
    if args.only:
        tasks = [task for task in tasks if task["run_id"] == args.only]
    # 如果指定 --max，就只取前 max 条。
    if args.max is not None:
        tasks = tasks[: args.max]
    # 没有任务可跑时直接退出并提示。
    if not tasks:
        raise SystemExit("没有任务可运行")

    # 失败任务计数。
    failed = 0
    # 顺序运行任务表中的每一条任务。
    for task in tasks:
        try:
            # 打印开始信息，flush=True 表示立刻写入日志。
            print(f"[开始] {task['run_id']}", flush=True)
            # 真正运行一条任务。
            result = run_task(root, config, task, args.steps)
            # 打印完成信息，包括溶剂粒子数和速度 TPS。
            print(f"[完成] {task['run_id']} solvent={result['solvent_count']} tps={result['tps']}", flush=True)
        except Exception as exc:
            # 捕获异常，避免一条任务失败时没有状态文件。
            failed += 1
            # payload 记录失败原因和完整 traceback。
            payload = {
                "run_id": task.get("run_id"),
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            # 把失败状态写入对应 status.json。
            write_json(status_path(root, task["run_id"]), payload)
            # 同时在终端/日志中打印失败摘要。
            print(f"[失败] {task['run_id']} {type(exc).__name__}: {exc}", flush=True)
    # 如果有任意任务失败，最后用非零退出码结束，方便外部发现问题。
    if failed:
        raise SystemExit(f"有 {failed} 条任务失败")


if __name__ == "__main__":
    # 直接运行 python scripts/run.py 时进入 main。
    main()
