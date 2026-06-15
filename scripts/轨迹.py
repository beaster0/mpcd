# 让类型注解延迟求值，保持不同 Python 小版本兼容。
from __future__ import annotations

"""运行一条轨迹。

这个脚本只做一件事：跑某个结构、某个流强、某个 seed 的一条模拟轨迹。
它不批量调度、不计算时间尺度、不画图。

可以改什么：
- 要改默认结构，改文件开头的 `DEFAULT_STRUCTURE`。
- 要改默认体力，改文件开头的 `DEFAULT_FORCE`。
- 要改默认 seed，改文件开头的 `DEFAULT_SEED`。
- 要改默认阶段，改文件开头的 `DEFAULT_STAGE`。
- 要临时改步数，运行时加 `--steps <步数>`。

输入：
- `config/base.json`：物理参数。
- `data/structures/g<n>.json`：凝胶结构；当 `--structure fluid` 时不需要凝胶结构。
- `tables/analysis/timescales.csv`：正式凝胶流动轨迹需要用它计算默认步数。

输出：
- `results/<中文目录名>/status.json`
- `results/<中文目录名>/summary.json`
- `results/<中文目录名>/state.npz`
- `results/<中文目录名>/timeseries.npz`
- `results/<中文目录名>/profiles.npz`

运行例子：
- 跑 g3 在真实体力 g=0.005、seed 101 的正式轨迹：
  `/home/zhangxh/students/sjl/miniconda3/envs/A/bin/python scripts/轨迹.py --structure g3 --force 0.005 --seed 101 --gpu 0`
- 跑 g2 的零流时间尺度轨迹：
  `/home/zhangxh/students/sjl/miniconda3/envs/A/bin/python scripts/轨迹.py --stage time --structure g2 --seed 101 --gpu 0`
"""

import argparse
import csv
from pathlib import Path
from typing import Any

from 显卡 import use
from 核心 import read_json, run_task
from 命名 import 目录名


# 默认阶段。`flow` 表示正式流动轨迹；`time` 表示零流时间尺度轨迹。
DEFAULT_STAGE = "flow"
# 默认结构。可写 `g1/g2/g3/g4/fluid`。
DEFAULT_STRUCTURE = "g1"
# 默认真实体力 g。零流就是 0。
DEFAULT_FORCE = 0.0
# 默认随机种子。
DEFAULT_SEED = 101
# 默认配置文件。
DEFAULT_CONFIG = Path("config/base.json")
# 默认时间尺度表。
DEFAULT_TAU = Path("tables/analysis/timescales.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 表格。"""
    # utf-8-sig 兼容 Excel 写出的 BOM。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # DictReader 把每一行变成字典。
        return list(csv.DictReader(handle))


def load_tau(path: Path, structure: str) -> dict[str, float]:
    """读取某个结构的时间尺度。"""
    # 遍历时间尺度表的每一行。
    for row in read_csv(path):
        # 找到目标结构，例如 g3。
        if row["structure"] == structure:
            # tau_int_r_used 是径向记忆时间。
            key = "tau_int_r_used" if row.get("tau_int_r_used") else "tau_r_used"
            # 返回正式运行需要的两个时间尺度。
            return {
                "tau_shape": float(row["tau_shape_used"]),
                "tau_int_r": float(row[key]),
            }
    # 没找到结构就说明时间尺度表不完整。
    raise ValueError(f"在 {path} 中找不到 {structure} 的时间尺度")


def flow_steps(config: dict[str, Any], tau: dict[str, float]) -> tuple[int, int, float]:
    """按正式运行规则计算凝胶轨迹步数和采样间隔。"""
    # dt 是 MD 基础时间步。
    dt = float(config["mpcd"]["dt"])
    # production 块控制正式运行长度。
    rule = config["production"]
    # 采样时间间隔 = 比例系数 * 形状弛豫时间。
    sample_time = float(rule["sample_dt_over_tau_shape"]) * tau["tau_shape"]
    # 采样步数间隔至少使用 output.sample_interval，避免小凝胶 tau_shape 太短导致过密采样。
    minimum_interval = int(config["output"].get("sample_interval", 2000))
    # 采样步数间隔至少为 1。
    sample_interval = max(1, minimum_interval, int((sample_time + dt - 1e-12) // dt))
    # 正式运行时间取形状要求、径向要求、最低时间三者最大值。
    total_time = max(
        float(rule["shape_tau_multiplier"]) * tau["tau_shape"],
        float(rule["radial_tau_multiplier"]) * tau["tau_int_r"],
        float(rule["min_time"]),
    )
    # max_time 是保险上限。
    total_time = min(total_time, float(rule["max_time"]))
    # 换算成步数。
    steps = int((total_time + dt - 1e-12) // dt)
    # 返回步数、采样间隔、设计时间。
    return steps, sample_interval, total_time


def case_name(radius: float, structure: str, force: float, seed: int, stage: str = "flow") -> str:
    """生成清楚的中文结果目录名。

    目录名把半径、结构、体力和 seed 都写出来，人工浏览 results 时不需要猜。
    """
    return 目录名(radius, structure, force, seed, stage)


def make_case(root: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """把命令行参数整理成 core.run_task 需要的一条 case。"""
    # 管半径写进 run_id，避免不同管径结果混在一起。
    pipe = config["pipe"]
    radius = float(pipe.get("analysis_radius", pipe.get("radius")))
    # 结构名来自命令行，例如 g3 或 fluid。
    structure = args.structure
    # 真实体力来自命令行。
    force = float(args.force)
    # seed 来自命令行。
    seed = int(args.seed)

    # time 阶段只用于零流凝胶时间尺度轨迹。
    if args.stage == "time":
        if structure == "fluid":
            raise ValueError("time 阶段只跑凝胶结构，不能使用 fluid")
        return {
            "run_id": case_name(radius, structure, 0.0, seed, "time"),
            "kind": "gel",
            "stage": "timescale",
            "pipe_radius": radius,
            "structure": structure,
            "force": 0.0,
            "seed": seed,
            "steps": int(args.steps if args.steps is not None else config["timescale"]["steps"]),
        }

    # flow 阶段允许空管。
    if structure == "fluid":
        # 空管步数来自 production.fluid_steps，除非命令行临时覆盖。
        steps = int(args.steps if args.steps is not None else config["production"]["fluid_steps"])
        return {
            "run_id": case_name(radius, "fluid", force, seed),
            "kind": "fluid",
            "stage": "production",
            "pipe_radius": radius,
            "structure": "fluid",
            "force": force,
            "seed": seed,
            "steps": steps,
            "sample_interval": int(config["output"]["sample_interval"]),
            "design_time": steps * float(config["mpcd"]["dt"]),
        }

    # flow 阶段的凝胶需要读取零流时间尺度结果。
    tau = load_tau(root / args.tau, structure)
    # 根据该结构自己的时间尺度计算步数。
    steps, sample_interval, design_time = flow_steps(config, tau)
    # 命令行 --steps 只用于临时检查，会覆盖正式设计步数。
    if args.steps is not None:
        steps = int(args.steps)
        design_time = steps * float(config["mpcd"]["dt"])
    return {
        "run_id": case_name(radius, structure, force, seed),
        "kind": "gel",
        "stage": "production",
        "pipe_radius": radius,
        "structure": structure,
        "force": force,
        "seed": seed,
        "steps": steps,
        "sample_interval": sample_interval,
        "design_time": design_time,
        "tau_shape_used": tau["tau_shape"],
        "tau_int_r_used": tau["tau_int_r"],
    }


def main() -> None:
    """命令行入口。"""
    # argparse 负责把命令行文字转成 Python 变量。
    parser = argparse.ArgumentParser()
    # --stage 选择跑零流时间尺度轨迹还是正式流动轨迹。
    parser.add_argument("--stage", choices=["time", "flow"], default=DEFAULT_STAGE)
    # --structure 指定结构，例如 g1/g2/g3/g4；空管写 fluid。
    parser.add_argument("--structure", default=DEFAULT_STRUCTURE)
    # --force 指定真实体力 g。
    parser.add_argument("--force", type=float, default=DEFAULT_FORCE)
    # --flow 只为兼容旧命令；如果填写，会被当作真实体力使用。
    parser.add_argument("--flow", type=float, default=None)
    # --seed 指定随机种子。
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # --gpu 指定使用哪张 GPU。
    parser.add_argument("--gpu", type=int, default=None)
    # --steps 临时覆盖步数，用于短检查。
    parser.add_argument("--steps", type=int, default=None)
    # --config 指定配置文件。
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    # --tau 指定时间尺度表。
    parser.add_argument("--tau", type=Path, default=DEFAULT_TAU)
    # 解析命令行参数。
    args = parser.parse_args()
    # 兼容旧命令：以前的 --flow 现在等价于 --force。
    if args.flow is not None:
        args.force = args.flow

    # 当前目录必须是项目根目录。
    root = Path.cwd()
    # 如果指定 GPU，就让当前进程只看到这一张卡。
    use(args.gpu)
    # 读取配置文件。
    config = read_json(args.config)
    # 生成单条 case。
    case = make_case(root, config, args)
    # 打印即将运行的结果目录名，方便确认没有跑错。
    print(f"[轨迹] {case['run_id']}")
    # 运行这一条轨迹；步数已经写进 case，所以这里不再传 steps。
    run_task(root, config, case, steps=None)


if __name__ == "__main__":
    main()
