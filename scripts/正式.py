from __future__ import annotations

"""运行正式轨迹矩阵。

这个文件只做一件事：按论文图版需要的数据矩阵，调用 `轨迹.py` 跑正式轨迹。
它不生成 CSV，不画图，不写新的物理模型。

可以改什么：
- STRUCTURES：要跑哪些整体网格结构。
- FORCES：要跑哪些真实体力。
- SEEDS：要跑哪些独立随机种子。
- GPUS：允许使用哪些显卡；默认只用 0/1，保留 2/3 给别人。

输入：
- config/base.json
- tables/analysis/timescales.csv
- data/structures/g<n>.json

输出：
- results/半径36_结构n<n>_体力<g>_种子<seed>/
"""

# json 用来读取 summary.json，判断已有结果是否已经满足正式步数。
import json
# multiprocessing 用来让多张 GPU 并行工作。
import multiprocessing as mp
# subprocess 用来调用现有的单条轨迹脚本，避免重复写模拟逻辑。
import subprocess
# Path 用来拼接项目路径和结果路径。
from pathlib import Path

# 从轨迹.py 复用正式步数计算逻辑，保证调度脚本和单条轨迹脚本完全一致。
from 轨迹 import DEFAULT_CONFIG, DEFAULT_TAU, flow_steps, load_tau, read_json
# 从命名.py 复用中文结果目录命名，避免出现多套目录规则。
from 命名 import 结果目录


# 项目根目录；正式运行必须在服务器项目目录下执行。
ROOT = Path("/home/zhangxh/students/sjl/mpcd_modular_v2")
# Python 解释器；使用项目已有 conda 环境。
PYTHON = "/home/zhangxh/students/sjl/miniconda3/envs/A/bin/python"
# 四个整体网格结构。
STRUCTURES = ["g1", "g2", "g3", "g4"]
# 五个真实体力；0 用作同结构同几何的零流基线。
FORCES = [0.0, 0.001, 0.003, 0.005, 0.01]
# 五个独立随机种子，用于 seed 间统计。
SEEDS = [101, 102, 103, 104, 105]
# 只用前两张卡，GPU 2 和 GPU 3 都保留给别人。
GPUS = [0, 1]


def target_steps(config: dict, structure: str) -> int:
    """计算某个结构的正式步数。"""
    # 读取该结构的零流时间尺度。
    tau = load_tau(ROOT / DEFAULT_TAU, structure)
    # flow_steps 返回步数、采样间隔和设计物理时间。
    steps, _sample_interval, _design_time = flow_steps(config, tau)
    # 调度时只需要步数。
    return int(steps)


def finished(config: dict, radius: float, structure: str, force: float, seed: int) -> bool:
    """判断已有结果是否已经满足正式运行要求。"""
    # 找到这条结果的 summary.json。
    summary = 结果目录(ROOT, radius, structure, force, seed) / "summary.json"
    # 没有摘要说明还没跑完。
    if not summary.exists():
        return False
    # 读取摘要；读失败就当作未完成，重新跑。
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    # 不是 done 说明不是完整结果。
    if data.get("status") != "done":
        return False
    # 已完成步数必须达到当前正式规则要求。
    return int(data.get("steps", 0)) >= target_steps(config, structure)


def build_tasks(config: dict) -> list[tuple[str, float, int]]:
    """生成需要运行的正式任务列表。"""
    # 管半径用于定位中文结果目录。
    pipe = config["pipe"]
    # analysis_radius 是论文声明和目录命名使用的半径。
    radius = float(pipe.get("analysis_radius", pipe.get("radius")))
    # tasks 保存所有未完成任务。
    tasks: list[tuple[str, float, int]] = []
    # 按结构、seed、体力排序；同一结构连续跑，便于日志检查。
    for structure in STRUCTURES:
        for seed in SEEDS:
            for force in FORCES:
                # 已经满足正式步数的结果直接跳过。
                if finished(config, radius, structure, force, seed):
                    continue
                # 未完成则加入任务。
                tasks.append((structure, force, seed))
    # 返回任务列表。
    return tasks


def run_one(gpu: int, structure: str, force: float, seed: int) -> None:
    """在指定 GPU 上运行一条轨迹。"""
    # 命令直接调用单条轨迹脚本，不使用 CSV 任务表。
    cmd = [
        PYTHON,
        "scripts/轨迹.py",
        "--structure",
        structure,
        "--force",
        str(force),
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
    ]
    # 打印任务开始信息，方便从日志里定位。
    print(f"[正式][GPU {gpu}] 开始 {structure} g={force:g} seed={seed}", flush=True)
    # subprocess.run 会等待单条轨迹结束。
    result = subprocess.run(cmd, cwd=ROOT)
    # 非零返回码表示模拟失败，直接中断当前 GPU 工作进程。
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    # 打印任务完成信息。
    print(f"[正式][GPU {gpu}] 完成 {structure} g={force:g} seed={seed}", flush=True)


def worker(gpu: int, tasks: list[tuple[str, float, int]]) -> None:
    """让一张 GPU 顺序处理分配给它的任务。"""
    # 逐条运行，避免同一张 GPU 上同时挤多个 HOOMD 进程。
    for structure, force, seed in tasks:
        run_one(gpu, structure, force, seed)


def main() -> None:
    """命令行入口。"""
    # 读取总配置。
    config = read_json(ROOT / DEFAULT_CONFIG)
    # 找出还需要跑的任务。
    tasks = build_tasks(config)
    # 没有任务时直接结束。
    if not tasks:
        print("[正式] 所有正式轨迹已经完成", flush=True)
        return
    # 打印总任务数，方便估计进度。
    print(f"[正式] 待运行 {len(tasks)} 条轨迹，使用 GPU {GPUS}，保留 GPU 3", flush=True)
    # 按轮转方式把任务分给多张 GPU。
    groups = [tasks[index:: len(GPUS)] for index in range(len(GPUS))]
    # processes 保存所有 GPU 工作进程。
    processes: list[mp.Process] = []
    # 启动每张 GPU 对应的进程。
    for gpu, group in zip(GPUS, groups):
        process = mp.Process(target=worker, args=(gpu, group))
        process.start()
        processes.append(process)
    # failed 记录是否有任一进程失败。
    failed = False
    # 等待所有 GPU 进程结束。
    for process in processes:
        process.join()
        failed = failed or process.exitcode != 0
    # 有失败就返回非零状态，方便日志判断。
    if failed:
        raise SystemExit(1)
    # 全部完成时打印最终信息。
    print("[正式] 全部完成", flush=True)


if __name__ == "__main__":
    main()
