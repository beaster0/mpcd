# MPCD 网格凝胶项目

这个项目只保留一条运行链条：先生成结构，再跑零流时间尺度，再由时间尺度直接运行正式任务。

## 运行顺序

1. 修改基础参数：`config/base.json`
2. 生成网格结构：`python scripts/build.py`
3. 启动零流任务：`python scripts/run_timescale.py 0 1`
4. 查看零流状态：`python scripts/run_timescale.py --status`
5. 由零流结果生成时间尺度表：`python scripts/timescales.py`
6. 启动正式任务：`python scripts/run_flow.py 0 1`

启动任务前先确认 GPU 空闲；不要在没有确认时直接启动正式任务。

## 文件职责

- `config/base.json`：唯一的基础参数文件。
- `scripts/build.py`：生成 `g1` 到 `g4` 四个整体网格结构。
- `scripts/run.py`：底层模拟函数文件，不直接运行。
- `scripts/run_timescale.py`：运行零流时间尺度数据；命令后面的数字就是要使用的 GPU。
- `scripts/run_flow.py`：运行正式流动数据；命令后面的数字就是要使用的 GPU。
- `scripts/gpu.py`：统一 GPU 分片调度函数；不单独运行，所有运行任务的脚本都应调用它。
- `scripts/timescales.py`：从零流结果估计 `tau_shape` 和 `tau_int_r`。
- `data/structures/`：网格结构 JSON。
- `tables/structures/metrics.csv`：结构指标表。
- `tables/analysis/`：时间尺度汇总表和单轨迹诊断表。
- `results/<run_id>/`：每条任务的输出目录。

## 运行脚本约定

所有真正耗时的运行脚本都必须复用 `run.py::run_task()`，不要自己重新写模拟推进循环。这样每条模拟都会自动显示统一进度条，包括完成百分比、步数、速度和预计剩余时间。

## 结构定义

四个凝胶结构是 `g1`、`g2`、`g3`、`g4`。

- `g1` 是 `1 x 1 x 1` 整体三维网格。
- `g2` 是 `2 x 2 x 2` 整体三维网格。
- `g3` 是 `3 x 3 x 3` 整体三维网格。
- `g4` 是 `4 x 4 x 4` 整体三维网格。

相邻网格单元共享同一个交联点，不是把小块简单堆在一起。

## 关键参数

`config/base.json` 不能写注释，所以主要字段解释放在这里。

- `pipe.radius`：圆管半径。
- `pipe.length`：圆管长度。
- `structure.n_values`：要生成的网格结构编号。
- `structure.segments_per_edge`：相邻交联点之间有多少个键段。
- `bead.model`：键模型，当前为 `fene`。
- `bead.fene_k`：FENE 键强度。
- `bead.fene_r0`：FENE 最大伸长长度。
- `bead.fene_delta`：FENE-WCA 的位移参数，当前为 `0.0`。
- `bead.bond_equilibrium`：生成初始网格时使用的相邻珠子几何间距。
- `bead.mass_ratio`：凝胶珠子质量相对溶剂粒子质量的倍数。
- `bead.wca_sigma`：WCA 排斥势长度参数。
- `bead.wca_epsilon`：WCA 排斥势能量参数。
- `mpcd.number_density`：MPCD 溶剂数密度。
- `mpcd.dt`：MD 基础时间步。
- `mpcd.stream_period`：每隔多少个 MD 步做一次 MPCD streaming。
- `mpcd.collision_period`：每隔多少个 MD 步做一次 MPCD collision。
- `mpcd.collision_angle_deg`：MPCD 随机旋转角。
- `timescale.steps`：零流时间尺度任务步数。
- `production.sample_dt_over_tau_shape`：正式任务采样间隔相对 `tau_shape` 的比例。
- `production.shape_tau_multiplier`：正式任务至少覆盖多少个形状弛豫时间。
- `production.radial_tau_multiplier`：正式任务至少覆盖多少个径向相关时间。
- `production.min_time`：正式任务最低物理时间。
- `production.max_time`：正式任务最高物理时间，防止异常时间尺度生成不可承受任务。

## 时间尺度逻辑

正式凝胶任务不手写运行长度，而是由零流结果决定：

```text
sample_interval = ceil(0.1 * tau_shape_used / dt)
production_time = max(100 * tau_shape_used, 50 * tau_int_r_used, min_time)
production_steps = ceil(production_time / dt)
```

`tau_int_r_used` 来自 `r_cm` 自相关；`tau_rad_diagnostic`（R²/D_perp）仅诊断，不参与步数。
`timescales.py` 会报告 `t_analyzed` 与 `t_over_tau_*`；轨迹过短时 flag 标 `short`。

## 结果文件

每条任务输出到 `results/<run_id>/`：

- `status.json`：运行状态。
- `summary.json`：参数和摘要。
- `profiles.npz`：时间平均溶剂径向剖面。
- `timeseries.npz`：凝胶质心、形变、取向和壁面间隙时间序列。
- `state.npz`：末态凝胶坐标和速度。
