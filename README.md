# MPCD 网格凝胶项目

这个项目按“一个文件只做一件事”组织。想要什么数据，就运行对应脚本。

## 运行顺序

1. 修改基础参数：`config/base.json`
2. 生成结构：`python scripts/结构.py`
3. 跑一条零流轨迹：`python scripts/轨迹.py --stage time --structure g3 --seed 101 --gpu 0`
4. 跑完所需零流轨迹后计算时间尺度和图：`python scripts/时间.py`
5. 跑一条正式流动轨迹：`python scripts/轨迹.py --stage flow --structure g3 --flow 1 --seed 101 --gpu 0`
6. 画径向占据图：`python scripts/径向.py`
7. 画形变图：`python scripts/形变.py`
8. 画翻滚图：`python scripts/翻滚.py`

启动模拟前先确认 GPU 空闲；不要在没有确认时直接启动正式长任务。

## 文件职责

- `config/base.json`：唯一基础参数文件。
- `scripts/结构.py`：只生成 `g1` 到 `g4` 的整体网格结构和结构表。
- `scripts/轨迹.py`：只运行一条轨迹，例如某个结构、某个流强、某个 seed。
- `scripts/时间.py`：只从零流轨迹计算时间尺度，并画时间尺度诊断图。
- `scripts/径向.py`：只计算质心径向占据并画径向图。
- `scripts/形变.py`：只计算形变指标并画形变图。
- `scripts/翻滚.py`：只计算主轴翻滚强度并画翻滚图。
- `scripts/核心.py`：底层模拟函数文件，不直接运行。
- `scripts/显卡.py`：GPU 选择工具，供运行脚本复用。
- `data/structures/`：网格结构 JSON。
- `tables/structures/metrics.csv`：结构指标表。
- `tables/analysis/`：时间尺度结果和逐轨迹诊断表。
- `results/<run_id>/`：每条轨迹的独立结果目录。
- `figures/radial/`：径向占据图。
- `figures/shape/`：形变图。
- `figures/tumble/`：翻滚图。

## 轨迹例子

零流时间尺度轨迹：

```bash
python scripts/轨迹.py --stage time --structure g1 --seed 101 --gpu 0
```

正式凝胶流动轨迹：

```bash
python scripts/轨迹.py --stage flow --structure g3 --flow 1 --seed 101 --gpu 0
```

空管流体轨迹：

```bash
python scripts/轨迹.py --stage flow --structure fluid --flow 3 --seed 301 --gpu 0
```

短测试：

```bash
python scripts/轨迹.py --stage flow --structure g1 --flow 1 --seed 101 --gpu 0 --steps 10000
```

## 运行脚本约定

所有真正耗时的运行脚本都必须复用 `核心.py` 里的 `run_task()`，不要自己重新写模拟推进循环。这样每条模拟都会自动显示统一进度条，包括完成百分比、步数、速度和预计剩余时间。

所有脚本文件名只用一个单词。新增脚本也必须遵守这个规则。

## 画图脚本约定

画图脚本不启动模拟，只从 `results/` 读取已经存在的 `timeseries.npz`。

- 想看质心往管壁还是管中心分布，运行 `python scripts/径向.py`。
- 想看凝胶是否被拉长、是否更非球形，运行 `python scripts/形变.py`。
- 想看主轴是否快速转动和翻滚，运行 `python scripts/翻滚.py`。

如果某个结构、流强或 seed 的结果还不存在，画图脚本会跳过那条数据，并在对应位置显示缺失，不会自动补跑模拟。

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
- `bead.bond_equilibrium`：生成初始网格时使用的相邻珠子几何间距。
- `mpcd.number_density`：MPCD 溶剂数密度。
- `mpcd.dt`：MD 基础时间步。
- `timescale.steps`：零流时间尺度轨迹默认步数。
- `production.gel_flow_strength`：正式凝胶轨迹建议流强列表。
- `production.gel_seeds`：正式凝胶轨迹建议 seed 列表。
- `production.sample_dt_over_tau_shape`：正式轨迹采样间隔相对 `tau_shape` 的比例。
- `production.shape_tau_multiplier`：正式轨迹至少覆盖多少个形状弛豫时间。
- `production.radial_tau_multiplier`：正式轨迹至少覆盖多少个径向相关时间。
- `production.min_time`：正式轨迹最低物理时间。
- `production.max_time`：正式轨迹最高物理时间。

## 结果文件

每条轨迹输出到 `results/<run_id>/`：

- `status.json`：运行状态。
- `summary.json`：参数和摘要。
- `profiles.npz`：时间平均溶剂径向剖面。
- `timeseries.npz`：凝胶质心、形变、取向和壁面间隙时间序列。
- `state.npz`：末态凝胶坐标和速度。

