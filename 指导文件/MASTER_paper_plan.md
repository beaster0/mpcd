# 模块化交联凝胶网络泊肃叶流动：理论与方法方案

## 0. 研究目标

本项目研究一组整体三维网格交联凝胶在圆管泊肃叶流中的径向占据、流向形变、主轴取向和翻滚动力学。四个凝胶结构采用同一套局部建模规则，只改变每个方向的网格单元数 $n$：

| 结构 | 网格 | 网格单元数 |
|---|---:|---:|
| G1 | $1\times1\times1$ | 1 |
| G2 | $2\times2\times2$ | 8 |
| G3 | $3\times3\times3$ | 27 |
| G4 | $4\times4\times4$ | 64 |

这里的结构不是小块堆积，也不是把独立小凝胶再用跨块键拼起来。每个结构都是一个整体 $n\times n\times n$ 三维网格；相邻网格单元共用同一个交联点。因此，G2--G4 的内部节点天然连续，不存在重复边界节点或人为跨块连接。

本文只在这一组同质网格凝胶家族内讨论规律。重点不是把尺寸、摩擦、渗透性和可变形性完全拆成单独变量，而是建立一条简单、可复现、物理含义清楚的结构序列，测量网格规模增加后流动响应如何变化。

---

## 1. 理论主线

每个方向的网格数为 $n$，网格单元数为

$$
N_{\mathrm{cell}}=n^3,\qquad n=1,2,3,4.
$$

随着 $n$ 增加，凝胶整体尺寸、形状弛豫时间、质心扩散时间和近壁可访问空间会一起改变。这些变化本身就是该结构家族的物理特征，不在主线中强行拆开。主问题是：

> 在同一局部网络规则和同一管流条件下，整体网格规模增加如何改变凝胶的径向分布、流向伸展、主轴取向和翻滚行为？

主结果使用零流条件作为同结构、同管道、同壁面相互作用下的基线。流动响应优先报告相对于零流基线的变化，例如

$$
R_0(r;\mathrm{Wi})=
P_{\mathrm{cm}}(r;\mathrm{Wi})-P_{\mathrm{cm}}(r;0),
$$

以及

$$
\Delta \bar r(\mathrm{Wi})=
\langle r_{\mathrm{cm}}/R\rangle_{\mathrm{Wi}}
-
\langle r_{\mathrm{cm}}/R\rangle_0.
$$

这样可以把零流下由壁面排斥、构型熵和几何可访问空间带来的径向分布先单独交代，再讨论流动引起的额外变化。

---

## 2. 符号说明

| 符号 | 含义 |
|---|---|
| $n$ | 每个方向的网格单元数 |
| $N_{\mathrm{cell}}$ | 网格单元数，$N_{\mathrm{cell}}=n^3$ |
| $N$ | 凝胶总珠子数 |
| $R$ | 管半径 |
| $L$ | 管长 |
| $R_g$ | 凝胶回转半径，表示珠子相对质心的均方根尺寸 |
| $R_g/R$ | 凝胶尺寸相对管半径的比例 |
| $r_{\mathrm{cm}}$ | 凝胶质心到管中心轴的径向距离 |
| $\mathbf{G}$ | 回转张量，用来描述凝胶形状 |
| $G_{zz}$ | 流向形变分量 |
| $G_{\perp}$ | 横向形变分量，$G_{\perp}=(G_{xx}+G_{yy})/2$ |
| $A$ | 非球形度，越接近 0 越接近球形 |
| $\theta$ | 主轴与流向之间的夹角 |
| $C_u(t)$ | 主轴取向自相关函数 |
| $N_{\mathrm{tumb}}$ | 翻滚事件数 |
| $f_{\mathrm{tumb}}$ | 翻滚频率 |
| $\mathrm{Re}$ | 雷诺数 |
| $\mathrm{Wi}$ | 魏森贝格数 |
| $\mathrm{Pe}$ | 佩克莱数 |
| $\dot\gamma_{\mathrm{wall}}$ | 管壁剪切率 |
| $\dot\gamma_{\mathrm{cm}}$ | 凝胶质心位置经历的局部剪切率 |
| MPCD | 多粒子碰撞动力学溶剂 |
| MD | 分子动力学凝胶积分 |
| HI | 流体动力学相互作用 |

---

## 3. 整体网格结构

每个结构由交联点和链段组成。交联点位于规则三维网格节点上，任意两个相邻交联点之间插入相同数量的链珠。当前代码采用

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `segments_per_edge` | 5 | 相邻交联点之间有 5 个键段 |
| `bond_r0` | 0.85 | 键平衡长度 |
| `bond_k` | 30.0 | 谐和键强度 |
| `wca_sigma` | 0.65 | 排除体积长度参数 |
| `wca_epsilon` | 1.0 | 排除体积能量参数 |

相邻交联点之间的空间距离为

$$
\ell_m = 5\times0.85=4.25.
$$

这个长度是网格孔径的直接几何尺度。由于所有结构使用相同的键长、相同的每边键段数和相同的相互作用参数，局部网络规则在 G1--G4 之间保持一致。

当前结构由 `scripts/build.py` 生成，输出为 `data/structures/g1.json` 到 `data/structures/g4.json`。结构指标写入 `data/structures/metrics.csv`。

当前几何指标为：

| 结构 | 交联点数 | 链珠数 | 总珠子数 | 键数 | $R_g/R$ | $R_{99}/R$ |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 8 | 48 | 56 | 60 | 0.180 | 0.167 |
| G2 | 27 | 216 | 243 | 270 | 0.305 | 0.334 |
| G3 | 64 | 576 | 640 | 720 | 0.427 | 0.501 |
| G4 | 125 | 1200 | 1325 | 1500 | 0.547 | 0.668 |

$R_{99}$ 是 99% 珠子到凝胶质心距离的分位数，用来检查少数外层珠子是否过度靠近管壁。G4 的 $R_g/R$ 和 $R_{99}/R$ 已经很大，因此它在结果解释中必须作为强受限大凝胶处理，不能简单等同于弱受限体系。

---

## 4. 流体和管道

管道半径为 $R=18$，管长为 $L=100$。流动采用圆管泊肃叶流：

$$
u_z(r)=u_{\max}\left(1-\frac{r^2}{R^2}\right).
$$

剪切率为

$$
\dot\gamma(r)=\left|\frac{du_z}{dr}\right|
=\frac{2u_{\max}}{R^2}r,
\qquad
\dot\gamma_{\mathrm{wall}}=\frac{2u_{\max}}{R}.
$$

雷诺数使用管半径定义：

$$
\mathrm{Re}_R=\frac{\rho u_{\max}R}{\eta}.
$$

主数据应保持在低雷诺数范围内：

$$
\mathrm{Re}_R<0.1.
$$

MPCD 使用约化单位。碰撞格长显式设为 $a=1.0$，溶剂数密度为 5.0，因此平均每个 MPCD 碰撞格约有 5 个溶剂粒子。温度和质量取 $k_BT=1.0$、$m=1.0$。时间步长为 $\Delta t=0.005$，streaming 和 collision 都每 20 个 MD 步执行一次，因此 MPCD 碰撞时间为

$$
h=20\Delta t=0.1.
$$

对应的热速度尺度为 $\sqrt{k_BT/m}=1$，平均自由程估计为

$$
\lambda \approx h\sqrt{k_BT/m}=0.1.
$$

这组 $a=1$、平均每格粒子数 5、旋转角 $130^\circ$、平均自由程约 0.1 的设置与常用 MPCD 参数区间一致，计算成本和流体输运稳定性比极小平均自由程设置更平衡。

溶剂粒子数不手写在脚本中，而是由圆管体积自动计算：

$$
N_{\mathrm{solvent}}=\rho\pi R^2L.
$$

当前参数下

$$
N_{\mathrm{solvent}}\approx 508938.
$$

---

## 5. 流强定义

参考魏森贝格数定义为

$$
\mathrm{Wi}_{\mathrm{ref}}=\dot\gamma_{\mathrm{wall}}\tau_{\mathrm{ref}}.
$$

$\tau_{\mathrm{ref}}$ 采用 G2 在零流条件下的形状弛豫时间。这个定义让所有结构共享同一个外部流强标尺。每个结构还应报告结构自身的

$$
\mathrm{Wi}_s=\dot\gamma_{\mathrm{wall}}\tau_{\mathrm{shape},s}.
$$

局部剪切暴露量定义为

$$
\dot\gamma_{\mathrm{cm}}(t)=
\left.\left|\frac{du_z}{dr}\right|\right|_{r=r_{\mathrm{cm}}(t)},
\qquad
\mathrm{Wi}_{\mathrm{loc},s}=\left\langle \dot\gamma_{\mathrm{cm}}(t)\right\rangle\tau_{\mathrm{shape},s}.
$$

当前代码中 `force_per_wi=0.0001` 是第一批运行使用的体力比例。纯流体结果出来后，应根据速度剖面重新确认 $g$、$\dot\gamma_{\mathrm{wall}}$、$\mathrm{Wi}_{\mathrm{ref}}$ 和 $\mathrm{Re}_R$ 的对应关系。

---

## 6. 运行计划

运行分成两步。第一步只跑零流时间尺度任务，用来测每个结构的形状弛豫时间和径向相关时间。第二步根据这些时间尺度自动生成正式任务，不手写凝胶任务步数。

### 6.1 运行编号

运行编号使用短名称：

```text
fluid_w<Wi>_s<seed>
g<n>_w<Wi>_s<seed>
```

示例：

```text
fluid_w1_s301
g4_w3_s105
```

这种命名短、清楚、便于在结果目录中直接查找。

### 6.2 零流时间尺度任务

时间尺度任务只包含凝胶、只跑零流。输出用于得到每个结构的 `tau_shape_used` 和 `tau_r_used`。

| 项目 | 设置 |
|---|---|
| 结构 | G1、G2、G3、G4 |
| 流强 | $\mathrm{Wi}_{\mathrm{ref}}=0$ |
| seed | 101、102、103、104、105 |
| 步数 | 800,000 |
| 物理时间 | 4,000 |
| 数量 | 20 |

时间尺度任务表由 `scripts/build.py` 生成：

```text
data/tasks_timescale.csv
data/tasks_timescale_gpu0.csv
data/tasks_timescale_gpu1.csv
```

### 6.3 正式任务

纯流体任务用于得到速度剖面、温度剖面、密度剖面和流强映射。

| 项目 | 设置 |
|---|---|
| 结构 | 空管 |
| 流强 | $\mathrm{Wi}_{\mathrm{ref}}=0,1,3$ |
| seed | 301、302、303 |
| 步数 | 1,000,000 |
| 物理时间 | 5,000 |
| 数量 | 9 |

凝胶正式任务同时包含零流基线和流动响应统计。$\mathrm{Wi}=0$ 的 5 个 seed 用来计算同结构零流基线，$\mathrm{Wi}=1,3$ 用来计算流动响应。每个结构的正式运行长度由 `data/timescales.csv` 自动决定：

$$
\Delta t_{\mathrm{sample}}=0.1\tau_{\mathrm{shape,used}},
$$

$$
T_{\mathrm{prod}}=
\max\left(
100\tau_{\mathrm{shape,used}},
50\tau_{r,\mathrm{used}},
T_{\min}
\right).
$$

| 项目 | 设置 |
|---|---|
| 结构 | G1、G2、G3、G4 |
| 流强 | $\mathrm{Wi}_{\mathrm{ref}}=0,1,3$ |
| seed | 101、102、103、104、105 |
| 步数 | 由 `scripts/plan.py` 根据时间尺度生成 |
| 采样间隔 | 由 `scripts/plan.py` 根据时间尺度生成 |
| 数量 | 60 |

正式任务表由 `scripts/plan.py` 生成：

```text
data/tasks_production.csv
data/tasks_production_gpu0.csv
data/tasks_production_gpu1.csv
```

正式任务总数为

$$
9 + 4\times3\times5 = 69.
$$

---

## 7. 每条任务的输出

每条任务输出到 `results/<run_id>/`。

| 文件 | 内容 |
|---|---|
| `status.json` | 运行状态、参数和摘要 |
| `summary.json` | 与状态相同的摘要备份 |
| `profiles.npz` | 时间平均溶剂径向剖面，包括每个半径壳层的粒子数、平均轴向速度和速度平方 |
| `state.npz` | 凝胶末态坐标、速度和类型 |
| `timeseries.npz` | 凝胶时间序列观测量，纯流体任务为空 |

`timeseries.npz` 是凝胶分析的核心输入。当前每 2,000 步记录一次，采样物理时间间隔为 10；800,000 步凝胶轨迹约记录 400 帧。记录量包括：

| 变量 | 含义 |
|---|---|
| `step` | 采样步数 |
| `time` | 采样对应的物理时间 |
| `com` | 凝胶质心坐标 |
| `r_cm` | 凝胶质心径向距离 |
| `Rg` | 回转半径 |
| `Gxx, Gyy, Gzz` | 回转张量对角分量 |
| `Gperp` | 横向形变分量 |
| `lambda1, lambda2, lambda3` | 回转张量三个本征值，按从大到小记录 |
| `lambda_gap` | 最大和次大本征值之差，用来判断主轴方向是否可靠 |
| `asphericity` | 非球形度 |
| `axis` | 最大主轴方向 |
| `theta` | 最大主轴与流向夹角 |
| `R95, R99` | 外层珠子径向分位数 |
| `max_bead_r` | 最大珠子径向距离 |
| `wall_clearance` | 珠子外缘到管壁的最小间隙 |

---

## 8. 时间尺度和采样长度

零流凝胶任务用于测量：

1. $R_g(t)$、$G_{zz}(t)$ 和 $A(t)$ 的稳定性。
2. 形状弛豫时间 $\tau_{\mathrm{shape}}$。
3. 质心径向自相关时间 $\tau_{\mathrm{int}}^r$。
4. 质心扩散系数 $D_{\mathrm{cm}}$。

径向扩散时间估计为

$$
\tau_{\mathrm{rad}}^{\mathrm{est}}=\frac{R^2}{D_{\mathrm{cm}}}.
$$

生产段长度用实际采样结果检查。每条轨迹至少报告采样帧数、$r_{\mathrm{cm}}(t)$ 是否持续漂移、$R_g(t)$ 前后半段是否一致、壁面间隙是否异常。

---

## 9. 径向分布分析

主文使用面积归一化径向分布。按环形 bin 计算：

$$
p_{\mathrm{area}}(r_k)=
\frac{P(r_k<r<r_{k+1})}
{\pi(r_{k+1}^2-r_k^2)}.
$$

零流基线为

$$
P_0(r)=P_{\mathrm{cm}}(r;\mathrm{Wi}=0).
$$

流动下的一阶残差为

$$
R_0(r;\mathrm{Wi})=
P_{\mathrm{cm}}(r;\mathrm{Wi})-P_{\mathrm{cm}}(r;0).
$$

分布级指标包括：

| 指标 | 含义 |
|---|---|
| $\Delta \bar r$ | 平均径向位置变化 |
| $L_1$ | 两个径向分布的整体差异 |
| $W_1$ | 径向分布搬运距离 |
| 近壁尾部概率 | 高 $r/R$ 区域的占据变化 |

---

## 10. 形变、取向和翻滚

形变由回转张量计算：

$$
G_{\alpha\beta}=\frac{1}{N}\sum_{i=1}^N
(r_{i\alpha}-r_{\mathrm{cm},\alpha})(r_{i\beta}-r_{\mathrm{cm},\beta}).
$$

流向形变为 $G_{zz}$，横向形变为

$$
G_{\perp}=\frac{G_{xx}+G_{yy}}{2}.
$$

归一化流向形变为

$$
\frac{G_{zz}(\mathrm{Wi})}{G_{zz}(0)}.
$$

主轴由回转张量最大本征值对应的单位向量 $\mathbf{u}_1(t)$ 表示。主轴正负等价，取向角定义为

$$
\theta(t)=\arccos\left(\left|\mathbf{u}_1(t)\cdot\hat{\mathbf{z}}\right|\right).
$$

主轴取向自相关定义为

$$
C_u(\Delta t)=
\left\langle
P_2\left[\mathbf{u}_1(t_0)\cdot\mathbf{u}_1(t_0+\Delta t)\right]
\right\rangle_{t_0},
\qquad
P_2(x)=\frac{3x^2-1}{2}.
$$

翻滚事件按 $\theta(t)$ 的完整大幅摆动统计。一次翻滚事件定义为平滑后的 $\theta(t)$ 从 $\theta<\pi/6$ 进入 $\theta>\pi/3$，再返回 $\theta<\pi/6$ 的完整过程。翻滚频率为

$$
f_{\mathrm{tumb}}=\frac{N_{\mathrm{tumb}}}{T_{\mathrm{prod}}}.
$$

---

## 11. 统计方法

统计单位是 seed，不是连续采样帧。每个 seed 内先完成时间平均，再在 seed 层面计算均值和误差。每个凝胶主条件有 5 个 seed。主要结果同时报告：

1. seed 均值。
2. 95% 置信区间。
3. 单个 seed 的离散点。
4. 采样帧数。
5. 运行状态。

时间序列相关性用自相关函数或 block 平均处理，不能把相邻采样帧直接当作完全独立样本。

---

## 12. 图和结果组织

### 图 1：模型结构和控制参数

展示圆管泊肃叶流几何、G1--G4 整体共享节点网格结构、$N_{\mathrm{cell}}$、$N$、$R_g/R$ 和 $\mathrm{Wi}_{\mathrm{ref}}$。

### 图 2：纯流体标定和零流基线

展示空管速度剖面、温度剖面、密度剖面，以及 G1--G4 的零流 $R_g/R$、$R_{99}/R$ 和 $P_{\mathrm{cm}}(r;0)$。

### 图 3：径向分布和流动残差

展示 $P_{\mathrm{cm}}(r;\mathrm{Wi})$、$R_0(r;\mathrm{Wi})$、$\Delta\bar r$、$L_1$、$W_1$ 和近壁尾部概率。

### 图 4：形变和取向

展示 $G_{zz}/G_{zz}(0)$、$G_{\perp}/G_{\perp}(0)$、非球形度 $A$ 和主轴取向角 $\theta$。

### 图 5：翻滚行为

展示代表性 $\theta(t)$、取向自相关 $C_u(t)$、翻滚事件数 $N_{\mathrm{tumb}}$ 和翻滚频率 $f_{\mathrm{tumb}}$。

### 图 6：时间序列和采样可靠性

展示 $r_{\mathrm{cm}}(t)$、$G_{zz}(t)$、$R_g(t)$、壁面间隙和采样帧数。

### 图 7：运行汇总

展示 69 条任务的完成状态、每个结构和流强下的 seed 覆盖情况、失败任务和异常壁面间隙记录。

---

## 13. 执行顺序

1. 修改 `config/base.json` 中的基础物理参数。
2. 运行 `python scripts/build.py` 生成结构和零流时间尺度任务表。
3. 检查 `data/structures/metrics.csv`，确认 G1--G4 的珠子数、键数和 $R_g/R$。
4. 在服务器上分别启动 `python scripts/gpu.py 0` 和 `python scripts/gpu.py 1`，先跑 `data/tasks_timescale.csv`。
5. 时间尺度任务完成后，从 `timeseries.npz` 计算每个结构的 `tau_shape_used` 和 `tau_r_used`，写入 `data/timescales.csv`。
6. 运行 `python scripts/plan.py`，生成 `data/tasks_production.csv`。
7. 用 `python scripts/gpu.py 0 --stage production` 和 `python scripts/gpu.py 1 --stage production` 跑正式任务。
8. 用 `python scripts/status.py --tasks data/tasks_production.csv` 查看正式任务状态。
9. 每条任务完成后检查 `summary.json`、`profiles.npz` 和 `timeseries.npz`。
10. 用后处理脚本从 `timeseries.npz` 计算径向分布、形变、取向和翻滚指标。
11. 按图 1--图 7 的顺序组织结果。

---

## 14. 参考依据

[1] Malevanets, A. & Kapral, R. Mesoscopic model for solvent dynamics. *J. Chem. Phys.* **110**, 8605--8613 (1999). DOI: 10.1063/1.478857.

[2] Malevanets, A. & Kapral, R. Solute molecular dynamics in a mesoscale solvent. *J. Chem. Phys.* **112**, 7260--7269 (2000). DOI: 10.1063/1.481289.

[3] Ihle, T. & Kroll, D. M. Stochastic rotation dynamics: A Galilean-invariant mesoscopic model for fluid flow. *Phys. Rev. E* **63**, 020201 (2001). DOI: 10.1103/PhysRevE.63.020201.

[4] Ripoll, M., Mussawisade, K., Winkler, R. G. & Gompper, G. Low-Reynolds-number hydrodynamics of complex fluids by multi-particle-collision dynamics. *EPL* **68**, 106--112 (2004). DOI: 10.1209/epl/i2003-10310-1.

[5] Gompper, G., Ihle, T., Kroll, D. M. & Winkler, R. G. Multi-Particle Collision Dynamics: A Particle-Based Mesoscale Simulation Approach to the Hydrodynamics of Complex Fluids. *Adv. Polym. Sci.* **221**, 1--87 (2009). DOI: 10.1007/978-3-540-87706-6_1.

[6] Noguchi, H. & Gompper, G. Transport coefficients of off-lattice mesoscale-hydrodynamics simulation techniques. *Phys. Rev. E* **78**, 016706 (2008). DOI: 10.1103/PhysRevE.78.016706.

[7] Huang, C.-C. & Winkler, R. G. Stress tensors of multiparticle collision dynamics fluids. *J. Chem. Phys.* **130**, 074907 (2009). DOI: 10.1063/1.3077860.

[8] Bolintineanu, D. S., Lechman, J. B., Plimpton, S. J. & Grest, G. S. No-slip boundary conditions and forced flow in multiparticle collision dynamics. *Phys. Rev. E* **86**, 066703 (2012). DOI: 10.1103/PhysRevE.86.066703.

[9] Howard, M. P., Nikoubashman, A. & Palmer, J. C. Modeling hydrodynamic interactions in soft materials with multiparticle collision dynamics. *Curr. Opin. Chem. Eng.* **23**, 34--43 (2019). DOI: 10.1016/j.coche.2019.02.007.

[10] Jendrejack, R. M., Schwartz, D. C., de Pablo, J. J. & Graham, M. D. Shear-induced migration in flowing polymer solutions: Simulation of long-chain DNA in microchannels. *J. Chem. Phys.* **120**, 2513--2529 (2004). DOI: 10.1063/1.1637331.

[11] Ma, H. & Graham, M. D. Theory of shear-induced migration in dilute polymer solutions near solid boundaries. *Phys. Fluids* **17**, 083103 (2005). DOI: 10.1063/1.2011367.

[12] Usta, O. B., Butler, J. E. & Ladd, A. J. C. Flow-induced migration of polymers in dilute solution. *Phys. Fluids* **18**, 031703 (2006). DOI: 10.1063/1.2186591.

[13] Saintillan, D., Shaqfeh, E. S. G. & Darve, E. Effect of flexibility on the shear-induced migration of short-chain polymers in parabolic channel flow. *J. Fluid Mech.* **557**, 297--306 (2006). DOI: 10.1017/S0022112006000243.

[14] Chelakkot, R., Winkler, R. G. & Gompper, G. Migration of semiflexible polymers in microcapillary flow. *EPL* **91**, 14001 (2010). DOI: 10.1209/0295-5075/91/14001.

[15] Chelakkot, R., Gompper, G. & Winkler, R. G. Semiflexible polymer conformation, distribution and migration in microcapillary flows. *J. Phys.: Condens. Matter* **23**, 184117 (2011). DOI: 10.1088/0953-8984/23/18/184117.

[16] Liu, A., Yang, Z., Liu, L., Chen, J. & An, L. Role of functionality in cross-stream migration, structures, and dynamics of star polymers in Poiseuille flow. *Macromolecules* **53**, 9993--10004 (2020). DOI: 10.1021/acs.macromol.0c00699.

[17] Peng, B., Yang, Z., Yang, L., Chen, J., Liu, L. & Wang, D. Reducing the solvent quality gives rise to the outward migration of a star polymer in Poiseuille flow. *Macromolecules* **55**, 3396--3407 (2022). DOI: 10.1021/acs.macromol.2c00031.

[18] Srivastva, D. & Nikoubashman, A. Flow behavior of chain and star polymers and their mixtures. *Polymers* **10**, 599 (2018). DOI: 10.3390/polym10060599.

[19] Doddi, S. K. & Bagchi, P. Lateral migration of a capsule in a plane Poiseuille flow in a channel. *Int. J. Multiphase Flow* **34**, 966--986 (2008). DOI: 10.1016/j.ijmultiphaseflow.2008.03.002.

[20] Danker, G., Vlahovska, P. M. & Misbah, C. Vesicles in Poiseuille flow. *Phys. Rev. Lett.* **102**, 148102 (2009). DOI: 10.1103/PhysRevLett.102.148102.

[21] Chen, Y.-L. Inertia- and deformation-driven migration of a soft particle in confined shear and Poiseuille flow. *RSC Adv.* **4**, 17908--17916 (2014). DOI: 10.1039/C4RA00837E.

[22] Chen, L., Wang, K. X. & Doyle, P. S. Effect of internal architecture on microgel deformation in microfluidic constrictions. *Soft Matter* **13**, 1920--1928 (2017). DOI: 10.1039/C6SM02674E.

[23] Rovigatti, L. et al. Internal structure and swelling behaviour of in silico microgel particles. *J. Phys.: Condens. Matter* **30**, 044001 (2018). DOI: 10.1088/1361-648X/aaa0f4.

[24] Grest, G. S. & Kremer, K. Molecular dynamics simulation for polymers in the presence of a heat bath. *Phys. Rev. A* **33**, 3628--3631 (1986). DOI: 10.1103/PhysRevA.33.3628.

[25] Kremer, K. & Grest, G. S. Dynamics of entangled linear polymer melts: A molecular-dynamics simulation. *J. Chem. Phys.* **92**, 5057--5086 (1990). DOI: 10.1063/1.458541.

[26] Grest, G. S. & Kremer, K. Statistical properties of random cross-linked rubbers. *Macromolecules* **23**, 4994--5000 (1990). DOI: 10.1021/ma00225a020.

[27] Likos, C. N. et al. Star polymers viewed as ultrasoft colloidal particles. *Phys. Rev. Lett.* **80**, 4450--4453 (1998). DOI: 10.1103/PhysRevLett.80.4450.

[28] Likos, C. N. Effective interactions in soft condensed matter physics. *Phys. Rep.* **348**, 267--439 (2001). DOI: 10.1016/S0370-1573(00)00141-1.

[29] Flyvbjerg, H. & Petersen, H. G. Error estimates on averages of correlated data. *J. Chem. Phys.* **91**, 461--466 (1989). DOI: 10.1063/1.457480.

[30] Efron, B. Bootstrap methods: Another look at the jackknife. *Ann. Stat.* **7**, 1--26 (1979). DOI: 10.1214/aos/1176344552.

[31] Kunsch, H. R. The jackknife and the bootstrap for general stationary observations. *Ann. Stat.* **17**, 1217--1241 (1989). DOI: 10.1214/aos/1176347265.
