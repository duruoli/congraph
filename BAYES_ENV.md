# 贝叶斯环境：各个量的定义、算式和算法

配合 `RESEARCH_ANCHOR.md` 第二圈"数学化框架"。这里写全公式和怎么算；anchor 里只留一句话的定义。

## 0. 符号

| 符号 | 含义 |
|---|---|
| $k = 1, \dots, K$ | 问题（出院 ICD 编码标出的疾病），每个是一个是 / 否开关 $s_k \in \{0, 1\}$ |
| $j$ | 一个检查项目（如血红蛋白、CT 上的"游离气"） |
| $o_j$ | 项目 $j$ 的结果：异常 / 正常。**"没做"不是观察，是行动**，不进 $o$ |
| $a$ | 一个检查行动（如"血常规"），会揭示一组项目 $J(a)$ |
| $b_k(t)$ | 时刻 $t$ 对"有问题 $k$"的信念 $q_t(s_k = 1)$ |
| $w_k$ | 问题 $k$ 的重要性（参数） |
| $H(p)$ | 二元熵 $-p \ln p - (1-p)\ln(1-p)$ |
| $\operatorname{logit}(p)$ | 对数几率 $\ln \frac{p}{1-p}$ |

v0 的简化：**每个问题单独算**（每个 $k$ 自成一个小模型），忽略问题之间的"解释掉"效应（explaining away）。要处理它，就升级到 noisy-OR，并用近似推断。

## 1. 似然 $p(o_j \mid s_k)$：这个病会在这个项目上留下什么

**定义**：在"**做了**项目 $j$"的前提下，有 / 没有问题 $k$ 的病人里，结果异常的比例。

$$
\theta_{kj1} = p(o_j = \text{异常} \mid s_k = 1, \text{做了 } j) = \frac{\#(\text{有 } k,\ \text{做了 } j,\ j \text{ 异常})}{\#(\text{有 } k,\ \text{做了 } j)}
$$

$$
\theta_{kj0} = p(o_j = \text{异常} \mid s_k = 0, \text{做了 } j) = \frac{\#(\text{无 } k,\ \text{做了 } j,\ j \text{ 异常})}{\#(\text{无 } k,\ \text{做了 } j)}
$$

- 分母必须限定"做了 $j$"，否则会把"医生开没开这个检查"混进"病会不会产生这个结果"里。
- 计数加伪计数平滑（如分子 +1、分母 +2），以免罕见组合出现 0 或 1。
- ICD 没有时间，所以这里的"有 $k$"指"这次住院有 $k$"，不是"此刻有 $k$"（见 anchor 待解决：s 的时间标签）。

**和 sameness 的关系**：

- **条件独立是模型的假设，不是 sameness 带来的**：给定 $s_k$，各项目的结果互相独立，

$$
p(o_A, o_B \mid s_k) = p(o_A \mid s_k)\, p(o_B \mid s_k).
$$

- **sameness = 共用同一个 $s_k$**：项目 $A$、$B$ 的似然比都明显偏离 1（对同一个 $k$）。所以它们在全体病人里相关，给定 $s_k$ 后不再相关。
- 归属不是硬分配：每个项目对每个问题都有一个似然比，$|\ln \mathrm{LR}|$ 越大，越"属于"这个问题。

**证据权重**（一条结果对问题 $k$ 的分量）：

$$
\mathrm{LR}_{kj}(\text{异常}) = \frac{\theta_{kj1}}{\theta_{kj0}}, \qquad
\mathrm{LR}_{kj}(\text{正常}) = \frac{1-\theta_{kj1}}{1-\theta_{kj0}}
$$

"查了正常"也有分量：它的似然比通常小于 1，即排除。

## 2. 信念 $q(s_k)$：到此刻为止，按同类病人的统计，这个病人有问题 $k$ 的可能

完全由贝叶斯算出来，用对数几率最方便：

$$
\operatorname{logit} b_k(0) = \operatorname{logit} p(s_k = 1) \qquad \text{（先验 = 海平面：同类病人里 } k \text{ 的比例）}
$$

$$
\operatorname{logit} b_k(t) = \operatorname{logit} b_k(0) + \sum_{j \in \mathcal{O}_t} \ln \mathrm{LR}_{kj}(o_j)
$$

其中 $\mathcal{O}_t$ 是到时刻 $t$ 为止已经观察到的项目。

- 它是"**一个掌握了同类病人统计的理想观察者**"的信念，不等于医生真实的信念（医生的信念看不到）。两者的差别正是可以研究的东西。
- 用在三处：地图的高度；预测下一个检查会看到什么（算认知价值）；每看到新事实就按上式累加更新。
- 抄写、同涨同落的项目（血红蛋白 / 压积）违反条件独立，加之前要先去重或合并，否则证据被算两次。

## 3. 意外 $-\ln p(o)$：这条事实对一个"像他这样的病人"有多反常

$$
p(o_j \mid \mathcal{O}_t) = \sum_{s_k} p(o_j \mid s_k)\, q_t(s_k)
= b_k(t)\, \theta_{kj}(o_j \mid 1) + \bigl(1 - b_k(t)\bigr)\, \theta_{kj}(o_j \mid 0)
$$

$$
\text{意外} = -\ln p(o_j \mid \mathcal{O}_t)
$$

其中 $\theta_{kj}(o_j \mid x) = p(o_j \mid s_k = x)$，结果异常时取 $\theta_{kjx}$，正常时取 $1-\theta_{kjx}$。

- 是对所有可能状态的**加权平均概率**再取对数，**不是**各状态下意外程度的加权平均。由 Jensen 不等式：

$$
-\ln \sum_{s} q(s)\, p(o \mid s) \;\le\; \sum_{s} q(s)\,\bigl(-\ln p(o \mid s)\bigr).
$$

- 用**当前信念** $q_t$ 加权，而不是先验：同一条结果，在信念已经转向之后再出现，就不那么意外了。
- 把一个病人所有事实按时间顺序的意外加起来，就等于 $-\ln p(\text{全部事实})$，即"整体上有多反常"。

## 4. 期望自由能：做检查之前，评估每个候选检查 $a$

### 4.1 认知价值（期望信息增益）

**单个项目 $j$、单个问题 $k$**（下式中 $b_k$ 即当前信念 $b_k(t)$）：

$$
m_{kj} = b_k\, \theta_{kj1} + (1 - b_k)\, \theta_{kj0} \qquad \text{（预计异常的概率）}
$$

$$
I_{kj} = H(m_{kj}) - \Bigl[\, b_k\, H(\theta_{kj1}) + (1 - b_k)\, H(\theta_{kj0}) \Bigr] \qquad \text{（预计能学到多少）}
$$

**一个检查揭示一组项目 $J(a)$**：给定 $s_k$ 时条件独立，但对 $s_k$ 的信息不能逐项相加（会重复计算）。准确算法是枚举或抽样所有结果组合 $o_J$：

$$
p(o_J \mid s_k = x) = \prod_{j \in J} p(o_j \mid s_k = x)
$$

$$
p(o_J) = b_k\, p(o_J \mid 1) + (1 - b_k)\, p(o_J \mid 0)
$$

$$
I(s_k; o_J) = \sum_{o_J} p(o_J)\; \mathrm{KL}\!\left[\, q(s_k \mid o_J) \,\middle\|\, q(s_k) \,\right]
$$

$|J| \le 10$ 时可直接枚举（不超过 $2^{10} = 1024$ 种组合），更多时用蒙特卡洛抽样。

**跨问题加权**（$w_k = 1$ 即不加权）：

$$
\mathrm{EV}(a) = \sum_{k=1}^{K} w_k\, I\bigl(s_k; o_{J(a)}\bigr)
$$

### 4.2 实用价值

- **v0（只有诊断行动）**：检查不改变 $s$，偏好对检查结果无所谓（均匀），这一项是常数，忽略。
- **v1（加入治疗）**：

$$
q(s' \mid a) = \sum_{s} p(s' \mid s, a)\, q(s) \qquad \text{（治疗后的状态，需要状态转移模型）}
$$

$$
q(o \mid a) = \sum_{s'} p(o \mid s')\, q(s' \mid a)
$$

$$
\mathrm{PV}(a) = \sum_{o} q(o \mid a)\, \ln \tilde p(o), \qquad \tilde p(o) = p(o \mid s = 0)
$$

### 4.3 合起来

$$
G(a) = -\,\mathrm{EV}(a) - \mathrm{PV}(a) + \mathrm{cost}(a)
$$

选择：$a^* = \arg\min_a G(a)$，或按 $P(a) \propto \exp\bigl(-\gamma\, G(a)\bigr)$ 抽样。

**和标准写法的关系**：只有一个问题、不加权、不计成本时，上式就是

$$
G(\pi) = -\, I(s; o \mid \pi) \;-\; \mathbb{E}_{q(o \mid \pi)}\bigl[\ln \tilde p(o)\bigr],
$$

等价于"风险 + 模糊"：

$$
G(\pi) = \mathrm{KL}\bigl[\, q(o \mid \pi) \,\big\|\, \tilde p(o) \,\bigr] + \mathbb{E}_{q(s \mid \pi)}\Bigl[ H\bigl[p(o \mid s, \pi)\bigr] \Bigr].
$$

## 5. 一轮的算法

1. **初始化**：对每个 $k$，令 $\operatorname{logit} b_k = \operatorname{logit} p(s_k = 1)$。
2. 对每一步 $t = 1, 2, \dots$：
   1. 对每个候选检查 $a$，用当前 $b_k$ 算 $\mathrm{EV}(a)$（4.1），得到 $G(a)$。
   2. 选行动：agent 取 $\arg\min_a G(a)$；回放模式下取医生实际做的检查。
   3. 观察结果 $o_J$；对每个 $k$ 记录意外 $-\ln p(o_J \mid \mathcal{O}_t)$（第 3 节）。
   4. 更新信念（第 2 节）：

$$
\operatorname{logit} b_k \leftarrow \operatorname{logit} b_k + \sum_{j \in J} \ln \mathrm{LR}_{kj}(o_j)
$$

   5. （RL）本步奖励 = 实际获得的信息减成本：

$$
r_t = \sum_{k} w_k\, \mathrm{KL}\!\left[\, q_{t+1}(s_k) \,\middle\|\, q_t(s_k) \,\right] - \mathrm{cost}(a)
$$

$r_t$ 对结果 $o_J$ 取期望，正好等于 $\mathrm{EV}(a) - \mathrm{cost}(a)$。所以规划时用期望值选行动，训练时用实际值作奖励，两者一致。

## 6. 第一个检验（不训练）

在回放模式下，每个医生做检查的时刻，比较医生选的检查和同一时刻其他候选检查的 $\mathrm{EV}$：医生的选择排在第几？先取 $w_k = 1$，再试加权。
