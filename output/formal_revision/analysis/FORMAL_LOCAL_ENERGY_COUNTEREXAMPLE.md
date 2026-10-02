# 局部能量非单调性的可复核反例

## 精确条件

正文采用不加权局部能量 E_i=(1/2)sum_l(x_i-x_l)^2。对三角更新 x_i'=x_i+mu Phi_ijk d，记 d=x_j-x_i，k_i 为节点 i 的度，则

Delta E_i = mu Phi_ijk d sum_l(x_i-x_l) + (k_i mu^2 Phi_ijk^2/2)d^2.

因此局部能量不增加当且仅当 d sum_l(x_i-x_l) <= -(k_i mu Phi_ijk/2)d^2。该条件取决于当前局部意见配置，并不是异步随机更新自动满足的性质。

## Full Model 数值反例

取三节点完全图K3，i=0、j=1、k=2，意见分别为0.4、0.6、0.0。三条边的Jaccard权重均为1/3，C_i=1。使用 epsilon_0=0.15、alpha=3.5、mu=0.5、beta=3.0 时，三条意见差0.2、0.4、0.6均小于有效阈值0.735307，所以三角更新合法。

Phi_ijk=0.301194，mu Phi_ijk=0.150597，更新后x_i'=0.430119。被选边能量由0.020000降至0.014430，但节点i的完整局部边能量由0.100000升至0.106931，变化量为+0.006931。由于边j-k未变化，全图无向边能量也增加相同数值。

该例同时满足模型触发、三边门控和状态空间约束，因而直接证明选中边收缩不能推出局部或全局能量在每次异步更新后单调下降。

## 建议论文表述

The selected interaction edge contracts whenever an accepted update occurs. This edgewise contraction does not define a Lyapunov function for the asynchronous network process, because moving the updated node toward the selected source can simultaneously increase its disagreement with other neighbours. The exact local-energy change contains a direction-dependent cross term. The exact configuration-dependent non-increase condition is given in the Supplementary Methods, together with a three-node counterexample in which all triadic gates are satisfied, the selected edge energy decreases, but both the complete incident-edge energy and the global unweighted edge energy increase.
