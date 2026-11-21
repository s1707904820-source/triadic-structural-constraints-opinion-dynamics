"""Construct a gate-valid counterexample to global energy monotonicity."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics import TrustThreshold  # noqa: E402
from network import NetworkGenerator  # noqa: E402


OUTPUT_DIR = ROOT / "output" / "formal_revision" / "analysis"
JSON_PATH = OUTPUT_DIR / "formal_local_energy_counterexample.json"
REPORT_PATH = OUTPUT_DIR / "FORMAL_LOCAL_ENERGY_COUNTEREXAMPLE.md"


def weighted_global_edge_energy(graph, weights, opinions):
    return float(sum(
        weights[(u, v)] * (opinions[u] - opinions[v]) ** 2
        for u, v in graph.edges()
    ))


def weighted_local_edge_energy(graph, weights, opinions, node):
    return float(sum(
        weights[(node, neighbor)]
        * (opinions[node] - opinions[neighbor]) ** 2
        for neighbor in graph.neighbors(node)
    ))


def main():
    graph = nx.complete_graph(3)
    weights = NetworkGenerator.compute_jaccard_weights(graph)
    clustering = NetworkGenerator.compute_clustering(graph)
    epsilon_0 = 0.15
    alpha = 3.5
    mu = 0.5
    beta = 3.0
    i, j, k = 0, 1, 2
    before_opinions = {i: 0.4, j: 0.6, k: 0.0}
    trust = TrustThreshold(epsilon_0, alpha)

    phi = math.exp(-beta * abs(before_opinions[i] - before_opinions[k]))
    update_fraction = mu * phi
    d = before_opinions[j] - before_opinions[i]
    after_opinions = dict(before_opinions)
    after_opinions[i] = before_opinions[i] + update_fraction * d

    gates = {}
    for u, v in ((i, j), (i, k), (j, k)):
        gap = abs(before_opinions[u] - before_opinions[v])
        threshold = float(trust.effective_threshold(weights[(u, v)]))
        gates["{}-{}".format(u, v)] = {
            "opinion_gap": float(gap),
            "effective_threshold": threshold,
            "passes": bool(gap <= threshold),
        }

    local_before = weighted_local_edge_energy(
        graph, weights, before_opinions, i)
    local_after = weighted_local_edge_energy(
        graph, weights, after_opinions, i)
    global_before = weighted_global_edge_energy(
        graph, weights, before_opinions)
    global_after = weighted_global_edge_energy(
        graph, weights, after_opinions)
    selected_before = weights[(i, j)] * (
        before_opinions[i] - before_opinions[j]) ** 2
    selected_after = weights[(i, j)] * (
        after_opinions[i] - after_opinions[j]) ** 2

    other_neighbors = [node for node in graph.neighbors(i) if node != j]
    W = float(sum(weights[(i, node)] for node in other_neighbors))
    S = float(sum(
        weights[(i, node)]
        * (before_opinions[i] - before_opinions[node])
        for node in other_neighbors
    ))
    weighted_other_mean = float(sum(
        weights[(i, node)] * before_opinions[node]
        for node in other_neighbors
    ) / W)
    bracket = float(
        weights[(i, j)] * (2.0 - update_fraction)
        - update_fraction * W)
    weak_deviation_bound = float(
        bracket * abs(d) / (2.0 * W)) if W > 0 and bracket >= 0 else None
    actual_deviation = float(abs(
        before_opinions[i] - weighted_other_mean))

    payload = {
        "graph": "three-node complete graph K3",
        "roles": {"updating_node_i": i, "source_j": j, "third_party_k": k},
        "parameters": {
            "epsilon_0": epsilon_0,
            "alpha": alpha,
            "mu": mu,
            "beta": beta,
            "local_clustering_i": float(clustering[i]),
            "edge_jaccard_weight": float(weights[(i, j)]),
        },
        "opinions_before": before_opinions,
        "opinions_after": after_opinions,
        "all_three_gates": gates,
        "third_party_factor_phi": float(phi),
        "effective_update_fraction_lambda": float(update_fraction),
        "selected_edge_energy": {
            "before": float(selected_before),
            "after": float(selected_after),
            "change": float(selected_after - selected_before),
        },
        "local_incident_edge_energy": {
            "before": local_before,
            "after": local_after,
            "change": float(local_after - local_before),
        },
        "global_undirected_edge_energy": {
            "before": global_before,
            "after": global_after,
            "change": float(global_after - global_before),
        },
        "weak_deviation_condition": {
            "other_neighbor_weight_sum_W": W,
            "cross_term_sum_S": S,
            "weighted_other_neighbor_mean": weighted_other_mean,
            "bracket_wij_2_minus_q_minus_qW": bracket,
            "sufficient_absolute_deviation_bound": weak_deviation_bound,
            "actual_absolute_deviation": actual_deviation,
            "bound_satisfied": bool(
                weak_deviation_bound is not None
                and actual_deviation <= weak_deviation_bound),
        },
        "validation": {
            "triangle_trigger_probability_is_one": bool(
                math.isclose(float(clustering[i]), 1.0)),
            "all_gates_pass": bool(all(item["passes"] for item in gates.values())),
            "selected_edge_decreases": bool(selected_after < selected_before),
            "local_energy_increases": bool(local_after > local_before),
            "global_energy_increases": bool(global_after > global_before),
        },
    }
    if not all(payload["validation"].values()):
        raise RuntimeError("counterexample validation failed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write("# 局部能量非单调性的可复核反例\n\n")
        handle.write("## 精确条件\n\n")
        handle.write(
            "令 q 表示一次更新的实际步长比例。普通边更新中 q=mu；"
            "完整三角更新中 q=mu Phi_ijk。记 d=x_j-x_i，"
            "W=sum_{l!=j} w_il，S=sum_{l!=j} w_il(x_i-x_l)。则\n\n")
        handle.write(
            "Delta V_E^(i) = q[-w_ij(2-q)d^2 + 2dS + qWd^2].\n\n")
        handle.write(
            "因此局部能量不增加的精确方向性条件是 "
            "2dS <= [w_ij(2-q)-qW]d^2。若以其他邻居的加权均值"
            " xbar 定义 S=W(x_i-xbar)，一个便于报告的充分条件是\n\n")
        handle.write(
            "w_ij(2-q)-qW >= 0，且 |x_i-xbar| <= "
            "[w_ij(2-q)-qW]|d|/(2W)。\n\n")
        handle.write(
            "这一定量条件说明原稿中的交叉项近似消失，实质上要求更新节点"
            "接近其余邻居的加权意见均值；它不是异步随机更新自动满足的性质。\n\n")
        handle.write("## Full Model 数值反例\n\n")
        handle.write(
            "取三节点完全图K3，i=0、j=1、k=2，意见分别为0.4、0.6、0.0。"
            "三条边的Jaccard权重均为1/3，C_i=1。使用 epsilon_0=0.15、"
            "alpha=3.5、mu=0.5、beta=3.0 时，三条意见差0.2、0.4、0.6"
            "均小于有效阈值{:.6f}，所以三角更新合法。\n\n".format(
                gates["0-1"]["effective_threshold"]))
        handle.write(
            "Phi_ijk={:.6f}，q=lambda={:.6f}，更新后x_i'={:.6f}。"
            "被选边能量由{:.6f}降至{:.6f}，但节点i的完整局部边能量"
            "由{:.6f}升至{:.6f}，变化量为+{:.6f}。由于边j-k未变化，"
            "全图无向边能量也增加相同数值。\n\n".format(
                phi, update_fraction, after_opinions[i], selected_before,
                selected_after, local_before, local_after,
                local_after - local_before))
        handle.write(
            "该例同时满足模型触发、三边门控和状态空间约束，因而直接证明"
            "选中边收缩不能推出局部或全局能量在每次异步更新后单调下降。\n\n")
        handle.write("## 建议论文表述\n\n")
        handle.write(
            "The selected interaction edge contracts whenever an accepted update "
            "occurs. This edgewise contraction does not define a Lyapunov function "
            "for the asynchronous network process, because moving the updated node "
            "toward the selected source can simultaneously increase its disagreement "
            "with other neighbours. The exact local-energy change contains a "
            "direction-dependent cross term. A sufficient weak-deviation condition "
            "is given in the Supplementary Methods, together with a three-node "
            "counterexample in which all triadic gates are satisfied, the selected "
            "edge energy decreases, but both the complete incident-edge energy and "
            "the global weighted edge energy increase.\n")

    print("VALIDATION=PASS")
    print("SELECTED_EDGE_CHANGE={:.12g}".format(
        selected_after - selected_before))
    print("LOCAL_ENERGY_CHANGE={:.12g}".format(local_after - local_before))
    print("GLOBAL_ENERGY_CHANGE={:.12g}".format(global_after - global_before))
    print("WROTE={}".format(JSON_PATH))
    print("WROTE={}".format(REPORT_PATH))


if __name__ == "__main__":
    main()
