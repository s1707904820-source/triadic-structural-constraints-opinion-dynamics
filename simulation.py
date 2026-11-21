"""
模块4+5: 仿真控制器
管理单次实验运行的完整流程：
网络生成 → 结构提取 → 意见初始化 → 动力学运行 → 指标计算
"""

import numpy as np
from network import NetworkGenerator
from dynamics import TrustThreshold, OpinionDynamics
from metrics import Metrics
from config import SIMULATION


class SingleRun:
    """单次仿真运行类
    
    管理一次完整的仿真实验，包括网络生成、动力学演化和指标计算。
    """

    def __init__(self, N=2000, m=4, p_hk=0.6, epsilon_0=0.15, alpha=3.5,
                 mu=0.5, beta=3.0, model_type='full_model',
                 num_steps=None, seed=None, convergence_check=None,
                 convergence_tol=None, convergence_patience=None,
                 convergence_min_steps=None, update_mode='asynchronous',
                 graph=None, initial_opinions=None,
                 network_type='holme_kim', network_params=None):
        """初始化单次仿真运行
        
        参数:
            N: 网络节点数（默认2000）
            m: Holme-Kim 模型参数，每个新节点连接的边数（默认4）
            p_hk: 三角形成概率（默认0.6）
            epsilon_0: 基础信任阈值（默认0.15）
            alpha: 权重敏感度指数（默认3.5）
            mu: 基础更新步长（默认0.5）
            beta: 第三方不一致衰减系数（默认3.0）
            model_type: 动力学模型类型（默认'full_model'）
            num_steps: 仿真总步数（默认 N × SIMULATION['main_sweeps']）
            seed: 随机种子（默认None）
        """
        self.graph = None if graph is None else graph.copy()
        self.N = (int(N) if self.graph is None
                  else int(self.graph.number_of_nodes()))
        self.m = m                           # 存储Holme-Kim参数m
        self.p_hk = p_hk                     # 存储三角形成概率
        self.epsilon_0 = epsilon_0           # 存储基础阈值
        self.alpha = alpha                   # 存储敏感度指数
        self.mu = mu                         # 存储更新步长
        self.beta = beta                     # 存储第三方不一致衰减系数
        self.model_type = model_type         # 存储模型类型
        self.num_steps = (                   # 存储仿真步数
            int(round(self.N * float(SIMULATION.get('main_sweeps', 200))))
            if num_steps is None else int(num_steps))
        self.seed = seed                     # 存储随机种子
        if update_mode not in ('asynchronous', 'synchronous'):
            raise ValueError(
                "update_mode must be 'asynchronous' or 'synchronous'")
        self.update_mode = update_mode
        self.initial_opinions = (
            None if initial_opinions is None else dict(initial_opinions))
        self.network_type = str(network_type)
        self.network_params = dict(network_params or {})
        self.convergence_check = (
            SIMULATION.get('use_convergence_check', False)
            if convergence_check is None else bool(convergence_check)
        )
        self.convergence_tol = (
            SIMULATION.get('convergence_tol', 1e-7)
            if convergence_tol is None else float(convergence_tol)
        )
        self.convergence_patience = (
            SIMULATION.get('convergence_patience', 5)
            if convergence_patience is None else int(convergence_patience)
        )
        self.convergence_min_steps = (
            0 if convergence_min_steps is None else int(convergence_min_steps)
        )

    @staticmethod
    def _build_edge_threshold_assignment(G, weights, trust, model_type, seed):
        """Build a fixed edge-threshold assignment for control variants."""
        if model_type not in ('mean_threshold_only',
                              'randomized_threshold_only'):
            return None

        def canonical_edge(u, v):
            return (u, v) if repr(u) <= repr(v) else (v, u)

        edges = sorted(
            (canonical_edge(u, v) for u, v in G.edges()),
            key=lambda edge: (repr(edge[0]), repr(edge[1])),
        )
        thresholds = [
            float(trust.effective_threshold(
                weights.get((u, v), weights.get((v, u), 0.0))))
            for u, v in edges
        ]
        if not thresholds:
            return {}

        if model_type == 'mean_threshold_only':
            assigned = [float(np.mean(thresholds))] * len(thresholds)
        else:
            # Use an independent stream so this control does not alter update RNG.
            shuffle_seed = (int(seed or 0) + 32452843) % (2 ** 32)
            assigned = [float(value) for value in
                        np.random.RandomState(shuffle_seed).permutation(thresholds)]

        edge_thresholds = {}
        for (u, v), threshold in zip(edges, assigned):
            edge_thresholds[(u, v)] = threshold
            edge_thresholds[(v, u)] = threshold
        return edge_thresholds

    def execute(self, record_interval=1000, verbose=False):
        """执行仿真
        
        这是主要的仿真执行方法，完成从网络生成到结果输出的完整流程。
        
        参数:
            record_interval: 数据记录间隔步数（默认1000）
            verbose: 是否打印详细输出（默认False）
        返回:
            包含完整仿真结果的字典
        """
        # 创建随机数生成器，使用指定种子保证可复现性
        rng = np.random.RandomState(self.seed)

        # --- 1. 网络生成 ---
        # 使用 Holme-Kim 模型生成网络
        if self.graph is None:
            if self.network_type != 'holme_kim':
                raise ValueError(
                    "a pre-generated graph is required for non-Holme-Kim runs")
            G = NetworkGenerator.generate_holme_kim(
                self.N, self.m, self.p_hk, seed=self.seed)
        else:
            G = self.graph.copy()

        # --- 2. 三角形提取 ---
        # 从网络中提取所有三角形结构
        triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)

        # --- 3. Jaccard权重计算 ---
        # 计算所有边的 Jaccard 相似度作为权重
        weights = NetworkGenerator.compute_jaccard_weights(G)

        # --- 4. 聚类系数计算 ---
        # 计算每个节点的局部聚类系数
        clustering = NetworkGenerator.compute_clustering(G)

        # --- 5. 初始意见初始化 ---
        # 为每个节点生成 [0,1] 区间均匀分布的初始意见
        if self.initial_opinions is None:
            opinions = {n: rng.uniform(0, 1) for n in G.nodes()}
        else:
            missing = set(G.nodes()) - set(self.initial_opinions)
            extra = set(self.initial_opinions) - set(G.nodes())
            if missing or extra:
                raise ValueError(
                    f"initial_opinions node mismatch: missing={len(missing)}, "
                    f"extra={len(extra)}")
            opinions = {
                node: float(np.clip(self.initial_opinions[node], 0.0, 1.0))
                for node in G.nodes()
            }

        # --- 6. 信任阈值模型初始化 ---
        # 创建信任阈值模型实例
        trust = TrustThreshold(self.epsilon_0, self.alpha)
        edge_thresholds = self._build_edge_threshold_assignment(
            G, weights, trust, self.model_type, self.seed)

        # --- 7. 动力学引擎初始化 ---
        # 创建意见动力学引擎
        engine = OpinionDynamics(
            G, opinions, weights, triangles_list, node_triangles,
            clustering, trust, self.mu, self.beta, self.model_type,
            edge_thresholds=edge_thresholds,
        )

        record_interval = int(record_interval)
        if record_interval <= 0:
            raise ValueError("record_interval must be a positive integer")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if (self.update_mode == 'synchronous'
                and self.num_steps % max(self.N, 1) != 0):
            raise ValueError(
                "synchronous runs require num_steps to be divisible by N")
        if (self.update_mode == 'synchronous'
                and record_interval % max(self.N, 1) != 0):
            raise ValueError(
                "synchronous record_interval must be divisible by N")

        # --- 8. 时间序列记录 ---
        # 初始化时间序列数据字典
        ts = {
            'step': [], 'sweep': [], 'n_clusters': [], 'variance': [],
            'order_param': [], 'triangle_alignment_fixed': [],
            'triangle_admissibility_adaptive': [], 'mean_abs_dx': [],
            'accepted_update_rate': [],
        }

        # 为计算每个记录间隔内的平均 dx 做准备
        dx_acc = 0.0
        dx_count = 0
        converged = False
        convergence_step = None
        stable_windows = 0
        last_mean_dx = None
        last_update_count = 0
        step = 0

        def record_state(current_step):
            nonlocal dx_acc, dx_count, last_mean_dx, last_update_count
            mean_dx = dx_acc / dx_count if dx_count > 0 else 0.0
            current_updates = engine.get_stats()['update_count']
            accepted_rate = ((current_updates - last_update_count) / dx_count
                             if dx_count > 0 else 0.0)
            last_update_count = current_updates
            last_mean_dx = mean_dx
            dx_acc = 0.0
            dx_count = 0

            ops_arr = engine.get_opinions_array()
            ops_dict = engine.get_opinions()
            ts['step'].append(int(current_step))
            ts['sweep'].append(float(current_step / max(self.N, 1)))
            ts['n_clusters'].append(Metrics.count_clusters(ops_arr))
            ts['variance'].append(Metrics.opinion_variance(ops_arr))
            ts['order_param'].append(Metrics.order_parameter(ops_arr))
            ts['triangle_alignment_fixed'].append(
                Metrics.triangle_alignment_fixed(
                    ops_dict, triangles_list, self.epsilon_0))
            ts['triangle_admissibility_adaptive'].append(
                Metrics.triangle_admissibility_adaptive(
                    ops_dict, triangles_list, weights, trust,
                    edge_thresholds=edge_thresholds))
            ts['mean_abs_dx'].append(float(mean_dx))
            ts['accepted_update_rate'].append(float(accepted_rate))
            return mean_dx

        # 主仿真循环。同步模式每次推进 N 个基本更新机会。
        increment = self.N if self.update_mode == 'synchronous' else 1
        while step < self.num_steps:
            if self.update_mode == 'synchronous':
                dx_acc += engine.synchronous_sweep(rng)
                dx_count += self.N
            else:
                dx_acc += engine.step(rng)
                dx_count += 1
            step += increment

            if step % record_interval == 0:
                mean_dx = record_state(step)

                if verbose and step % (record_interval * 10) == 0:
                    print(f"    Step {step}: clusters={ts['n_clusters'][-1]}, "
                          f"OP={ts['order_param'][-1]:.4f}, var={ts['variance'][-1]:.6f}")

                if self.convergence_check and step >= self.convergence_min_steps:
                    if mean_dx <= self.convergence_tol:
                        stable_windows += 1
                    else:
                        stable_windows = 0
                    if stable_windows >= self.convergence_patience:
                        converged = True
                        convergence_step = step
                        if verbose:
                            print(f"    Converged at step {step}: mean_dx={mean_dx:.3e}")
                        break

        # Retain a terminal-window diagnostic when the horizon is not an exact
        # multiple of the recording interval.
        if dx_count > 0:
            record_state(step)

        # --- 9. 最终指标计算 ---
        # 获取最终意见状态
        final_ops_dict = engine.get_opinions()
        final_ops_arr = engine.get_opinions_array()
        # 计算全局三角形占比
        f_tri = NetworkGenerator.compute_f_triangle(G)

        # 组装结果字典
        # Keep metric subsampling independent from the model-specific
        # dynamics stream.  Otherwise two paired variants with the same seed
        # can evaluate edge disagreement on different random edge samples,
        # adding avoidable noise to the paired comparison.
        metric_seed = (0 if self.seed is None else int(self.seed))
        metric_seed = (metric_seed + 104729) % (2 ** 32)
        metric_rng = np.random.RandomState(metric_seed)

        result = {
            # 仿真参数
            'params': {
                'N': self.N, 'm': self.m, 'p_hk': self.p_hk,
                'epsilon_0': self.epsilon_0, 'alpha': self.alpha,
                'mu': self.mu, 'beta': self.beta,
                'model_type': self.model_type, 'seed': self.seed,
                'threshold_assignment': (
                    self.model_type if edge_thresholds is not None
                    else 'network_edge_specific'),
                'update_mode': self.update_mode,
                'network_type': self.network_type,
                'network_params': dict(self.network_params),
                'num_steps': self.num_steps,
                'num_sweeps': float(self.num_steps / max(self.N, 1)),
                'record_interval_steps': record_interval,
                'record_interval_sweeps': float(record_interval / max(self.N, 1)),
            },
            # 网络属性
            'network': {
                'num_triangles': len(triangles_list),
                'f_triangle': f_tri,
                'avg_clustering': float(np.mean(list(clustering.values()))),
                'threshold_diagnostics': NetworkGenerator.compute_threshold_diagnostics(
                    G, weights, trust),
            },
            # 最终状态指标
            'final': {
                'n_clusters': Metrics.count_clusters(final_ops_arr),
                'order_param': Metrics.order_parameter(final_ops_arr),
                'variance': Metrics.opinion_variance(final_ops_arr),
                'triangle_alignment_fixed': Metrics.triangle_alignment_fixed(
                    final_ops_dict, triangles_list, self.epsilon_0),
                'triangle_admissibility_adaptive':
                    Metrics.triangle_admissibility_adaptive(
                        final_ops_dict, triangles_list, weights, trust,
                        edge_thresholds=edge_thresholds),
                'edge_disagreement_penalty': Metrics.edge_disagreement_penalty(
                    final_ops_dict, G, self.beta, rng=metric_rng),
                'opinions': final_ops_arr.tolist(),
            },
            # 动力学统计
            'dynamics_stats': engine.get_stats(),
            'convergence': {
                'enabled': bool(self.convergence_check),
                'converged': bool(converged),
                'steps_run': int(step),
                'sweeps_run': float(step / max(self.N, 1)),
                'convergence_step': None if convergence_step is None else int(convergence_step),
                'final_mean_dx': None if last_mean_dx is None else float(last_mean_dx),
                'tol': float(self.convergence_tol),
                'patience': int(self.convergence_patience),
                'min_steps': int(self.convergence_min_steps),
            },
            # 时间序列数据
            'time_series': ts,
        }
        return result
