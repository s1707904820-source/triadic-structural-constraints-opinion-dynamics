"""
模块5: 关键观测指标
- 意见簇数量
- 序参数(最大簇相对大小)
- 意见方差
- 固定尺度三角对齐率与规则相关可交互率
- 边平均意见不一致惩罚
- 磁滞面积
- 峰值灵敏度
"""

import numpy as np
from decimal import Decimal, getcontext
try:
    from .config import METRICS, get_triangle_consensus_eps
except ImportError:  # pragma: no cover
    from config import METRICS, get_triangle_consensus_eps

getcontext().prec = 50


def check_order_param(order_param_value):
    """将 order parameter 限制在 [0, 1] 区间
    
    用于处理数值计算中的精度问题，确保有序参数在有效范围内。
    
    参数:
        order_param_value: 输入的有序参数值
    返回:
        限制在 [0,1] 范围内的值（Decimal 输入返回 Decimal，其他返回 float）
    """
    # 处理 Decimal 类型输入
    if isinstance(order_param_value, Decimal):
        if order_param_value < Decimal(0):
            return Decimal(0)
        if order_param_value > Decimal(1):
            return Decimal(1)
        return order_param_value

    # 尝试转换为 float
    try:
        v = float(order_param_value)
    except Exception:
        return order_param_value

    # 限制在 [0,1] 范围
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class Metrics:
    """观测指标计算类
    
    提供各种静态方法用于计算意见动力学系统的观测指标。
    """

    @staticmethod
    def _consensus_width(width=None):
        """Return the opinion diameter used to define one consensus cluster."""
        if width is None:
            width = METRICS.get('cluster_diameter_threshold',
                                METRICS.get('cluster_gap_threshold', 0.05))
        return float(width)

    @staticmethod
    def _clusters_by_diameter(opinions_array, width=None):
        """Greedy 1D clusters whose internal opinion diameter is <= width.

        This avoids the single-linkage/density chaining problem: a dense but
        broadly spread distribution is no longer counted as one consensus
        cluster merely because adjacent samples are close when N is large.
        """
        width = Metrics._consensus_width(width)
        x = np.sort(np.asarray(opinions_array, dtype=float))
        n = len(x)
        if n == 0:
            return []

        clusters = []
        start = 0
        for i in range(1, n):
            if x[i] - x[start] > width:
                clusters.append(i - start)
                start = i
        clusters.append(n - start)
        return clusters

    @staticmethod
    def _max_window_count(opinions_array, width=None):
        """Largest number of agents inside any interval of length width."""
        width = Metrics._consensus_width(width)
        x = np.sort(np.asarray(opinions_array, dtype=float))
        n = len(x)
        if n == 0:
            return 0

        best = 1
        left = 0
        for right in range(n):
            while x[right] - x[left] > width:
                left += 1
            best = max(best, right - left + 1)
        return best

    @staticmethod
    def count_clusters(opinions_array, **kwargs):
        clusters = Metrics._clusters_by_diameter(
            opinions_array, kwargs.get('width')
        )
        return len(clusters)

    @staticmethod
    def order_parameter(opinions_array, **kwargs):
        n = len(opinions_array)
        if n == 0:
            return 0.0
        return float(Metrics._max_window_count(
            opinions_array, kwargs.get('width')
        ) / n)
    
    @staticmethod
    def effective_number_of_clusters(opinions_array, gap=None):
        """有效簇数（基于 Herfindahl 指数的倒数）
        
        N_eff = 1 / Σ(s_k²)，其中 s_k 是第 k 个簇的相对大小
        完全共识: N_eff=1；完全碎片化(均匀k簇): N_eff=k
        
        参数:
            opinions_array: 意见值数组
            gap: 簇间间隔阈值（默认使用 METRICS 配置）
        返回:
            有效簇数（浮点数）
        """
        # 使用默认阈值
        if gap is None:
            gap = METRICS['cluster_gap_threshold']
        # 空数组返回0
        if len(opinions_array) == 0:
            return 0.0
        # 排序意见值
        s = np.sort(opinions_array)
        # 划分簇
        clusters = []
        current_size = 1
        for idx in range(1, len(s)):
            if s[idx] - s[idx - 1] <= gap:
                current_size += 1
            else:
                clusters.append(current_size)
                current_size = 1
        clusters.append(current_size)

        # 计算 Herfindahl 指数
        total = len(opinions_array)
        fracs = np.array(clusters) / total
        hhi = np.sum(fracs ** 2)
        # 返回有效簇数
        return 1.0 / hhi if hhi > 0 else 0.0

    @staticmethod
    def opinion_variance(opinions_array):
        """意见方差
        
        衡量意见分布的离散程度。
        
        参数:
            opinions_array: 意见值数组
        返回:
            方差值（浮点数）
        """
        return float(np.var(opinions_array))

    @staticmethod
    def vector_compact_order(opinions_array, diameter=None, chunk_size=256):
        """Largest observed-centred Euclidean ball occupancy.

        The ball radius is half of the one-dimensional consensus-window
        diameter, making the reported spatial scale explicit.
        """
        points = np.asarray(opinions_array, dtype=float)
        if points.ndim != 2:
            raise ValueError("vector opinions must be a 2D array")
        n = len(points)
        if n == 0:
            return 0.0
        radius = Metrics._consensus_width(diameter) / 2.0
        radius_sq = radius * radius
        best = 1
        for start in range(0, n, int(chunk_size)):
            centers = points[start:start + int(chunk_size)]
            dist_sq = np.sum(
                (centers[:, None, :] - points[None, :, :]) ** 2,
                axis=2,
            )
            best = max(best, int(np.max(np.sum(
                dist_sq <= radius_sq + 1e-15, axis=1))))
        return float(best / n)

    @staticmethod
    def vector_total_variance(opinions_array):
        points = np.asarray(opinions_array, dtype=float)
        if points.ndim != 2 or len(points) == 0:
            return 0.0
        return float(np.sum(np.var(points, axis=0)))

    @staticmethod
    def vector_mean_pairwise_distance(opinions_array, rng=None,
                                      max_pairs=10000):
        points = np.asarray(opinions_array, dtype=float)
        n = len(points)
        if n < 2:
            return 0.0
        total_pairs = n * (n - 1) // 2
        if total_pairs <= max_pairs:
            values = [
                np.linalg.norm(points[i] - points[j])
                for i in range(n) for j in range(i + 1, n)
            ]
        else:
            if rng is None:
                rng = np.random.RandomState(0)
            first = rng.randint(0, n, size=int(max_pairs))
            second = rng.randint(0, n - 1, size=int(max_pairs))
            second = second + (second >= first)
            values = np.linalg.norm(points[first] - points[second], axis=1)
        return float(np.mean(values))

    @staticmethod
    def vector_triangle_alignment_fixed(opinions_dict, triangles_list,
                                        epsilon):
        if not triangles_list:
            return 0.0
        count = 0
        for i, j, k in triangles_list:
            if (np.linalg.norm(opinions_dict[i] - opinions_dict[j]) <= epsilon
                    and np.linalg.norm(opinions_dict[j] - opinions_dict[k]) <= epsilon
                    and np.linalg.norm(opinions_dict[i] - opinions_dict[k]) <= epsilon):
                count += 1
        return float(count / len(triangles_list))

    @staticmethod
    def vector_triangle_admissibility_adaptive(
            opinions_dict, triangles_list, weights, trust_model):
        if not triangles_list:
            return 0.0
        count = 0
        for i, j, k in triangles_list:
            ok = True
            for a, b in ((i, j), (j, k), (i, k)):
                threshold = trust_model.effective_threshold(
                    weights.get((a, b), weights.get((b, a), 0.0)))
                if np.linalg.norm(
                        opinions_dict[a] - opinions_dict[b]) > threshold:
                    ok = False
                    break
            count += int(ok)
        return float(count / len(triangles_list))

    @staticmethod
    def triangle_alignment_fixed(opinions_dict, triangles_list, epsilon=None):
        """固定尺度三角形对齐率：三人互差均不超过同一个 ε 的比例。
        
        衡量三角形结构中达成共识的比例，反映高阶结构的稳定性。
        
        参数:
            opinions_dict: 意见字典 {节点ID: 意见值}
            triangles_list: 三角形列表 [(i,j,k), ...]
            epsilon: 共识阈值（默认使用配置值）
        返回:
            共识率（0-1之间的浮点数）
        """
        # 使用默认阈值
        if epsilon is None:
            epsilon = get_triangle_consensus_eps()
        # 无三角形返回0
        if len(triangles_list) == 0:
            return 0.0
        # 统计满足共识条件的三角形
        count = 0
        for tri in triangles_list:
            i, j, k = tri
            oi, oj, ok = opinions_dict[i], opinions_dict[j], opinions_dict[k]
            # 检查三人意见是否都在阈值范围内
            if (abs(oi - oj) <= epsilon and
                    abs(oj - ok) <= epsilon and
                    abs(oi - ok) <= epsilon):
                count += 1
        return count / len(triangles_list)

    @staticmethod
    def triangle_admissibility_adaptive(
            opinions_dict, triangles_list, weights, trust_model,
            edge_thresholds=None):
        """
        规则相关的三角形可交互率：三条边均满足各自的结构依赖阈值。
        
        参数:
            opinions_dict: 意见字典 {节点ID: 意见值}
            triangles_list: 三角形列表 [(i,j,k), ...]
            weights: 边权重字典 {(u,v): w, ...}
            trust_model: 信任模型实例，提供 effective_threshold 方法
        返回:
            共识率（0-1之间的浮点数）
        """
        if len(triangles_list) == 0:
            return 0.0

        count = 0
        for i, j, k in triangles_list:
            pairs = [(i, j), (j, k), (i, k)]
            ok = True
            for a, b in pairs:
                if edge_thresholds is not None and (a, b) in edge_thresholds:
                    eps = edge_thresholds[(a, b)]
                elif edge_thresholds is not None and (b, a) in edge_thresholds:
                    eps = edge_thresholds[(b, a)]
                else:
                    w = weights.get((a, b), weights.get((b, a), 0.0))
                    eps = trust_model.effective_threshold(w)
                if abs(opinions_dict[a] - opinions_dict[b]) > eps:
                    ok = False
                    break
            if ok:
                count += 1
        return count / len(triangles_list)

    @staticmethod
    def edge_disagreement_penalty(opinions_dict, G, beta, rng=None,
                                  max_edges=5000):
        """边平均意见不一致惩罚，而非第三方三元反馈指标。

        越高表示相邻意见差异越大。
        计算公式：1 - mean(exp(-β * |oi - oj|))
        
        参数:
            opinions_dict: 意见字典 {节点ID: 意见值}
            G: NetworkX 图对象
            beta: 意见差异的指数缩放系数
        返回:
            边平均不一致惩罚（0-1之间的浮点数）
        """
        # 获取所有边
        edges = list(G.edges())
        if len(edges) == 0:
            return 0.0
        # 边数过多时使用显式随机数生成器采样，以保证复现。
        if len(edges) <= max_edges or max_edges is None:
            sample = edges
        else:
            if rng is None:
                rng = np.random.RandomState(0)
            sample = [edges[i] for i in rng.choice(
                len(edges), int(max_edges), replace=False)]
        # 计算每条边的不一致惩罚
        penalties = []
        for u, v in sample:
            diff = abs(opinions_dict[u] - opinions_dict[v])
            penalties.append(1.0 - np.exp(-beta * diff))
        return float(np.mean(penalties))

    # ---- Backward-compatible aliases for old result-generation scripts ----
    @staticmethod
    def triangle_consensus_rate(opinions_dict, triangles_list, epsilon=None):
        return Metrics.triangle_alignment_fixed(
            opinions_dict, triangles_list, epsilon
        )

    @staticmethod
    def triangle_consensus_rate_adaptive(opinions_dict, triangles_list,
                                         weights, trust_model):
        return Metrics.triangle_admissibility_adaptive(
            opinions_dict, triangles_list, weights, trust_model
        )

    @staticmethod
    def reputation_fragmentation(opinions_dict, G, beta, rng=None):
        return Metrics.edge_disagreement_penalty(
            opinions_dict, G, beta, rng=rng
        )

    @staticmethod
    def hysteresis_area(forward, backward, params):
        """磁滞面积：正/逆向扫描曲线间面积（梯形法）
        
        用于衡量系统的路径依赖性，面积越大表示磁滞效应越强。
        
        参数:
            forward: 正向扫描的序参数值列表
            backward: 逆向扫描的序参数值列表
            params: 扫描参数值列表
        返回:
            磁滞面积（浮点数）
        """
        if len(forward) < 2:
            return 0.0
        area = 0.0
        # 使用梯形法计算面积
        for idx in range(len(params) - 1):
            dp = params[idx + 1] - params[idx]
            # 梯形高度（两端的平均值）
            h = abs(forward[idx] - backward[idx]) + abs(forward[idx + 1] - backward[idx + 1])
            area += 0.5 * h * abs(dp)
        return area

    @staticmethod
    def peak_sensitivity(values, params):
        """峰值灵敏度：序参数对参数的最大有限差分导数
        
        用于识别相变点的陡峭程度。
        
        参数:
            values: 序参数值列表
            params: 参数值列表
        返回:
            最大灵敏度（浮点数）
        """
        if len(values) < 2:
            return 0.0
        max_d = 0.0
        # 计算相邻点的差分导数
        for idx in range(len(values) - 1):
            dp = abs(params[idx + 1] - params[idx])
            if dp > 0:
                d = abs(values[idx + 1] - values[idx]) / dp
                max_d = max(max_d, d)
        return max_d
