"""
模块1: 网络拓扑与高阶结构
- 1.1 Holme-Kim网络生成
- 1.2 三角形提取
- 1.3 Jaccard权重计算
- 7.  网络属性验证
"""

import hashlib

import numpy as np
import networkx as nx
from collections import defaultdict


class NetworkGenerator:
    """网络生成与高阶结构提取类
    
    提供静态方法用于生成 Holme-Kim 网络、提取三角形结构、
    计算边权重和网络拓扑属性。
    """

    @staticmethod
    def generate_holme_kim(N, m, p, seed=None):
        """1.1 Holme-Kim 模型网络生成
        
        使用 networkx 的 powerlaw_cluster_graph 实现 Holme-Kim 模型，
        该模型是 Barabási-Albert 模型的扩展，增加了三角形形成机制。
        
        参数:
            N: 网络节点总数
            m: 每个新节点连接的边数
            p: 三角形成概率（Holme-Kim 模型的关键参数）
            seed: 随机种子（用于结果复现）
        返回:
            G: 生成的 NetworkX 图对象
        """
        # 调用 NetworkX 的 Holme-Kim 模型生成函数
        G = nx.powerlaw_cluster_graph(N, m, p, seed=seed)
        return G

    @staticmethod
    def generate_two_block_sbm(N, target_degree=8.0, mixing=0.1,
                               seed=None):
        """Generate a balanced two-community stochastic block model.

        ``mixing`` is the expected fraction of a node's edges that go to the
        other community. Probabilities are derived to keep the expected mean
        degree close to ``target_degree`` across mixing conditions.
        """
        if N < 4:
            raise ValueError("N must be at least 4 for a two-block SBM")
        if not 0.0 <= mixing <= 1.0:
            raise ValueError("mixing must lie in [0, 1]")
        n0 = int(N // 2)
        n1 = int(N - n0)
        k_in = (1.0 - float(mixing)) * float(target_degree)
        k_out = float(mixing) * float(target_degree)
        p_in_0 = min(1.0, k_in / max(n0 - 1, 1))
        p_in_1 = min(1.0, k_in / max(n1 - 1, 1))
        p_out = min(1.0, k_out / max((n0 + n1) / 2.0, 1.0))
        probabilities = [
            [p_in_0, p_out],
            [p_out, p_in_1],
        ]
        G = nx.stochastic_block_model(
            [n0, n1], probabilities, seed=seed,
            selfloops=False, sparse=True,
        )
        membership = {
            node: int(G.nodes[node].get('block', 0)) for node in G.nodes()
        }
        return G, membership, {
            'sizes': [n0, n1],
            'target_degree': float(target_degree),
            'mixing': float(mixing),
            'probability_matrix': probabilities,
        }

    @staticmethod
    def load_empirical_benchmark(name):
        """Load a small, public social-network benchmark bundled with NetworkX."""
        key = str(name).strip().lower()
        if key in {'karate', 'karate_club'}:
            raw = nx.karate_club_graph()
            citation = 'Zachary karate club social network'
        elif key in {'davis', 'southern_women', 'davis_southern_women'}:
            bipartite_graph = nx.davis_southern_women_graph()
            women = [node for node, attrs in bipartite_graph.nodes(data=True)
                     if attrs.get('bipartite') == 0]
            raw = nx.algorithms.bipartite.projected_graph(
                bipartite_graph, women)
            citation = 'Davis, Gardner and Gardner Southern Women projection'
        elif key in {'florentine', 'florentine_families'}:
            raw = nx.florentine_families_graph()
            citation = 'Padgett Florentine families social network'
        else:
            raise ValueError(
                "unknown empirical benchmark; choose karate, davis, or florentine")
        relabeled = nx.convert_node_labels_to_integers(
            nx.Graph(raw), ordering='sorted', label_attribute='original_label')
        # Some NetworkX projections build adjacency dictionaries from sets.
        # Their insertion order can therefore vary across Python processes as
        # string hash randomization changes.  Since the dynamics samples a
        # random neighbour index, canonical node/edge insertion is required
        # for a seed to identify the same trajectory on every process.
        graph = nx.Graph()
        for node in sorted(relabeled.nodes()):
            graph.add_node(node, **dict(relabeled.nodes[node]))
        canonical_edges = sorted(
            (min(u, v), max(u, v), dict(attrs))
            for u, v, attrs in relabeled.edges(data=True)
            if u != v
        )
        graph.add_edges_from(canonical_edges)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        edge_payload = ';'.join(
            f'{min(u, v)}-{max(u, v)}' for u, v in sorted(graph.edges()))
        return graph, {
            'benchmark': key,
            'citation_label': citation,
            'topology_replicates': 1,
            'num_nodes': int(graph.number_of_nodes()),
            'num_edges': int(graph.number_of_edges()),
            'canonical_edge_sha256': hashlib.sha256(
                edge_payload.encode('ascii')).hexdigest(),
        }

    @staticmethod
    def extract_triangles(G):
        """1.2 高阶结构提取 - 枚举图中所有三角形
        
        三角形是三个节点互相连接形成的闭合结构，
        是高阶网络结构的基本单元。
        
        参数:
            G: NetworkX 图对象
        返回:
            triangles_list: 三角形列表，每个元素是 (i,j,k) 元组
            node_triangles: 字典，键为节点ID，值为包含该节点的三角形索引列表
        """
        # 使用集合存储三角形，自动去重
        triangle_set = set()
        # 遍历所有节点
        for node in G.nodes():
            # 获取当前节点的邻居列表
            neighbors = list(G.neighbors(node))
            # 遍历邻居对，检查是否形成三角形
            for a_idx in range(len(neighbors)):
                for b_idx in range(a_idx + 1, len(neighbors)):
                    # 获取邻居对
                    a, b = neighbors[a_idx], neighbors[b_idx]
                    # 如果两个邻居之间有边，则形成三角形
                    if G.has_edge(a, b):
                        # 对三角形节点排序后转为元组，确保唯一性
                        tri = tuple(sorted([node, a, b]))
                        triangle_set.add(tri)

        # 将集合转换为列表，建立索引
        # Set iteration order is not guaranteed across Python processes.
        triangles_list = sorted(triangle_set)

        # 建立节点到三角形的索引映射
        node_triangles = defaultdict(list)
        for idx, tri in enumerate(triangles_list):
            # 三角形的三个节点都添加该三角形索引
            for node in tri:
                node_triangles[node].append(idx)

        return triangles_list, dict(node_triangles)

    @staticmethod
    def compute_jaccard_weights(G):
        """1.3 Jaccard 相似度作为边权重计算
        
        Jaccard 相似度衡量两条边的共同邻居比例：
        w_ij = |N(i)∩N(j)| / |N(i)∪N(j)|
        
        参数:
            G: NetworkX 图对象
        返回:
            weights: 字典，键为 (u,v) 元组，值为 Jaccard 权重
        """
        weights = {}
        # 遍历图中所有边
        for u, v in G.edges():
            # 获取两个节点的邻居集合
            nu = set(G.neighbors(u))
            nv = set(G.neighbors(v))
            # 计算并集大小
            union_size = len(nu | nv)
            # 计算 Jaccard 相似度，避免除零
            w = len(nu & nv) / union_size if union_size > 0 else 0.0
            # 存储双向权重（无向图）
            weights[(u, v)] = w
            weights[(v, u)] = w
        return weights

    @staticmethod
    def compute_clustering(G):
        """计算每个节点的局部聚类系数
        
        局部聚类系数衡量一个节点的邻居之间互相连接的程度。
        
        参数:
            G: NetworkX 图对象
        返回:
            字典，键为节点ID，值为局部聚类系数
        """
        return nx.clustering(G)

    @staticmethod
    def compute_f_triangle(G):
        """计算全局三角形占比 f_△
        
        f_△ = 属于至少一个三角形的边数 / 总边数
        
        这个指标反映了网络中高阶结构的丰富程度。
        
        参数:
            G: NetworkX 图对象
        返回:
            f_triangle: 全局三角形占比（0-1之间的浮点数）
        """
        # 使用集合存储属于三角形的边
        edges_in_tri = set()
        # 遍历所有节点
        for node in G.nodes():
            # 获取邻居列表
            neighbors = list(G.neighbors(node))
            # 检查邻居对是否形成三角形
            for a_idx in range(len(neighbors)):
                for b_idx in range(a_idx + 1, len(neighbors)):
                    a, b = neighbors[a_idx], neighbors[b_idx]
                    if G.has_edge(a, b):
                        # 将三角形的三条边都加入集合
                        edges_in_tri.add(tuple(sorted([node, a])))
                        edges_in_tri.add(tuple(sorted([node, b])))
                        edges_in_tri.add(tuple(sorted([a, b])))
        # 获取总边数
        total_edges = G.number_of_edges()
        # 计算比例，避免除零
        return len(edges_in_tri) / total_edges if total_edges > 0 else 0.0

    @staticmethod
    def compute_full_properties(G, seed=None):
        """7. 网络属性验证 - 完整拓扑属性计算
        
        计算网络的各种拓扑属性，用于验证网络生成和分析。
        
        参数:
            G: NetworkX 图对象
        返回:
            props: 字典，包含各种网络属性
        """
        props = {}
        # 基本统计量
        props['num_nodes'] = G.number_of_nodes()  # 节点数
        props['num_edges'] = G.number_of_edges()  # 边数
        props['avg_degree'] = 2.0 * G.number_of_edges() / G.number_of_nodes()  # 平均度数

        # 聚类系数相关
        clustering = nx.clustering(G)  # 每个节点的局部聚类系数
        props['avg_clustering'] = float(np.mean(list(clustering.values())))  # 平均聚类系数
        props['global_clustering'] = nx.transitivity(G)  # 全局聚类系数（传递性）

        # 三角形统计
        tri_per_node = nx.triangles(G)  # 每个节点参与的三角形数量
        props['num_triangles'] = sum(tri_per_node.values()) // 3  # 三角形总数（每个被计数3次）
        props['f_triangle'] = NetworkGenerator.compute_f_triangle(G)  # 全局三角形占比

        # 平均路径长度（采样计算，避免大图计算过慢）
        try:
            # 检查图是否连通
            if nx.is_connected(G):
                subG = G  # 连通则使用全图
            else:
                # 不连通则使用最大连通分量
                largest_cc = max(nx.connected_components(G), key=len)
                subG = G.subgraph(largest_cc).copy()

            # 采样节点计算平均路径长度
            nodes_sample = list(subG.nodes())
            if len(nodes_sample) > 80:
                # 大图时随机采样80个节点，避免网络验证过慢
                rng = np.random.RandomState(seed)
                nodes_sample = list(rng.choice(nodes_sample, 80, replace=False))
            lengths = []
            # 计算从每个采样节点到其他所有节点的最短路径
            for s in nodes_sample:
                sp = nx.single_source_shortest_path_length(subG, s)
                lengths.extend(sp.values())
            props['avg_path_length'] = float(np.mean(lengths))
        except Exception:
            # 计算失败时返回 NaN
            props['avg_path_length'] = float('nan')

        # 度分布统计
        degrees = [d for _, d in G.degree()]
        props['degrees'] = degrees  # 所有节点的度数列表
        props['max_degree'] = max(degrees)  # 最大度数

        # 模块度（社区结构强度）
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            from networkx.algorithms.community.quality import modularity

            # 用简单无向图计算，避免自环或多重边影响
            G_mod = nx.Graph(G)
            try:
                G_mod.remove_edges_from(nx.selfloop_edges(G_mod))
            except Exception:
                pass

            if G_mod.number_of_nodes() > 0 and G_mod.number_of_edges() > 0:
                comms = list(greedy_modularity_communities(G_mod))
                props['modularity'] = float(modularity(G_mod, comms))
            else:
                props['modularity'] = None

        except Exception as e:
            print(f"[WARN] modularity calculation failed: {e}")
            props['modularity'] = None

        return props

    @staticmethod
    def compute_threshold_diagnostics(G, weights, trust_model):
        """Describe edge weights and the nonlinearity of the effective threshold.

        The threshold map is concave, so evaluating it at the mean weight is not
        the same as averaging edge-specific thresholds. ``jensen_gap`` reports
        ``epsilon(mean(w)) - mean(epsilon(w))``.
        """
        undirected_edges = list(G.edges())
        if not undirected_edges:
            return {
                'n_edges': 0,
                'n_nontriangle_edges': 0,
                'fraction_nontriangle_edges': 0.0,
                'mean_weight': 0.0,
                'mean_effective_threshold': float(trust_model.epsilon_0),
                'effective_threshold_at_mean_weight': float(trust_model.epsilon_0),
                'jensen_gap': 0.0,
            }

        edge_weights = np.asarray(
            [weights.get((u, v), weights.get((v, u), 0.0))
             for u, v in undirected_edges],
            dtype=float,
        )
        edge_thresholds = np.asarray(
            [trust_model.effective_threshold(w) for w in edge_weights],
            dtype=float,
        )
        mean_weight = float(np.mean(edge_weights))
        threshold_at_mean = float(trust_model.effective_threshold(mean_weight))
        mean_threshold = float(np.mean(edge_thresholds))
        n_nontriangle = int(np.count_nonzero(edge_weights == 0.0))
        return {
            'n_edges': int(edge_weights.size),
            'n_nontriangle_edges': n_nontriangle,
            'fraction_nontriangle_edges': float(n_nontriangle / edge_weights.size),
            'mean_weight': mean_weight,
            'mean_effective_threshold': mean_threshold,
            'effective_threshold_at_mean_weight': threshold_at_mean,
            'jensen_gap': float(threshold_at_mean - mean_threshold),
        }
