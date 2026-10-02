"""
模块2: 信任阈值机制
模块3: 意见动力学更新规则
包含5种模型变体用于消融实验(Ablation Study)
"""

import numpy as np


class TrustThreshold:
    """2. 饱和信任阈值模型
    
    实现基于边权重的动态阈值机制：
    ε_eff(w_ij) = ε₀ + (1 - ε₀) · (1 - exp(-α · w_ij))
    
    其中：
    - ε₀: 基础容忍阈值
    - α: 权重敏感度指数
    - w_ij: 边 (i,j) 的权重
    
    有效阈值严格限制在 [0, 1] 区间
    """

    def __init__(self, epsilon_0, alpha=3.5):
        """初始化信任阈值模型
        
        参数:
            epsilon_0: 基础容忍阈值
            alpha: 权重敏感度指数（默认3.5）
        """
        self.epsilon_0 = epsilon_0  # 存储基础阈值
        self.alpha = alpha          # 存储敏感度指数

    def effective_threshold(self, w_ij):
        """计算有效阈值
        
        根据边权重计算动态调整后的信任阈值。
        
        参数:
            w_ij: 边权重（0-1之间的值）
        返回:
            有效阈值（ε₀ 到 1 之间的值）
        """
        # 使用指数函数实现平滑过渡
        return self.epsilon_0 + (1.0 - self.epsilon_0) * (1.0 - np.exp(-self.alpha * w_ij))


class OpinionDynamics:
    """3. 意见动力学更新规则
    
    支持5种模型变体用于消融实验:
      - baseline / baseline_dw:       异步单节点固定阈值 BCM
      - weighted_bcm / weighted_dw:   仅Jaccard权重影响步长
      - threshold_only:               仅结构依赖、边特异阈值
      - bcm_triangle / triangle_fixed: 固定阈值 + 三角形触发（无第三方衰减）
      - full_model:                   完整模型（结构阈值 + C_i触发 + 第三方衰减）
      - no_triadic_feedback:          保留三角触发，但关闭第三方不一致衰减
    """

    # ---- 统一别名表：所有可能的外部名称 → 内部方法名后缀 ----
    # 这个字典用于将用户输入的模型类型映射到内部方法名
    _METHOD_ALIAS = {
        # 内部原始名
        'baseline':       '_step_baseline',
        'weighted_bcm':   '_step_weighted_bcm',
        'threshold_only': '_step_threshold_only',
        'bcm_triangle':   '_step_bcm_triangle',
        'full_model':     '_step_full_model',
        'no_triadic_feedback': '_step_no_triadic_feedback',
        # config 中可能使用的别名
        'baseline_dw':    '_step_baseline',
        'weighted_dw':    '_step_weighted_bcm',
        'trust':          '_step_threshold_only',
        'trust_only':     '_step_threshold_only',
        'triangle_fixed': '_step_bcm_triangle',
        'triangle_only':  '_step_no_triadic_feedback',
        # Legacy compatibility only; new runs must use the explicit name.
        'no_structure':   '_step_no_triadic_feedback',
        'triangle':       '_step_no_triadic_feedback',
        'full':           '_step_full_model',
        # 更多安全别名
        'weighted':       '_step_weighted_bcm',
    }

    def __init__(self, G, opinions, weights, triangles_list, node_triangles,
                 clustering, trust_model, mu=0.5, beta=3.0,
                 model_type='full_model'):
        """初始化意见动力学引擎
        
        参数:
            G: NetworkX 图对象
            opinions: 初始意见字典 {节点ID: 意见值}
            weights: 边权重字典 {(u,v): 权重值}
            triangles_list: 三角形列表 [(i,j,k), ...]
            node_triangles: 节点三角形索引字典 {节点ID: [三角形索引列表]}
            clustering: 局部聚类系数字典 {节点ID: 聚类系数}
            trust_model: TrustThreshold 实例
            mu: 基础更新步长（默认0.5）
            beta: 第三方不一致衰减系数（默认3.0）
            model_type: 模型类型字符串（默认'full_model'）
        """
        self.G = G                           # 存储网络图
        self.opinions = dict(opinions)       # 复制意见字典
        self.weights = weights               # 存储边权重
        self.triangles_list = triangles_list # 存储三角形列表
        self.node_triangles = node_triangles # 存储节点三角形映射
        self.clustering = clustering         # 存储聚类系数
        self.trust = trust_model             # 存储信任阈值模型
        self.mu = mu                         # 存储基础步长
        self.beta = beta                     # 存储第三方不一致衰减系数
        self.model_type = model_type         # 存储模型类型
        self.nodes = list(G.nodes())         # 获取节点列表
        self.N = len(self.nodes)             # 存储节点总数
        # 预计算邻居列表，避免重复查询
        self.neighbors = {n: list(G.neighbors(n)) for n in G.nodes()}

        # 预解析：在初始化时就确定要调用的方法，避免每步都查找（性能优化）
        self._step_fn = self._resolve_step_fn(model_type)

        # 交互统计字典，用于记录各种交互事件次数
        self.stats = {
            'edge_count': 0,      # 边交互次数
            'tri_count': 0,       # 三角形交互尝试次数
            'tri_success': 0,     # 三角形交互成功次数
            'tri_fallback': 0,    # 三角形交互失败回退次数
            'update_count': 0     # 意见更新总次数
        }

    def _resolve_step_fn(self, model_type):
        """根据 model_type 字符串，解析出对应的内部方法
        
        参数:
            model_type: 模型类型字符串
        返回:
            对应的方法引用
        抛出:
            KeyError: 如果模型类型无效
        """
        # 首先尝试精确匹配
        method_name = self._METHOD_ALIAS.get(model_type)
        if model_type == 'no_structure':
            import warnings
            warnings.warn(
                "model alias 'no_structure' is deprecated; use "
                "'no_triadic_feedback'",
                DeprecationWarning, stacklevel=2)
        if method_name is not None:
            return getattr(self, method_name)

        # 模糊匹配（大小写不敏感）
        mt_low = model_type.lower()
        for key, mname in self._METHOD_ALIAS.items():
            if mt_low in key or key in mt_low:
                return getattr(self, mname)

        # 未找到匹配，抛出错误
        raise KeyError(
            f"未知 model_type: '{model_type}'。"
            f"可用: {list(set(self._METHOD_ALIAS.keys()))}"
        )

    def step(self, rng):
        """执行一步异步更新
        
        这是仿真的核心方法，每步随机选择一个节点进行更新。
        
        参数:
            rng: NumPy 随机数生成器
        返回:
            本步意见更新幅度 dx
        """
        # 随机选择一个节点
        i = self.nodes[rng.randint(self.N)]
        # 如果节点没有邻居，跳过
        if len(self.neighbors[i]) == 0:
            return 0.0 # 没有更新，返回0
        
        old_x = self.opinions[i] # 记录更新前的意见
        # 调用预解析的更新方法
        self._step_fn(i, rng)
        
        return abs(self.opinions[i] - old_x) # 返回更新幅度

    def synchronous_sweep(self, rng):
        """Execute one synchronous sweep from a frozen opinion snapshot.

        Every node receives exactly one update opportunity. Interaction
        partners are sampled using the same model-specific rules as in the
        asynchronous engine, but every proposal is evaluated against the
        opinions at the beginning of the sweep and committed simultaneously.

        Returns
        -------
        float
            Sum of absolute node-level changes in this sweep. Dividing by N
            gives the mean absolute change per elementary update opportunity.
        """
        snapshot = dict(self.opinions)
        proposals = dict(snapshot)
        total_abs_change = 0.0
        # Random order is used only to consume random variates without a fixed
        # node-order artifact; proposals never see earlier proposals.
        for raw_i in rng.permutation(self.nodes):
            i = int(raw_i) if isinstance(raw_i, np.integer) else raw_i
            if len(self.neighbors[i]) == 0:
                continue
            self.opinions[i] = snapshot[i]
            self._step_fn(i, rng)
            proposals[i] = self.opinions[i]
            total_abs_change += abs(proposals[i] - snapshot[i])
            self.opinions[i] = snapshot[i]
        self.opinions = proposals
        return float(total_abs_change)

    # ----- Asynchronous unilateral fixed-threshold BCM -----
    def _step_baseline(self, i, rng):
        """异步单节点固定阈值有界信任基线。
        
        最基本的意见动力学模型，使用固定阈值。
        
        参数:
            i: 当前更新的节点
            rng: 随机数生成器
        """
        # 随机选择一个邻居
        j = self.neighbors[i][rng.randint(len(self.neighbors[i]))]
        # 检查意见差异是否在阈值内
        if abs(self.opinions[i] - self.opinions[j]) <= self.trust.epsilon_0:
            # 执行意见更新（向邻居意见靠拢）
            self.opinions[i] += self.mu * (self.opinions[j] - self.opinions[i])
            self.stats['update_count'] += 1
        self.stats['edge_count'] += 1

    # ----- Weighted BCM: Jaccard权重影响步长 -----
    def _step_weighted_bcm(self, i, rng):
        """Weighted BCM 模型
        
        在经典 BCM 基础上，使用 Jaccard 权重调制更新步长。
        
        参数:
            i: 当前更新的节点
            rng: 随机数生成器
        """
        # 随机选择邻居
        j = self.neighbors[i][rng.randint(len(self.neighbors[i]))]
        # 获取边权重
        w = self.weights.get((i, j), 0.0)
        # 使用固定阈值检查
        if abs(self.opinions[i] - self.opinions[j]) <= self.trust.epsilon_0:
            # 更新步长乘以权重
            self.opinions[i] += self.mu * w * (self.opinions[j] - self.opinions[i])
            self.stats['update_count'] += 1
        self.stats['edge_count'] += 1

    def _step_threshold_only(self, i, rng):
        """边特异结构依赖阈值；无三角触发或第三方衰减。"""
        self.stats['edge_count'] += 1
        self._edge_interaction(i, rng)

    # ----- bcm_triangle: 固定阈值 + 三角触发，无第三方衰减 -----
    def _step_bcm_triangle(self, i, rng):
        """BCM + Triangle 模型
        
        使用固定阈值 epsilon_0 的标准 BCM 更新，但有三角形触发机制，无第三方衰减。
        
        参数:
            i: 当前更新的节点
            rng: 随机数生成器
        """
        C_i = self.clustering.get(i, 0.0)
        has_triangles = (i in self.node_triangles and len(self.node_triangles[i]) > 0)

        if rng.rand() < C_i and has_triangles:
            self.stats['tri_count'] += 1
            self._triangle_interaction_fixed_threshold(i, rng)
        else:
            self.stats['edge_count'] += 1
            self._edge_interaction_fixed_threshold(i, rng)

    def _edge_interaction_fixed_threshold(self, i, rng):
        """使用固定阈值的边交互"""
        j = self.neighbors[i][rng.randint(len(self.neighbors[i]))]
        if abs(self.opinions[i] - self.opinions[j]) <= self.trust.epsilon_0:
            self.opinions[i] += self.mu * (self.opinions[j] - self.opinions[i])
            self.stats['update_count'] += 1

    def _triangle_interaction_fixed_threshold(self, i, rng):
        """使用固定阈值的三角形交互（无第三方衰减）"""
        tri_indices = self.node_triangles[i]
        tri_idx = tri_indices[rng.randint(len(tri_indices))]
        triangle = self.triangles_list[tri_idx]
        nodes_tri = list(triangle)

        pairs = [(nodes_tri[0], nodes_tri[1]),
                 (nodes_tri[1], nodes_tri[2]),
                 (nodes_tri[0], nodes_tri[2])]

        all_ok = True
        for a, b in pairs:
            if abs(self.opinions[a] - self.opinions[b]) > self.trust.epsilon_0:
                all_ok = False
                break

        if all_ok:
            self.stats['tri_success'] += 1
            others = [n for n in nodes_tri if n != i]
            rng.shuffle(others)
            j = others[0]
            self.opinions[i] += self.mu * (self.opinions[j] - self.opinions[i])
            self.stats['update_count'] += 1
        else:
            self.stats['tri_fallback'] += 1
            self._edge_interaction_fixed_threshold(i, rng)

    # ----- Full Model: 完整模型 -----
    def _step_full_model(self, i, rng):
        """完整模型
        
        包含所有机制：结构依赖阈值 + 局部聚类触发 + 第三方不一致衰减。
        
        参数:
            i: 当前更新的节点
            rng: 随机数生成器
        """
        # 获取节点的局部聚类系数
        C_i = self.clustering.get(i, 0.0)
        # 检查节点是否有三角形
        has_triangles = (i in self.node_triangles and len(self.node_triangles[i]) > 0)

        # 以概率 C_i 尝试三角形交互（高阶结构优先）
        if rng.rand() < C_i and has_triangles:
            self.stats['tri_count'] += 1
            self._triangle_interaction(i, rng)
        else:
            # 否则执行普通边交互
            self.stats['edge_count'] += 1
            self._edge_interaction(i, rng)

    # ----- No Triadic Feedback: 保留三角触发，关闭第三方不一致衰减 -----
    def _step_no_triadic_feedback(self, i, rng):
        """保留三角形触发机制，但关闭第三方不一致衰减（Phi=1）。

        与 full_model 相同的 C_i 概率触发真实三角形；
        三角形交互成功后使用 _triangle_interaction_no_feedback，
        即不计算 rep_factor，直接执行标准 BCM 更新。
        """
        C_i = self.clustering.get(i, 0.0)
        has_triangles = (i in self.node_triangles and len(self.node_triangles[i]) > 0)

        if rng.rand() < C_i and has_triangles:
            self.stats['tri_count'] += 1
            self._triangle_interaction_no_feedback(i, rng)
        else:
            self.stats['edge_count'] += 1
            self._edge_interaction(i, rng)
        
    # ----- 3.2 普通边交互 -----
    def _edge_interaction(self, i, rng):
        """执行普通边交互
        
        使用动态饱和阈值的边更新规则。
        
        参数:
            i: 当前更新的节点
            rng: 随机数生成器
        """
        # 随机选择邻居
        j = self.neighbors[i][rng.randint(len(self.neighbors[i]))]
        # 获取边权重
        w = self.weights.get((i, j), 0.0)
        # 计算动态阈值
        eps = self.trust.effective_threshold(w)
        # 检查意见差异
        if abs(self.opinions[i] - self.opinions[j]) <= eps:
            self.opinions[i] += self.mu * (self.opinions[j] - self.opinions[i])
            self.stats['update_count'] += 1

    # ----- 3.3 三角形交互（第三方不一致衰减） -----
    def _triangle_interaction(self, i, rng):
        """执行三角形交互
        
        在三角形结构中进行三方交互，包含第三方不一致衰减。
        
        参数:
            i: 当前更新的节点
            rng: 随机数生成器
        """
        # 获取节点参与的三角形列表
        tri_indices = self.node_triangles[i]
        # 随机选择一个三角形
        tri_idx = tri_indices[rng.randint(len(tri_indices))]
        triangle = self.triangles_list[tri_idx]
        nodes_tri = list(triangle)

        # 获取三角形的三条边
        pairs = [(nodes_tri[0], nodes_tri[1]),
                 (nodes_tri[1], nodes_tri[2]),
                 (nodes_tri[0], nodes_tri[2])]

        # 检查三角形内所有边是否满足阈值条件
        all_ok = True
        for a, b in pairs:
            w_ab = self.weights.get((a, b), 0.0)
            eps_ab = self.trust.effective_threshold(w_ab)
            if abs(self.opinions[a] - self.opinions[b]) > eps_ab:
                all_ok = False
                break

        if all_ok:
            # 三角形交互成功
            self.stats['tri_success'] += 1
            # 获取三角形中的其他两个节点
            others = [n for n in nodes_tri if n != i]
            rng.shuffle(others)  # 随机打乱
            j, k = others[0], others[1]

            # 衰减由更新者 i 与第三方 k 的意见差异决定。
            attenuation = np.exp(-self.beta * abs(
                self.opinions[i] - self.opinions[k]))
            self.opinions[i] += self.mu * (
                self.opinions[j] - self.opinions[i]) * attenuation
            self.stats['update_count'] += 1
        else:
            # 三角形条件不满足，回退到边交互
            self.stats['tri_fallback'] += 1
            self._edge_interaction(i, rng)

    def _triangle_interaction_no_feedback(self, i, rng):
        """执行无第三方不一致衰减的三角形交互。

        三角形选择、三边动态阈值检查和失败回退与 Full Model 相同；
        唯一区别是成功后三角更新使用 Phi=1。
        """
        tri_indices = self.node_triangles[i]
        tri_idx = tri_indices[rng.randint(len(tri_indices))]
        triangle = self.triangles_list[tri_idx]
        nodes_tri = list(triangle)

        pairs = [(nodes_tri[0], nodes_tri[1]),
                 (nodes_tri[1], nodes_tri[2]),
                 (nodes_tri[0], nodes_tri[2])]

        all_ok = True
        for a, b in pairs:
            w_ab = self.weights.get((a, b), 0.0)
            eps_ab = self.trust.effective_threshold(w_ab)
            if abs(self.opinions[a] - self.opinions[b]) > eps_ab:
                all_ok = False
                break

        if all_ok:
            self.stats['tri_success'] += 1
            others = [n for n in nodes_tri if n != i]
            rng.shuffle(others)
            j = others[0]
            self.opinions[i] += self.mu * (self.opinions[j] - self.opinions[i])
            self.stats['update_count'] += 1
        else:
            self.stats['tri_fallback'] += 1
            self._edge_interaction(i, rng)

    def get_opinions(self):
        """获取当前意见字典的副本
        
        返回:
            意见字典的副本
        """
        return dict(self.opinions)

    def get_opinions_array(self):
        """获取当前意见值的数组形式
        
        返回:
            NumPy 数组，按节点顺序排列的意见值
        """
        return np.array([self.opinions[n] for n in self.nodes])

    def get_stats(self):
        """获取交互统计信息
        
        返回:
            统计字典的副本
        """
        return dict(self.stats)
