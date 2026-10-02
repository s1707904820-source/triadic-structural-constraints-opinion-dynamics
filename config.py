"""
全局配置中心: 网络参数、动力学参数、信任参数、模型变体、
实验设置、指标阈值、绘图样式 —— 全部集中管理
"""

import os
import numpy as np
from collections import OrderedDict

# ================================================================
# JSON key 格式化函数
# ================================================================

def format_p_key(p):
    return f"{float(p):.1f}"

def format_beta_key(b):
    return f"{float(b):.1f}"

def format_eps_key(e):
    # Three decimals are required for the canonical 0.025-spaced grid.
    return f"{float(e):.3f}"

__all__ = list(globals().get('__all__', [])) + [
    'format_p_key', 'format_beta_key', 'format_eps_key'
]

# One publication-wide source for every epsilon_0 scan.  Keep three decimal
# places: rounding this grid to two decimals changes 0.125, 0.175, ... and can
# make stored labels disagree with the values that were actually simulated.
EPSILON_0_GRID = tuple(round(0.10 + x * 0.025, 3) for x in range(19))
__all__.append('EPSILON_0_GRID')

# ================================================================
# 1. 网络参数 - Holme-Kim 网络生成模型的参数配置
# ================================================================

NETWORK = {
    'N': 2000,                           # 网络节点总数
    'm': 4,                              # 每个新节点连接的边数（平均度数的一半）
    'p_values': [round(x * 0.05, 2) for x in range(21)],  # 三角形成概率的扫描值列表 (0.0到1.0，21个点)
    'p_default': 0.6,                    # 默认三角形成概率
}

# ================================================================
# 2. 信任阈值参数 - 动态阈值机制的核心参数
# ================================================================

TRUST = {
    'epsilon_0_default': 0.15,          # ★ 基础容忍阈值 (测试用)
    'alpha': 3.5,                       # ★ 权重敏感度指数
    # Backward-compatible alias for the single canonical epsilon_0 grid.
    'epsilon_values': list(EPSILON_0_GRID),
                                        # ★ ε₀扫描范围 (从0.02到0.40，20个点)
    'epsilon_0_range': list(EPSILON_0_GRID),
}

# ================================================================
# 3. 动力学参数 - 意见演化过程的基本参数
# ================================================================

DYNAMICS = {
    'mu': 0.5,                           # 基础更新步长（收敛速度参数）
    'beta_default': 3.0,                 # 默认第三方不一致衰减系数
    'beta_range': [round(x * 0.25, 2) for x in range(21)],  # beta 扫描范围 (0.0到5.0，21个点)
}

# ================================================================
# 4. 模型变体（六组比较，其中 Weighted BCM 为外部比较）
# ================================================================

# 使用 OrderedDict 保持模型定义的顺序
MODEL_TYPES = OrderedDict([
    # 基线模型：经典 Deffuant 模型，无权重调制、无三角形机制
    ('baseline', {
        'label': 'Asynchronous unilateral BCM',
        'description': '异步单节点、固定阈值的有界信任基线（仅更新被抽中的节点）',
        'engine_key': 'baseline',        # 仿真引擎标识
        'color': '#6B7280',              # 绘图颜色（灰色）
        'marker': 's',                   # 绘图标记（方形）
        'use_trust': False,              # 不使用信任阈值
        'use_triangle': False,           # 不使用三角形机制
        'use_weight': False,             # 不使用边权重
    }),
    # 加权 BCM 模型：仅加入 Jaccard 权重影响步长
    ('weighted_bcm', {
        'label': 'Weighted BCM',
        'description': '仅加入Jaccard权重影响步长',
        'engine_key': 'weighted_bcm',
        'color': '#2563EB',              # 蓝色
        'marker': '^',                   # 三角形
        'use_trust': False,
        'use_triangle': False,
        'use_weight': True,              # 使用边权重
    }),
    # 仅结构依赖阈值：用于隔离边特异阈值本身的贡献
    ('threshold_only', {
        'label': 'Threshold Only',
        'description': '仅使用结构依赖、边特异的有效阈值；无三角触发和第三方衰减',
        'engine_key': 'threshold_only',
        'color': '#7C3AED',
        'marker': 'P',
        'use_trust': True,
        'use_triangle': False,
        'use_weight': True,
    }),
    # BCM + Triangle 模型：固定阈值 + 三角触发，无第三方衰减
    ('bcm_triangle', {
        'label': 'BCM + Triangle',
        'description': '固定阈值 ε_0 + C_i 三角触发，标准 BCM 更新（无第三方衰减）',
        'engine_key': 'bcm_triangle',
        'color': '#EA580C',              # 橙色
        'marker': 'D',                   # 菱形
        'use_trust': True,               # 使用信任阈值（但用固定阈值）
        'use_triangle': True,            # 使用三角形机制
        'use_weight': False,
    }),
    # 完整模型：包含所有机制
    ('full_model', {
        'label': 'Full Model',
        'description': '结构依赖阈值 + 局部聚类触发 + 第三方不一致衰减',
        'engine_key': 'full_model',
        'color': '#DC2626',              # 红色
        'marker': 'o',                   # 圆形
        'use_trust': True,
        'use_triangle': True,            # 使用三角形机制
        'use_weight': True,
    }),
    # 无结构模型：关闭高阶结构验证
    ('no_triadic_feedback', {
        'label': 'No Triadic Feedback',
        'description': '保留 C_i 三角触发和三边阈值检查，但关闭第三方不一致衰减（Phi=1）',
        'engine_key': 'no_triadic_feedback',
        'color': '#0D9488',              # 青色
        'marker': 'v',                   # 倒三角
        'use_trust': True,
        'use_triangle': True,
        'use_weight': True,
    }),
])

# 模型别名映射 - 允许使用不同的名称引用同一模型
MODEL_ALIASES = {
    'baseline_dw': 'baseline',
    'weighted': 'weighted_bcm',
    'weighted_dw': 'weighted_bcm',
    'trust': 'threshold_only',
    'trust_only': 'threshold_only',
    'triangle_fixed': 'bcm_triangle',
    'triangle': 'no_triadic_feedback',
    'triangle_only': 'no_triadic_feedback',
    'full': 'full_model',
}

# Accepted only when reproducing legacy scripts.  These names are scientifically
# misleading and must not appear in new result files or manuscript text.
DEPRECATED_MODEL_ALIASES = {
    'no_structure': 'no_triadic_feedback',
}

# 模型显示顺序列表
MODEL_DISPLAY_ORDER = list(MODEL_TYPES.keys())

def resolve_model_key(key: str) -> str:
    """解析模型键名，处理别名映射
    
    参数:
        key: 输入的模型键名或别名
    返回:
        标准化的模型键名
    抛出:
        KeyError: 如果键名无效
    """
    if key in MODEL_TYPES:
        return key
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if key in DEPRECATED_MODEL_ALIASES:
        import warnings
        warnings.warn(
            "model alias {!r} is deprecated; use {!r}".format(
                key, DEPRECATED_MODEL_ALIASES[key]),
            DeprecationWarning, stacklevel=2)
        return DEPRECATED_MODEL_ALIASES[key]
    raise KeyError(f"未知 model_type: {key}. 可选: {list(MODEL_TYPES.keys())}")

# 将模型相关函数和变量添加到公共接口
__all__ += ['MODEL_ALIASES', 'DEPRECATED_MODEL_ALIASES',
            'MODEL_DISPLAY_ORDER', 'resolve_model_key']

# ================================================================
# 5. 仿真控制 - 蒙特卡洛仿真的基本设置
# ================================================================

SIMULATION = {
    # 所有主实验统一使用 Monte Carlo sweep 作为时间单位；1 sweep = N 次异步更新。
    'main_sweeps': 200,
    'num_steps': 400000,                 # 向后兼容；主实验按 N * main_sweeps 动态计算
    'sweep_sweeps': 200,                 # epsilon/磁滞扫描每个 epsilon 点的 Monte Carlo sweeps
    'num_runs': 20,                      # 蒙特卡洛运行次数（用于统计平均）
    'random_seed': 42,                   # 随机种子（用于结果复现）
    'seed': 42,                          # 备用随机种子
    'record_sweeps': 5,                  # 每隔多少个 sweep 记录一次（降低长时扫描记录开销）
    'record_interval': 10000,            # 向后兼容；默认按 N * record_sweeps 动态计算
    'use_convergence_check': False,      # 默认不改变全局实验；规模实验会单独启用
    'convergence_tol': 1e-7,             # 最近记录窗口平均 dx 低于该值视为近似稳定
    'convergence_patience': 5,           # 连续多少个记录窗口满足阈值后提前停止
    'convergence_min_sweeps': 20,        # 至少运行多少个 Monte Carlo sweeps 后才检查收敛
    'size_sweeps': 200,                  # 网络规模实验的最大 sweeps
    'size_record_sweeps': 1,             # 网络规模实验每隔多少 sweeps 记录一次
    'num_networks_validation': 50,       # 网络验证时的网络实例数量
    'backward_perturb_sigma': 0.02,
    # 主分析与理论设定一致：仅在逆向扫描起点注入一次扰动。
    'backward_perturb_each_step': False,
    # 从无扰动到强扰动，用于检验逆向稳定性是否只是小冲击下的吸收态现象。
    'perturb_sigma_values': [0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
}

# ================================================================
# 6. 敏感性分析参数 - 参数扫描范围
# ================================================================

SENSITIVITY = {
    'alpha_range': [round(0.5 + x * 0.25, 2) for x in range(19)],  # 结构调制灵敏度扫描 (0.5到5.0，19个点)
    'mu_range':    [round(0.05 + x * 0.05, 2) for x in range(19)],  # 更新步长扫描 (0.05到0.95，19个点)
    'N_sizes':     [50, 100, 200, 500, 1000, 2000],  # 网络规模扫描
    'm_values':    [2, 3, 4, 5, 6, 8],  # 平均度数扫描
}

# ================================================================
# 7. 联合扫描参数 - 多参数组合扫描
# ================================================================

JOINT_SWEEP = {
    'eps_values': list(EPSILON_0_GRID),  # 与所有 epsilon_0 实验共用同一网格
    'beta_values': [round(x * 0.25, 2) for x in range(21)],  # beta 扫描值 (0.0到5.0，21个点)
    'p_hk': 0.6,                         # 联合扫描时的默认 p 值
    'p_beta_p_values': [round(x * 0.05, 2) for x in range(21)],  # p-beta 联合扫描的 p 值 (0.0到1.0，21个点)
    'p_beta_beta_values': [round(x * 0.25, 2) for x in range(21)],  # p-beta 联合扫描的 beta 值 (0.0到5.0，21个点)
}

# ================================================================
# 8. 指标阈值 - 各种度量和判断的阈值
# ================================================================

METRICS = {
    'cluster_gap_threshold': 0.05,       # 聚类间距离阈值（用于识别不同团体）
    'cluster_diameter_threshold': 0.05,  # 共识簇最大意见直径（用于OP，避免大N链式虚高）
    'cluster_min_size': 1,               # 最小聚类大小
    'triangle_consensus_eps': None,      # 三角形共识阈值（None 表示使用默认值）
    'order_param_consensus_threshold': 0.9,  # 有序参数共识阈值
    'order_param_polarized_threshold': 0.3,  # 有序参数极化阈值
    'hysteresis_significant_threshold': 0.01,  # 磁滞显著性阈值
}

# ================================================================
# 9. 绘图样式 - matplotlib 的全局样式设置
# ================================================================

PLOT = {
    'rcParams': {                        # matplotlib 运行配置参数
        'figure.dpi': 150,               # 显示分辨率
        'savefig.dpi': 300,              # 保存图片分辨率
        'font.size': 11,                 # 基础字体大小
        'axes.labelsize': 13,            # 坐标轴标签字体大小
        'axes.titlesize': 14,            # 标题字体大小
        'legend.fontsize': 9,            # 图例字体大小
        'xtick.labelsize': 10,           # X轴刻度字体大小
        'ytick.labelsize': 10,           # Y轴刻度字体大小
        'figure.figsize': (8, 5),        # 默认图大小
        'axes.grid': True,               # 显示网格
        'grid.alpha': 0.3,               # 网格透明度
    },
    'color_pool': [                      # 颜色池，用于多组数据绘图
        '#e74c3c', '#2ecc71', '#3498db', '#f39c12',
        '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
    ],
    'marker_pool': ['o', 's', '^', 'D', 'v', 'P', 'X', 'h'],  # 标记池
    # 不同 p 值对应的颜色映射
    'p_colors': {
        '0.0': '#e74c3c',                # 红色
        '0.2': '#e67e22',                # 橙色
        '0.4': '#2ecc71',                # 绿色
        '0.6': '#3498db',                # 蓝色
        '0.8': '#9b59b6',                # 紫色
        '1.0': '#1abc9c',                # 青色
    },
    # 不同 p 值对应的标记映射
    'p_markers': {
        '0.0': 'o', '0.2': 's', '0.4': '^',
        '0.6': 'D', '0.8': 'v', '1.0': 'P',
    },
    # 不同初始分布对应的颜色
    'dist_colors': {
        'uniform': '#3498db',            # 均匀分布 - 蓝色
        'normal':  '#e74c3c',            # 正态分布 - 红色
        'bimodal': '#2ecc71',            # 双峰分布 - 绿色
    },
    'heatmap_cmap_op': 'RdYlBu_r',       # 有序参数热力图配色
    'heatmap_cmap_cl': 'YlOrRd',         # 聚类数量热力图配色
    'heatmap_text_threshold': 0.5,       # 热力图文字显示阈值
    'bar_alpha': 0.85,                   # 柱状图透明度
    'bar_edgecolor': 'white',            # 柱状图边缘颜色
    'bar_edgewidth': 0.5,                # 柱状图边缘宽度
    'errorbar_capsize': 4,               # 误差线帽大小
    'line_width': 1.8,                   # 线宽
    'marker_size': 6,                    # 标记大小
    'save_format': 'png',                # 保存格式
    'transparent_bg': False,             # 透明背景
}

# ================================================================
# 10. 输出路径 - 数据和图表的保存位置
# ================================================================

# 获取当前文件所在目录，确保输出路径正确
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_CONFIG_DIR, 'output')   # 输出根目录（绝对路径）
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')        # 数据文件目录
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')   # 图表文件目录

# ================================================================
# 工具函数（样式、阈值等）- 辅助函数定义
# ================================================================

def get_model_styles():
    """获取所有模型的样式配置（标签、颜色、标记）
    
    返回:
        字典，键为模型名，值为样式配置字典
    """
    styles = {}
    for key, info in MODEL_TYPES.items():
        styles[key] = {
            'label': info['label'],
            'color': info.get('color', '#34495e'),  # 默认深灰色
            'marker': info.get('marker', 'o'),      # 默认圆形
        }
    return styles

def get_p_style(p_str):
    """获取指定 p 值的颜色和标记样式
    
    参数:
        p_str: p 值的字符串表示
    返回:
        (颜色, 标记) 元组
    """
    c = PLOT['p_colors'].get(p_str, '#34495e')  # 默认深灰色
    m = PLOT['p_markers'].get(p_str, 'x')       # 默认叉号
    return c, m

def get_triangle_consensus_eps():
    """获取三角形共识阈值
    
    如果 METRICS 中未设置，则使用 TRUST 中的默认值
    
    返回:
        阈值数值
    """
    val = METRICS['triangle_consensus_eps']
    return val if val is not None else TRUST['epsilon_0_default']

def get_metrics_threshold(name):
    """获取指定名称的指标阈值
    
    参数:
        name: 阈值名称
    返回:
        阈值数值，如果不存在则返回 None
    """
    return METRICS.get(name)

# ================================================================
# 自检摘要
# ================================================================

def _print_summary():
    """打印配置摘要信息，用于调试和验证"""
    print("=" * 50)
    print("CONFIG SUMMARY")
    print(f"  Network: N={NETWORK['N']}, m={NETWORK['m']}, p_values={NETWORK['p_values']}")
    print(f"  Trust:   epsilon_0={TRUST['epsilon_0_default']}, "
          f"alpha={TRUST['alpha']}")
    print(f"  Dynamics: μ={DYNAMICS['mu']}, β={DYNAMICS['beta_default']}")
    print(f"  Sim:     steps={SIMULATION['num_steps']}, runs={SIMULATION['num_runs']}")
    print(f"  Models:  {list(MODEL_TYPES.keys())}")
    print(f"  Cluster gap threshold: {METRICS['cluster_gap_threshold']}")
    print(f"  Fixed triangle-alignment ε: {get_triangle_consensus_eps()}")
    print(f"  Plot format: {PLOT['save_format']}")
    print("=" * 50)

# 取消下面一行的注释以在导入时打印配置摘要
# _print_summary()
