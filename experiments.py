"""
实验管理器: 实现文档中所有实验任务
- 实验7: 网络属性验证
- 实验6: 对比实验 (Ablation Study)
- 核心实验: ε₀ 正/逆向扫描 (磁滞检测)
- 实验8: 参数敏感性分析
- 联合扫描热图
- 鲁棒性测试
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import os
import json
import time
import numpy as np

try:
    from .config import (
        NETWORK,
        TRUST,
        DYNAMICS,
        SIMULATION,
        MODEL_TYPES,
        resolve_model_key,
        SENSITIVITY,
        JOINT_SWEEP,
        METRICS,
        OUTPUT_DIR,
        FIGURE_DIR,
        DATA_DIR,
    )
except ImportError:  # pragma: no cover
    from config import (
        NETWORK,
        TRUST,
        DYNAMICS,
        SIMULATION,
        MODEL_TYPES,
        resolve_model_key,
        SENSITIVITY,
        JOINT_SWEEP,
        METRICS,
        OUTPUT_DIR,
        FIGURE_DIR,
        DATA_DIR,
    )
from network import NetworkGenerator
from dynamics import TrustThreshold, OpinionDynamics
from simulation import SingleRun
from metrics import Metrics
from vector_dynamics import VectorOpinionDynamics
from statistics_utils import (summarize, summarize_proportion, paired_difference, summarize_aligned_series)
from provenance import build_metadata, execution_source_signature


_RUNTIME_CONFIG_NAMES = (
    'NETWORK', 'TRUST', 'DYNAMICS', 'MODEL_TYPES', 'MODEL_ALIASES',
    'MODEL_DISPLAY_ORDER', 'SIMULATION', 'SENSITIVITY', 'JOINT_SWEEP',
    'METRICS', 'PLOT', 'OUTPUT_DIR', 'DATA_DIR', 'FIGURE_DIR',
)


def _capture_runtime_config():
    """Capture mutable configuration for Windows ``spawn`` workers."""
    try:
        import config as config_module
    except ImportError:  # pragma: no cover - package import fallback
        config_module = None
    snapshot = {}
    for name in _RUNTIME_CONFIG_NAMES:
        if name in globals():
            snapshot[name] = copy.deepcopy(globals()[name])
        elif config_module is not None and hasattr(config_module, name):
            snapshot[name] = copy.deepcopy(getattr(config_module, name))
    return snapshot


def _apply_runtime_config(snapshot):
    """Apply the parent's configuration inside a newly spawned worker."""
    if not isinstance(snapshot, dict):
        raise TypeError('runtime configuration snapshot must be a dictionary')

    try:
        import config as config_module
    except ImportError:  # pragma: no cover - package import fallback
        config_module = None

    for name, value in snapshot.items():
        copied = copy.deepcopy(value)
        current = globals().get(name)
        if isinstance(current, dict) and isinstance(copied, dict):
            current.clear()
            current.update(copied)
        elif isinstance(current, list) and isinstance(copied, list):
            current[:] = copied
        else:
            globals()[name] = copied

        if config_module is None or not hasattr(config_module, name):
            continue
        config_value = getattr(config_module, name)
        if isinstance(config_value, dict) and isinstance(copied, dict):
            config_value.clear()
            config_value.update(copy.deepcopy(copied))
        elif isinstance(config_value, list) and isinstance(copied, list):
            config_value[:] = copy.deepcopy(copied)
        else:
            setattr(config_module, name, copy.deepcopy(copied))


def _process_pool_kwargs(runtime_config):
    """Return common executor arguments that preserve test/runtime config."""
    return {
        'initializer': _apply_runtime_config,
        'initargs': (runtime_config,),
    }

def run_experiment(run_fn, params, n_runs=10, base_seed=0, out_file=None, param_keys_formats=None):
    """
    通用运行框架：多次运行、设置随机种子、收集均值/方差并保存 JSON。
    run_fn(params, seed) -> dict (需要返回一组指标)
    """
    results = []
    # Windows/Anaconda 下多进程容易触发 BrokenProcessPool；默认顺序执行更稳定。
    seeds = [int(base_seed) + i for i in range(n_runs)]
    for seed in seeds:
        try:
            res = run_fn(params, seed)
            results.append(res)
        except Exception as e:
            results.append({'error': str(e)})
    agg = {}
    keys = set()
    for r in results:
        keys.update(r.keys())
    for k in keys:
        vals = [r[k] for r in results if isinstance(r.get(k, None), (int, float))]
        if vals:
            agg[k] = summarize(vals)
        else:
            agg[k] = results[-1].get(k, None)
    if out_file:
        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        meta = {'params': params, 'n_runs': n_runs, 'base_seed': base_seed}
        save_obj = {'meta': meta, 'agg': agg, 'raw': results}
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(save_obj, f, indent=2, ensure_ascii=False)
    return agg
class ExperimentManager:
    def __init__(self):
        for d in [OUTPUT_DIR, FIGURE_DIR, DATA_DIR]:
            os.makedirs(d, exist_ok=True)
    # ================================================================
    # 工具方法
    # ================================================================
    @staticmethod
    def _core_parameter_signature():
        """Parameters that must match before partial results may be resumed."""
        return {
            'epsilon_0': float(TRUST['epsilon_0_default']),
            'alpha': float(TRUST['alpha']),
            'mu': float(DYNAMICS['mu']),
            'beta': float(DYNAMICS['beta_default']),
            'execution_source_signature': execution_source_signature(),
        }

    @staticmethod
    def _ordered_numeric_mapping(mapping):
        """Return a numerically keyed mapping in deterministic JSON order."""
        return {
            key: mapping[key]
            for key in sorted(mapping, key=lambda value: float(value))
        }

    @staticmethod
    def _check_checkpoint_target(path, label, resume):
        """Reject links, non-files, and accidental no-resume overwrites."""
        if not os.path.lexists(path):
            return
        if os.path.islink(path):
            raise FileExistsError(
                f"{label} checkpoint target is a symlink; original preserved")
        if not os.path.isfile(path):
            raise FileExistsError(
                f"{label} checkpoint target is not a regular file; original preserved")
        if not resume:
            raise FileExistsError(
                f"{label} checkpoint already exists; use a new filename")

    def _load_controlled_results(self, filename, control, resume=True):
        """Load a checkpoint only when its full experiment control matches."""
        control = dict(control)
        control['execution_source_signature'] = execution_source_signature()
        fresh = {'_control': control}
        path = os.path.join(DATA_DIR, filename)
        if not resume or not os.path.exists(path):
            return fresh
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                loaded = json.load(handle)
            if (isinstance(loaded, dict)
                    and loaded.get('_control') == control):
                print(f"  [续跑] 已载入参数匹配的断点: {path}")
                return loaded
            raise FileExistsError(
                f"checkpoint control/source mismatch: {path}; "
                "choose a new output_filename or remove the checkpoint "
                "after archiving it"
            )
        except Exception as exc:
            if isinstance(exc, FileExistsError):
                raise
            raise FileExistsError(
                f"checkpoint unreadable: {path}; original file preserved"
            ) from exc
        return fresh

    def save_json(self, data, filename, metadata=None):
        """保存结果，并写出同名 ``.meta.json`` 可复现性清单。"""
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def write_atomic(obj, target):
            """Write complete JSON before atomically replacing the target."""
            if os.path.lexists(target) and os.path.islink(target):
                raise FileExistsError(
                    f"refusing to replace symlink checkpoint: {target}")
            temp_path = f"{target}.tmp-{os.getpid()}"
            try:
                with open(temp_path, 'w', encoding='utf-8') as handle:
                    json.dump(obj, handle, default=convert, indent=2,
                              ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, target)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        filepath = os.path.join(DATA_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        write_atomic(data, filepath)
        stem, _ = os.path.splitext(filepath)
        metadata_path = stem + '.meta.json'
        metadata_obj = build_metadata(
            experiment=os.path.splitext(filename)[0],
            n_runs=(metadata or {}).get('n_runs'),
            extra=metadata,
        )
        write_atomic(metadata_obj, metadata_path)
        print(f"  [保存] {filepath}")
        print(f"  [元数据] {metadata_path}")
    def _run_batch(self, param_dict, num_runs, num_steps=None, verbose_label="",
                   record_interval=None, convergence_check=None,
                   convergence_min_steps=None):
        """
        通用批量运行：对同一参数组合执行num_runs次独立仿真
        返回所有运行结果列表
        """
        run_N = int(param_dict.get('N', NETWORK['N']))
        if num_steps is None:
            num_steps = int(round(
                run_N * float(SIMULATION.get('main_sweeps', 200))))
        if record_interval is None:
            record_interval = max(1, int(round(
                run_N * float(SIMULATION.get('record_sweeps', 5)))))
        all_results = []
        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            sim = SingleRun(
                N=run_N,
                m=param_dict.get('m', NETWORK['m']),
                p_hk=param_dict.get('p_hk', 0.5),
                epsilon_0=param_dict.get('epsilon_0', TRUST['epsilon_0_default']),
                alpha=param_dict.get('alpha', TRUST['alpha']),
                mu=param_dict.get('mu', DYNAMICS['mu']),
                beta=param_dict.get('beta', DYNAMICS['beta_default']),
                model_type=param_dict.get('model_type', 'full_model'),
                num_steps=num_steps,
                seed=seed,
                convergence_check=convergence_check,
                convergence_min_steps=convergence_min_steps,
                update_mode=param_dict.get('update_mode', 'asynchronous'),
            )
            result = sim.execute(record_interval=record_interval)
            all_results.append(result)
            if verbose_label:
                print(f"    {verbose_label} run {run_i+1}/{num_runs}: "
                      f"clusters={result['final']['n_clusters']}, "
                      f"OP={result['final']['order_param']:.4f}", flush=True)
        return all_results
    @staticmethod
    def _aggregate(all_results, key_path):
        """
        从结果列表中提取嵌套key的值并聚合
        key_path: 如 ['final', 'order_param']
        """
        vals = []
        for r in all_results:
            obj = r
            for k in key_path:
                obj = obj[k]
            vals.append(obj)
        return summarize(vals)
    # ================================================================
    # 带状态继承的 ε₀ 扫描（类方法）
    # ================================================================
    @staticmethod
    def _aggregate_scalar_values(vals):
        """聚合一组标量值，返回均值、标准差、极值和原始值。"""
        return summarize(vals)

    @staticmethod
    def _aggregate_time_series(all_results):
        """Aggregate aligned repeated trajectories, including 95% CIs."""
        if not all_results:
            return None
        time_series = [result.get('time_series', {}) for result in all_results]
        steps = [series.get('step', []) for series in time_series]
        if not steps or any(row != steps[0] for row in steps[1:]):
            return None
        output = {
            'step': steps[0],
            'sweep': time_series[0].get('sweep', []),
            'n_runs': len(time_series),
        }
        metric_keys = (
            'n_clusters', 'variance', 'order_param',
            'triangle_alignment_fixed',
            'triangle_admissibility_adaptive',
            'mean_abs_dx', 'accepted_update_rate',
        )
        for key in metric_keys:
            rows = [series.get(key, []) for series in time_series]
            if rows and rows[0] and all(len(row) == len(rows[0]) for row in rows):
                output[key] = summarize_aligned_series(rows)
        return output

    @staticmethod
    def _scan_step_config():
        """Return step count for epsilon/hysteresis scans using sweeps when configured."""
        sweep_sweeps = SIMULATION.get('sweep_sweeps')
        if sweep_sweeps is None:
            steps = int(SIMULATION['num_steps'])
            sweeps = steps / max(int(NETWORK['N']), 1)
            return steps, float(sweeps)

        sweeps = float(sweep_sweeps)
        steps = int(round(int(NETWORK['N']) * sweeps))
        return max(1, steps), sweeps

    @staticmethod
    def _perturb_opinions(opinions, rng, sigma, return_stats=False):
        """加入高斯扰动并裁剪回 [0,1]；可返回截断比例。"""
        if sigma is None or sigma <= 0:
            result = dict(opinions)
            stats = {'n_perturbed': 0, 'n_clipped': 0, 'clipped_fraction': 0.0}
            return (result, stats) if return_stats else result
        result = {}
        n_clipped = 0
        for node, opinion in opinions.items():
            raw = float(opinion + rng.normal(0.0, sigma))
            if raw < 0.0 or raw > 1.0:
                n_clipped += 1
            result[node] = float(np.clip(raw, 0.0, 1.0))
        n_perturbed = len(result)
        stats = {
            'n_perturbed': int(n_perturbed),
            'n_clipped': int(n_clipped),
            'clipped_fraction': float(n_clipped / n_perturbed)
            if n_perturbed else 0.0,
        }
        return (result, stats) if return_stats else result

    @staticmethod
    def _apply_standardized_perturbation(opinions, z_values, sigma,
                                          return_stats=False):
        """Apply a pre-drawn standard-normal shock at a chosen amplitude.

        The perturbation vector is deliberately supplied by the caller.  This
        makes every ``sigma`` branch use the same random direction and keeps
        the backward dynamics on a common random-number stream.  ``sigma=0``
        still follows the same design path, but leaves opinions unchanged.
        """
        sigma = float(sigma)
        if len(opinions) != len(z_values):
            raise ValueError("opinions and perturbation vector must align")
        result = {}
        n_clipped = 0
        for node, opinion in opinions.items():
            raw = float(opinion) + sigma * float(z_values[node])
            if sigma > 0.0 and (raw < 0.0 or raw > 1.0):
                n_clipped += 1
            result[node] = float(np.clip(raw, 0.0, 1.0))
        n_perturbed = len(result) if sigma > 0.0 else 0
        stats = {
            'n_perturbed': int(n_perturbed),
            'n_clipped': int(n_clipped),
            'clipped_fraction': float(n_clipped / n_perturbed)
            if n_perturbed else 0.0,
        }
        return (result, stats) if return_stats else result

    # ================================================================
    # 带状态继承的 ε₀ 扫描（类方法）
    # ================================================================
    def _run_sweep_with_continuation(
            self, eps_sequence, p_hk, num_runs,
            backward_perturb_sigma=0.0,
            perturb_each_step=False,
            collect_clusters=False,
            scan_label='scan'):
        """
        带状态继承的 ε₀ 扫描。

        对每个独立 run，依次遍历 eps_sequence，并将上一个 ε₀ 的终态
        作为下一个 ε₀ 的初始意见。若 backward_perturb_sigma > 0，则在
        进入某个 ε₀ 的动力学演化前对当前意见注入高斯扰动。

        参数
        ----
        backward_perturb_sigma : float
            后向扫描扰动强度。0.0 等价于旧版无扰动扫描。
        perturb_each_step : bool
            True：每个 ε₀ 步骤前都扰动；False：只在扫描起点扰动。
        collect_clusters : bool
            True 时同时返回 n_clusters 的聚合结果。
        scan_label : str
            日志标签。
        """
        N = NETWORK['N']
        m = NETWORK['m']
        num_steps, sweep_sweeps = self._scan_step_config()
        eps_results = {str(eps): [] for eps in eps_sequence}
        cluster_results = {str(eps): [] for eps in eps_sequence}

        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            rng = np.random.RandomState(seed)
            G = NetworkGenerator.generate_holme_kim(N, m, p_hk, seed=seed)
            triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
            weights = NetworkGenerator.compute_jaccard_weights(G)
            clustering = NetworkGenerator.compute_clustering(G)
            nodes = list(G.nodes())
            opinions = {n: rng.uniform(0, 1) for n in nodes}

            for eps_idx, eps in enumerate(eps_sequence):
                should_perturb = (
                    backward_perturb_sigma is not None
                    and backward_perturb_sigma > 0
                    and (perturb_each_step or eps_idx == 0)
                )
                if should_perturb:
                    opinions = self._perturb_opinions(
                        opinions, rng, backward_perturb_sigma
                    )

                trust = TrustThreshold(eps, TRUST['alpha'])
                engine = OpinionDynamics(
                    G, opinions, weights, triangles_list, node_triangles,
                    clustering, trust,
                    DYNAMICS['mu'], DYNAMICS['beta_default'],
                    'full_model',
                )
                for _ in range(num_steps):
                    engine.step(rng)

                final_arr = engine.get_opinions_array()
                op = Metrics.order_parameter(final_arr)
                n_clusters = Metrics.count_clusters(final_arr)
                eps_results[str(eps)].append(op)
                cluster_results[str(eps)].append(n_clusters)
                opinions = engine.get_opinions()

                suffix = ''
                if should_perturb:
                    suffix = f', perturb_sigma={backward_perturb_sigma:.4f}'
                print(f"      [{scan_label}] run {run_i+1}/{num_runs}, "
                      f"epsilon_0={eps:.3f}: OP={op:.4f}, "
                      f"clusters={n_clusters}, sweeps={sweep_sweeps:g}{suffix}")

        aggregated = {}
        clusters_aggregated = {}
        for eps in eps_sequence:
            aggregated[str(eps)] = self._aggregate_scalar_values(
                eps_results[str(eps)]
            )
            clusters_aggregated[str(eps)] = self._aggregate_scalar_values(
                cluster_results[str(eps)]
            )

        if collect_clusters:
            return aggregated, clusters_aggregated
        return aggregated

    # ================================================================
    # 实验7: 网络属性验证
    # ================================================================

    def _run_network_validation_for_p(self, p, N, m, num_nets):
        print(f"\n  p = {p:.1f}: 生成 {num_nets} 个网络...")
        t0 = time.time()
        props_all = []

        for net_i in range(num_nets):
            G = NetworkGenerator.generate_holme_kim(N, m, p, seed=net_i * 137 + 42)
            props = NetworkGenerator.compute_full_properties(
                G, seed=net_i * 137 + 42)
            if props is None:
                print(f"    [WARN] network properties returned None at net {net_i+1}; skipped")
                continue
            props.setdefault('num_edges', G.number_of_edges())
            props.setdefault('num_triangles', 0)
            props.setdefault('f_triangle', np.nan)
            props.setdefault('avg_clustering', np.nan)
            props.setdefault('global_clustering', np.nan)
            props.setdefault('avg_degree', np.nan)
            props.setdefault('avg_path_length', np.nan)
            props.setdefault('modularity', None)
            props.setdefault('degrees', [d for _, d in G.degree()])
            props_all.append(props)
            print(f"    [{net_i+1}/{num_nets}] edges={props['num_edges']}, "
                  f"tri={props['num_triangles']}, "
                  f"f△={props['f_triangle']:.4f}, "
                  f"avgC={props['avg_clustering']:.4f}, "
                  f"Q={props['modularity'] if props['modularity'] is not None else float('nan'):.4f}")

        # 聚合统计
        keys = ['f_triangle', 'avg_clustering', 'global_clustering',
                'avg_degree', 'avg_path_length', 'num_triangles', 'num_edges']
        stats = {}
        for k in keys:
            vals = []
            for p_item in props_all:
                v = p_item.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            if vals:
                stats[k] = summarize(vals)
            else:
                stats[k] = {'mean': None, 'std': None}

        # 模块度单独处理
        mod_vals = [
            p_item['modularity']
            for p_item in props_all
            if p_item.get('modularity') is not None
            and np.isfinite(p_item.get('modularity'))
        ]
        if mod_vals:
            stats['modularity'] = summarize(mod_vals)
        else:
            stats['modularity'] = {'mean': None, 'std': None}

        # 度分布（合并所有网络的度序列）
        all_degrees = []
        for p_item in props_all:
            all_degrees.extend(p_item.get('degrees', []))
        stats['degree_histogram'] = np.histogram(
            all_degrees, bins=50, density=True)[0].tolist() if all_degrees else []

        elapsed = time.time() - t0
        def _fmt_stat(key, digits=4):
            m = stats.get(key, {}).get('mean')
            s = stats.get(key, {}).get('std')
            if m is None or s is None:
                return 'NA'
            return f"{m:.{digits}f}±{s:.{digits}f}"

        print(f"  p={p:.1f} 完成 ({elapsed:.1f}s): "
              f"f△={_fmt_stat('f_triangle', 4)}, "
              f"avgC={_fmt_stat('avg_clustering', 4)}, "
              f"avgDeg={_fmt_stat('avg_degree', 2)}, "
              f"Q={_fmt_stat('modularity', 4)}")
        return {str(p): stats}

    def experiment_network_validation(
            self, num_networks=None, p_values=None, resume=True,
            output_filename='exp7_network_validation.json'):
        """
        对每个p值生成num_networks个网络，统计拓扑属性。
        默认顺序执行，避免 Windows/Anaconda 多进程 BrokenProcessPool。
        quick 模式可传入较少的 p_values 以快速验证流程。
        """
        print("\n" + "=" * 70)
        print("实验7: 网络属性验证")
        print("=" * 70)

        N = NETWORK['N']
        m = NETWORK['m']
        num_nets = int(num_networks) if num_networks is not None else SIMULATION['num_networks_validation']
        if p_values is None:
            p_values = NETWORK['p_values']
        p_values = [float(p) for p in p_values]
        control = {
            'experiment': 'network_validation',
            'N': int(N), 'm': int(m),
            'num_networks': int(num_nets),
            'p_values': p_values,
            'network_seeds': [net_i * 137 + 42
                              for net_i in range(num_nets)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)

        for p in p_values:
            key = str(p)
            if resume and key in results:
                print(f"  [续跑] 网络验证 p={p:.2f} 已完成")
                continue
            p_result = self._run_network_validation_for_p(p, N, m, num_nets)
            results.update(p_result)
            self.save_json(
                results, output_filename,
                metadata={'n_networks_per_p': num_nets,
                          'network_seed_scheme': 'net_i * 137 + 42'},
            )

        self.save_json(
            results, output_filename,
            metadata={'n_networks_per_p': num_nets,
                      'network_seed_scheme': 'net_i * 137 + 42'},
        )
        print("\n  [实验7完成] 结果已保存")
        return results

    # ================================================================
    # 实验6: 对比实验 (Ablation Study) —— 同一参数五条线
    # ================================================================
    def _run_ablation_for_p(self, p, num_runs, model_order):
        print(f"\n  === p = {p:.2f} ===")
        p_results = {}
        for model_key in model_order:
            cfg = MODEL_TYPES[model_key]
            label = cfg['label']
            engine_key = cfg['engine_key']
            print(f"    [{label}] 运行 {num_runs} 次...")
            t0 = time.time()
            batch = self._run_batch(
                {'p_hk': p, 'model_type': engine_key},
                num_runs=num_runs,
                verbose_label=f"p={p:.2f}, {label}",
            )
            agg = {
                'order_param': self._aggregate(batch, ['final', 'order_param']),
                'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
                'variance': self._aggregate(batch, ['final', 'variance']),
                'triangle_alignment_fixed': self._aggregate(
                    batch, ['final', 'triangle_alignment_fixed']),
                'triangle_admissibility_adaptive': self._aggregate(
                    batch, ['final', 'triangle_admissibility_adaptive']),
                'edge_disagreement_penalty': self._aggregate(
                    batch, ['final', 'edge_disagreement_penalty']),
            }
            # 跨重复平均的轨迹，而不是仅保存第一个 run。
            time_series_summary = self._aggregate_time_series(batch)
            if time_series_summary is not None:
                agg['time_series_summary'] = time_series_summary

            dyn_keys = ['edge_count', 'tri_count', 'tri_success', 'tri_fallback']
            for dk in dyn_keys:
                vals = [r['dynamics_stats'][dk] for r in batch]
                agg[dk] = summarize(vals)
            p_results[model_key] = agg
            elapsed = time.time() - t0
            print(f"      完成 ({elapsed:.1f}s): "
                  f"OP={agg['order_param']['mean']:.4f}±{agg['order_param']['std']:.4f}, "
                  f"clusters={agg['n_clusters']['mean']:.2f}", flush=True)
        full_key = 'full_model'
        paired = {}
        if full_key in p_results:
            for model_key in model_order:
                if model_key == full_key:
                    continue
                comparison = {}
                for metric in ('order_param', 'n_clusters', 'variance',
                               'triangle_alignment_fixed',
                               'triangle_admissibility_adaptive',
                               'edge_disagreement_penalty'):
                    comparison[metric] = paired_difference(
                        p_results[full_key][metric]['values'],
                        p_results[model_key][metric]['values'],
                    )
                paired[f'{full_key}_minus_{model_key}'] = comparison
        p_results['_paired_comparisons'] = paired
        p_results['_control'] = {
            'N': int(NETWORK['N']),
            'm': int(NETWORK['m']),
            'n_runs': int(num_runs),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'model_order': list(model_order),
            'core_parameters': self._core_parameter_signature(),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        return {str(p): p_results}

    def experiment_ablation(self, num_runs=None, p_values=None, resume=True,
                            output_filename='exp6_ablation.json', workers=1):
        """
        消融实验：5种模型变体在多个 p 值下的对比。
        输出结构 results[str(p)][model_key]，用于“一张图五条线”。
        """
        print("\n" + "=" * 70)
        print("实验6: 对比实验 (Ablation Study)")
        print("=" * 70)
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        worker_count = max(1, int(workers))
        p_values_ablation = (NETWORK['p_values'] if p_values is None
                             else [float(p) for p in p_values])
        # 从 config 中安全读取 MODEL_DISPLAY_ORDER（若未定义则使用默认顺序）
        try:
            from .config import MODEL_DISPLAY_ORDER as MODEL_DISPLAY_ORDER_CFG
        except Exception:
            try:
                from config import MODEL_DISPLAY_ORDER as MODEL_DISPLAY_ORDER_CFG
            except Exception:
                MODEL_DISPLAY_ORDER_CFG = [
                    'baseline', 'weighted_bcm', 'threshold_only',
                    'bcm_triangle', 'no_triadic_feedback', 'full_model'
                ]
        # 从配置中读取显示顺序（若存在）并映射为标准模型 key
        raw_order = MODEL_DISPLAY_ORDER_CFG
        model_order = [resolve_model_key(m) for m in raw_order]
        expected_seeds = [run_i * 997 + 7 for run_i in range(num_runs)]

        def compatible_ablation_block(block):
            if not isinstance(block, dict):
                return False
            control = block.get('_control', {})
            return (
                int(control.get('N', -1)) == int(NETWORK['N'])
                and int(control.get('m', -1)) == int(NETWORK['m'])
                and int(control.get('n_runs', -1)) == int(num_runs)
                and float(control.get('main_sweeps', -1))
                == float(SIMULATION.get('main_sweeps', 200))
                and float(control.get('record_sweeps', -1))
                == float(SIMULATION.get('record_sweeps', 5))
                and list(control.get('model_order', [])) == model_order
                and control.get('core_parameters')
                == self._core_parameter_signature()
                and list(control.get('seeds', [])) == expected_seeds
                and int(control.get('workers', 1)) == worker_count
            )

        results = {}
        output_path = os.path.join(DATA_DIR, output_filename)
        self._check_checkpoint_target(output_path, 'ablation', resume)
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if not isinstance(loaded, dict):
                    raise ValueError("top-level checkpoint is not an object")
                incompatible = [
                    str(p) for p in p_values_ablation
                    if str(p) in loaded
                    and not compatible_ablation_block(loaded[str(p)])
                ]
                if incompatible:
                    raise FileExistsError(
                        "ablation checkpoint control/source mismatch for "
                        + ", ".join(incompatible)
                        + "; original file preserved"
                    )
                # Retain only currently requested p values; otherwise a
                # resumed run with a changed grid can silently preserve
                # stale, unrequested blocks in the output file.
                results.update({
                    str(p): loaded[str(p)]
                    for p in p_values_ablation
                    if str(p) in loaded
                })
                print(f"  [续跑] 已读取 {len(results)} 个参数匹配的 p 块")
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"ablation checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        pending = []
        for p in p_values_ablation:
            p_key = str(p)
            existing = results.get(p_key, {})
            matches = compatible_ablation_block(existing)
            if resume and matches:
                print(f"  [续跑] p={p:.2f} 参数匹配，跳过")
                continue
            pending.append(p)

        def complete_block(p, p_result):
            results.update(p_result)
            ordered = self._ordered_numeric_mapping(results)
            results.clear()
            results.update(ordered)
            self.save_json(
                results, output_filename,
                metadata={'n_runs': num_runs,
                          'main_sweeps': SIMULATION.get('main_sweeps'),
                          'resume_enabled': bool(resume)},
            )

        if worker_count == 1 or len(pending) <= 1:
            for p in pending:
                complete_block(p, self._run_ablation_for_p(
                    p, num_runs, model_order))
        else:
            with ProcessPoolExecutor(
                    max_workers=min(worker_count, len(pending)),
                    **_process_pool_kwargs(_capture_runtime_config())) as pool:
                futures = {
                    pool.submit(
                        self._run_ablation_for_p, p, num_runs, model_order): p
                    for p in pending
                }
                for future in as_completed(futures):
                    complete_block(futures[future], future.result())

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'main_sweeps': SIMULATION.get('main_sweeps')},
        )
        print("\n  [实验6完成] 结果已保存")
        return results

    def _run_threshold_assignment_control_for_p(
            self, p, num_runs, model_order, metric_paths, expected_seeds):
        p_block = {}
        for model_key in model_order:
            batch = self._run_batch(
                {'p_hk': p, 'model_type': model_key},
                num_runs=num_runs,
                verbose_label=f"阈值对照 p={p:.2f}, {model_key}",
            )
            p_block[model_key] = {
                metric: self._aggregate(batch, path)
                for metric, path in metric_paths.items()
            }
            p_block[model_key]['time_series_summary'] = (
                self._aggregate_time_series(batch))

        reference = p_block['threshold_only']
        p_block['_paired_comparisons'] = {}
        for comparator in ('mean_threshold_only',
                           'randomized_threshold_only'):
            p_block['_paired_comparisons'][
                f"threshold_only_minus_{comparator}"] = {
                    metric: paired_difference(
                        reference[metric]['values'],
                        p_block[comparator][metric]['values'])
                    for metric in metric_paths
                }
        p_block['_control'] = {
            'complete': True,
            'seeds': expected_seeds,
            'model_order': list(model_order),
        }
        return p_block

    def experiment_threshold_assignment_controls(
            self, num_runs=None, p_values=None, resume=True,
            output_filename='exp_threshold_assignment_controls.json',
            workers=1):
        """Compare structured thresholds with mean and shuffled assignments."""
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        p_values = (NETWORK['p_values'] if p_values is None
                    else [float(p) for p in p_values])
        worker_count = max(1, int(workers))
        model_order = (
            'threshold_only', 'mean_threshold_only',
            'randomized_threshold_only')
        metric_paths = {
            'order_param': ['final', 'order_param'],
            'n_clusters': ['final', 'n_clusters'],
            'variance': ['final', 'variance'],
            'triangle_alignment_fixed': [
                'final', 'triangle_alignment_fixed'],
            'triangle_admissibility_adaptive': [
                'final', 'triangle_admissibility_adaptive'],
            'edge_disagreement_penalty': [
                'final', 'edge_disagreement_penalty'],
        }
        control = {
            'N': int(NETWORK['N']),
            'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'p_values': list(p_values),
            'model_order': list(model_order),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'core_parameters': self._core_parameter_signature(),
            'seed_scheme': 'run_i * 997 + 7; edge permutation uses an independent seed offset',
            'threshold_assignment': {
                'mean_threshold_only': 'unweighted mean over undirected network edges',
                'randomized_threshold_only': 'uniform permutation of realized edge thresholds',
            },
            'workers': int(worker_count),
        }
        results = {'control': control, 'data': {}}
        output_path = os.path.join(DATA_DIR, output_filename)
        self._check_checkpoint_target(
            output_path, 'threshold-assignment', resume)
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if loaded.get('control') == control:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个阈值对照块")
                else:
                    raise FileExistsError(
                        f"threshold-assignment checkpoint control/source "
                        f"mismatch: {output_path}; original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"threshold-assignment checkpoint unreadable: "
                    f"{output_path}; original file preserved"
                ) from exc

        expected_seeds = [run_i * 997 + 7 for run_i in range(num_runs)]
        pending = []
        for p in p_values:
            p_key = f"{p:.2f}"
            existing = results['data'].get(p_key, {})
            if (resume and existing.get('_control', {}).get('complete')
                    and existing.get('_control', {}).get('seeds')
                    == expected_seeds):
                continue
            pending.append(p)

        def complete_block(p, p_block):
            results['data'][f"{p:.2f}"] = p_block
            results['data'] = self._ordered_numeric_mapping(results['data'])
            self.save_json(results, output_filename, metadata={
                'n_runs': num_runs,
                'experiment_design': 'threshold_assignment_controls',
                'models': list(model_order),
            })

        if worker_count == 1 or len(pending) <= 1:
            for p in pending:
                complete_block(
                    p, self._run_threshold_assignment_control_for_p(
                        p, num_runs, model_order, metric_paths, expected_seeds))
        else:
            with ProcessPoolExecutor(
                    max_workers=min(worker_count, len(pending)),
                    **_process_pool_kwargs(_capture_runtime_config())) as pool:
                futures = {
                    pool.submit(
                        self._run_threshold_assignment_control_for_p,
                        p, num_runs, model_order, metric_paths,
                        expected_seeds): p
                    for p in pending
                }
                for future in as_completed(futures):
                    complete_block(futures[future], future.result())

        self.save_json(results, output_filename, metadata={
            'n_runs': num_runs,
            'experiment_design': 'threshold_assignment_controls',
            'models': list(model_order),
        })
        return results

    def _run_triangle_failure_control_for_p(
            self, p, num_runs, model_order, metric_paths, dynamics_paths,
            expected_seeds):
        p_block = {}
        for model_key in model_order:
            batch = self._run_batch(
                {'p_hk': p, 'model_type': model_key},
                num_runs=num_runs,
                verbose_label=f"三角失败对照 p={p:.2f}, {model_key}",
            )
            p_block[model_key] = {
                metric: self._aggregate(batch, path)
                for metric, path in metric_paths.items()
            }
            p_block[model_key]['dynamics_stats'] = {
                metric: self._aggregate(batch, path)
                for metric, path in dynamics_paths.items()
            }
            p_block[model_key]['time_series_summary'] = (
                self._aggregate_time_series(batch))

        reference = p_block['full_model']
        comparator = p_block['full_model_no_triangle_fallback']
        p_block['_paired_comparisons'] = {
            'full_model_minus_full_model_no_triangle_fallback': {
                metric: paired_difference(
                    reference[metric]['values'],
                    comparator[metric]['values'])
                for metric in metric_paths
            },
            'full_model_minus_full_model_no_triangle_fallback_dynamics': {
                metric: paired_difference(
                    reference['dynamics_stats'][metric]['values'],
                    comparator['dynamics_stats'][metric]['values'])
                for metric in dynamics_paths
            },
        }
        p_block['_control'] = {
            'complete': True,
            'seeds': expected_seeds,
            'model_order': list(model_order),
        }
        return p_block

    def experiment_triangle_failure_control(
            self, num_runs=None, p_values=None, resume=True,
            output_filename='exp_triangle_failure_control.json', workers=1):
        """Compare dyadic fallback with no update after a failed triangle."""
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        p_values = (NETWORK['p_values'] if p_values is None
                    else [float(p) for p in p_values])
        worker_count = max(1, int(workers))
        model_order = (
            'full_model', 'full_model_no_triangle_fallback')
        metric_paths = {
            'order_param': ['final', 'order_param'],
            'n_clusters': ['final', 'n_clusters'],
            'variance': ['final', 'variance'],
            'triangle_alignment_fixed': [
                'final', 'triangle_alignment_fixed'],
            'triangle_admissibility_adaptive': [
                'final', 'triangle_admissibility_adaptive'],
            'edge_disagreement_penalty': [
                'final', 'edge_disagreement_penalty'],
        }
        dynamics_paths = {
            'tri_count': ['dynamics_stats', 'tri_count'],
            'tri_success': ['dynamics_stats', 'tri_success'],
            'tri_fallback': ['dynamics_stats', 'tri_fallback'],
            'tri_fallback_edge_update': [
                'dynamics_stats', 'tri_fallback_edge_update'],
            'tri_fallback_suppressed': [
                'dynamics_stats', 'tri_fallback_suppressed'],
            'update_count': ['dynamics_stats', 'update_count'],
        }
        control = {
            'N': int(NETWORK['N']),
            'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'p_values': list(p_values),
            'model_order': list(model_order),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'core_parameters': self._core_parameter_signature(),
            'seed_scheme': 'run_i * 997 + 7; matched network, initial opinions, and RNG draws',
            'failed_triangle_rule': {
                'full_model': 'sample a neighbor and apply the ordinary edge threshold',
                'full_model_no_triangle_fallback': 'make no opinion change',
                'other_rules_held_fixed': [
                    'triangle trigger probability', 'edge-specific thresholds',
                    'successful-triangle update', 'third-party damping',
                ],
            },
            'workers': int(worker_count),
        }
        results = {'control': control, 'data': {}}
        output_path = os.path.join(DATA_DIR, output_filename)
        self._check_checkpoint_target(output_path, 'triangle-failure', resume)
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if loaded.get('control') == control:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个三角失败对照块")
                else:
                    raise FileExistsError(
                        f"triangle-failure checkpoint control/source mismatch: "
                        f"{output_path}; original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"triangle-failure checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        expected_seeds = [run_i * 997 + 7 for run_i in range(num_runs)]
        pending = []
        for p in p_values:
            p_key = f"{p:.2f}"
            existing = results['data'].get(p_key, {})
            if (resume and existing.get('_control', {}).get('complete')
                    and existing.get('_control', {}).get('seeds')
                    == expected_seeds):
                continue
            pending.append(p)

        def complete_block(p, p_block):
            results['data'][f"{p:.2f}"] = p_block
            results['data'] = self._ordered_numeric_mapping(results['data'])
            self.save_json(results, output_filename, metadata={
                'n_runs': num_runs,
                'experiment_design': 'triangle_failure_fallback_control',
                'models': list(model_order),
            })

        if worker_count == 1 or len(pending) <= 1:
            for p in pending:
                complete_block(
                    p, self._run_triangle_failure_control_for_p(
                        p, num_runs, model_order, metric_paths,
                        dynamics_paths, expected_seeds))
        else:
            with ProcessPoolExecutor(
                    max_workers=min(worker_count, len(pending)),
                    **_process_pool_kwargs(_capture_runtime_config())) as pool:
                futures = {
                    pool.submit(
                        self._run_triangle_failure_control_for_p,
                        p, num_runs, model_order, metric_paths,
                        dynamics_paths, expected_seeds): p
                    for p in pending
                }
                for future in as_completed(futures):
                    complete_block(futures[future], future.result())

        self.save_json(results, output_filename, metadata={
            'n_runs': num_runs,
            'experiment_design': 'triangle_failure_fallback_control',
            'models': list(model_order),
        })
        return results

    # ================================================================
    # 核心实验: ε₀ 正/逆向扫描 (磁滞检测)
    # ================================================================

    def _run_epsilon_sweep_for_p(
            self, p, num_runs, eps_values, eps_reverse,
            backward_perturb_sigma=None, perturb_each_step=None):
        print(f"\n  === p = {p:.2f} ===")
        if backward_perturb_sigma is None:
            backward_perturb_sigma = SIMULATION.get(
                'backward_perturb_sigma', 0.0)
        backward_perturb_sigma = float(backward_perturb_sigma)
        if perturb_each_step is None:
            perturb_each_step = SIMULATION.get(
                'backward_perturb_each_step', False)
        perturb_each_step = bool(perturb_each_step)
        scan_num_steps, scan_sweeps = self._scan_step_config()

        p_results = {
            'N': int(NETWORK['N']),
            'm': int(NETWORK['m']),
            'p_hk': float(p),
            'num_runs': int(num_runs),
            'num_steps': int(scan_num_steps),
            'sweep_sweeps': float(scan_sweeps),
            'backward_starts_from_forward': True,
            'backward_perturb_sigma': backward_perturb_sigma,
            'backward_perturb_each_step': perturb_each_step,
            'eps_values': [float(e) for e in eps_values],
            'forward': {},
            'forward_clusters': {},
            'backward': {},
            'backward_clusters': {},
            'hysteresis_area': None,
            'peak_sensitivity': None,
            'backward_low_eps_order_param': None,
            'backward_low_eps_clusters': None,
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
            'core_parameters': self._core_parameter_signature(),
        }

        forward_vals = {str(eps): [] for eps in eps_values}
        forward_cluster_vals = {str(eps): [] for eps in eps_values}
        backward_vals = {str(eps): [] for eps in eps_reverse}
        backward_cluster_vals = {str(eps): [] for eps in eps_reverse}
        perturbation_by_run = []

        print(f"  [forward sweep] epsilon: {eps_values[0]} -> {eps_values[-1]}")
        print(f"  [backward sweep] epsilon: {eps_reverse[0]} -> {eps_reverse[-1]}, "
              f"start=forward terminal state, perturb_sigma={backward_perturb_sigma}")

        N = NETWORK['N']
        m = NETWORK['m']
        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            rng = np.random.RandomState(seed)
            G = NetworkGenerator.generate_holme_kim(N, m, p, seed=seed)
            triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
            weights = NetworkGenerator.compute_jaccard_weights(G)
            clustering = NetworkGenerator.compute_clustering(G)
            nodes = list(G.nodes())
            opinions = {n: rng.uniform(0, 1) for n in nodes}
            run_perturbed = 0
            run_clipped = 0

            for eps in eps_values:
                trust = TrustThreshold(eps, TRUST['alpha'])
                engine = OpinionDynamics(
                    G, opinions, weights, triangles_list, node_triangles,
                    clustering, trust,
                    DYNAMICS['mu'], DYNAMICS['beta_default'],
                    'full_model',
                )
                for _ in range(scan_num_steps):
                    engine.step(rng)

                final_arr = engine.get_opinions_array()
                op = Metrics.order_parameter(final_arr)
                n_clusters = Metrics.count_clusters(final_arr)
                forward_vals[str(eps)].append(op)
                forward_cluster_vals[str(eps)].append(n_clusters)
                opinions = engine.get_opinions()
                print(f"      [forward] run {run_i+1}/{num_runs}, "
                      f"epsilon={eps:.3f}: OP={op:.4f}, "
                      f"clusters={n_clusters}, sweeps={scan_sweeps:g}")

            # True hysteresis: backward starts from the terminal state reached
            # at the largest forward epsilon in this same run.
            for eps_idx, eps in enumerate(eps_reverse):
                should_perturb = (
                    backward_perturb_sigma is not None
                    and backward_perturb_sigma > 0
                    and (perturb_each_step or eps_idx == 0)
                )
                if should_perturb:
                    opinions, perturb_stats = self._perturb_opinions(
                        opinions, rng, backward_perturb_sigma,
                        return_stats=True,
                    )
                    run_perturbed += perturb_stats['n_perturbed']
                    run_clipped += perturb_stats['n_clipped']

                trust = TrustThreshold(eps, TRUST['alpha'])
                engine = OpinionDynamics(
                    G, opinions, weights, triangles_list, node_triangles,
                    clustering, trust,
                    DYNAMICS['mu'], DYNAMICS['beta_default'],
                    'full_model',
                )
                for _ in range(scan_num_steps):
                    engine.step(rng)

                final_arr = engine.get_opinions_array()
                op = Metrics.order_parameter(final_arr)
                n_clusters = Metrics.count_clusters(final_arr)
                backward_vals[str(eps)].append(op)
                backward_cluster_vals[str(eps)].append(n_clusters)
                opinions = engine.get_opinions()

                suffix = ''
                if should_perturb:
                    suffix = f', perturb_sigma={backward_perturb_sigma:.4f}'
                print(f"      [backward] run {run_i+1}/{num_runs}, "
                      f"epsilon={eps:.3f}: OP={op:.4f}, "
                      f"clusters={n_clusters}, sweeps={scan_sweeps:g}{suffix}")

            perturbation_by_run.append({
                'seed': int(seed),
                'n_perturbed': int(run_perturbed),
                'n_clipped': int(run_clipped),
                'clipped_fraction': float(run_clipped / run_perturbed)
                if run_perturbed else 0.0,
            })

        p_results['forward'] = {
            str(eps): self._aggregate_scalar_values(forward_vals[str(eps)])
            for eps in eps_values
        }
        p_results['forward_clusters'] = {
            str(eps): self._aggregate_scalar_values(forward_cluster_vals[str(eps)])
            for eps in eps_values
        }
        p_results['backward'] = {
            str(eps): self._aggregate_scalar_values(backward_vals[str(eps)])
            for eps in eps_reverse
        }
        p_results['backward_clusters'] = {
            str(eps): self._aggregate_scalar_values(backward_cluster_vals[str(eps)])
            for eps in eps_reverse
        }

        forward_ops = [
            p_results['forward'][str(e)]['mean']
            for e in eps_values
        ]
        backward_ops_sorted = [
            p_results['backward'][str(e)]['mean']
            for e in eps_values
        ]
        hyst = Metrics.hysteresis_area(
            forward_ops, backward_ops_sorted, eps_values
        )
        p_results['hysteresis_area'] = float(hyst)
        peak = Metrics.peak_sensitivity(forward_ops, eps_values)
        p_results['peak_sensitivity'] = float(peak)
        run_hysteresis = []
        run_peak_sensitivity = []
        for run_i in range(num_runs):
            forward_run = [forward_vals[str(e)][run_i] for e in eps_values]
            backward_run = [backward_vals[str(e)][run_i] for e in eps_values]
            run_hysteresis.append(Metrics.hysteresis_area(
                forward_run, backward_run, eps_values))
            run_peak_sensitivity.append(Metrics.peak_sensitivity(
                forward_run, eps_values))
        p_results['hysteresis_area_stats'] = summarize(run_hysteresis)
        p_results['peak_sensitivity_stats'] = summarize(
            run_peak_sensitivity)
        # The historical scalar fields are retained as the mean for old plots.
        p_results['hysteresis_area'] = float(
            p_results['hysteresis_area_stats']['mean'])
        p_results['peak_sensitivity'] = float(
            p_results['peak_sensitivity_stats']['mean'])
        p_results['perturbation_diagnostics'] = {
            'by_run': perturbation_by_run,
            'clipped_fraction': summarize(
                [row['clipped_fraction'] for row in perturbation_by_run]),
        }
        low_eps_key = str(eps_values[0])
        p_results['backward_low_eps_order_param'] = float(
            p_results['backward'][low_eps_key]['mean']
        )
        p_results['backward_low_eps_clusters'] = float(
            p_results['backward_clusters'][low_eps_key]['mean']
        )

        print(f"  p={p:.2f}: hysteresis_area={hyst:.6f}, "
              f"peak_sensitivity={peak:.4f}, "
              f"backward_low_OP={p_results['backward_low_eps_order_param']:.4f}, "
              f"backward_low_clusters={p_results['backward_low_eps_clusters']:.2f}")
        return {f"{p:.2f}": p_results}

    def _run_perturbation_sweep_for_p_shared(
            self, p, num_runs, sigma_values, eps_values, eps_reverse,
            perturb_each_step=False):
        """Run one shared forward scan and branch the backward scan by sigma.

        The old implementation repeated the complete forward scan for every
        sigma.  This routine performs that scan once per ``(p, seed)`` and
        then forks from the identical terminal state.  Each sigma receives a
        common standardized Gaussian shock vector (or vectors when
        ``perturb_each_step`` is enabled), scaled only by sigma.  The backward
        engine also starts from the same RNG state for every sigma, so
        differences are attributable to the perturbation amplitude rather
        than to a changed random stream.
        """
        scan_num_steps, scan_sweeps = self._scan_step_config()
        sigma_values = [float(s) for s in sigma_values]
        p_results = {}
        for sigma in sigma_values:
            p_results[f"{sigma:.3f}"] = {
                'N': int(NETWORK['N']), 'm': int(NETWORK['m']),
                'p_hk': float(p), 'num_runs': int(num_runs),
                'num_steps': int(scan_num_steps),
                'sweep_sweeps': float(scan_sweeps),
                'backward_starts_from_forward': True,
                'backward_perturb_sigma': float(sigma),
                'backward_perturb_each_step': bool(perturb_each_step),
                'eps_values': [float(e) for e in eps_values],
                'forward': {}, 'forward_clusters': {},
                'backward': {}, 'backward_clusters': {},
                'forward_tail_mean_abs_dx': {},
                'backward_tail_mean_abs_dx': {},
                'hysteresis_area': None, 'peak_sensitivity': None,
                'backward_low_eps_order_param': None,
                'backward_low_eps_clusters': None,
                'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
                'core_parameters': self._core_parameter_signature(),
                'shared_forward_scan': True,
                'common_random_numbers': True,
            }
            for field in ('forward', 'forward_clusters',
                          'forward_tail_mean_abs_dx', 'backward',
                          'backward_clusters', 'backward_tail_mean_abs_dx'):
                p_results[f"{sigma:.3f}"][field] = {
                    str(e): [] for e in (
                        eps_values if field.startswith('forward') else eps_reverse)
                }
            p_results[f"{sigma:.3f}"]['perturbation_diagnostics'] = []

        N, m = NETWORK['N'], NETWORK['m']
        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            rng_forward = np.random.RandomState(seed)
            G = NetworkGenerator.generate_holme_kim(N, m, p, seed=seed)
            triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
            weights = NetworkGenerator.compute_jaccard_weights(G)
            clustering = NetworkGenerator.compute_clustering(G)
            nodes = list(G.nodes())
            opinions = {n: rng_forward.uniform(0, 1) for n in nodes}
            forward_run = {str(e): None for e in eps_values}
            forward_cluster_run = {str(e): None for e in eps_values}
            forward_tail_dx_run = {str(e): None for e in eps_values}
            tail_steps = min(
                scan_num_steps,
                max(1, int(round(N * float(SIMULATION.get('record_sweeps', 5))))))
            for eps in eps_values:
                engine = OpinionDynamics(
                    G, opinions, weights, triangles_list, node_triangles,
                    clustering, TrustThreshold(eps, TRUST['alpha']),
                    DYNAMICS['mu'], DYNAMICS['beta_default'], 'full_model')
                tail_dx_sum = 0.0
                for step_idx in range(scan_num_steps):
                    dx = engine.step(rng_forward)
                    if step_idx >= scan_num_steps - tail_steps:
                        tail_dx_sum += dx
                arr = engine.get_opinions_array()
                forward_run[str(eps)] = Metrics.order_parameter(arr)
                forward_cluster_run[str(eps)] = Metrics.count_clusters(arr)
                forward_tail_dx_run[str(eps)] = float(tail_dx_sum / tail_steps)
                opinions = engine.get_opinions()

            terminal_opinions = dict(opinions)
            forward_rng_state = rng_forward.get_state()
            shock_rng = np.random.RandomState(seed + 104729)
            shocks = []
            n_shocks = len(eps_reverse) if perturb_each_step else 1
            for _ in range(n_shocks):
                shocks.append({node: float(shock_rng.normal()) for node in nodes})

            # Store the shared forward observations in every sigma block.
            for sigma in sigma_values:
                block = p_results[f"{sigma:.3f}"]
                for e in eps_values:
                    block['forward'][str(e)].append(forward_run[str(e)])
                    block['forward_clusters'][str(e)].append(
                        forward_cluster_run[str(e)])
                    block['forward_tail_mean_abs_dx'][str(e)].append(
                        forward_tail_dx_run[str(e)])

            for sigma in sigma_values:
                sigma_key = f"{sigma:.3f}"
                block = p_results[sigma_key]
                rng_backward = np.random.RandomState()
                rng_backward.set_state(forward_rng_state)
                branch_opinions = dict(terminal_opinions)
                run_perturbed = run_clipped = 0
                for eps_idx, eps in enumerate(eps_reverse):
                    should_perturb = sigma > 0.0 and (
                        perturb_each_step or eps_idx == 0)
                    if should_perturb:
                        branch_opinions, stats = self._apply_standardized_perturbation(
                            branch_opinions, shocks[eps_idx if perturb_each_step else 0],
                            sigma, return_stats=True)
                        run_perturbed += stats['n_perturbed']
                        run_clipped += stats['n_clipped']
                    engine = OpinionDynamics(
                        G, branch_opinions, weights, triangles_list,
                        node_triangles, clustering,
                        TrustThreshold(eps, TRUST['alpha']),
                        DYNAMICS['mu'], DYNAMICS['beta_default'], 'full_model')
                    tail_dx_sum = 0.0
                    for step_idx in range(scan_num_steps):
                        dx = engine.step(rng_backward)
                        if step_idx >= scan_num_steps - tail_steps:
                            tail_dx_sum += dx
                    arr = engine.get_opinions_array()
                    block['backward'][str(eps)].append(Metrics.order_parameter(arr))
                    block['backward_clusters'][str(eps)].append(
                        Metrics.count_clusters(arr))
                    block['backward_tail_mean_abs_dx'][str(eps)].append(
                        float(tail_dx_sum / tail_steps))
                    branch_opinions = engine.get_opinions()
                block['perturbation_diagnostics'].append({
                    'seed': int(seed), 'n_perturbed': int(run_perturbed),
                    'n_clipped': int(run_clipped),
                    'clipped_fraction': float(run_clipped / run_perturbed)
                    if run_perturbed else 0.0,
                })

        for sigma in sigma_values:
            block = p_results[f"{sigma:.3f}"]
            for field, eps_seq in (
                    ('forward', eps_values), ('forward_clusters', eps_values),
                    ('forward_tail_mean_abs_dx', eps_values),
                    ('backward', eps_reverse), ('backward_clusters', eps_reverse),
                    ('backward_tail_mean_abs_dx', eps_reverse)):
                raw = block[field]
                block[field] = {str(e): self._aggregate_scalar_values(raw[str(e)])
                                for e in eps_seq}
            run_hysteresis = []
            run_peak = []
            for run_i in range(num_runs):
                f_run = [block['forward'][str(e)]['values'][run_i] for e in eps_values]
                b_run = [block['backward'][str(e)]['values'][run_i] for e in eps_values]
                run_hysteresis.append(Metrics.hysteresis_area(f_run, b_run, eps_values))
                run_peak.append(Metrics.peak_sensitivity(f_run, eps_values))
            block['hysteresis_area_stats'] = summarize(run_hysteresis)
            block['peak_sensitivity_stats'] = summarize(run_peak)
            block['hysteresis_area'] = float(block['hysteresis_area_stats']['mean'])
            block['peak_sensitivity'] = float(block['peak_sensitivity_stats']['mean'])
            block['perturbation_diagnostics'] = {
                'by_run': block['perturbation_diagnostics'],
                'clipped_fraction': summarize([
                    row['clipped_fraction'] for row in block['perturbation_diagnostics']
                ]),
            }
            low_key = str(eps_values[0])
            block['backward_low_eps_order_param'] = float(block['backward'][low_key]['mean'])
            block['backward_low_eps_clusters'] = float(block['backward_clusters'][low_key]['mean'])
        return p_results

    def experiment_epsilon_sweep(
            self, num_runs=None, p_values=None, eps_values=None,
            resume=True, output_filename='exp_epsilon_sweep.json'):
        """
        核心实验：扫描ε₀观察相变行为
        正向/逆向扫描均带状态继承以测量磁滞
        """
        print("\n" + "=" * 70)
        print("核心实验: epsilon_0 正/逆向扫描 (磁滞检测)")
        print("=" * 70)

        if num_runs is None:
            num_runs = SIMULATION['num_runs']

        # 容错读取 epsilon 范围：优先 TRUST，其次 JOINT_SWEEP，最后回退生成
        if eps_values is None:
            eps_values = TRUST.get('epsilon_0_range')
        if not eps_values:
            eps_values = JOINT_SWEEP.get('eps_values') if 'JOINT_SWEEP' in globals() else None
        if not eps_values:
            eps_values = [round(0.10 + x * 0.025, 3) for x in range(19)]  # 0.10到0.55，19个点
        # 保证为浮点列表
        eps_values = [float(x) for x in eps_values]
        eps_reverse = list(reversed(eps_values))    # 0.55 → 0.10
        if p_values is None:
            p_values = list(NETWORK["p_values"])
        p_values_sweep = [float(value) for value in p_values]
        scan_num_steps, scan_sweeps = self._scan_step_config()
        control = {
            'experiment': 'epsilon_sweep',
            'N': int(NETWORK['N']), 'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'scan_num_steps': int(scan_num_steps),
            'sweep_sweeps': float(scan_sweeps),
            'p_values': p_values_sweep, 'eps_values': eps_values,
            'backward_perturb_sigma': float(
                SIMULATION.get('backward_perturb_sigma', 0.0)),
            'backward_perturb_each_step': bool(
                SIMULATION.get('backward_perturb_each_step', False)),
            'core_parameters': self._core_parameter_signature(),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)

        for p in p_values_sweep:
            key = f"{p:.2f}"
            if resume and key in results:
                print(f"  [续跑] epsilon 扫描 p={p:.2f} 已完成")
                continue
            p_result = self._run_epsilon_sweep_for_p(p, num_runs, eps_values, eps_reverse)
            results.update(p_result)
            self.save_json(
                results, output_filename,
                metadata={'n_runs': num_runs,
                          'sweep_sweeps': scan_sweeps})

        self.save_json(results, output_filename,
                       metadata={'n_runs': num_runs,
                                 'sweep_sweeps': scan_sweeps})
        print("\n  [epsilon_0 扫描实验完成]")
        return results


    # ================================================================
    # Reviewer control: paired fixed-versus-structure threshold scan
    # ================================================================
    def _run_transition_prediction_seed(
            self, p_hk, seed, eps_values, sweeps_per_epsilon,
            alpha_target):
        """Run matched alpha=0 and structural-threshold forward branches."""
        N = int(NETWORK["N"])
        m = int(NETWORK["m"])
        seed = int(seed)
        graph = NetworkGenerator.generate_holme_kim(
            N, m, float(p_hk), seed=seed)
        triangles_list, node_triangles = NetworkGenerator.extract_triangles(graph)
        weights = NetworkGenerator.compute_jaccard_weights(graph)
        clustering = NetworkGenerator.compute_clustering(graph)
        nodes = list(graph.nodes())
        opinion_rng = np.random.RandomState(seed)
        initial = {
            node: float(opinion_rng.uniform(0.0, 1.0))
            for node in nodes
        }
        edge_weights = np.asarray([
            weights.get((u, v), weights.get((v, u), 0.0))
            for u, v in graph.edges()
        ], dtype=float)
        if edge_weights.size == 0:
            raise ValueError("transition scan network has no edges")
        expansion = 1.0 - np.exp(-float(alpha_target) * edge_weights)
        gamma = float(np.mean(expansion))
        degrees = dict(graph.degree())
        edge_probabilities = np.asarray([
            (1.0 / float(N)) * (1.0 / degrees[u] + 1.0 / degrees[v])
            for u, v in graph.edges()
        ], dtype=float)
        if abs(float(np.sum(edge_probabilities)) - 1.0) > 1e-10:
            raise ValueError("dyadic edge-proposal probabilities do not sum to one")
        gamma_dyadic_proposal = float(np.dot(
            edge_probabilities, expansion))
        num_steps = int(round(N * float(sweeps_per_epsilon)))
        variants = {
            "uniform_reference": ("baseline", 0.0),
            "structure_only": ("threshold_only", float(alpha_target)),
            "structure_full": ("full_model", float(alpha_target)),
        }
        variant_results = {}
        for variant, (model_type, alpha) in variants.items():
            opinions = dict(initial)
            order_values = []
            interaction_rows = []
            for eps_index, epsilon in enumerate(eps_values):
                trust = TrustThreshold(float(epsilon), float(alpha))
                engine = OpinionDynamics(
                    graph, opinions, weights, triangles_list, node_triangles,
                    clustering, trust, DYNAMICS["mu"],
                    DYNAMICS["beta_default"], model_type)
                # Each epsilon point gets a deterministic stream shared by
                # both variants; the opinion state itself continues forward.
                update_seed = int(
                    ((seed + 1) * 1000003 + (eps_index + 1) * 9176 + 104729)
                    % (2 ** 32 - 1))
                rng = np.random.RandomState(update_seed)
                for _ in range(num_steps):
                    engine.step(rng)
                opinions = engine.get_opinions()
                order_values.append(Metrics.order_parameter(
                    engine.get_opinions_array(),
                    width=float(METRICS['transition_order_window_width'])))
                interaction_rows.append({
                    key: int(engine.stats[key])
                    for key in (
                        "edge_count", "tri_count", "tri_success",
                        "tri_fallback", "tri_fallback_edge_update",
                        "update_count")
                })
            variant_results[variant] = {
                "alpha": float(alpha),
                "order_parameter": order_values,
                "interactions": interaction_rows,
            }
        return {
            "seed": seed,
            "n_edges": int(graph.number_of_edges()),
            "gamma_G": gamma,
            "gamma_dyadic_proposal": gamma_dyadic_proposal,
            "variants": variant_results,
        }

    def experiment_threshold_transition_prediction(
            self, num_runs=None, p_values=None, eps_values=None,
            sweeps_per_epsilon=None, alpha_target=None, workers=1,
            resume=True,
            output_filename=(
                "exp_formal_threshold_transition_prediction_revision_v3.json")):
        """Test the mean-threshold transition proxy against paired simulations.

        The pairwise uniform reference and the full model use identical
        network and initial opinion realizations. The reference is the
        asynchronous unilateral BCM with a uniform threshold; the comparison
        adds the formal structural-threshold value. The full model additionally
        exposes the deviation caused by triangle selection and three-edge gates.
        """
        if num_runs is None:
            num_runs = SIMULATION["num_runs"]
        if p_values is None:
            p_values = [0.0, 0.4, 0.8]
        p_values = [float(value) for value in p_values]
        if eps_values is None:
            eps_values = [round(index * 0.025, 3) for index in range(23)]
        eps_values = [float(value) for value in eps_values]
        if not eps_values or sorted(eps_values) != eps_values:
            raise ValueError("epsilon grid must be non-empty and increasing")
        if sweeps_per_epsilon is None:
            sweeps_per_epsilon = SIMULATION.get("sweep_sweeps", 100.0)
        sweeps_per_epsilon = float(sweeps_per_epsilon)
        if sweeps_per_epsilon <= 0:
            raise ValueError("sweeps_per_epsilon must be positive")
        if alpha_target is None:
            alpha_target = TRUST["alpha"]
        alpha_target = float(alpha_target)
        if alpha_target <= 0:
            raise ValueError("alpha_target must be positive")

        seeds = [run_i * 997 + 7 for run_i in range(int(num_runs))]
        control = {
            "experiment": "paired_forward_transition_prediction",
            "N": int(NETWORK["N"]),
            "m": int(NETWORK["m"]),
            "num_runs": int(num_runs),
            "seeds": seeds,
            "p_values": p_values,
            "eps_values": eps_values,
            "sweeps_per_epsilon": sweeps_per_epsilon,
            "num_steps_per_epsilon": int(round(
                int(NETWORK["N"]) * sweeps_per_epsilon)),
            "transition_metric": (
                "maximum fraction of opinions in any interval of configured width"),
            "transition_order_window_width": float(
                METRICS['transition_order_window_width']),
            "transition_cutoff": float(METRICS['transition_order_cutoff']),
            "transition_order_parameter_formula": (
                "max_x count_i(x_i in [x, x + window_width]) / N"),
            "transition_criterion": (
                "per-seed first forward-grid epsilon whose final order parameter "
                ">= {:.6g}".format(float(METRICS['transition_order_cutoff']))),
            "threshold_expansion_factor": "Gamma_G = mean_edges[1-exp(-alpha_target*w_ij)]",
            "dyadic_proposal_weighting": "P(edge uv)=(1/N)(1/k_u+1/k_v)",
            "prediction": "max(0,(epsilon_c_uniform-Gamma_G)/(1-Gamma_G))",
            "variants": {
                "uniform_reference": {
                    "model_type": "baseline",
                    "alpha": 0.0,
                    "triangle_protocol": "none; asynchronous pairwise BCM",
                },
                "structure_only": {
                    "model_type": "threshold_only",
                    "alpha": alpha_target,
                    "triangle_protocol": "none; edge-specific threshold on each proposed edge",
                },
                "structure_full": {
                    "model_type": "full_model",
                    "alpha": alpha_target,
                    "triangle_protocol": "full-model trigger, three-edge gate and fallback",
                },
            },
            "initialization": "shared iid Uniform(0,1) opinions within each network seed",
            "update_streams": (
                "same deterministic epsilon-specific update seed across variants; "
                "opinion states continue forward separately"),
            "core_parameters": {
                "mu": float(DYNAMICS["mu"]),
                "beta": float(DYNAMICS["beta_default"]),
            },
            "execution_source_signature": execution_source_signature(),
        }
        output_path = os.path.join(DATA_DIR, output_filename)
        results = {"control": control, "data": {}}
        if os.path.exists(output_path):
            if not resume:
                raise FileExistsError(
                    "transition-scan checkpoint already exists; choose a new "
                    "output_filename rather than overwriting it")
            try:
                with open(output_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except Exception as exc:
                raise RuntimeError(
                    "unable to read transition-scan checkpoint; original file "
                    "was preserved: {}".format(exc))
            if not isinstance(loaded, dict) or loaded.get("control") != control:
                raise ValueError(
                    "transition-scan checkpoint controls differ; original file "
                    "was preserved. Use a new output_filename for a new design")
            results = loaded
            print("  [resume] loaded matching transition-scan checkpoint")

        worker_count = max(1, int(workers))
        for p_hk in p_values:
            p_key = "{:.2f}".format(p_hk)
            p_block = results["data"].setdefault(
                p_key, {"p_hk": p_hk, "seeds": seeds, "runs": {}})
            pending = [
                seed for seed in seeds
                if not (resume and str(seed) in p_block["runs"])
            ]
            if worker_count > 1 and len(pending) > 1:
                with ProcessPoolExecutor(
                        max_workers=min(worker_count, len(pending)),
                        **_process_pool_kwargs(_capture_runtime_config())) as pool:
                    future_to_seed = {
                        pool.submit(
                            self._run_transition_prediction_seed,
                            p_hk, seed, eps_values, sweeps_per_epsilon,
                            alpha_target): seed
                        for seed in pending
                    }
                    completed = 0
                    for future in as_completed(future_to_seed):
                        seed = future_to_seed[future]
                        p_block["runs"][str(seed)] = future.result()
                        p_block["runs"] = self._ordered_numeric_mapping(
                            p_block["runs"])
                        completed += 1
                        self.save_json(results, output_filename, metadata={
                            "n_runs": int(num_runs),
                            "experiment_design":
                                "paired_forward_transition_prediction",
                            "p_values": p_values,
                            "eps_values": eps_values,
                            "sweeps_per_epsilon": sweeps_per_epsilon,
                            "alpha_target": alpha_target,
                            "workers": worker_count,
                            "transition_metric_config": {
                                "metric": control["transition_metric"],
                                "order_window_width": control[
                                    "transition_order_window_width"],
                                "cutoff": control["transition_cutoff"],
                                "formula": control["transition_order_parameter_formula"],
                            },
                        })
                        if completed % 5 == 0 or completed == len(pending):
                            print(
                                "  [progress] p_HK={:.2f}: {}/{} new seeds; "
                                "total={}/{}".format(
                                    p_hk, completed, len(pending),
                                    len(p_block["runs"]), len(seeds)),
                                flush=True)
            else:
                for seed in pending:
                    p_block["runs"][str(seed)] = (
                        self._run_transition_prediction_seed(
                            p_hk, seed, eps_values, sweeps_per_epsilon,
                            alpha_target))
                    p_block["runs"] = self._ordered_numeric_mapping(
                        p_block["runs"])
                    self.save_json(results, output_filename, metadata={
                        "n_runs": int(num_runs),
                        "experiment_design":
                            "paired_forward_transition_prediction",
                        "p_values": p_values,
                        "eps_values": eps_values,
                        "sweeps_per_epsilon": sweeps_per_epsilon,
                        "alpha_target": alpha_target,
                        "workers": worker_count,
                        "transition_metric_config": {
                            "metric": control["transition_metric"],
                            "order_window_width": control[
                                "transition_order_window_width"],
                            "cutoff": control["transition_cutoff"],
                            "formula": control["transition_order_parameter_formula"],
                        },
                    })
                    print(
                        "  [progress] p_HK={:.2f}: {}/{} seeds".format(
                            p_hk, len(p_block["runs"]), len(seeds)),
                        flush=True)
            print("  [complete] transition scan p_HK={:.2f} ({}/{} seeds)".format(
                p_hk, len(p_block["runs"]), len(seeds)), flush=True)
        results["data"] = self._ordered_numeric_mapping(results["data"])
        self.save_json(results, output_filename, metadata={
            "n_runs": int(num_runs),
            "experiment_design": "paired_forward_transition_prediction",
            "p_values": p_values,
            "eps_values": eps_values,
            "sweeps_per_epsilon": sweeps_per_epsilon,
            "alpha_target": alpha_target,
            "workers": worker_count,
            "transition_metric_config": {
                "metric": control["transition_metric"],
                "order_window_width": control[
                    "transition_order_window_width"],
                "cutoff": control["transition_cutoff"],
                "formula": control["transition_order_parameter_formula"],
            },
        })
        return results


    # ================================================================
    # 扩展实验: 全 p 值磁滞面积扫描
    # ================================================================
    @staticmethod
    def _hysteresis_summary_rows(results, p_values):
        """将磁滞扫描结果整理为绘图/摘要所需的数组。"""
        rows = []
        for p in p_values:
            p_key = f"{float(p):.2f}"
            item = results.get(p_key)
            if item is None:
                rows.append({
                    'p': float(p),
                    'hysteresis_area': float('nan'),
                    'peak_sensitivity': float('nan'),
                    'backward_low_eps_order_param': float('nan'),
                    'backward_low_eps_clusters': float('nan'),
                })
                continue
            rows.append({
                'p': float(p),
                'hysteresis_area': float(item.get('hysteresis_area', np.nan)),
                'peak_sensitivity': float(item.get('peak_sensitivity', np.nan)),
                'backward_low_eps_order_param': float(item.get('backward_low_eps_order_param', np.nan)),
                'backward_low_eps_clusters': float(item.get('backward_low_eps_clusters', np.nan)),
            })
        return rows

    def plot_hysteresis_sweep(self, results, p_values=None,
                              filename='fig_hysteresis_sweep.png'):
        """
        绘制完整 p_HK 扫描下的磁滞面积、峰值灵敏度和后向低阈值韧性。

        该函数被 experiment_hysteresis_sweep 调用，也可单独用于重画图。
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if p_values is None:
            p_values = NETWORK['p_values']
        p_values = [float(p) for p in p_values]
        rows = self._hysteresis_summary_rows(results, p_values)

        ps = [r['p'] for r in rows]
        areas = [r['hysteresis_area'] for r in rows]
        sensitivities = [r['peak_sensitivity'] for r in rows]
        low_ops = [r['backward_low_eps_order_param'] for r in rows]
        low_clusters = [r['backward_low_eps_clusters'] for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        ax.plot(ps, areas, 'o-', linewidth=2, markersize=5)
        ax.fill_between(ps, 0, areas, alpha=0.2)
        ax.set_xlabel('$p_{HK}$')
        ax.set_ylabel('Hysteresis Area')
        ax.set_title('Hysteresis Area vs $p_{HK}$')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(ps, sensitivities, 's-', linewidth=2, markersize=5)
        ax.set_xlabel('$p_{HK}$')
        ax.set_ylabel('Peak Sensitivity')
        ax.set_title('Peak Sensitivity vs $p_{HK}$')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(ps, low_ops, '^-', linewidth=2, markersize=5)
        ax.axhline(0.9, linestyle='--', linewidth=1)
        ax.set_xlabel('$p_{HK}$')
        ax.set_ylabel('Backward Low-$\\epsilon_0$ Order Parameter')
        ax.set_title('Residual Consensus after Backward Perturbation')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(ps, low_clusters, 'D-', linewidth=2, markersize=5)
        ax.axhline(1.0, linestyle='--', linewidth=1)
        ax.set_xlabel('$p_{HK}$')
        ax.set_ylabel('Backward Low-$\\epsilon_0$ Clusters')
        ax.set_title('Backward Fragmentation under Perturbation')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(FIGURE_DIR, filename)
        os.makedirs(FIGURE_DIR, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [保存] {out_path}")
        return out_path

    def experiment_hysteresis_sweep(
            self, num_runs=None, p_values=None, eps_values=None,
            save_figure=True, resume=True):
        """
        完整磁滞面积扫描：对 NETWORK['p_values'] 中的全部 p_HK 值计算磁滞面积。

        稳定版特性：
        - 每完成一个 p_HK 就立即写入 exp_hysteresis_sweep.json，避免中断后全部重跑；
        - 默认 resume=True，会跳过已经完成且 num_runs/num_steps 匹配的 p_HK；
        - 如果要强制重跑，先删除 output/data/exp_hysteresis_sweep.json，或把 resume=False。
        """
        print("\n" + "=" * 70)
        print("扩展实验: 全 p_HK 磁滞面积扫描")
        print("=" * 70)

        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if p_values is None:
            p_values = NETWORK['p_values']
        p_values = [float(p) for p in p_values]

        if eps_values is None:
            eps_values = TRUST.get('epsilon_0_range')
        if not eps_values:
            eps_values = JOINT_SWEEP.get('eps_values') if 'JOINT_SWEEP' in globals() else None
        if not eps_values:
            eps_values = [round(0.10 + x * 0.025, 3) for x in range(19)]
        eps_values = [float(x) for x in eps_values]
        eps_reverse = list(reversed(eps_values))

        print(f"  p值数量: {len(p_values)} ({min(p_values):.2f} → {max(p_values):.2f})")
        print(f"  epsilon_0 数量: {len(eps_values)} "
              f"({min(eps_values):.3f} -> {max(eps_values):.3f})")
        print(f"  每个p运行次数: {num_runs}")
        scan_num_steps, scan_sweeps = self._scan_step_config()
        print(f"  每个epsilon点扫描sweeps: {scan_sweeps:g}")
        print(f"  每个epsilon点扫描步数: {scan_num_steps}")
        print(f"  后向扰动强度: {SIMULATION.get('backward_perturb_sigma', 0.0)}")
        print(f"  后向每步扰动: {SIMULATION.get('backward_perturb_each_step', True)}")
        print(f"  断点续跑: {resume}")

        out_name = 'exp_hysteresis_sweep.json'
        out_path = os.path.join(DATA_DIR, out_name)

        def compatible_hysteresis_block(existing):
            if not isinstance(existing, dict):
                return False
            return (
                int(existing.get('num_runs', -1)) == int(num_runs)
                and int(existing.get('N', -1)) == int(NETWORK['N'])
                and int(existing.get('m', -1)) == int(NETWORK['m'])
                and int(existing.get('num_steps', -1)) == int(scan_num_steps)
                and float(existing.get('sweep_sweeps', -1.0))
                == float(scan_sweeps)
                and list(existing.get('eps_values', [])) == eps_values
                and existing.get('core_parameters')
                == self._core_parameter_signature()
                and bool(existing.get('backward_starts_from_forward', False))
                and float(existing.get('backward_perturb_sigma', -999))
                == float(SIMULATION.get('backward_perturb_sigma', 0.0))
                and bool(existing.get('backward_perturb_each_step', True))
                == bool(SIMULATION.get('backward_perturb_each_step', False))
                and existing.get('hysteresis_area_stats') is not None
            )

        results = {}
        if resume and os.path.exists(out_path):
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("top-level checkpoint is not an object")
                incompatible = [
                    f"{p:.2f}" for p in p_values
                    if f"{p:.2f}" in loaded
                    and not compatible_hysteresis_block(loaded[f"{p:.2f}"])
                ]
                if incompatible:
                    raise FileExistsError(
                        "hysteresis checkpoint control/source mismatch for "
                        + ", ".join(incompatible)
                        + "; original file preserved"
                    )
                results.update({
                    f"{p:.2f}": loaded[f"{p:.2f}"]
                    for p in p_values
                    if f"{p:.2f}" in loaded
                })
                print(f"  [续跑] 已读取已有结果: {out_path}，"
                      f"包含 {len(results)} 个参数匹配的 p 值")
            except Exception as e:
                if isinstance(e, FileExistsError):
                    raise
                raise FileExistsError(
                    f"hysteresis checkpoint unreadable: {out_path}; "
                    "original file preserved"
                ) from e

        t0 = time.time()
        try:
            for idx, p in enumerate(p_values, start=1):
                p_key = f"{p:.2f}"
                existing = results.get(p_key)
                if resume and compatible_hysteresis_block(existing):
                    print(f"\n  [{idx}/{len(p_values)}] p_HK={p:.2f} 已完成，跳过")
                    continue

                print(f"\n  [{idx}/{len(p_values)}] p_HK={p:.2f}")
                p_result = self._run_epsilon_sweep_for_p(
                    p, num_runs, eps_values, eps_reverse
                )
                results.update(p_result)

                # 每完成一个 p_HK 立即保存一次，防止中途 KeyboardInterrupt 后丢失全部进度。
                self.save_json(results, out_name,
                               metadata={'n_runs': num_runs,
                                         'sweep_sweeps': scan_sweeps})

        except KeyboardInterrupt:
            print("\n  [中断] 检测到 KeyboardInterrupt。已保留已完成 p_HK 的 JSON 结果。")
            self.save_json(results, out_name,
                           metadata={'n_runs': num_runs,
                                     'sweep_sweeps': scan_sweeps})
            raise

        self.save_json(results, out_name,
                       metadata={'n_runs': num_runs,
                                 'sweep_sweeps': scan_sweeps})

        if save_figure:
            self.plot_hysteresis_sweep(
                results, p_values, filename='fig_hysteresis_sweep.png'
            )

        elapsed = time.time() - t0
        print("\n  [全p磁滞扫描完成]")
        print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"{'p_HK':>8} {'Hysteresis':>14} {'Sensitivity':>14} {'BackLowOP':>12} {'BackLowCls':>12}")
        print("-" * 68)
        for row in self._hysteresis_summary_rows(results, p_values):
            print(f"{row['p']:>8.2f} {row['hysteresis_area']:>14.6f} "
                  f"{row['peak_sensitivity']:>14.4f} "
                  f"{row['backward_low_eps_order_param']:>12.4f} "
                  f"{row['backward_low_eps_clusters']:>12.2f}")
        return results


    # ================================================================
    # Reviewer control: independent perturbation and recovery at each epsilon
    # ================================================================
    def _run_threshold_recovery_for_p(
            self, p, num_runs, sigma_values, eps_values,
            recovery_order_threshold=1.0, recovery_persistence=3):
        """Measure recovery after an independent shock at every threshold.

        Reference states come from one unperturbed high-to-low scan. At each
        epsilon, every sigma branch starts from the state entering that point,
        receives a matched standardized Gaussian shock, and evolves at that
        fixed epsilon. Unrecovered trajectories are right-censored at the
        configured per-epsilon horizon.
        """
        N = int(NETWORK['N'])
        scan_num_steps, scan_sweeps = self._scan_step_config()
        record_sweeps = float(SIMULATION.get('record_sweeps', 5))
        record_interval = max(1, int(round(N * record_sweeps)))
        measurement_steps = list(range(record_interval, scan_num_steps + 1,
                                       record_interval))
        if not measurement_steps or measurement_steps[-1] != scan_num_steps:
            measurement_steps.append(scan_num_steps)
        time_sweeps = [0.0] + [float(step) / N for step in measurement_steps]
        eps_values = [float(value) for value in eps_values]
        eps_reverse = sorted(eps_values, reverse=True)
        sigma_values = [float(value) for value in sigma_values]
        if not sigma_values or not eps_values:
            raise ValueError("sigma_values and eps_values must be non-empty")
        if recovery_persistence < 1:
            raise ValueError("recovery_persistence must be at least one")

        def metric_values(opinions):
            values = np.asarray(list(opinions.values()), dtype=float)
            return {
                'opinion_sd': float(np.std(values)),
                'opinion_range': float(np.max(values) - np.min(values))
                if len(values) else 0.0,
                'order_parameter': float(Metrics.order_parameter(values)),
            }

        raw = {}
        reference_raw = {}
        reference_input_raw = {}
        for eps in eps_values:
            eps_key = str(eps)
            raw[eps_key] = {}
            reference_raw[eps_key] = {
                'opinion_sd': [], 'opinion_range': [], 'order_parameter': [],
            }
            reference_input_raw[eps_key] = {
                'opinion_sd': [], 'opinion_range': [], 'order_parameter': [],
            }
            for sigma in sigma_values:
                sigma_key = "{:.3f}".format(sigma)
                raw[eps_key][sigma_key] = {
                    'trajectory': {
                        'opinion_sd': [], 'opinion_range': [],
                        'order_parameter': [],
                    },
                    'recovery_time_sweeps': [],
                    'recovered': [],
                    'initial': {
                        'opinion_sd': [], 'opinion_range': [],
                        'order_parameter': [],
                    },
                    'endpoint': {
                        'opinion_sd': [], 'opinion_range': [],
                        'order_parameter': [],
                    },
                    'perturbation_clipped_fraction': [],
                    'perturbation_n_clipped': [],
                }

        for run_i in range(int(num_runs)):
            seed = run_i * 997 + 7
            G = NetworkGenerator.generate_holme_kim(
                N, NETWORK['m'], p, seed=seed)
            triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
            weights = NetworkGenerator.compute_jaccard_weights(G)
            clustering = NetworkGenerator.compute_clustering(G)
            nodes = list(G.nodes())
            rng_initial = np.random.RandomState(seed)
            opinions = {node: float(rng_initial.uniform(0.0, 1.0))
                        for node in nodes}
            rng_reference = np.random.RandomState(seed + 104729)
            reference_states = {}
            reference_rng_states = {}

            # Branch at each epsilon from the state entering that threshold.
            # The clean path continues independently to define the next state.
            for eps in eps_reverse:
                reference_states[str(eps)] = dict(opinions)
                reference_rng_states[str(eps)] = rng_reference.get_state()
                for metric, value in metric_values(opinions).items():
                    reference_input_raw[str(eps)][metric].append(value)
                engine = OpinionDynamics(
                    G, opinions, weights, triangles_list, node_triangles,
                    clustering, TrustThreshold(eps, TRUST['alpha']),
                    DYNAMICS['mu'], DYNAMICS['beta_default'], 'full_model')
                for _ in range(scan_num_steps):
                    engine.step(rng_reference)
                opinions = engine.get_opinions()
                for metric, value in metric_values(opinions).items():
                    reference_raw[str(eps)][metric].append(value)

            for eps_idx, eps in enumerate(eps_values):
                eps_key = str(eps)
                reference = reference_states[eps_key]
                shock_rng = np.random.RandomState(
                    seed + 200003 + eps_idx * 1009)
                z_values = {node: float(shock_rng.normal()) for node in nodes}
                for sigma in sigma_values:
                    sigma_key = "{:.3f}".format(sigma)
                    branch, perturb_stats = self._apply_standardized_perturbation(
                        reference, z_values, sigma, return_stats=True)
                    branch_rng = np.random.RandomState()
                    branch_rng.set_state(reference_rng_states[eps_key])
                    engine = OpinionDynamics(
                        G, branch, weights, triangles_list, node_triangles,
                        clustering, TrustThreshold(eps, TRUST['alpha']),
                        DYNAMICS['mu'], DYNAMICS['beta_default'], 'full_model')

                    trajectory = {
                        'opinion_sd': [], 'opinion_range': [],
                        'order_parameter': [],
                    }
                    initial = metric_values(branch)
                    for metric, value in initial.items():
                        trajectory[metric].append(value)
                        raw[eps_key][sigma_key]['initial'][metric].append(value)

                    measurement_index = 0
                    for step_idx in range(1, scan_num_steps + 1):
                        engine.step(branch_rng)
                        if step_idx != measurement_steps[measurement_index]:
                            continue
                        values = metric_values(engine.get_opinions())
                        for metric, value in values.items():
                            trajectory[metric].append(value)
                        measurement_index += 1
                        if measurement_index >= len(measurement_steps):
                            break

                    for metric in trajectory:
                        raw[eps_key][sigma_key]['trajectory'][metric].append(
                            trajectory[metric])
                        raw[eps_key][sigma_key]['endpoint'][metric].append(
                            trajectory[metric][-1])

                    consecutive = 0
                    recovery_time = None
                    for point_idx, order_value in enumerate(
                            trajectory['order_parameter']):
                        if order_value >= recovery_order_threshold:
                            consecutive += 1
                            if (consecutive >= recovery_persistence
                                    and recovery_time is None):
                                recovery_time = time_sweeps[
                                    point_idx - recovery_persistence + 1]
                        else:
                            consecutive = 0
                    raw[eps_key][sigma_key]['recovery_time_sweeps'].append(
                        recovery_time)
                    raw[eps_key][sigma_key]['recovered'].append(
                        recovery_time is not None)
                    raw[eps_key][sigma_key][
                        'perturbation_clipped_fraction'].append(
                            perturb_stats['clipped_fraction'])
                    raw[eps_key][sigma_key]['perturbation_n_clipped'].append(
                        perturb_stats['n_clipped'])

            print("  p={:.2f}: recovery reference run {}/{}".format(
                p, run_i + 1, num_runs), flush=True)

        p_result = {
            'p_hk': float(p),
            'num_runs': int(num_runs),
            'seeds': [run_i * 997 + 7 for run_i in range(int(num_runs))],
            'time_sweeps': time_sweeps,
            'reference_input_by_epsilon': {},
            'reference_state_by_epsilon': {},
            'by_epsilon': {},
        }
        for eps in eps_values:
            eps_key = str(eps)
            p_result['reference_input_by_epsilon'][eps_key] = {
                metric: summarize(values)
                for metric, values in reference_input_raw[eps_key].items()
            }
            p_result['reference_state_by_epsilon'][eps_key] = {
                metric: summarize(values)
                for metric, values in reference_raw[eps_key].items()
            }
            p_result['by_epsilon'][eps_key] = {}
            for sigma in sigma_values:
                sigma_key = "{:.3f}".format(sigma)
                item = raw[eps_key][sigma_key]
                recovered = [bool(value) for value in item['recovered']]
                recovered_times = [
                    float(value) for value in item['recovery_time_sweeps']
                    if value is not None
                ]
                restricted_times = [
                    float(value) if value is not None else float(scan_sweeps)
                    for value in item['recovery_time_sweeps']
                ]
                p_result['by_epsilon'][eps_key][sigma_key] = {
                    'sigma': float(sigma),
                    'trajectory': {
                        metric: summarize_aligned_series(series)
                        for metric, series in item['trajectory'].items()
                    },
                    'initial': {
                        metric: summarize(values)
                        for metric, values in item['initial'].items()
                    },
                    'endpoint': {
                        metric: summarize(values)
                        for metric, values in item['endpoint'].items()
                    },
                    'recovery': {
                        'order_parameter_threshold': float(
                            recovery_order_threshold),
                        'persistence_measurements': int(
                            recovery_persistence),
                        'probability': summarize_proportion(recovered),
                        'restricted_mean_time_sweeps': summarize(
                            restricted_times),
                        'time_sweeps_recovered_only': summarize(recovered_times)
                        if recovered_times else None,
                        'n_recovered': int(sum(recovered)),
                        'n_censored': int(len(recovered) - sum(recovered)),
                        'time_by_run': item['recovery_time_sweeps'],
                        'recovered_by_run': recovered,
                    },
                    'perturbation': {
                        'clipped_fraction': summarize(
                            item['perturbation_clipped_fraction']),
                        'n_clipped': summarize(
                            item['perturbation_n_clipped']),
                        'clipping_rule': '[0, 1]',
                    },
                }
        return p_result

    @staticmethod
    def _validate_threshold_recovery_checkpoint_block(
            block, p, seeds, eps_values, sigma_values, time_sweeps,
            recovery_persistence):
        """Reject an incomplete or inconsistent whole-p recovery checkpoint."""
        if not isinstance(block, dict) or block.get('p_hk') != p \
                or block.get('num_runs') != len(seeds) \
                or block.get('seeds') != seeds \
                or block.get('time_sweeps') != time_sweeps:
            return False
        eps_keys = set(str(value) for value in eps_values)
        if (set(block.get('by_epsilon', {})) != eps_keys
                or set(block.get('reference_input_by_epsilon', {})) != eps_keys
                or set(block.get('reference_state_by_epsilon', {})) != eps_keys):
            return False
        sigma_keys = set('{:.3f}'.format(value) for value in sigma_values)
        n = len(seeds)
        for eps_key in eps_keys:
            for reference_name in ('reference_input_by_epsilon',
                                   'reference_state_by_epsilon'):
                reference = block[reference_name][eps_key]
                if any(reference.get(metric, {}).get('n') != n
                       or len(reference.get(metric, {}).get('values', [])) != n
                       for metric in ('opinion_sd', 'opinion_range',
                                      'order_parameter')):
                    return False
            cells = block['by_epsilon'][eps_key]
            if not isinstance(cells, dict) or set(cells) != sigma_keys:
                return False
            for sigma_key in sigma_keys:
                cell = cells[sigma_key]
                recovery = cell.get('recovery', {})
                times = recovery.get('time_by_run', [])
                events = recovery.get('recovered_by_run', [])
                if (cell.get('sigma') != float(sigma_key)
                        or not isinstance(times, list) or len(times) != n
                        or not isinstance(events, list) or len(events) != n
                        or any(type(event) is not bool for event in events)
                        or any((time_value is not None) != event
                               for time_value, event in zip(times, events))
                        or any(time_value not in time_sweeps
                               for time_value in times if time_value is not None)
                        or any(time_sweeps.index(time_value)
                               > len(time_sweeps) - recovery_persistence
                               for time_value in times if time_value is not None)
                        or recovery.get('n_recovered') != sum(events)
                        or recovery.get('n_censored') != n - sum(events)
                        or recovery.get('probability', {}).get('n') != n
                        or recovery.get('probability', {}).get('values')
                        != [float(event) for event in events]):
                    return False
                if any(cell.get(section, {}).get(metric, {}).get('n') != n
                       or len(cell.get(section, {}).get(metric, {}).get(
                           'values', [])) != n
                       for section in ('initial', 'endpoint')
                       for metric in ('opinion_sd', 'opinion_range',
                                      'order_parameter')):
                    return False
                if any(len(cell.get('trajectory', {}).get(metric, {}).get(
                        field, [])) != len(time_sweeps)
                       for metric in ('opinion_sd', 'opinion_range',
                                      'order_parameter')
                       for field in ('mean', 'ci95_low', 'ci95_high')):
                    return False
                if (cell.get('perturbation', {}).get('clipped_fraction', {}).get(
                        'n') != n or cell.get('perturbation', {}).get(
                            'n_clipped', {}).get('n') != n):
                    return False
        return True

    def experiment_threshold_recovery(
            self, num_runs=None, p_values=None, sigma_values=None,
            eps_values=None, recovery_order_threshold=1.0,
            recovery_persistence=3, resume=True,
            output_filename='exp_threshold_recovery.json', workers=1):
        """Apply an independent perturbation at every epsilon and measure recovery."""
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if p_values is None:
            p_values = [0.0, 0.4, 0.8]
        if sigma_values is None:
            sigma_values = SIMULATION.get(
                'perturb_sigma_values', [0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
        if eps_values is None:
            eps_values = TRUST.get('epsilon_0_range') or JOINT_SWEEP.get(
                'eps_values')
        p_values = [float(value) for value in p_values]
        sigma_values = [float(value) for value in sigma_values]
        eps_values = [float(value) for value in eps_values]
        worker_count = max(1, int(workers))
        if not p_values or not sigma_values or not eps_values:
            raise ValueError("p_values, sigma_values, and eps_values must be non-empty")
        scan_num_steps, scan_sweeps = self._scan_step_config()
        control = {
            'schema_version': 1,
            'experiment': 'independent_perturbation_recovery_at_each_epsilon',
            'N': int(NETWORK['N']),
            'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'p_values': p_values,
            'sigma_values': sigma_values,
            'eps_values': eps_values,
            'num_steps_per_epsilon': int(scan_num_steps),
            'sweeps_per_epsilon': float(scan_sweeps),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'recovery_order_parameter_threshold': float(
                recovery_order_threshold),
            'recovery_persistence_measurements': int(
                recovery_persistence),
            'recovery_definition': (
                'first recorded sweep with order_parameter >= threshold '
                'for the configured number of consecutive measurements'),
            'censoring': 'right-censored at sweeps_per_epsilon',
            'reference_state': (
                'state entering each epsilon on one unperturbed high-to-low '
                'continuation scan; clean path then evolves for one interval'),
            'perturbation': (
                'independent threshold-level shock; matched standardized '
                'Gaussian direction across sigma conditions; clipped to [0,1]'),
            'paired_on': (
                'network seed, baseline reference path, epsilon-specific '
                'shock direction, and epsilon-specific dynamics RNG stream'),
            'core_parameters': self._core_parameter_signature(),
            'execution_source_signature': execution_source_signature(),
            'workers': int(worker_count),
        }
        output_path = os.path.join(DATA_DIR, output_filename)
        metadata_path = os.path.splitext(output_path)[0] + '.meta.json'
        if (os.path.lexists(metadata_path)
                and (os.path.islink(metadata_path)
                     or not os.path.isfile(output_path))):
            raise ValueError('orphaned or linked recovery metadata; preserved')
        results = {'control': control, 'data': {}}
        if os.path.lexists(output_path):
            if os.path.islink(output_path):
                raise ValueError('recovery checkpoint target is a symlink')
            if not resume:
                raise FileExistsError(
                    'recovery checkpoint already exists; use a new filename')
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
            except Exception as exc:
                raise RuntimeError(
                    'unable to read recovery checkpoint; file preserved: {}'
                    .format(exc))
            if not isinstance(loaded, dict) or loaded.get('control') != control:
                raise ValueError(
                    'recovery checkpoint control/source mismatch; file preserved')
            results['data'] = loaded.get('data', {})
            print("  [续跑] 已读取 {} 个 p 块".format(len(results['data'])))

        N = int(NETWORK['N'])
        steps, _ = self._scan_step_config()
        record_interval = max(1, int(round(
            N * float(SIMULATION.get('record_sweeps', 5)))))
        measurement_steps = list(range(record_interval, steps + 1,
                                       record_interval))
        if not measurement_steps or measurement_steps[-1] != steps:
            measurement_steps.append(steps)
        expected_times = [0.0] + [float(step) / N
                                  for step in measurement_steps]
        seeds = [run_i * 997 + 7 for run_i in range(int(num_runs))]
        for p_key, block in results['data'].items():
            if p_key not in set('{:.2f}'.format(value) for value in p_values):
                raise ValueError('unexpected p block in recovery checkpoint: '
                                 + p_key)
            if not self._validate_threshold_recovery_checkpoint_block(
                    block, float(p_key), seeds, eps_values,
                    sigma_values, expected_times, recovery_persistence):
                raise ValueError(
                    'incomplete recovery checkpoint p block {}; file preserved'
                    .format(p_key))

        expected_p_keys = ["{:.2f}".format(value) for value in p_values]
        pending = []
        for p in p_values:
            p_key = "{:.2f}".format(p)
            if resume and p_key in results['data']:
                print("  [续跑] p={} 已完成".format(p_key))
                continue
            pending.append(p)

        def complete_block(p, p_result):
            p_key = "{:.2f}".format(p)
            print("  [逐阈值恢复] p={}, eps={}, sigma={}, runs={}".format(
                p_key, len(eps_values), len(sigma_values), num_runs),
                flush=True)
            results['data'][p_key] = p_result
            results['data'] = self._ordered_numeric_mapping(results['data'])
            self.save_json(
                results, output_filename,
                metadata={
                    'n_runs': num_runs,
                    'experiment_design': control['experiment'],
                    'sweeps_per_epsilon': scan_sweeps,
                })

        if worker_count == 1 or len(pending) <= 1:
            for p in pending:
                complete_block(
                    p, self._run_threshold_recovery_for_p(
                        p, num_runs, sigma_values, eps_values,
                        recovery_order_threshold=recovery_order_threshold,
                        recovery_persistence=recovery_persistence))
        else:
            with ProcessPoolExecutor(
                    max_workers=min(worker_count, len(pending)),
                    **_process_pool_kwargs(_capture_runtime_config())) as pool:
                futures = {
                    pool.submit(
                        self._run_threshold_recovery_for_p,
                        p, num_runs, sigma_values, eps_values,
                        recovery_order_threshold, recovery_persistence): p
                    for p in pending
                }
                for future in as_completed(futures):
                    complete_block(futures[future], future.result())
        results['data'] = {
            key: results['data'][key]
            for key in expected_p_keys if key in results['data']
        }
        self.save_json(
            results, output_filename,
            metadata={
                'n_runs': num_runs,
                'experiment_design': control['experiment'],
                'sweeps_per_epsilon': scan_sweeps,
            })
        return results

    # ================================================================
    # 扩展实验: 扰动强度梯度与逆向稳定性
    # ================================================================
    def experiment_perturbation_sweep(
            self, num_runs=None, p_values=None, sigma_values=None,
            eps_values=None,
            perturb_each_step=False, resume=True,
            output_filename='exp_formal_perturbation_sweep_revision_v2.json',
            workers=1):
        """扫描一次性高斯扰动强度，检验逆向分支的稳定边界。

        每个 ``(sigma, p_HK)`` 组合均保存完整前/后向曲线、逐重复磁滞
        面积及95%置信区间。默认只在逆向扫描起点扰动一次，与论文理论
        部分的有限冲击设定一致；重复逐点扰动可显式传入
        ``perturb_each_step=True`` 作为另一种实验设计。
        """
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if p_values is None:
            p_values = [0.0, 0.4, 0.8]
        if sigma_values is None:
            sigma_values = SIMULATION.get(
                'perturb_sigma_values', [0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
        p_values = [float(p) for p in p_values]
        sigma_values = [float(sigma) for sigma in sigma_values]
        worker_count = max(1, int(workers))

        if eps_values is None:
            eps_values = TRUST.get('epsilon_0_range') or JOINT_SWEEP.get(
                'eps_values')
        eps_values = [float(eps) for eps in eps_values]
        eps_reverse = list(reversed(eps_values))
        scan_num_steps, scan_sweeps = self._scan_step_config()
        output_path = os.path.join(DATA_DIR, output_filename)

        results = {
            'control': {
                'schema_version': 2,
                'N': int(NETWORK['N']),
                'm': int(NETWORK['m']),
                'num_runs': int(num_runs),
                'p_values': p_values,
                'sigma_values': sigma_values,
                'eps_values': eps_values,
                'num_steps_per_epsilon': int(scan_num_steps),
                'sweeps_per_epsilon': float(scan_sweeps),
                'tail_diagnostic_sweeps': float(
                    min(scan_sweeps, SIMULATION.get('record_sweeps', 5))),
                'perturb_each_step': bool(perturb_each_step),
                'clipping': '[0, 1]',
                'core_parameters': self._core_parameter_signature(),
                'workers': int(worker_count),
            },
            'data': {},
        }
        self._check_checkpoint_target(output_path, 'perturbation', resume)
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                old_control = loaded.get('control', {})
                control_matches = old_control == results['control']
                if control_matches:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个 sigma 块")
                else:
                    raise FileExistsError(
                        f"perturbation checkpoint control/source mismatch: "
                        f"{output_path}; original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"perturbation checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        print("\n" + "=" * 70)
        print("扩展实验: 扰动强度梯度与逆向稳定性")
        print("=" * 70)
        pending = []
        for p in p_values:
            p_complete = all(
                isinstance(results['data'].get(f"{sigma:.3f}", {}).get(f"{p:.2f}"), dict)
                and results['data'][f"{sigma:.3f}"][f"{p:.2f}"].get('shared_forward_scan')
                and results['data'][f"{sigma:.3f}"][f"{p:.2f}"].get('hysteresis_area_stats') is not None
                for sigma in sigma_values
            )
            if resume and p_complete:
                print(f"  [续跑] p={p:.2f} 的全部 sigma 分支已完成")
                continue
            pending.append(p)

        def complete_block(p, p_blocks):
            print(f"\n  p={p:.2f}: shared forward scan, {len(sigma_values)} sigma branches")
            for sigma in sigma_values:
                sigma_key = f"{sigma:.3f}"
                results['data'].setdefault(sigma_key, {})[f"{p:.2f}"] = p_blocks[sigma_key]
                results['data'][sigma_key] = self._ordered_numeric_mapping(
                    results['data'][sigma_key])
            results['data'] = self._ordered_numeric_mapping(results['data'])
            self.save_json(
                results, output_filename,
                metadata={
                    'n_runs': num_runs,
                    'experiment_design': 'perturbation_strength_gradient_shared_forward',
                    'perturb_each_step': bool(perturb_each_step),
                    'sweep_sweeps': scan_sweeps,
                },
            )

        if worker_count == 1 or len(pending) <= 1:
            for p in pending:
                complete_block(
                    p, self._run_perturbation_sweep_for_p_shared(
                        p, num_runs, sigma_values, eps_values, eps_reverse,
                        perturb_each_step=perturb_each_step))
        else:
            with ProcessPoolExecutor(
                    max_workers=min(worker_count, len(pending)),
                    **_process_pool_kwargs(_capture_runtime_config())) as pool:
                futures = {
                    pool.submit(
                        self._run_perturbation_sweep_for_p_shared,
                        p, num_runs, sigma_values, eps_values, eps_reverse,
                        perturb_each_step): p
                    for p in pending
                }
                for future in as_completed(futures):
                    complete_block(futures[future], future.result())

        self.save_json(
            results, output_filename,
            metadata={
                'n_runs': num_runs,
                'experiment_design': 'perturbation_strength_gradient_shared_forward',
                'perturb_each_step': bool(perturb_each_step),
                'sweep_sweeps': scan_sweeps,
            },
        )
        print("\n  [扰动强度扫描完成]")
        return results

    # ================================================================
    # 扩展实验: 异步与同步更新对照
    # ================================================================
    def experiment_update_mode_control(
            self, num_runs=None, p_values=None, model_type='full_model',
            resume=True, output_filename='exp_update_mode_control.json'):
        """Compare asynchronous and frozen-snapshot synchronous updating."""
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if p_values is None:
            p_values = [0.0, 0.4, 0.8, 1.0]
        p_values = [float(p) for p in p_values]
        model_type = resolve_model_key(model_type)
        modes = ('asynchronous', 'synchronous')
        output_path = os.path.join(DATA_DIR, output_filename)
        results = {
            'control': {
                'N': int(NETWORK['N']),
                'm': int(NETWORK['m']),
                'num_runs': int(num_runs),
                'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
                'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
                'p_values': p_values,
                'model_type': model_type,
                'synchronous_definition': (
                    'each node proposes once per sweep from the same frozen '
                    'start-of-sweep opinion snapshot; proposals commit together'
                ),
                'paired_on': 'network seed and initial-opinion seed',
                'core_parameters': self._core_parameter_signature(),
            },
            'data': {},
        }
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                old = loaded.get('control', {})
                same = (
                    int(old.get('N', -1)) == int(NETWORK['N'])
                    and int(old.get('m', -1)) == int(NETWORK['m'])
                    and int(old.get('num_runs', -1)) == int(num_runs)
                    and float(old.get('main_sweeps', -1))
                    == float(SIMULATION.get('main_sweeps', 200))
                    and float(old.get('record_sweeps', -1))
                    == float(SIMULATION.get('record_sweeps', 5))
                    and list(old.get('p_values', [])) == p_values
                    and old.get('model_type') == model_type
                    and old.get('core_parameters')
                    == self._core_parameter_signature()
                )
                if same:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个 p 块")
                else:
                    raise FileExistsError(
                        f"update-mode checkpoint control/source mismatch: "
                        f"{output_path}; original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"update-mode checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        metric_paths = {
            'order_param': ['final', 'order_param'],
            'n_clusters': ['final', 'n_clusters'],
            'variance': ['final', 'variance'],
            'triangle_alignment_fixed': [
                'final', 'triangle_alignment_fixed'],
            'triangle_admissibility_adaptive': [
                'final', 'triangle_admissibility_adaptive'],
            'edge_disagreement_penalty': [
                'final', 'edge_disagreement_penalty'],
        }

        print("\n" + "=" * 70)
        print("扩展实验: 异步与同步更新对照")
        print("=" * 70)
        for p in p_values:
            p_key = f"{p:.2f}"
            existing = results['data'].get(p_key, {})
            if (resume and existing.get('_control', {}).get('complete')
                    and int(existing['_control'].get('num_runs', -1))
                    == int(num_runs)):
                print(f"  [续跑] p={p:.2f} 已完成")
                continue

            p_block = {}
            batches = {}
            for mode in modes:
                print(f"\n  p={p:.2f}, update_mode={mode}")
                batch = self._run_batch(
                    {'p_hk': p, 'model_type': model_type,
                     'update_mode': mode},
                    num_runs=num_runs,
                    verbose_label=f"p={p:.2f}, {mode}",
                )
                batches[mode] = batch
                mode_block = {
                    metric: self._aggregate(batch, path)
                    for metric, path in metric_paths.items()
                }
                mode_block['time_series_summary'] = (
                    self._aggregate_time_series(batch))
                p_block[mode] = mode_block

            p_block['paired_synchronous_minus_asynchronous'] = {
                metric: paired_difference(
                    p_block['synchronous'][metric]['values'],
                    p_block['asynchronous'][metric]['values'],
                )
                for metric in metric_paths
            }
            p_block['_control'] = {
                'complete': True,
                'num_runs': int(num_runs),
                'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
            }
            results['data'][p_key] = p_block
            self.save_json(
                results, output_filename,
                metadata={
                    'n_runs': num_runs,
                    'experiment_design': 'update_mode_control',
                    'main_sweeps': SIMULATION.get('main_sweeps'),
                },
            )

        self.save_json(
            results, output_filename,
            metadata={
                'n_runs': num_runs,
                'experiment_design': 'update_mode_control',
                'main_sweeps': SIMULATION.get('main_sweeps'),
            },
        )
        print("\n  [更新模式对照完成]")
        return results

    # ================================================================
    # 扩展实验: 双社区对抗初态（SBM）
    # ================================================================
    def experiment_sbm_community_conflict(
            self, num_runs=None, mixing_values=None, target_degree=8.0,
            community_centers=(0.25, 0.75), init_sd=0.05,
            model_order=None, resume=True,
            output_filename='exp_sbm_community_conflict.json'):
        """Test structural-threshold claims under explicit community conflict."""
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if mixing_values is None:
            mixing_values = [0.02, 0.05, 0.10, 0.20, 0.35]
        mixing_values = [float(value) for value in mixing_values]
        if model_order is None:
            model_order = [
                'threshold_only', 'no_triadic_feedback', 'full_model']
        model_order = [resolve_model_key(model) for model in model_order]
        N = int(NETWORK['N'])
        main_sweeps = float(SIMULATION.get('main_sweeps', 200))
        record_sweeps = float(SIMULATION.get('record_sweeps', 5))
        num_steps = int(round(N * main_sweeps))
        record_interval = max(1, int(round(N * record_sweeps)))
        output_path = os.path.join(DATA_DIR, output_filename)

        results = {
            'control': {
                'N': N,
                'num_runs': int(num_runs),
                'main_sweeps': main_sweeps,
                'record_sweeps': record_sweeps,
                'target_degree': float(target_degree),
                'mixing_values': mixing_values,
                'community_centers': [float(x) for x in community_centers],
                'initial_sd': float(init_sd),
                'model_order': model_order,
                'network_type': 'balanced_two_block_sbm',
                'paired_on': 'network, community initial opinions, and seed',
                'core_parameters': self._core_parameter_signature(),
            },
            'data': {},
        }
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                old = loaded.get('control', {})
                same = (
                    int(old.get('N', -1)) == N
                    and int(old.get('num_runs', -1)) == int(num_runs)
                    and float(old.get('main_sweeps', -1)) == main_sweeps
                    and float(old.get('record_sweeps', -1)) == record_sweeps
                    and float(old.get('target_degree', -1))
                    == float(target_degree)
                    and list(old.get('mixing_values', [])) == mixing_values
                    and list(old.get('community_centers', []))
                    == [float(x) for x in community_centers]
                    and float(old.get('initial_sd', -1)) == float(init_sd)
                    and list(old.get('model_order', [])) == model_order
                    and old.get('core_parameters')
                    == self._core_parameter_signature()
                )
                if same:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个 mixing 块")
                else:
                    raise FileExistsError(
                        f"SBM checkpoint control/source mismatch: {output_path}; "
                        "original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"SBM checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        metric_paths = {
            'order_param': ['final', 'order_param'],
            'n_clusters': ['final', 'n_clusters'],
            'variance': ['final', 'variance'],
            'triangle_alignment_fixed': [
                'final', 'triangle_alignment_fixed'],
            'triangle_admissibility_adaptive': [
                'final', 'triangle_admissibility_adaptive'],
            'edge_disagreement_penalty': [
                'final', 'edge_disagreement_penalty'],
            'community_mean_gap': ['community', 'mean_gap'],
            'mean_within_community_variance': [
                'community', 'mean_within_variance'],
            'cross_edge_fraction': ['network', 'cross_edge_fraction'],
            'actual_average_degree': ['network', 'actual_average_degree'],
        }

        print("\n" + "=" * 70)
        print("扩展实验: 双社区SBM与对抗初态")
        print("=" * 70)
        for mixing in mixing_values:
            mixing_key = f"{mixing:.3f}"
            existing = results['data'].get(mixing_key, {})
            if (resume and existing.get('_control', {}).get('complete')
                    and int(existing['_control'].get('num_runs', -1))
                    == int(num_runs)):
                print(f"  [续跑] mixing={mixing:.3f} 已完成")
                continue

            mixing_block = {}
            for model_key in model_order:
                print(f"\n  mixing={mixing:.3f}, model={model_key}")
                batch = []
                for run_i in range(num_runs):
                    seed = run_i * 997 + 7
                    G, membership, sbm_params = (
                        NetworkGenerator.generate_two_block_sbm(
                            N, target_degree=target_degree,
                            mixing=mixing, seed=seed))
                    init_rng = np.random.RandomState(seed + 104729)
                    initial_opinions = {
                        node: float(np.clip(init_rng.normal(
                            community_centers[membership[node]], init_sd),
                            0.0, 1.0))
                        for node in G.nodes()
                    }
                    sim = SingleRun(
                        N=N, m=NETWORK['m'], p_hk=0.0,
                        epsilon_0=TRUST['epsilon_0_default'],
                        alpha=TRUST['alpha'], mu=DYNAMICS['mu'],
                        beta=DYNAMICS['beta_default'],
                        model_type=model_key, num_steps=num_steps,
                        seed=seed, update_mode='asynchronous', graph=G,
                        initial_opinions=initial_opinions,
                        network_type='two_block_sbm',
                        network_params=sbm_params,
                    )
                    result = sim.execute(record_interval=record_interval)
                    node_order = list(G.nodes())
                    final_by_node = dict(zip(
                        node_order, result['final']['opinions']))
                    community_values = [
                        np.asarray([
                            final_by_node[node] for node in node_order
                            if membership[node] == block
                        ], dtype=float)
                        for block in (0, 1)
                    ]
                    community_means = [
                        float(np.mean(values)) for values in community_values]
                    within_variances = [
                        float(np.var(values)) for values in community_values]
                    cross_edges = sum(
                        membership[u] != membership[v] for u, v in G.edges())
                    result['community'] = {
                        'means': community_means,
                        'mean_gap': float(abs(
                            community_means[0] - community_means[1])),
                        'within_variances': within_variances,
                        'mean_within_variance': float(np.mean(
                            within_variances)),
                    }
                    result['network']['cross_edge_fraction'] = float(
                        cross_edges / max(G.number_of_edges(), 1))
                    result['network']['actual_average_degree'] = float(
                        2.0 * G.number_of_edges() / max(N, 1))
                    batch.append(result)
                    print(f"    run {run_i+1}/{num_runs}: "
                          f"R={result['final']['order_param']:.4f}, "
                          f"gap={result['community']['mean_gap']:.4f}")

                model_block = {
                    metric: self._aggregate(batch, path)
                    for metric, path in metric_paths.items()
                }
                model_block['time_series_summary'] = (
                    self._aggregate_time_series(batch))
                mixing_block[model_key] = model_block

            paired = {}
            for comparator in model_order:
                if comparator == 'full_model' or 'full_model' not in mixing_block:
                    continue
                paired[f'full_model_minus_{comparator}'] = {
                    metric: paired_difference(
                        mixing_block['full_model'][metric]['values'],
                        mixing_block[comparator][metric]['values'],
                    )
                    for metric in metric_paths
                }
            mixing_block['_paired_comparisons'] = paired
            mixing_block['_control'] = {
                'complete': True,
                'num_runs': int(num_runs),
                'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
            }
            results['data'][mixing_key] = mixing_block
            self.save_json(
                results, output_filename,
                metadata={
                    'n_runs': num_runs,
                    'experiment_design': 'two_block_sbm_conflict',
                    'main_sweeps': main_sweeps,
                },
            )

        self.save_json(
            results, output_filename,
            metadata={
                'n_runs': num_runs,
                'experiment_design': 'two_block_sbm_conflict',
                'main_sweeps': main_sweeps,
            },
        )
        print("\n  [SBM社区冲突实验完成]")
        return results

    # ================================================================
    # 扩展实验: 双社区对抗初态下的正/逆向磁滞扫描（R4.2）
    # ================================================================
    def _run_sbm_hysteresis_for_mixing(
            self, mixing, num_runs, eps_values, eps_reverse, sigma_values,
            target_degree, community_centers, init_sd, model_type,
            perturb_each_step, scan_num_steps, sweep_sweeps):
        """One mixing value: shared forward scan, then a backward branch per
        sigma from the identical forward terminal state.

        Independent unit = seed-level replicate. The forward scan inherits the
        terminal state across increasing epsilon on a two-block SBM with
        adversarial community opinions; the backward scan restarts from the
        forward terminal state and injects a standardized Gaussian perturbation
        (clipped to [0,1]) at the backward start.
        """
        N = int(NETWORK['N'])
        forward_vals = {str(eps): [] for eps in eps_values}
        forward_cluster_vals = {str(eps): [] for eps in eps_values}
        forward_gap_vals = {str(eps): [] for eps in eps_values}
        backward_vals = {
            f"{sigma:.3f}": {str(eps): [] for eps in eps_reverse}
            for sigma in sigma_values
        }
        backward_cluster_vals = {
            f"{sigma:.3f}": {str(eps): [] for eps in eps_reverse}
            for sigma in sigma_values
        }
        backward_gap_vals = {
            f"{sigma:.3f}": {str(eps): [] for eps in eps_reverse}
            for sigma in sigma_values
        }
        perturbation_diag = {f"{sigma:.3f}": [] for sigma in sigma_values}
        cross_edge_fractions = []

        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            G, membership, sbm_params = NetworkGenerator.generate_two_block_sbm(
                N, target_degree=target_degree, mixing=mixing, seed=seed)
            triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
            weights = NetworkGenerator.compute_jaccard_weights(G)
            clustering = NetworkGenerator.compute_clustering(G)
            init_rng = np.random.RandomState(seed + 104729)
            opinions = {
                node: float(np.clip(init_rng.normal(
                    community_centers[membership[node]], init_sd), 0.0, 1.0))
                for node in G.nodes()
            }
            cross_edges = sum(
                membership[u] != membership[v] for u, v in G.edges())
            cross_edge_fractions.append(
                float(cross_edges / max(G.number_of_edges(), 1)))

            rng = np.random.RandomState(seed)

            def _evolve(eps, start_opinions, engine_rng):
                engine = OpinionDynamics(
                    G, start_opinions, weights, triangles_list,
                    node_triangles, clustering,
                    TrustThreshold(eps, TRUST['alpha']),
                    DYNAMICS['mu'], DYNAMICS['beta_default'], model_type)
                for _ in range(scan_num_steps):
                    engine.step(engine_rng)
                arr = engine.get_opinions_array()
                block_means = [
                    float(np.mean([arr[node] for node in G.nodes()
                                   if membership[node] == block]))
                    for block in (0, 1)
                ]
                gap = abs(block_means[0] - block_means[1])
                return (float(Metrics.order_parameter(arr)),
                        float(Metrics.count_clusters(arr)), gap,
                        engine.get_opinions())

            # Forward scan with state inheritance across increasing epsilon
            for eps in eps_values:
                op, nc, gap, opinions = _evolve(eps, opinions, rng)
                forward_vals[str(eps)].append(op)
                forward_cluster_vals[str(eps)].append(nc)
                forward_gap_vals[str(eps)].append(gap)

            terminal_opinions = dict(opinions)
            forward_rng_state = rng.get_state()
            # Distinct shock stream so sigma branches share one common
            # standardized direction without reusing the initial-opinion draws.
            shock_rng = np.random.RandomState(seed + 2 * 104729)
            n_shocks = len(eps_reverse) if perturb_each_step else 1
            shocks = [
                {node: float(shock_rng.normal()) for node in G.nodes()}
                for _ in range(n_shocks)
            ]

            for sigma in sigma_values:
                sigma_key = f"{sigma:.3f}"
                branch_rng = np.random.RandomState()
                branch_rng.set_state(forward_rng_state)
                branch_opinions = dict(terminal_opinions)
                n_clipped = 0
                n_perturbed = 0
                for eps_idx, eps in enumerate(eps_reverse):
                    should_perturb = sigma > 0.0 and (
                        perturb_each_step or eps_idx == 0)
                    if should_perturb:
                        shock_idx = eps_idx if perturb_each_step else 0
                        branch_opinions, stats = (
                            self._apply_standardized_perturbation(
                                branch_opinions, shocks[shock_idx], sigma,
                                return_stats=True))
                        n_clipped += stats['n_clipped']
                        n_perturbed += stats['n_perturbed']
                    op, nc, gap, branch_opinions = _evolve(
                        eps, branch_opinions, branch_rng)
                    backward_vals[sigma_key][str(eps)].append(op)
                    backward_cluster_vals[sigma_key][str(eps)].append(nc)
                    backward_gap_vals[sigma_key][str(eps)].append(gap)
                perturbation_diag[sigma_key].append({
                    'seed': int(seed),
                    'n_perturbed': int(n_perturbed),
                    'n_clipped': int(n_clipped),
                    'clipped_fraction': float(
                        n_clipped / n_perturbed if n_perturbed else 0.0),
                })

        low_eps_key = str(eps_values[0])
        block = {
            'N': N,
            'mixing': float(mixing),
            'num_runs': int(num_runs),
            'sweep_sweeps': float(sweep_sweeps),
            'eps_values': [float(e) for e in eps_values],
            'sigma_values': [float(s) for s in sigma_values],
            'perturb_each_step': bool(perturb_each_step),
            'model_type': model_type,
            'cross_edge_fraction': summarize(cross_edge_fractions),
            'forward': {
                'order_param': {
                    str(eps): self._aggregate_scalar_values(forward_vals[str(eps)])
                    for eps in eps_values
                },
                'n_clusters': {
                    str(eps): self._aggregate_scalar_values(
                        forward_cluster_vals[str(eps)])
                    for eps in eps_values
                },
                'community_gap': {
                    str(eps): self._aggregate_scalar_values(
                        forward_gap_vals[str(eps)])
                    for eps in eps_values
                },
            },
            'backward': {},
            'perturbation_diagnostics': {},
        }
        for sigma in sigma_values:
            sigma_key = f"{sigma:.3f}"
            run_hysteresis = []
            for run_i in range(num_runs):
                f_run = [forward_vals[str(e)][run_i] for e in eps_values]
                b_run = [backward_vals[sigma_key][str(e)][run_i]
                         for e in eps_values]
                run_hysteresis.append(
                    Metrics.hysteresis_area(f_run, b_run, eps_values))
            block['backward'][sigma_key] = {
                'order_param': {
                    str(eps): self._aggregate_scalar_values(
                        backward_vals[sigma_key][str(eps)])
                    for eps in eps_reverse
                },
                'n_clusters': {
                    str(eps): self._aggregate_scalar_values(
                        backward_cluster_vals[sigma_key][str(eps)])
                    for eps in eps_reverse
                },
                'community_gap': {
                    str(eps): self._aggregate_scalar_values(
                        backward_gap_vals[sigma_key][str(eps)])
                    for eps in eps_reverse
                },
                'hysteresis_area_stats': summarize(run_hysteresis),
                'backward_low_eps_order_param': float(summarize(
                    backward_vals[sigma_key][low_eps_key])['mean']),
                'backward_low_eps_clusters': float(summarize(
                    backward_cluster_vals[sigma_key][low_eps_key])['mean']),
            }
            block['perturbation_diagnostics'][sigma_key] = {
                'by_run': perturbation_diag[sigma_key],
                'clipped_fraction': summarize([
                    row['clipped_fraction'] for row in perturbation_diag[sigma_key]
                ]),
            }
        return block

    def experiment_sbm_hysteresis(
            self, num_runs=None, mixing_values=None, sigma_values=None,
            eps_values=None, perturb_each_step=False, target_degree=8.0,
            community_centers=(0.25, 0.75), init_sd=0.05,
            model_type='full_model', resume=True,
            output_filename='exp_sbm_hysteresis_revision_v2.json',
            workers=1):
        """双社区对抗初态的正/逆向磁滞扫描（R4.2）。

        Forward scan inherits terminal state across increasing epsilon on a
        two-block SBM with adversarial community opinions; backward scan
        restarts from the forward terminal state and injects Gaussian
        perturbation (clipped to [0,1]) at the backward start. Set
        ``perturb_each_step=True`` to inject a fresh, matched standardized
        perturbation before every reverse epsilon point. This answers the
        reviewer request for a multi-community hysteresis path under strong
        perturbation, complementing the terminal-state mixing experiment.
        """
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if mixing_values is None:
            mixing_values = [0.02, 0.05, 0.10, 0.20, 0.35]
        mixing_values = [float(value) for value in mixing_values]
        if sigma_values is None:
            sigma_values = [0.0, 0.05, 0.10, 0.20]
        sigma_values = [float(value) for value in sigma_values]
        if eps_values is None:
            eps_values = (TRUST.get('epsilon_0_range')
                          or JOINT_SWEEP.get('eps_values'))
        if not eps_values:
            eps_values = [round(0.10 + x * 0.025, 3) for x in range(19)]
        eps_values = [float(eps) for eps in eps_values]
        eps_reverse = list(reversed(eps_values))
        model_type = resolve_model_key(model_type)
        worker_count = max(1, int(workers))
        scan_num_steps, sweep_sweeps = self._scan_step_config()
        N = int(NETWORK['N'])
        output_path = os.path.join(DATA_DIR, output_filename)
        self._check_checkpoint_target(output_path, 'SBM hysteresis', resume)

        results = {
            'control': {
                'N': N,
                'num_runs': int(num_runs),
                'sweeps_per_epsilon': float(sweep_sweeps),
                'target_degree': float(target_degree),
                'mixing_values': mixing_values,
                'sigma_values': sigma_values,
                'eps_values': eps_values,
                'perturb_each_step': bool(perturb_each_step),
                'community_centers': [float(x) for x in community_centers],
                'initial_sd': float(init_sd),
                'model_type': model_type,
                'network_type': 'balanced_two_block_sbm',
                'clipping': '[0, 1]',
                'state_inheritance': True,
                'backward_starts_from_forward': True,
                'core_parameters': self._core_parameter_signature(),
                'workers': int(worker_count),
            },
            'data': {},
        }
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if loaded.get('control', {}) == results['control']:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个 mixing 块")
                else:
                    raise FileExistsError(
                        f"SBM hysteresis checkpoint control/source mismatch: "
                        f"{output_path}; original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"SBM hysteresis checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        print("\n" + "=" * 70)
        print("扩展实验: 双社区对抗初态正/逆向磁滞扫描")
        print("=" * 70)
        pending = []
        for mixing in mixing_values:
            mixing_key = f"{mixing:.3f}"
            existing = results['data'].get(mixing_key, {})
            block_matches_perturbation = (
                bool(existing.get('perturb_each_step', False))
                == bool(perturb_each_step))
            if (resume and block_matches_perturbation
                    and existing.get('_control', {}).get('complete')):
                print(f"  [续跑] mixing={mixing:.3f} 已完成")
                continue
            pending.append(mixing)

        def complete_block(mixing, block):
            mixing_key = f"{mixing:.3f}"
            block['_control'] = {'complete': True, 'num_runs': int(num_runs)}
            results['data'][mixing_key] = block
            results['data'] = self._ordered_numeric_mapping(results['data'])
            self.save_json(
                results, output_filename,
                metadata={
                    'n_runs': num_runs,
                    'experiment_design': 'sbm_forward_backward_hysteresis',
                    'sweep_sweeps': sweep_sweeps,
                })

        if worker_count == 1 or len(pending) <= 1:
            for mixing in pending:
                block = self._run_sbm_hysteresis_for_mixing(
                    mixing, num_runs, eps_values, eps_reverse, sigma_values,
                    target_degree, community_centers, init_sd, model_type,
                    perturb_each_step, scan_num_steps, sweep_sweeps)
                complete_block(mixing, block)
        else:
            with ProcessPoolExecutor(
                    max_workers=min(worker_count, len(pending)),
                    **_process_pool_kwargs(_capture_runtime_config())) as pool:
                futures = {
                    pool.submit(
                        self._run_sbm_hysteresis_for_mixing,
                        mixing, num_runs, eps_values, eps_reverse, sigma_values,
                        target_degree, community_centers, init_sd, model_type,
                        perturb_each_step, scan_num_steps, sweep_sweeps): mixing
                    for mixing in pending
                }
                for future in as_completed(futures):
                    mixing = futures[future]
                    complete_block(mixing, future.result())

        self.save_json(
            results, output_filename,
            metadata={
                'n_runs': num_runs,
                'experiment_design': 'sbm_forward_backward_hysteresis',
                'sweep_sweeps': sweep_sweeps,
            })
        print("\n  [SBM正逆向磁滞扫描完成]")
        return results

    # ================================================================
    # 扩展实验: 二维意见鲁棒性
    # ================================================================
    def experiment_vector_opinions(
            self, num_runs=None, p_values=None, dimension=2,
            model_order=None, resume=True,
            output_filename='exp_vector_opinions.json'):
        """Run the core mechanisms with Euclidean vector-valued opinions."""
        if dimension < 2:
            raise ValueError("dimension must be at least 2")
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if p_values is None:
            p_values = [0.0, 0.4, 0.8, 1.0]
        p_values = [float(p) for p in p_values]
        if model_order is None:
            model_order = [
                'threshold_only', 'no_triadic_feedback', 'full_model']
        model_order = [resolve_model_key(model) for model in model_order]
        N = int(NETWORK['N'])
        m = int(NETWORK['m'])
        main_sweeps = float(SIMULATION.get('main_sweeps', 200))
        record_sweeps = float(SIMULATION.get('record_sweeps', 5))
        num_steps = int(round(N * main_sweeps))
        record_interval = max(1, int(round(N * record_sweeps)))
        diameter = float(METRICS.get('cluster_diameter_threshold', 0.05))
        output_path = os.path.join(DATA_DIR, output_filename)
        results = {
            'control': {
                'N': N, 'm': m, 'dimension': int(dimension),
                'num_runs': int(num_runs),
                'main_sweeps': main_sweeps,
                'record_sweeps': record_sweeps,
                'p_values': p_values,
                'model_order': model_order,
                'initial_distribution': f'Uniform([0,1]^{dimension})',
                'distance': 'Euclidean norm',
                'compact_order_definition': (
                    'largest fraction inside a ball of radius diameter/2 '
                    'centred on an observed opinion'),
                'compact_order_diameter': diameter,
                'core_parameters': self._core_parameter_signature(),
            },
            'data': {},
        }
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                old = loaded.get('control', {})
                same = (
                    int(old.get('N', -1)) == N
                    and int(old.get('m', -1)) == m
                    and int(old.get('dimension', -1)) == int(dimension)
                    and int(old.get('num_runs', -1)) == int(num_runs)
                    and float(old.get('main_sweeps', -1)) == main_sweeps
                    and float(old.get('record_sweeps', -1)) == record_sweeps
                    and list(old.get('p_values', [])) == p_values
                    and list(old.get('model_order', [])) == model_order
                    and float(old.get('compact_order_diameter', -1))
                    == diameter
                    and old.get('core_parameters')
                    == self._core_parameter_signature()
                )
                if same:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个 p 块")
                else:
                    raise FileExistsError(
                        f"vector-opinion checkpoint control/source mismatch: "
                        f"{output_path}; original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"vector-opinion checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        metrics = (
            'compact_order', 'total_variance', 'mean_pairwise_distance',
            'triangle_alignment_fixed',
            'triangle_admissibility_adaptive', 'terminal_mean_abs_dx',
        )
        print("\n" + "=" * 70)
        print(f"扩展实验: {dimension}维意见鲁棒性")
        print("=" * 70)
        for p in p_values:
            p_key = f"{p:.2f}"
            existing = results['data'].get(p_key, {})
            if (resume and existing.get('_control', {}).get('complete')
                    and int(existing['_control'].get('num_runs', -1))
                    == int(num_runs)):
                print(f"  [续跑] p={p:.2f} 已完成")
                continue

            p_block = {}
            for model_key in model_order:
                print(f"\n  p={p:.2f}, model={model_key}")
                raw = []
                for run_i in range(num_runs):
                    seed = run_i * 997 + 7
                    rng = np.random.RandomState(seed)
                    G = NetworkGenerator.generate_holme_kim(
                        N, m, p, seed=seed)
                    triangles, node_triangles = (
                        NetworkGenerator.extract_triangles(G))
                    weights = NetworkGenerator.compute_jaccard_weights(G)
                    clustering = NetworkGenerator.compute_clustering(G)
                    opinions = {
                        node: rng.uniform(0.0, 1.0, size=int(dimension))
                        for node in G.nodes()
                    }
                    trust = TrustThreshold(
                        TRUST['epsilon_0_default'], TRUST['alpha'])
                    engine = VectorOpinionDynamics(
                        G, opinions, weights, triangles, node_triangles,
                        clustering, trust, DYNAMICS['mu'],
                        DYNAMICS['beta_default'], model_key)
                    dx_acc = 0.0
                    dx_count = 0
                    last_window_dx = None
                    for step in range(1, num_steps + 1):
                        dx_acc += engine.step(rng)
                        dx_count += 1
                        if step % record_interval == 0:
                            last_window_dx = dx_acc / dx_count
                            dx_acc = 0.0
                            dx_count = 0
                    if dx_count:
                        last_window_dx = dx_acc / dx_count
                    final_array = engine.get_opinions_array()
                    final_dict = engine.get_opinions()
                    row = {
                        'compact_order': Metrics.vector_compact_order(
                            final_array, diameter=diameter),
                        'total_variance': Metrics.vector_total_variance(
                            final_array),
                        'mean_pairwise_distance':
                            Metrics.vector_mean_pairwise_distance(
                                final_array,
                                rng=np.random.RandomState(seed + 32452843)),
                        'triangle_alignment_fixed':
                            Metrics.vector_triangle_alignment_fixed(
                                final_dict, triangles,
                                TRUST['epsilon_0_default']),
                        'triangle_admissibility_adaptive':
                            Metrics.vector_triangle_admissibility_adaptive(
                                final_dict, triangles, weights, trust),
                        'terminal_mean_abs_dx': float(last_window_dx or 0.0),
                        'seed': int(seed),
                    }
                    raw.append(row)
                    print(f"    run {run_i+1}/{num_runs}: "
                          f"R{dimension}={row['compact_order']:.4f}, "
                          f"var={row['total_variance']:.4f}")
                model_block = {
                    metric: summarize([row[metric] for row in raw])
                    for metric in metrics
                }
                model_block['raw'] = raw
                p_block[model_key] = model_block

            paired = {}
            if 'full_model' in p_block:
                for comparator in model_order:
                    if comparator == 'full_model':
                        continue
                    paired[f'full_model_minus_{comparator}'] = {
                        metric: paired_difference(
                            p_block['full_model'][metric]['values'],
                            p_block[comparator][metric]['values'])
                        for metric in metrics
                    }
            p_block['_paired_comparisons'] = paired
            p_block['_control'] = {
                'complete': True,
                'num_runs': int(num_runs),
                'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
            }
            results['data'][p_key] = p_block
            self.save_json(
                results, output_filename,
                metadata={
                    'n_runs': num_runs,
                    'experiment_design': f'{dimension}d_vector_opinions',
                    'main_sweeps': main_sweeps,
                },
            )

        self.save_json(
            results, output_filename,
            metadata={
                'n_runs': num_runs,
                'experiment_design': f'{dimension}d_vector_opinions',
                'main_sweeps': main_sweeps,
            },
        )
        print(f"\n  [{dimension}维意见实验完成]")
        return results

    # ================================================================
    # 扩展实验: 内置真实社会网络拓扑
    # ================================================================
    def experiment_empirical_topologies(
            self, num_runs=None, topology_names=None, model_order=None,
            resume=True, output_filename='exp_empirical_topologies.json'):
        """Run dynamics on fixed public empirical social-network benchmarks."""
        if num_runs is None:
            num_runs = SIMULATION['num_runs']
        if topology_names is None:
            topology_names = ['karate', 'davis', 'florentine']
        topology_names = [str(name) for name in topology_names]
        if model_order is None:
            model_order = [
                'threshold_only', 'no_triadic_feedback', 'full_model']
        model_order = [resolve_model_key(model) for model in model_order]
        main_sweeps = float(SIMULATION.get('main_sweeps', 200))
        record_sweeps = float(SIMULATION.get('record_sweeps', 5))
        output_path = os.path.join(DATA_DIR, output_filename)
        results = {
            'control': {
                'num_runs_per_fixed_topology': int(num_runs),
                'topology_replicates_per_benchmark': 1,
                'main_sweeps': main_sweeps,
                'record_sweeps': record_sweeps,
                'topology_names': topology_names,
                'model_order': model_order,
                'initial_distribution': 'Uniform[0,1]',
                'interpretation': (
                    'uncertainty reflects initial opinions and update '
                    'sequences, not independent empirical networks'),
                'core_parameters': self._core_parameter_signature(),
            },
            'data': {},
        }
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                old = loaded.get('control', {})
                same = (
                    int(old.get('num_runs_per_fixed_topology', -1))
                    == int(num_runs)
                    and float(old.get('main_sweeps', -1)) == main_sweeps
                    and float(old.get('record_sweeps', -1)) == record_sweeps
                    and list(old.get('topology_names', [])) == topology_names
                    and list(old.get('model_order', [])) == model_order
                    and old.get('core_parameters')
                    == self._core_parameter_signature()
                )
                if same:
                    results['data'].update(loaded.get('data', {}))
                    print(f"  [续跑] 已载入 {len(results['data'])} 个真实拓扑")
                else:
                    raise FileExistsError(
                        f"real-topology checkpoint control/source mismatch: "
                        f"{output_path}; original file preserved"
                    )
            except Exception as exc:
                if isinstance(exc, FileExistsError):
                    raise
                raise FileExistsError(
                    f"real-topology checkpoint unreadable: {output_path}; "
                    "original file preserved"
                ) from exc

        metric_paths = {
            'order_param': ['final', 'order_param'],
            'n_clusters': ['final', 'n_clusters'],
            'variance': ['final', 'variance'],
            'triangle_alignment_fixed': [
                'final', 'triangle_alignment_fixed'],
            'triangle_admissibility_adaptive': [
                'final', 'triangle_admissibility_adaptive'],
            'edge_disagreement_penalty': [
                'final', 'edge_disagreement_penalty'],
        }
        print("\n" + "=" * 70)
        print("扩展实验: 真实社会网络拓扑")
        print("=" * 70)
        for topology_name in topology_names:
            key = topology_name.lower()
            existing = results['data'].get(key, {})
            if (resume and existing.get('_control', {}).get('complete')
                    and int(existing['_control'].get('num_runs', -1))
                    == int(num_runs)):
                print(f"  [续跑] topology={key} 已完成")
                continue
            graph, benchmark_meta = (
                NetworkGenerator.load_empirical_benchmark(topology_name))
            N = graph.number_of_nodes()
            num_steps = int(round(N * main_sweeps))
            record_interval = max(1, int(round(N * record_sweeps)))
            topology_block = {
                'network': {
                    **benchmark_meta,
                    **NetworkGenerator.compute_full_properties(
                        graph, seed=SIMULATION.get('random_seed', 42)),
                },
            }
            # The raw degree vector is unnecessary in the result bundle.
            topology_block['network'].pop('degrees', None)
            for model_key in model_order:
                print(f"\n  topology={key}, model={model_key}")
                batch = []
                for run_i in range(num_runs):
                    seed = run_i * 997 + 7
                    sim = SingleRun(
                        N=N, m=0, p_hk=0.0,
                        epsilon_0=TRUST['epsilon_0_default'],
                        alpha=TRUST['alpha'], mu=DYNAMICS['mu'],
                        beta=DYNAMICS['beta_default'],
                        model_type=model_key, num_steps=num_steps,
                        seed=seed, graph=graph,
                        network_type='empirical_benchmark',
                        network_params=benchmark_meta,
                    )
                    result = sim.execute(record_interval=record_interval)
                    batch.append(result)
                    print(f"    run {run_i+1}/{num_runs}: "
                          f"R={result['final']['order_param']:.4f}")
                model_block = {
                    metric: self._aggregate(batch, path)
                    for metric, path in metric_paths.items()
                }
                model_block['time_series_summary'] = (
                    self._aggregate_time_series(batch))
                topology_block[model_key] = model_block

            paired = {}
            if 'full_model' in topology_block:
                for comparator in model_order:
                    if comparator == 'full_model':
                        continue
                    paired[f'full_model_minus_{comparator}'] = {
                        metric: paired_difference(
                            topology_block['full_model'][metric]['values'],
                            topology_block[comparator][metric]['values'])
                        for metric in metric_paths
                    }
            topology_block['_paired_comparisons'] = paired
            topology_block['_control'] = {
                'complete': True,
                'num_runs': int(num_runs),
                'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
            }
            results['data'][key] = topology_block
            self.save_json(
                results, output_filename,
                metadata={
                    'n_runs': num_runs,
                    'experiment_design': 'fixed_empirical_topologies',
                    'main_sweeps': main_sweeps,
                },
            )

        self.save_json(
            results, output_filename,
            metadata={
                'n_runs': num_runs,
                'experiment_design': 'fixed_empirical_topologies',
                'main_sweeps': main_sweeps,
            },
        )
        print("\n  [真实拓扑实验完成]")
        return results

    # ================================================================
    # 实验8.1: β 单参数扫描
    # ================================================================

    def _run_beta_sweep_for_p(self, p, num_runs, beta_values):
        p_results = {}
        print(f"\n  === p = {p:.1f} ===")
        for beta in beta_values:
            batch = self._run_batch(
                {'p_hk': p, 'beta': beta, 'model_type': 'full_model'},
                num_runs=num_runs,
                verbose_label=f"β={beta:.1f}",
            )
            agg = {
                'order_param': self._aggregate(batch, ['final', 'order_param']),
                'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
                'triangle_alignment_fixed': self._aggregate(
                    batch, ['final', 'triangle_alignment_fixed']),
                'triangle_admissibility_adaptive': self._aggregate(
                    batch, ['final', 'triangle_admissibility_adaptive']),
                'edge_disagreement_penalty': self._aggregate(
                    batch, ['final', 'edge_disagreement_penalty']),
            }
            p_results[str(beta)] = agg
            print(f"    β={beta:.1f}: OP={agg['order_param']['mean']:.4f}, "
                  f"clusters={agg['n_clusters']['mean']:.1f}")
        return {str(p): p_results}

    def experiment_beta_sweep(
            self, num_runs=None, beta_values=None, p_values=None, resume=True,
            output_filename='exp8_beta_sweep.json'):
        """β第三方不一致衰减强度扫描"""
        print("\n" + "=" * 70)
        print("实验8.1: β 扫描")
        print("=" * 70)

        if num_runs is None:
            num_runs = SIMULATION['num_runs']

        if beta_values is None:
            beta_values = DYNAMICS['beta_range']
        if p_values is None:
            p_values = [0.0, 0.4, 0.8]
        beta_values = [float(value) for value in beta_values]
        p_values_sweep = [float(value) for value in p_values]
        control = {
            'experiment': 'beta_sweep',
            'N': int(NETWORK['N']), 'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'p_values': p_values_sweep,
            'beta_values': [float(value) for value in beta_values],
            'epsilon_0': float(TRUST['epsilon_0_default']),
            'alpha': float(TRUST['alpha']),
            'mu': float(DYNAMICS['mu']),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)
        for p in p_values_sweep:
            p_key = str(p)
            p_results = results.setdefault(p_key, {})
            print(f"\n  === p = {p:.1f} ===")
            for beta in beta_values:
                beta_key = str(beta)
                if resume and beta_key in p_results:
                    print(f"    [续跑] p={p:.1f}, β={beta:.2f} 已完成")
                    continue
                batch = self._run_batch(
                    {'p_hk': p, 'beta': beta,
                     'model_type': 'full_model'},
                    num_runs=num_runs,
                    verbose_label=f"β={beta:.2f}",
                )
                p_results[beta_key] = {
                    'order_param': self._aggregate(
                        batch, ['final', 'order_param']),
                    'n_clusters': self._aggregate(
                        batch, ['final', 'n_clusters']),
                    'triangle_alignment_fixed': self._aggregate(
                        batch, ['final', 'triangle_alignment_fixed']),
                    'triangle_admissibility_adaptive': self._aggregate(
                        batch, ['final',
                                'triangle_admissibility_adaptive']),
                    'edge_disagreement_penalty': self._aggregate(
                        batch, ['final', 'edge_disagreement_penalty']),
                }
                self.save_json(
                    results, output_filename,
                    metadata={'n_runs': num_runs,
                              'experiment_design': 'beta_sweep',
                              'main_sweeps': control['main_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'beta_sweep',
                      'main_sweeps': control['main_sweeps']})
        print("\n  [β扫描完成]")
        return results

    # ================================================================
    # 实验8.1续: α 灵敏度扫描
    # ================================================================

    def _run_alpha_sweep_for_alpha(self, alpha, num_runs):
        batch = self._run_batch(
            {'alpha': alpha, 'p_hk': 0.6, 'model_type': 'full_model'},
            num_runs=num_runs,
            verbose_label=f"α={alpha:.1f}",
        )
        alpha_results = {
            'order_param': self._aggregate(batch, ['final', 'order_param']),
            'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
        }
        print(f"  α={alpha:.1f}: OP={alpha_results['order_param']['mean']:.4f}")
        return {str(alpha): alpha_results}

    def experiment_alpha_sweep(
            self, num_runs=None, alpha_values=None, resume=True,
            output_filename='exp8_alpha_sweep.json'):
        """α调制灵敏度扫描"""
        print("\n" + "=" * 70)
        print("实验8.1续: α 扫描")
        print("=" * 70)

        if num_runs is None:
            num_runs = SIMULATION['num_runs']

        if alpha_values is None:
            alpha_values = SENSITIVITY['alpha_range']
        alpha_values = [float(value) for value in alpha_values]
        control = {
            'experiment': 'alpha_sweep',
            'N': int(NETWORK['N']), 'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'alpha_values': [float(value) for value in alpha_values],
            'p_hk': 0.6,
            'epsilon_0': float(TRUST['epsilon_0_default']),
            'mu': float(DYNAMICS['mu']),
            'beta': float(DYNAMICS['beta_default']),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)

        for alpha in alpha_values:
            key = str(alpha)
            if resume and key in results:
                print(f"  [续跑] α={alpha:.2f} 已完成")
                continue
            alpha_result = self._run_alpha_sweep_for_alpha(alpha, num_runs)
            results.update(alpha_result)
            self.save_json(
                results, output_filename,
                metadata={'n_runs': num_runs,
                          'experiment_design': 'alpha_sweep',
                          'main_sweeps': control['main_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'alpha_sweep',
                      'main_sweeps': control['main_sweeps']})
        print("\n  [α扫描完成]")
        return results

    # ================================================================
    # 实验8.1续: μ 步长扫描
    # ================================================================

    def _run_mu_sweep_for_mu(self, mu, num_runs):
        batch = self._run_batch(
            {'mu': mu, 'p_hk': 0.6, 'model_type': 'full_model'},
            num_runs=num_runs,
            verbose_label=f"μ={mu:.1f}",
        )
        mu_results = {
            'order_param': self._aggregate(batch, ['final', 'order_param']),
            'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
        }
        print(f"  μ={mu:.1f}: OP={mu_results['order_param']['mean']:.4f}")
        return {str(mu): mu_results}

    def experiment_mu_sweep(
            self, num_runs=None, mu_values=None, resume=True,
            output_filename='exp8_mu_sweep.json'):
        """μ更新步长扫描"""
        print("\n" + "=" * 70)
        print("实验8.1续: μ 扫描")
        print("=" * 70)

        if num_runs is None:
            num_runs = SIMULATION['num_runs']

        if mu_values is None:
            mu_values = SENSITIVITY['mu_range']
        mu_values = [float(value) for value in mu_values]
        control = {
            'experiment': 'mu_sweep',
            'N': int(NETWORK['N']), 'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'mu_values': [float(value) for value in mu_values],
            'p_hk': 0.6,
            'epsilon_0': float(TRUST['epsilon_0_default']),
            'alpha': float(TRUST['alpha']),
            'beta': float(DYNAMICS['beta_default']),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)

        for mu in mu_values:
            key = str(mu)
            if resume and key in results:
                print(f"  [续跑] μ={mu:.2f} 已完成")
                continue
            mu_result = self._run_mu_sweep_for_mu(mu, num_runs)
            results.update(mu_result)
            self.save_json(
                results, output_filename,
                metadata={'n_runs': num_runs,
                          'experiment_design': 'mu_sweep',
                          'main_sweeps': control['main_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'mu_sweep',
                      'main_sweeps': control['main_sweeps']})
        print("\n  [μ扫描完成]")
        return results

    # ================================================================
    # 实验8.2: ε₀ vs β 联合扫描 (二维热图数据)
    # ================================================================

    def _run_joint_eps_beta_for_eps(self, eps, num_runs, beta_values, p_fixed, total_combinations):
        eps_data = {}
        for beta in beta_values:
            # count += 1  # 这里的 count 需要全局管理或重新计算，暂时移除
            label = f"epsilon_0={eps:.2f}, beta={beta:.1f}"
            batch = self._run_batch(
                {'p_hk': p_fixed, 'epsilon_0': eps, 'beta': beta,
                 'model_type': 'full_model'},
                num_runs=num_runs,
                verbose_label=label,
            )
            key = f"{eps:.2f}_{beta:.2f}"
            eps_data[key] = {
                'order_param': self._aggregate(batch, ['final', 'order_param']),
                'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
                'triangle_alignment_fixed': self._aggregate(
                    batch, ['final', 'triangle_alignment_fixed']),
                'triangle_admissibility_adaptive': self._aggregate(
                    batch, ['final', 'triangle_admissibility_adaptive']),
                'edge_disagreement_penalty': self._aggregate(
                    batch, ['final', 'edge_disagreement_penalty']),
            }
            op = eps_data[key]['order_param']['mean']
            print(f"  {label}: OP={op:.4f}")
        return eps_data

    def experiment_joint_eps_beta(
            self, num_runs=None, eps_values=None, beta_values=None,
            p_fixed=0.6, resume=True,
            output_filename='exp8_joint_eps_beta.json'):
        """ε₀ × β 联合扫描，在固定p值下生成二维热图数据"""
        print("\n" + "=" * 70)
        print("实验8.2: epsilon_0 x beta 联合扫描")
        print("=" * 70)

        if num_runs is None:
            num_runs = max(SIMULATION['num_runs'] // 2, 5)

        if eps_values is None:
            eps_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                          0.40, 0.45, 0.50]
        if beta_values is None:
            beta_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        eps_values = [float(value) for value in eps_values]
        beta_values = [float(value) for value in beta_values]
        control = {
            'experiment': 'joint_eps_beta',
            'N': int(NETWORK['N']), 'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'eps_values': eps_values, 'beta_values': beta_values,
            'p_hk': float(p_fixed),
            'alpha': float(TRUST['alpha']),
            'mu': float(DYNAMICS['mu']),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)
        results.update({
            'eps_values': eps_values,
            'beta_values': beta_values,
            'p_hk': p_fixed,
        })
        data = results.setdefault('data', {})

        total = len(eps_values) * len(beta_values)

        for eps in eps_values:
            for beta in beta_values:
                key = f"{eps:.2f}_{beta:.2f}"
                if resume and key in data:
                    print(f"  [续跑] epsilon_0={eps:.2f}, "
                          f"beta={beta:.1f} 已完成")
                    continue
                data.update(self._run_joint_eps_beta_for_eps(
                    eps, num_runs, [beta], p_fixed, total))
                self.save_json(
                    results, output_filename,
                    metadata={'n_runs': num_runs,
                              'experiment_design': 'joint_eps_beta',
                              'main_sweeps': control['main_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'joint_eps_beta',
                      'main_sweeps': control['main_sweeps']})
        print("\n  [epsilon_0 x beta 联合扫描完成]")
        return results

    # ================================================================
    # 实验8.2续: f△ vs β 联合扫描
    # ================================================================

    def _run_joint_p_beta_for_p(self, p, num_runs, beta_values, total_combinations):
        p_data = {}
        for beta in beta_values:
            # count += 1 # 这里的 count 需要全局管理或重新计算，暂时移除
            label = f"p={p:.2f},β={beta:.2f}"
            batch = self._run_batch(
                {'p_hk': p, 'beta': beta, 'model_type': 'full_model'},
                num_runs=num_runs,
                verbose_label=label,
            )
            key = f"{p:.2f}_{beta:.2f}"
            p_data[key] = {
                'order_param': self._aggregate(batch, ['final', 'order_param']),
                'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
                'f_triangle': self._aggregate(batch, ['network', 'f_triangle']),
            }
            print(f"  {label}: OP={p_data[key]['order_param']['mean']:.4f}, "
                  f"f△={p_data[key]['f_triangle']['mean']:.4f}")
        return p_data

    def experiment_joint_p_beta(
            self, num_runs=None, p_values=None, beta_values=None,
            resume=True, output_filename='exp8_joint_p_beta.json'):
        """HK p (f△) × β 联合扫描"""
        print("\n" + "=" * 70)
        print("实验8.2续: p(f△) × β 联合扫描")
        print("=" * 70)

        if num_runs is None:
            num_runs = max(SIMULATION['num_runs'] // 2, 5)

        if p_values is None:
            p_values = [round(x * 0.05, 2) for x in range(21)]
        if beta_values is None:
            beta_values = [round(x * 0.5, 2) for x in range(21)]
        p_values = [float(value) for value in p_values]
        beta_values = [float(value) for value in beta_values]
        control = {
            'experiment': 'joint_p_beta',
            'N': int(NETWORK['N']), 'm': int(NETWORK['m']),
            'num_runs': int(num_runs),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'p_values': p_values, 'beta_values': beta_values,
            'epsilon_0': float(TRUST['epsilon_0_default']),
            'alpha': float(TRUST['alpha']),
            'mu': float(DYNAMICS['mu']),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)
        results.update({
            'p_values': p_values,
            'beta_values': beta_values,
        })
        data = results.setdefault('data', {})

        total = len(p_values) * len(beta_values)

        for p in p_values:
            for beta in beta_values:
                key = f"{p:.2f}_{beta:.2f}"
                if resume and key in data:
                    print(f"  [续跑] p={p:.2f}, β={beta:.2f} 已完成")
                    continue
                data.update(self._run_joint_p_beta_for_p(
                    p, num_runs, [beta], total))
                self.save_json(
                    results, output_filename,
                    metadata={'n_runs': num_runs,
                              'experiment_design': 'joint_p_beta',
                              'main_sweeps': control['main_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'joint_p_beta',
                      'main_sweeps': control['main_sweeps']})
        print("\n  [p×β联合扫描完成]")
        return results

    # ================================================================
    # 实验8.3: 网络规模鲁棒性
    # ================================================================

    def _run_size_robustness_for_N(self, N, num_runs, p_fixed):
        print(f"\n  N = {N}:")
        max_sweeps = int(SIMULATION.get('size_sweeps', 200))
        record_sweeps = int(SIMULATION.get('size_record_sweeps', 1))
        min_sweeps = int(SIMULATION.get('convergence_min_sweeps', 20))
        steps = int(N * max_sweeps)
        record_interval = max(1, int(N * record_sweeps))
        min_steps = int(N * min_sweeps)
        batch = self._run_batch(
            {'N': N, 'p_hk': p_fixed, 'model_type': 'full_model'},
            num_runs=num_runs,
            num_steps=steps,
            record_interval=record_interval,
            convergence_check=True,
            convergence_min_steps=min_steps,
            verbose_label=f"N={N}",
        )
        N_results = {
            'order_param': self._aggregate(batch, ['final', 'order_param']),
            'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
            'variance': self._aggregate(batch, ['final', 'variance']),
            'triangle_alignment_fixed': self._aggregate(
                batch, ['final', 'triangle_alignment_fixed']),
            'triangle_admissibility_adaptive': self._aggregate(
                batch, ['final', 'triangle_admissibility_adaptive']),
            'edge_disagreement_penalty': self._aggregate(
                batch, ['final', 'edge_disagreement_penalty']),
            'converged': self._aggregate(batch, ['convergence', 'converged']),
            'steps_run': self._aggregate(batch, ['convergence', 'steps_run']),
            'sweeps_run': self._aggregate(batch, ['convergence', 'sweeps_run']),
            'final_mean_dx': self._aggregate(batch, ['convergence', 'final_mean_dx']),
            'simulation_control': {
                'max_sweeps': max_sweeps,
                'max_steps': steps,
                'record_sweeps': record_sweeps,
                'record_interval': record_interval,
                'convergence_min_sweeps': min_sweeps,
                'convergence_min_steps': min_steps,
                'convergence_tol': SIMULATION.get('convergence_tol', 1e-7),
                'convergence_patience': SIMULATION.get('convergence_patience', 5),
            },
        }
        print(f"  N={N}: OP={N_results['order_param']['mean']:.4f}, "
              f"converged={N_results['converged']['mean']:.2f}, "
              f"sweeps={N_results['sweeps_run']['mean']:.1f}")
        return {str(N): N_results}

    def experiment_size_robustness(
            self, num_runs=None, N_sizes=None, p_fixed=0.6, resume=True,
            output_filename='exp8_size_robustness.json'):
        print("\n" + "=" * 70)
        print("实验8.3: 网络规模鲁棒性")
        print("=" * 70)

        if num_runs is None:
            num_runs = max(SIMULATION['num_runs'] // 2, 5)

        if N_sizes is None:
            N_sizes = SENSITIVITY['N_sizes']
        N_sizes = [int(value) for value in N_sizes]
        control = {
            'experiment': 'size_robustness',
            'N_sizes': N_sizes, 'm': int(NETWORK['m']),
            'num_runs': int(num_runs), 'p_hk': float(p_fixed),
            'size_sweeps': float(SIMULATION.get('size_sweeps', 200)),
            'size_record_sweeps': float(
                SIMULATION.get('size_record_sweeps', 1)),
            'convergence_min_sweeps': float(
                SIMULATION.get('convergence_min_sweeps', 20)),
            'convergence_tol': float(
                SIMULATION.get('convergence_tol', 1e-7)),
            'convergence_patience': int(
                SIMULATION.get('convergence_patience', 5)),
            'core_parameters': self._core_parameter_signature(),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)

        for N in N_sizes:
            key = str(N)
            if resume and key in results:
                print(f"  [续跑] N={N} 已完成")
                continue
            N_result = self._run_size_robustness_for_N(N, num_runs, p_fixed)
            results.update(N_result)
            self.save_json(
                results, output_filename,
                metadata={'n_runs': num_runs,
                          'experiment_design': 'size_robustness',
                          'size_sweeps': control['size_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'size_robustness',
                      'size_sweeps': control['size_sweeps']})
        print("\n  [规模鲁棒性实验完成]")
        return results

    # ================================================================
    # 实验8.3续: 平均度变体
    # ================================================================

    def _run_m_robustness_for_m(self, m, num_runs, p_fixed):
        print(f"\n  m = {m}:")
        batch = self._run_batch(
            {'m': m, 'p_hk': p_fixed, 'model_type': 'full_model'},
            num_runs=num_runs,
            verbose_label=f"m={m}",
        )
        m_results = {
            'order_param': self._aggregate(batch, ['final', 'order_param']),
            'n_clusters': self._aggregate(batch, ['final', 'n_clusters']),
            'f_triangle': self._aggregate(batch, ['network', 'f_triangle']),
        }
        print(f"  m={m}: OP={m_results['order_param']['mean']:.4f}")
        return {str(m): m_results}

    def experiment_m_robustness(
            self, num_runs=None, m_values=None, p_fixed=0.6, resume=True,
            output_filename='exp8_m_robustness.json'):
        """m = 2, 3, 4 的平均度变体测试"""
        print("\n" + "=" * 70)
        print("实验8.3续: 平均度 m 鲁棒性")
        print("=" * 70)

        if num_runs is None:
            num_runs = max(SIMULATION['num_runs'] // 2, 5)

        if m_values is None:
            m_values = SENSITIVITY['m_values']
        m_values = [int(value) for value in m_values]
        control = {
            'experiment': 'm_robustness',
            'N': int(NETWORK['N']), 'm_values': m_values,
            'num_runs': int(num_runs), 'p_hk': float(p_fixed),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'record_sweeps': float(SIMULATION.get('record_sweeps', 5)),
            'core_parameters': self._core_parameter_signature(),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)

        for m in m_values:
            key = str(m)
            if resume and key in results:
                print(f"  [续跑] m={m} 已完成")
                continue
            m_result = self._run_m_robustness_for_m(m, num_runs, p_fixed)
            results.update(m_result)
            self.save_json(
                results, output_filename,
                metadata={'n_runs': num_runs,
                          'experiment_design': 'm_robustness',
                          'main_sweeps': control['main_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'm_robustness',
                      'main_sweeps': control['main_sweeps']})
        print("\n  [m鲁棒性实验完成]")
        return results

    # ================================================================
    # 实验8.3续: 初始意见分布鲁棒性
    # ================================================================

    def _run_init_distribution_for_dist(self, dist_name, num_runs, p_fixed, N, m, num_steps):
        print(f"\n  分布: {dist_name}")
        batch_results = []

        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            rng = np.random.RandomState(seed)

            # ---- 生成网络 ----
            G = NetworkGenerator.generate_holme_kim(N, m, p_fixed, seed=seed)
            triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
            weights = NetworkGenerator.compute_jaccard_weights(G)
            clustering = NetworkGenerator.compute_clustering(G)

            # ---- 根据分布类型生成初始意见 ----
            nodes = list(G.nodes())
            n = len(nodes)

            if dist_name == 'uniform':
                raw = rng.uniform(0, 1, n)
            elif dist_name == 'normal':
                raw = rng.normal(0.5, 0.2, n)
            elif dist_name == 'bimodal':
                mask = rng.rand(n) < 0.5
                raw = np.where(
                    mask,
                    rng.normal(0.25, 0.1, n),
                    rng.normal(0.75, 0.1, n),
                )
            else:
                raw = rng.uniform(0, 1, n)

            # 裁剪到 [0, 1]
            raw = np.clip(raw, 0.0, 1.0)
            opinions = {node: float(raw[idx]) for idx, node in enumerate(nodes)}

            # ---- 构建动力学引擎并运行 ----
            trust = TrustThreshold(
                TRUST['epsilon_0_default'], TRUST['alpha']
            )
            engine = OpinionDynamics(
                G, opinions, weights, triangles_list, node_triangles,
                clustering, trust,
                DYNAMICS['mu'], DYNAMICS['beta_default'],
                'full_model',
            )

            for _ in range(num_steps):
                engine.step(rng)

            # ---- 收集终态指标 ----
            final_ops_dict = engine.get_opinions()
            final_ops_arr = engine.get_opinions_array()

            res_i = {
                'final': {
                    'order_param': float(
                        Metrics.order_parameter(final_ops_arr)),
                    'n_clusters': int(
                        Metrics.count_clusters(final_ops_arr)),
                    'variance': float(
                        Metrics.opinion_variance(final_ops_arr)),
                    'triangle_alignment_fixed': float(
                        Metrics.triangle_alignment_fixed(
                            final_ops_dict, triangles_list,
                            TRUST['epsilon_0_default'])),
                    'triangle_admissibility_adaptive': float(
                        Metrics.triangle_admissibility_adaptive(
                            final_ops_dict, triangles_list, weights, trust)),
                    'edge_disagreement_penalty': float(
                        Metrics.edge_disagreement_penalty(
                            final_ops_dict, G, DYNAMICS['beta_default'],
                            rng=rng)),
                }
            }
            batch_results.append(res_i)
            print(f"    run {run_i + 1}/{num_runs}: "
                  f"clusters={res_i['final']['n_clusters']}, "
                  f"OP={res_i['final']['order_param']:.4f}")

        # ---- 聚合该分布的所有 run ----
        dist_results = {
            'order_param': self._aggregate(
                batch_results, ['final', 'order_param']),
            'n_clusters': self._aggregate(
                batch_results, ['final', 'n_clusters']),
            'variance': self._aggregate(
                batch_results, ['final', 'variance']),
            'triangle_alignment_fixed': self._aggregate(
                batch_results, ['final', 'triangle_alignment_fixed']),
            'triangle_admissibility_adaptive': self._aggregate(
                batch_results, ['final', 'triangle_admissibility_adaptive']),
            'edge_disagreement_penalty': self._aggregate(
                batch_results, ['final', 'edge_disagreement_penalty']),
        }
        op = dist_results['order_param']
        nc = dist_results['n_clusters']
        print(f"  [{dist_name}] "
              f"OP={op['mean']:.4f}±{op['std']:.4f}, "
              f"clusters={nc['mean']:.1f}±{nc['std']:.1f}")
        return {dist_name: dist_results}

    def experiment_init_distribution(
            self, num_runs=None, distributions=None, p_fixed=0.6,
            resume=True, output_filename='exp8_init_distribution.json'):
        """
        初始意见分布变体：Uniform, Normal, Bimodal
        通过预生成意见再手动构建动力学引擎的方式实现
        """
        print("\n" + "=" * 70)
        print("实验8.3续: 初始意见分布鲁棒性")
        print("=" * 70)

        if num_runs is None:
            num_runs = SIMULATION['num_runs']

        if distributions is None:
            distributions = ['uniform', 'normal', 'bimodal']
        distributions = list(distributions)
        N = NETWORK['N']
        m = NETWORK['m']
        num_steps = int(round(
            N * float(SIMULATION.get('main_sweeps', 200))))
        control = {
            'experiment': 'init_distribution',
            'N': int(N), 'm': int(m), 'num_runs': int(num_runs),
            'main_sweeps': float(SIMULATION.get('main_sweeps', 200)),
            'num_steps': int(num_steps), 'p_hk': float(p_fixed),
            'distributions': distributions,
            'core_parameters': self._core_parameter_signature(),
            'seeds': [run_i * 997 + 7 for run_i in range(num_runs)],
        }
        results = self._load_controlled_results(
            output_filename, control, resume=resume)

        for dist_name in distributions:
            if resume and dist_name in results:
                print(f"  [续跑] 初始分布 {dist_name} 已完成")
                continue
            dist_result = self._run_init_distribution_for_dist(
                dist_name, num_runs, p_fixed, N, m, num_steps)
            results.update(dist_result)
            self.save_json(
                results, output_filename,
                metadata={'n_runs': num_runs,
                          'experiment_design': 'init_distribution',
                          'main_sweeps': control['main_sweeps']})

        self.save_json(
            results, output_filename,
            metadata={'n_runs': num_runs,
                      'experiment_design': 'init_distribution',
                      'main_sweeps': control['main_sweeps']})
        print("\n  [初始分布鲁棒性实验完成]")
        return results
    # ================================================================
    # 运行全部实验的便捷入口
    # ================================================================

    def run_all(self, quick=False, resume=True, num_runs=None):
        """
        依次执行所有实验
        quick=True 时减少 num_runs 加速调试
        """
        nr = 3 if quick else num_runs
        nv = 1 if quick else None  # quick 模式每个 p 只生成 1 个网络
        quick_p_values = [0.0, 0.25, 0.5, 0.75, 1.0] if quick else None

        t_total = time.time()
        results = {}

        print("\n" + "#" * 70)
        print("  开始全部实验")
        print("  稳定版默认顺序执行，避免 Windows/Anaconda 多进程 BrokenProcessPool")
        print("#" * 70)

        results['exp7'] = self.experiment_network_validation(
            num_networks=nv, p_values=quick_p_values, resume=resume)
        results['exp6'] = self.experiment_ablation(
            num_runs=nr, p_values=quick_p_values, resume=resume)
        results['eps_sweep'] = self.experiment_epsilon_sweep(
            num_runs=nr,
            p_values=quick_p_values[:3] if quick else None,
            eps_values=[0.05, 0.20, 0.40] if quick else None,
            resume=resume)
        results['hysteresis_sweep'] = self.experiment_hysteresis_sweep(
            num_runs=nr, p_values=quick_p_values,
            eps_values=[0.05, 0.20, 0.40] if quick else None,
            resume=resume)
        results['perturbation'] = self.experiment_perturbation_sweep(
            num_runs=nr,
            p_values=quick_p_values,
            sigma_values=[0.0, 0.02, 0.10] if quick else None,
            eps_values=[0.05, 0.20, 0.40] if quick else None,
            resume=resume,
        )
        results['update_mode'] = self.experiment_update_mode_control(
            num_runs=nr, p_values=quick_p_values, resume=resume)
        results['community'] = self.experiment_sbm_community_conflict(
            num_runs=nr,
            mixing_values=[0.05, 0.20] if quick else None,
            resume=resume,
        )
        results['sbm_hysteresis'] = self.experiment_sbm_hysteresis(
            num_runs=nr,
            mixing_values=[0.02, 0.20] if quick else None,
            sigma_values=[0.0, 0.10] if quick else None,
            eps_values=[0.10, 0.20, 0.40] if quick else None,
            resume=resume,
        )
        results['vector'] = self.experiment_vector_opinions(
            num_runs=nr, p_values=quick_p_values, resume=resume)
        results['empirical'] = self.experiment_empirical_topologies(
            num_runs=nr, resume=resume)
        results['beta_sweep'] = self.experiment_beta_sweep(
            num_runs=nr,
            beta_values=[0.0, 2.5, 5.0] if quick else None,
            resume=resume)
        results['alpha_sweep'] = self.experiment_alpha_sweep(
            num_runs=nr,
            alpha_values=[0.5, 2.75, 5.0] if quick else None,
            resume=resume)
        results['mu_sweep'] = self.experiment_mu_sweep(
            num_runs=nr,
            mu_values=[0.05, 0.5, 0.95] if quick else None,
            resume=resume)
        results['joint_eps_beta'] = self.experiment_joint_eps_beta(
            num_runs=nr,
            eps_values=[0.10, 0.30, 0.50] if quick else None,
            beta_values=[2.0, 6.0, 10.0] if quick else None,
            resume=resume)
        results['joint_p_beta'] = self.experiment_joint_p_beta(
            num_runs=nr,
            p_values=[0.0, 0.5, 1.0] if quick else None,
            beta_values=[0.0, 5.0, 10.0] if quick else None,
            resume=resume)
        results['size_robust'] = self.experiment_size_robustness(
            num_runs=nr, N_sizes=[100, 300] if quick else None,
            resume=resume)
        results['m_robust'] = self.experiment_m_robustness(
            num_runs=nr, resume=resume)
        results['init_dist'] = self.experiment_init_distribution(
            num_runs=nr, resume=resume)

        elapsed = time.time() - t_total
        print(f"\n{'#' * 70}")
        print(f"  全部实验完成！总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"{'#' * 70}")
        return results
