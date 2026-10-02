# ============================================================
# 稳定版运行入口
# ============================================================
"""
运行实验并生成 JSON 结果。

推荐先做快速测试：
    python run_all.py --quick --only network hysteresis
    python run_all.py --plot

完整运行（必须显式确认）：
    python run_all.py --all

只运行指定实验：
    python run_all.py --only hysteresis
    python run_all.py --only network ablation epsilon

运行实验后立刻生成图片：
    python run_all.py --quick --only hysteresis --plot

说明：命令行默认 workers=4；可用 --workers K 调整 Windows spawn-safe 多进程数。
每条状态继承路径内部仍保持顺序，只有独立重复或独立参数块并行。
"""

import argparse
import os
import time
import config
import experiments as experiments_module
from config import NETWORK, SIMULATION
from experiments import ExperimentManager


def build_parser():
    parser = argparse.ArgumentParser(description="Run structural confinement experiments.")
    parser.add_argument(
        "--quick", action="store_true",
        help=("仅用于冒烟测试：默认设 N=200、num_runs=3、主实验和扫描均为2 sweeps，"
              "并缩减参数网格；显式命令行覆盖值优先"))
    parser.add_argument("--plot", "--figures", action="store_true", help="实验结束后生成图片")
    parser.add_argument("--runs", type=int, default=None, help="覆盖 SIMULATION['num_runs']，例如 --runs 10")
    parser.add_argument(
        "--workers", type=int, default=4,
        help=("Number of worker processes for independent simulations; "
              "default 4; use 1 for serial troubleshooting."),
    )
    parser.add_argument("--main-sweeps", type=float, default=None,
                        help="覆盖所有主实验的 Monte Carlo sweeps")
    parser.add_argument("--steps", type=int, default=None,
                        help="兼容旧命令：换算为 steps/N 个主实验 sweeps")
    parser.add_argument("--N", type=int, default=None, help="覆盖网络节点数")
    parser.add_argument("--sweep-sweeps", type=float, default=None,
                        help="覆盖 epsilon/磁滞扫描每个 epsilon 点的 Monte Carlo sweeps，例如 --sweep-sweeps 120")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="所有实验禁用断点续跑，按当前配置重新计算并覆盖对应结果")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印有效配置和预计规模，不运行实验")
    parser.add_argument(
        "--output-dir", default=None,
        help=("覆盖输出根目录；quick模式默认使用 output/smoke，"
              "避免覆盖正式结果"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--only", nargs="*", default=None,
        help=(
            "只运行指定实验。可选：network ablation epsilon hysteresis beta alpha mu "
            "perturbation update_mode community vector empirical joint_eps_beta joint_p_beta size m init"
        ),
    )
    selection.add_argument(
        "--all", action="store_true",
        help="显式运行全部17类实验；运行前建议先用相同选项加 --dry-run")
    return parser


def configure_output_root(output_root):
    """Point config and already-imported experiment code at one output root."""
    root = os.path.abspath(output_root)
    data = os.path.join(root, 'data')
    figures = os.path.join(root, 'figures')
    config.OUTPUT_DIR = root
    config.DATA_DIR = data
    config.FIGURE_DIR = figures
    experiments_module.OUTPUT_DIR = root
    experiments_module.DATA_DIR = data
    experiments_module.FIGURE_DIR = figures
    return root


def estimate_elementary_updates(selected, quick, runs):
    """Return transparent, conservative workload estimates for dry runs."""
    aliases = {
        'exp7': 'network', 'exp6': 'ablation', 'eps': 'epsilon',
        'eps_sweep': 'epsilon', 'hysteresis_sweep': 'hysteresis',
        'perturb': 'perturbation', 'sigma': 'perturbation',
        'synchronous': 'update_mode', 'sync': 'update_mode',
        'sbm': 'community', 'community_conflict': 'community',
        '2d': 'vector', 'multidimensional': 'vector',
        'real': 'empirical', 'real_network': 'empirical',
        'size_robust': 'size', 'm_robust': 'm', 'init_dist': 'init',
    }
    names = [aliases.get(name, name) for name in selected]
    p_count = 5 if quick else len(NETWORK['p_values'])
    epsilon_count = 3 if quick else len(config.TRUST['epsilon_0_range'])
    beta_count = 3 if quick else len(config.DYNAMICS['beta_range'])
    alpha_count = 3 if quick else len(config.SENSITIVITY['alpha_range'])
    mu_count = 3 if quick else len(config.SENSITIVITY['mu_range'])
    main_updates = int(round(
        NETWORK['N'] * float(SIMULATION['main_sweeps'])))
    scan_updates = int(round(
        NETWORK['N'] * float(SIMULATION['sweep_sweeps'])))
    estimates = {
        'network': 0,
        'ablation': p_count * 6 * runs * main_updates,
        'epsilon': 3 * 2 * epsilon_count * runs * scan_updates,
        'hysteresis': p_count * 2 * epsilon_count * runs * scan_updates,
        # One forward scan is shared by every sigma branch; each sigma then
        # runs one backward scan.  Do not count a separate forward scan for
        # every perturbation amplitude.
        'perturbation': ((5 if quick else 3)
                         * (1 + (3 if quick else len(SIMULATION.get(
                             'perturb_sigma_values', []))))
                         * epsilon_count * runs * scan_updates),
        'update_mode': (5 if quick else 4) * 2 * runs * main_updates,
        'community': (2 if quick else 5) * 3 * runs * main_updates,
        'vector': (5 if quick else 4) * 3 * runs * main_updates,
        # Bundled empirical graphs have 34 + 18 + 15 nodes.
        'empirical': int(67 * float(SIMULATION['main_sweeps']) * 3 * runs),
        'beta': 3 * beta_count * runs * main_updates,
        'alpha': alpha_count * runs * main_updates,
        'mu': mu_count * runs * main_updates,
        'joint_eps_beta': ((9 if quick else
                            len(config.JOINT_SWEEP['eps_values'])
                            * len(config.JOINT_SWEEP['beta_values']))
                           * runs * main_updates),
        'joint_p_beta': ((9 if quick else
                          len(config.JOINT_SWEEP['p_beta_p_values'])
                          * len(config.JOINT_SWEEP['p_beta_beta_values']))
                         * runs * main_updates),
        'size': int((sum([100, 300]) if quick else
                     sum(config.SENSITIVITY['N_sizes']))
                    * SIMULATION.get('size_sweeps', 200) * runs),
        'm': len(config.SENSITIVITY['m_values']) * runs * main_updates,
        'init': 3 * runs * main_updates,
    }
    if 'all' in names:
        names = list(estimates)
    return {name: estimates[name] for name in names if name in estimates}


def run_selected(manager, names, quick=False, runs=None, resume=True):
    nr = (3 if runs is None else int(runs)) if quick else runs
    nv = 1 if quick else None
    quick_p_values = [0.0, 0.25, 0.5, 0.75, 1.0] if quick else None
    quick_eps_values = (
        [float(config.EPSILON_0_GRID[index])
         for index in (0, len(config.EPSILON_0_GRID) // 2,
                       len(config.EPSILON_0_GRID) - 1)]
        if quick else None
    )
    results = {}
    tasks = {
        "network": lambda: manager.experiment_network_validation(
            num_networks=nv, p_values=quick_p_values, resume=resume),
        "ablation": lambda: manager.experiment_ablation(
            num_runs=nr, p_values=quick_p_values if quick else None,
            resume=resume),
        "epsilon": lambda: manager.experiment_epsilon_sweep(
            num_runs=nr,
            p_values=quick_p_values[:3] if quick else None,
            eps_values=quick_eps_values,
            resume=resume),
        "hysteresis": lambda: manager.experiment_hysteresis_sweep(
            num_runs=nr,
            p_values=quick_p_values if quick else None,
            eps_values=quick_eps_values,
            resume=resume),
        "perturbation": lambda: manager.experiment_perturbation_sweep(
            num_runs=nr,
            p_values=quick_p_values if quick else None,
            sigma_values=[0.0, 0.02, 0.10] if quick else None,
            eps_values=quick_eps_values,
            resume=resume),
        "update_mode": lambda: manager.experiment_update_mode_control(
            num_runs=nr,
            p_values=quick_p_values if quick else None,
            resume=resume),
        "community": lambda: manager.experiment_sbm_community_conflict(
            num_runs=nr,
            mixing_values=[0.05, 0.20] if quick else None,
            resume=resume),
        "vector": lambda: manager.experiment_vector_opinions(
            num_runs=nr,
            p_values=quick_p_values if quick else None,
            resume=resume),
        "empirical": lambda: manager.experiment_empirical_topologies(
            num_runs=nr, resume=resume),
        "beta": lambda: manager.experiment_beta_sweep(
            num_runs=nr,
            beta_values=[0.0, 2.5, 5.0] if quick else None,
            resume=resume),
        "alpha": lambda: manager.experiment_alpha_sweep(
            num_runs=nr,
            alpha_values=[0.5, 2.75, 5.0] if quick else None,
            resume=resume),
        "mu": lambda: manager.experiment_mu_sweep(
            num_runs=nr,
            mu_values=[0.05, 0.5, 0.95] if quick else None,
            resume=resume),
        "joint_eps_beta": lambda: manager.experiment_joint_eps_beta(
            num_runs=nr,
            eps_values=[0.10, 0.30, 0.50] if quick else None,
            beta_values=[2.0, 6.0, 10.0] if quick else None,
            resume=resume),
        "joint_p_beta": lambda: manager.experiment_joint_p_beta(
            num_runs=nr,
            p_values=[0.0, 0.5, 1.0] if quick else None,
            beta_values=[0.0, 5.0, 10.0] if quick else None,
            resume=resume),
        "size": lambda: manager.experiment_size_robustness(
            num_runs=nr,
            N_sizes=[100, 300] if quick else None,
            resume=resume),
        "m": lambda: manager.experiment_m_robustness(
            num_runs=nr, resume=resume),
        "init": lambda: manager.experiment_init_distribution(
            num_runs=nr, resume=resume),
    }
    aliases = {
        "exp7": "network", "exp6": "ablation", "eps": "epsilon",
        "eps_sweep": "epsilon", "hysteresis_sweep": "hysteresis",
        "perturb": "perturbation", "sigma": "perturbation",
        "synchronous": "update_mode", "sync": "update_mode",
        "sbm": "community", "community_conflict": "community",
        "2d": "vector", "multidimensional": "vector",
        "real": "empirical", "real_network": "empirical",
        "size_robust": "size", "m_robust": "m", "init_dist": "init",
    }
    resolved = [aliases.get(name, name) for name in names]
    unknown = [name for name in resolved if name not in tasks]
    if unknown:
        raise SystemExit(f"未知实验名: {unknown}. 可选: {list(tasks.keys())}")

    t0 = time.time()
    for name in resolved:
        print("\n" + "#" * 70)
        print(f"  RUN: {name}")
        print("#" * 70)
        results[name] = tasks[name]()
    elapsed = time.time() - t0
    print(f"\n完成所选实验，总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return results


if __name__ == "__main__":
    args = build_parser().parse_args()

    if not args.only and not args.all:
        raise SystemExit(
            "安全停止：请用 --only <实验名...> 选择实验，或用 --all 显式确认全部实验。"
            "长任务启动前建议先添加 --dry-run 核对规模。")

    if args.N is not None:
        NETWORK['N'] = int(args.N)
        print(f"[CONFIG] NETWORK['N'] = {NETWORK['N']}")
    if args.main_sweeps is not None:
        SIMULATION['main_sweeps'] = float(args.main_sweeps)
        print(f"[CONFIG] SIMULATION['main_sweeps'] = {SIMULATION['main_sweeps']}")
    if args.steps is not None:
        SIMULATION['num_steps'] = int(args.steps)
        SIMULATION['main_sweeps'] = float(args.steps) / max(NETWORK['N'], 1)
        print("[DEPRECATED] --steps converted to "
              f"SIMULATION['main_sweeps'] = {SIMULATION['main_sweeps']}")
    if args.sweep_sweeps is not None:
        SIMULATION['sweep_sweeps'] = float(args.sweep_sweeps)
        print(f"[CONFIG] SIMULATION['sweep_sweeps'] = {SIMULATION['sweep_sweeps']}")
    if args.runs is not None:
        SIMULATION['num_runs'] = int(args.runs)
        print(f"[CONFIG] SIMULATION['num_runs'] = {SIMULATION['num_runs']}")

    if args.quick:
        if args.N is None:
            NETWORK['N'] = 200
        if args.main_sweeps is None and args.steps is None:
            SIMULATION['main_sweeps'] = 2.0
        if args.sweep_sweeps is None:
            SIMULATION['sweep_sweeps'] = 2.0
        print(f"[QUICK] N={NETWORK['N']}, main_sweeps={SIMULATION['main_sweeps']}, "
              f"scan_sweeps={SIMULATION['sweep_sweeps']}")

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    print(f"[PARALLEL] workers={args.workers}")

    if args.output_dir is not None:
        output_root = configure_output_root(args.output_dir)
    elif args.quick:
        output_root = configure_output_root(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'output', 'smoke'))
    else:
        output_root = configure_output_root(config.OUTPUT_DIR)
    print(f"[OUTPUT] {output_root}")

    if args.dry_run:
        selected = args.only if args.only else ['all']
        effective_runs = (int(args.runs) if args.runs is not None else
                          3 if args.quick else
                          int(SIMULATION['num_runs']))
        main_updates = int(round(
            NETWORK['N'] * float(SIMULATION['main_sweeps'])))
        scan_updates = int(round(
            NETWORK['N'] * float(SIMULATION['sweep_sweeps'])))
        print("[DRY RUN] no experiment was executed")
        print(f"  selected={selected}")
        print(f"  N={NETWORK['N']}, m={NETWORK['m']}, runs={effective_runs}")
        print(f"  main_sweeps={SIMULATION['main_sweeps']} ({main_updates} updates/run)")
        print(f"  sweep_sweeps={SIMULATION['sweep_sweeps']} ({scan_updates} updates/epsilon point)")
        print(f"  resume={not args.no_resume}, plot={args.plot}")
        print(f"  output_root={output_root}")
        print(f"  workers={args.workers}")
        estimates = estimate_elementary_updates(
            selected, args.quick, effective_runs)
        if estimates:
            print("  estimated elementary updates:")
            for name, count in estimates.items():
                print(f"    {name}: {count:,}")
            print(f"    TOTAL: {sum(estimates.values()):,}")
        raise SystemExit(0)

    manager = ExperimentManager(workers=args.workers)

    if args.only:
        run_selected(manager, args.only, quick=args.quick, runs=args.runs, resume=(not args.no_resume))
    elif args.all:
        manager.run_all(
            quick=args.quick, resume=(not args.no_resume),
            num_runs=args.runs)

    if args.plot:
        from plotting import main as plot_main
        plot_main([])
