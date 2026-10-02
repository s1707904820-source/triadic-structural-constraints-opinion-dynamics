"""
该文件只保留为命令行入口。核心逻辑已整合进 experiments.py:
    ExperimentManager.experiment_hysteresis_sweep()

运行方式：
    python hysteresis_sweep.py

输出：
    output/data/exp_hysteresis_sweep.json
    output/figures/fig_hysteresis_sweep.png
"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NETWORK, SIMULATION, DATA_DIR, FIGURE_DIR
from experiments import ExperimentManager


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    p_values = NETWORK['p_values']
    num_runs = SIMULATION.get('num_runs', 20)

    print("\n参数设置:")
    print(f"  p值数量: {len(p_values)} (范围: {min(p_values):.2f} ~ {max(p_values):.2f})")
    print(f"  每点运行次数: {num_runs}")
    print(f"  每个epsilon点扫描sweeps: {SIMULATION.get('sweep_sweeps')}")
    print(f"  后向扰动强度: {SIMULATION.get('backward_perturb_sigma', 0.0)}")
    print(f"  后向每步扰动: {SIMULATION.get('backward_perturb_each_step', True)}")

    manager = ExperimentManager()
    manager.experiment_hysteresis_sweep(
        num_runs=num_runs,
        p_values=p_values,
        save_figure=True,
    )

    print("\n磁滞扫描完成！")


if __name__ == '__main__':
    main()
