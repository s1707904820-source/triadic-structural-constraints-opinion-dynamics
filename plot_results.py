#!/usr/bin/env python3
"""兼容入口：历史文件名 plot_results.py。现在统一转发到 plotting.py。"""
try:
    from .plotting import *  # type: ignore
    from .plotting import main  # type: ignore
except ImportError:  # pragma: no cover
    from plotting import *  # type: ignore
    from plotting import main  # type: ignore

if __name__ == "__main__":
    main()
