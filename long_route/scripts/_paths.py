"""公共路径设置：以只读方式导入冻结的 evsim_v9 模拟器/训练器，不修改原代码。"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVSIM = ROOT / 'evsim_v9'
REFERENCE = ROOT / 'reference'
RESULTS = ROOT / 'results'


def use_long_route():
    """冻结代码在导入时读取 EVSIM_ROUTE；20 km 长程路线要求该变量未设置（设为 mini 会切到 4 km）。"""
    os.environ.pop('EVSIM_ROUTE', None)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    if str(EVSIM) not in sys.path:
        sys.path.insert(0, str(EVSIM))
    os.chdir(EVSIM)
