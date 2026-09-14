"""Measure this machine; do not assume twelve free cores or a fixed ETA."""
import os
import platform
from pathlib import Path
import shutil
import sys
import time
print(f'Python {platform.python_version()} ({sys.executable})')
try:
    import numpy as np
    import torch
except ImportError as err:
    raise SystemExit(f'{err}; install requirements.txt first')
print(f'NumPy {np.__version__}; PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}')
cpu=float(os.cpu_count() or 1)
try:
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if quota!='max':cpu=min(cpu,int(quota)/int(period))
except (OSError,ValueError):pass
mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
try:
    cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if cap!='max':mem=min(mem,int(cap))
except (OSError,ValueError):pass
print(f'Effective CPU quota {cpu:g}; memory limit {mem/2**30:.1f} GiB; disk free {shutil.disk_usage(".").free/2**30:.1f} GiB')
import env20 as E
import route20 as R
from td3_run import MLP
E.CURVE_MODE='envelope';E.REWARD_MODE='L'
e=E.Route20();e.reset();n=0;t0=time.monotonic()
while time.monotonic()-t0<2:
    _,_,done,_=e.step(1. if n%40<20 else -1.);n+=1
    if done:e.reset()
rate=n/(time.monotonic()-t0)
torch.set_num_threads(1)
net=MLP(E.OBS_DIM+1,1);opt=torch.optim.Adam(net.parameters(),lr=3e-4)
x=torch.randn(256,E.OBS_DIM+1);n=0;t0=time.monotonic()
while time.monotonic()-t0<2:
    loss=net(x).square().mean();opt.zero_grad();loss.backward();opt.step();n+=1
updates=n/(time.monotonic()-t0)
print(f'Raw integration {rate:.0f} steps/s; one-critic update {updates:.0f}/s')
print('Use a measured 40k-step smoke log for full-TD3 ETA; evaluation and concurrent jobs add overhead.')
print('Default run_all.sh: ONE 40k-step run, not a detached 12-seed experiment.')
print(f'Default replay: 125,000 decision transitions = about 500,000 integration steps; {(2*E.OBS_DIM+3)*4*125000/2**20:.1f} MiB plus metadata.')
