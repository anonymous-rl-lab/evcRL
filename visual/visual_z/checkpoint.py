"""Atomic checkpoint envelope; exact physical resume depends on backend state hooks."""
from pathlib import Path
import os,random
import numpy as np
import torch

REQUIRED_EXTERNAL={'environment','camera','collector','replay','frame_store','samplers','schedulers'}

def save_checkpoint(path,learner,external,identity):
    if not REQUIRED_EXTERNAL<=external.keys():raise ValueError('incomplete collector/camera/replay checkpoint')
    state={'schema':1,'identity':identity,'learner':learner.state_dict(),'external':external,
      'runtime':{'torch':torch.__version__,'numpy':np.__version__},
      'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
             'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    if p.exists():os.replace(p,p.with_name(p.name+'.previous'))
    os.replace(tmp,p)
    if os.name=='posix':
        fd=os.open(str(p.parent),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def load_checkpoint(path,learner,identity):
    # Load only trusted locally generated research checkpoints (contains Python RNG objects).
    s=torch.load(path,map_location=learner.device,weights_only=False)
    if s['schema']!=1 or s['identity']!=identity:raise ValueError('code/data/protocol identity mismatch')
    if s['runtime']!={'torch':torch.__version__,'numpy':np.__version__}:raise ValueError('runtime mismatch')
    if not REQUIRED_EXTERNAL<=s['external'].keys():raise ValueError('incomplete external state')
    learner.load_state_dict(s['learner'])
    random.setstate(s['rng']['python']);np.random.set_state(s['rng']['numpy']);torch.set_rng_state(s['rng']['torch'].cpu())
    if s['rng']['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in s['rng']['cuda']])
    return s['external']
