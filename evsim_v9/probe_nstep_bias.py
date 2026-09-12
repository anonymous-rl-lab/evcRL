"""Frozen-policy probe of uncorrected noisy n-step returns.

At each visited state, keep the initial action fixed. Compare the exact
noise-free return with 19 noisy continuation actions followed by an exact
noise-free Monte Carlo tail. This replaces the learned bootstrap with a real
rollout and isolates behavior/target-policy mismatch. It does not prove the
cause or size of the trained critic's overall bias.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import env20 as E
import td3_run as T


def action(actor, obs):
    with torch.no_grad():return float(actor(torch.as_tensor(obs).unsqueeze(0))[0,0])


def value(actor, state, noisy_decisions, rng):
    e=copy.deepcopy(state);o=e.obs();ret=0.;ou=0.;k=0
    while True:
        u=action(actor,o)
        if 1<=k<noisy_decisions:
            ou=.85*ou+.2*rng.normal();u+=ou
        o,r,d,_=T.env_step(e,float(np.clip(u,-1,1)),4);ret+=r;k+=1
        if d:return ret


def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path)
    ap.add_argument('--trials',type=int,default=8);ap.add_argument('--output',type=Path,default=Path('out/nstep_probe.json'))
    a=ap.parse_args();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    E.CURVE_MODE='envelope';E.REWARD_MODE='L';E.SHAPE_C=0.
    actor=T.MLP(E.OBS_DIM,1,True);actor.load_state_dict(ck['actor']);actor.eval()
    e=E.Route20(.85,288.15,offsets=[0.,0.]);o=e.reset()
    targets=[1000,6000,10000,14000,18000];states=[]
    while targets:
        if e.x>=targets[0]:states.append(copy.deepcopy(e));targets.pop(0)
        o,_,d,_=T.env_step(e,action(actor,o),4)
        if d:break
    rng=np.random.default_rng(20260909);rows=[]
    for st in states:
        true=value(actor,st,0,rng)
        vals=[value(actor,st,20,rng) for _ in range(a.trials)]
        rows.append(dict(x=st.x,true_Q=true,noisy_20step_targets=vals,
                         mean_shift=float(np.mean(vals)-true),sd=float(np.std(vals,ddof=1))))
        print(f'x={st.x:.0f}: true={true:.3f}, noisy-20 target shift={rows[-1]["mean_shift"]:+.3f}',flush=True)
    a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint),trials=a.trials,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__':main()
