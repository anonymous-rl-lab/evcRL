"""Frozen-actor signal-information experiment; no training code is called."""
import argparse, copy, hashlib, json, os, pickle, random, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'deps'))
import study as S
import curve_layer as C

MODES=('color','timing')
RANGE=1000.

def frozen_files():
    return sorted(p for d in ('code','deps','weights','protocol')
                  for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts)

def source_hash():
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):S.digest(p)
        for p in frozen_files()},sort_keys=True).encode()).hexdigest()

def load_actor():
    actor=S.MLP(13,True)
    actor.load_state_dict(torch.load(ROOT/'weights/actor_A.pt',map_location='cpu',weights_only=True)['actor'])
    actor.eval().requires_grad_(False)
    return actor

def execute(v,a,cmd,targets,fallback,signal):
    """Pure executor. No global clock, phase, environment, or signal offset."""
    action,info=C.project(v,a,cmd,targets,fallback,vmax=S.R.V_FREE)
    action=min(action,C.brake_bound(max(S.R.V_FREE-v,0.)))
    info=dict(info)
    info['timing_test']=False
    if not info['fallback'] and signal['green'] is True and signal['remaining'] is not None:
        info['timing_test']=True
        out,extra=C.signal_project(v,a,cmd,(info['lower'],info['upper']),
            signal['distance'],signal['remaining'],S.R.V_FREE)
        info.update(extra)
        if out is None:
            info['fallback']=True
            action=fallback
        else:
            action=out
    return float(action),info

class ProbeEnv(S.StudyEnv):
    def __init__(self,soc,temp,offset,mode):
        assert mode in MODES
        assert len(S.R.SIGNALS)==1,'This experiment is restricted to the one-signal mini route.'
        self.mode=mode
        self.layer_log=[]
        super().__init__(soc,temp,offset)

    def signal_view(self,mode):
        distance=S.R.SIGNALS[0]-self.x
        in_range=0.<distance<=RANGE
        green=bool(S.R.signal_green(S.R.SIGNALS[0],self.t,self.offsets[0])) if in_range else None
        remaining=float(S.R.time_to_change(S.R.SIGNALS[0],self.t,self.offsets[0])) if in_range and mode=='timing' else None
        return dict(distance=float(distance),green=green,remaining=remaining)

    def _stop_targets(self):
        targets=[]
        # Both arms use exactly the same current-color visibility and static map.
        for k,xs in enumerate(S.R.SIGNALS):
            if 0.<xs-self.x<=RANGE and not S.R.signal_green(xs,self.t,self.offsets[k]):
                targets.append((xs-S.E.STOP_MARGIN,0.))
        for ca,cb,vc in S.R.CURVES:
            if cb>self.x-1e-9:
                targets.append((max(ca-S.E.MARGIN,self.x),vc,2.))
        targets.append((S.R.LENGTH,0.))
        return targets

    def project(self,cmd):
        fallback=super().project(cmd)
        targets=[(max(t[0]-self.x,0.),t[1]) for t in self._stop_targets()]
        actual_view=self.signal_view(self.mode)
        action,info=execute(self.v,self.a,cmd,targets,fallback,actual_view)
        # Read-only diagnostic computed AFTER actual action selection. It cannot
        # modify the command, actor, physics, or selected action.
        other='timing' if self.mode=='color' else 'color'
        shadow,shadow_info=execute(self.v,self.a,cmd,targets,fallback,self.signal_view(other))
        pair={self.mode:action,other:shadow}
        self.layer_log.append(dict(t=float(self.t),x=float(self.x),v=float(self.v),a_prev=float(self.a),
            command=float(cmd),applied=action,color_action=pair['color'],timing_action=pair['timing'],
            paired_difference=pair['timing']-pair['color'],fallback=bool(info['fallback']),
            timing_test=info['timing_test'],signal_reason=info.get('signal_reason'),
            green=actual_view['green'],countdown_visible=actual_view['remaining'] is not None,
            signal_distance=actual_view['distance'],remaining=actual_view['remaining']))
        return action

def extra_metrics(env):
    m=S.episode_metrics(env)
    z=np.asarray(env.trace);dt=z[:,1]-z[:,0];mid=(z[:,2]+z[:,3])/2
    mask=(mid>=2200)&(mid<3300)
    lim=np.where((z[:,3]>=1450-1e-9)&(z[:,3]<=1550+1e-9),35/3.6,80/3.6)
    over=np.maximum(z[:,5]-lim,0.)
    m.update(settled=bool(z[-1,22] and env.v<=1e-6 and abs(env.a)<=1e-6),
        local_overspeed_time_s=float(np.sum(dt[over>1e-6])),local_overspeed_max=float(over.max()),
        layer_fallback_steps=sum(r['fallback'] for r in env.layer_log),
        countdown_visible_steps=sum(r['countdown_visible'] for r in env.layer_log),
        timing_test_steps=sum(r['timing_test'] for r in env.layer_log),
        paired_action_difference_steps=sum(abs(r['paired_difference'])>1e-6 for r in env.layer_log),
        paired_action_difference_max=max(abs(r['paired_difference']) for r in env.layer_log),
        signal_Ij=float(np.sum(z[mask,9]**2*dt[mask])),signal_window_complete=bool(env.x>=3300),
        signal_time_s=float(dt[mask].sum()),signal_energy_Wh=float(z[mask,10].sum()/3600),
        signal_friction_Wh=float(z[mask,11].sum()/3600),
        signal_strong_braking_s=float(dt[mask&(z[:,8]<-2)].sum()),
        signal_peak_jerk=float(np.max(np.abs(z[mask,9]))) if mask.any() else None,
        signal_min_acceleration=float(np.min(z[mask,8])) if mask.any() else None,
        strong_braking_s=float(dt[z[:,8]<-2].sum()))
    return m

def atomic_pickle(p,state):
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(state,f,protocol=5);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def rollout(actor,condition,mode,checkpoint=None,fingerprint=None,stop_after=None):
    soc,temp,v0,offset=condition
    env=ProbeEnv(soc,temp,offset,mode)
    obs=env.reset(v0)
    wall_acc=0.;start=time.monotonic()
    if checkpoint is not None and checkpoint.exists():
        with checkpoint.open('rb') as f:s=pickle.load(f)
        assert s['fingerprint']==fingerprint and s['condition']==list(condition) and s['mode']==mode
        env.__dict__=s['environment'];obs=s['observation'];wall_acc=s['wall_s']
        random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'])
    finished=False
    while env.t<600:
        with torch.inference_mode():cmd=S.command(float(actor(torch.as_tensor(obs).unsqueeze(0))[0,0]))
        for _ in range(4):
            if env.x>=3999 and env.v<=.3:cmd=0.
            obs,r,done,info=env.step(cmd)
            finished=bool(info['red_crossing'] or (info['arrived'] and env.v<=1e-6 and abs(env.a)<=1e-6) or env.t>=600)
            if finished:break
        # Checkpoints are at decision boundaries: the next command is recomputed
        # from this observation, so no unrecorded command-hold remainder exists.
        if checkpoint is not None and (env.steps%128==0 or finished or (stop_after and env.steps>=stop_after)):
            atomic_pickle(checkpoint,dict(fingerprint=fingerprint,condition=list(condition),mode=mode,
                environment=env.__dict__,observation=obs,command_hold_remaining=0,
                wall_s=wall_acc+time.monotonic()-start,python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state()))
        if finished:break
        if stop_after and env.steps>=stop_after:return None,env
    m=extra_metrics(env);m['wall_s']=wall_acc+time.monotonic()-start
    return m,env

def run_stage(stage):
    fp=source_hash()
    freeze=json.loads((ROOT/'audit/freeze.json').read_text())
    assert freeze['source_hash']==fp,'Frozen code/protocol/weight changed.'
    assert json.loads((ROOT/'audit/audit.json').read_text())['passed']
    if stage=='pilot':assert json.loads((ROOT/'reports/smoke.json').read_text())['gate_passed']
    if stage=='reporting':
        assert json.loads((ROOT/'reports/pilot.json').read_text())['gate_passed']
        assert json.loads((ROOT/'reports/reporting_budget.json').read_text())['authorized_scope']=='2 configurations x 18 conditions; no training'
    split='reporting' if stage=='reporting' else 'development'
    ids=[0] if stage=='smoke' else list(range(len(S.conditions(split))))
    out=ROOT/'runs'/split;out.mkdir(exist_ok=True)
    pf=out/'progress.json'
    progress=json.loads(pf.read_text()) if pf.exists() else dict(source_hash=fp,rows=[])
    assert progress['source_hash']==fp
    completed={(r['mode'],r['condition_id']) for r in progress['rows']}
    actor=load_actor();start=time.monotonic();budget=600 if stage=='smoke' else 3600
    for i in ids:
        for mode in MODES:
            if (mode,i) in completed:continue
            if time.monotonic()-start>=budget or (ROOT/'STOP').exists():return
            condition=S.conditions(split)[i];stem=f'{mode}_{i:02d}'
            cp=out/(stem+'.resume.pkl')
            m,env=rollout(actor,condition,mode,cp,fp)
            m.update(mode=mode,condition_id=i,condition=list(condition))
            S.save_trace(out/(stem+'.npz'),env.trace)
            S.json_save(out/(stem+'_layer.json'),env.layer_log)
            progress['rows'].append(m);S.json_save(pf,progress)
            cp.unlink(missing_ok=True)
            print(json.dumps({k:m[k] for k in ('mode','condition_id','settled','signal_Ij','Ij','time_s','violations','layer_fallback_steps','paired_action_difference_steps','wall_s')}),flush=True)
    S.json_save(ROOT/'reports'/f'{stage}_runtime.json',dict(stage=stage,complete=True,wall_s=time.monotonic()-start,source_hash=fp))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('smoke','pilot','reporting'))
    run_stage(ap.parse_args().stage)
