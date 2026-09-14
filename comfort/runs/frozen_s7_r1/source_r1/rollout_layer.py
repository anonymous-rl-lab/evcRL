"""Frozen-policy development probe. Resumes at completed-condition boundaries."""
import json,sys,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT.parent/'TIV_comfort_v1';sys.path.insert(0,str(BASE/'code'))
import study as S
import curve_layer as C

class CurveEnv(S.StudyEnv):
    def __init__(self,*args,**kw):
        self.layer_log=[]
        super().__init__(*args,**kw)
    def project(self,cmd):
        fallback=super().project(cmd)
        targets=[(max(t[0]-self.x,0.),t[1]) for t in self._stop_targets()]
        result,info=C.project(self.v,self.a,cmd,targets,fallback)
        # Preserve the original emergency upper free-flow bound, with a
        # jerk-aware velocity margin as for the lower zero-speed bound.
        limit=S.R.V_FREE+S.E.V_OVER_HARD
        result=min(result,C.brake_bound(max(limit-self.v,0.)))
        if not info['fallback']:
            for k,xs in enumerate(S.R.SIGNALS):
                if self.x<xs and S.R.signal_green(xs,self.t,self.offsets[k]):
                    remaining=S.R.time_to_change(xs,self.t,self.offsets[k])
                    action,extra=C.signal_project(self.v,self.a,cmd,
                        (info['lower'],min(info['upper'],C.brake_bound(max(limit-self.v,0.)))),
                        xs-self.x,remaining,limit)
                    info.update(extra)
                    if action is None:
                        info['fallback']=True;result=fallback
                    else:result=action
        info.update(x=self.x,v=self.v,a_prev=self.a,command=float(cmd),applied=float(result))
        self.layer_log.append(info)
        return result

def digest():
    return hashlib.sha256(''.join((ROOT/p).read_text() for p in ['code/curve_layer.py','code/rollout_layer.py','protocol/PROBE_v2.md','protocol/IMPLEMENTATION_FIXES.md']).encode()).hexdigest()

def rollout(actor,condition):
    soc,temp,v0,offset=condition;env=CurveEnv(soc,temp,offset);obs=env.reset(v0)
    settled=False;violation=False
    while env.t<600:
        with torch.no_grad():cmd=S.command(float(actor(torch.as_tensor(obs).unsqueeze(0))[0,0]))
        for _ in range(4):
            if env.x>=3999 and env.v<=.3:cmd=0.
            obs,r,done,info=env.step(cmd)
            violation=bool(info['red_crossing'])
            settled=bool(info['arrived'] and env.v<=1e-6 and abs(env.a)<=1e-6)
            if violation or settled or env.t>=600:break
        if violation or settled:break
    metrics=S.episode_metrics(env);metrics['settled']=settled
    metrics['layer_fallback_steps']=sum(x['fallback'] for x in env.layer_log)
    a=np.asarray(env.trace);lim=np.where((a[:,3]>=1450-1e-9)&(a[:,3]<=1550+1e-9),35/3.6,80/3.6)
    excess=np.maximum(a[:,5]-lim,0.)
    metrics['local_overspeed_time_s']=float((excess>1e-6).sum()*.5)
    metrics['local_overspeed_max']=float(excess.max())
    return metrics,env

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','pilot'],required=True);args=ap.parse_args()
    audit=json.loads((ROOT/'audit/layer_audit.json').read_text());assert audit['passed']
    codehash=digest();out=ROOT/'runs/frozen_s7_r1';out.mkdir(parents=True,exist_ok=True)
    f=out/'progress.json'
    progress=json.loads(f.read_text()) if f.exists() else dict(source_hash=codehash,rows=[])
    assert progress['source_hash']==codehash,'Source changed after frozen probe started'
    completed={(r['arm'],r['condition_id']) for r in progress['rows']}
    if args.stage=='pilot':
        assert ('A',0) in completed and ('B',0) in completed
        assert all(r['settled'] and r['violations']==0 and r['physical_violation_steps']==0 for r in progress['rows'])
    began=time.monotonic();budget=600 if args.stage=='smoke' else 3600
    for arm in ['A','B']:
        trainer=S.Trainer.load(BASE/'runs/development/s7'/arm/'resume.pt')
        for i,c in enumerate(S.conditions('development')):
            if args.stage=='smoke' and i!=0:continue
            if (arm,i) in completed:continue
            if time.monotonic()-began>budget or (ROOT/'STOP').exists():return
            start=time.monotonic();m,env=rollout(trainer.actor,c)
            m.update(arm=arm,condition_id=i,wall_s=time.monotonic()-start)
            S.save_trace(out/f'{arm}_{i:02d}.npz',env.trace);S.json_save(out/f'{arm}_{i:02d}_layer.json',env.layer_log)
            progress['rows'].append(m);S.json_save(f,progress)
            print(json.dumps({k:m[k] for k in ['arm','condition_id','settled','Ij','jerk_max','time_s','E_Wh','violations','layer_fallback_steps','wall_s']}),flush=True)
    S.json_save(out/f'{args.stage}_stage.json',dict(completed=True,source_hash=codehash,wall_s=time.monotonic()-began,rows=len(progress['rows'])))

if __name__=='__main__':main()
