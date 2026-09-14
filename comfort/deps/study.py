"""Paired 4 km Actor comfort regularization study; no visual encoder.

Scientific protocol: ../protocol/PROTOCOL.md. Only lambda changes after a
shared frozen-Actor prefix. A=0, B=.1. Checkpoints include the whole process.
"""
import os
os.environ['EVSIM_ROUTE'] = 'mini'  # Must precede the simulator imports.
import argparse
import copy
import hashlib
import json
import math
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
import env20 as E
import route20 as R
import models as M

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
E.REWARD_MODE='L'; E.SHAPE_C=0.; E.CURVE_MODE='envelope'
E.CURVE_ENVELOPE=False; E.CLAMP_LIMIT=False; E.BOOTSTRAP_MODE=False
E.CLOCK_OFF=False; E.CUTOFF_VMIN=None; E.OBS_DIM=13
ROOT=Path(__file__).resolve().parent.parent
PACKS=[(.85,288.15),(.90,263.15),(.95,263.15)]
WINDOWS={'launch':(0.,800.),'curve':(850.,1850.),'signal':(2200.,3300.),'terminal':(3500.,4000.)}
STOP=False
TRACE_COLS=['t0','t1','x0','x1','v0','v1','a_prev','a_cmd','a','jerk',
            'dE_J','dF_J','reward','soc','safe_ceiling','nominal_limited',
            'safe_override','jerk_override','red_crossing','signal_green',
            'signal_phase','deadline','arrived','physical_violation']

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,p)
def append_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def source_hash():
    names=['code/study.py','code/env20.py','code/route20.py','code/models.py',
           'code/plant.py','code/effcal.py','code/scn/arterial_mini_profile.npy',
           'protocol/PROTOCOL.md','protocol/initialization.json']
    return hashlib.sha256(json.dumps({n:digest(ROOT/n) for n in names},sort_keys=True).encode()).hexdigest()
def config(seed,**overrides):
    c=dict(seed=seed,obs_dim=13,hidden=64,repeat=4,replay_capacity=125000,
           nstep=20,gamma=1.,batch=256,warmup=4000,prefix_steps=40000,
           actor_lr=3e-5,critic_lr=3e-4,tau=.005,policy_delay=2,
           target_noise=.2,target_clip=.5,ou_sigma=.2,ou_theta=.15,
           lambda_c=0.,delta_a=1.,phase='prefix',initial_seed=seed)
    c.update(overrides);return c
def command(u):
    u=float(np.clip(u,-1.,1.));return u*(2.6 if u>0. else 3.5)
def command_tensor(u):return torch.where(u>0.,2.6*u,3.5*u)
def comfort_loss(ob,u):return (command_tensor(u)-3.5*ob[:,1:2]).square().mean()

class StudyEnv(E.Route20):
    def __init__(self,soc0=.85,T=288.15,offset=0.):
        super().__init__(soc0,T,offsets=[float(offset)],t_budget=480.)
    def reset(self,v0=16.):
        super().reset();self.v=float(v0);self._was_moving=self.v>=.1
        self.trace=[];return self.obs()
    def step(self,a_cmd):
        t,x,v,ap,eb,ef=self.t,self.x,self.v,self.a,self.e_batt,self.e_fric
        safe=self.a_safe();green=R.signal_green(3000.,t,self.offsets[0])
        a1=float(np.clip(a_cmd,ap-E.J_MAX*E.DT,ap+E.J_MAX*E.DT))
        a2=float(np.clip(a1,E.A_LO,E.A_HI))
        o,r,d,info=super().step(a_cmd)
        row=[t,self.t,x,self.x,v,self.v,ap,float(a_cmd),self.a,(self.a-ap)/E.DT,
             self.e_batt-eb,self.e_fric-ef,r,self.soc,float(safe),float(abs(a1-a_cmd)>1e-7),
             float(safe<a2-1e-7),float(abs(self.a-ap)>E.J_MAX*E.DT+1e-7),
             float(info['red_crossing']),float(green),(t+self.offsets[0])%90.,
             float(info['deadline']),float(info['arrived']),
             float(self.a<E.A_LO-1e-7 or self.a>E.A_HI+1e-7)]
        if not np.isfinite(row).all() or not np.isfinite(o).all():raise FloatingPointError('Environment nonfinite')
        self.trace.append(row);return o,r,d,info

def trace_metrics(trace):
    a=np.asarray(trace,dtype=np.float64)
    if a.ndim!=2 or len(a)==0:raise ValueError('empty trace')
    dt=a[:,1]-a[:,0];j=a[:,9];duration=float(dt.sum())
    if not np.allclose(j,(a[:,8]-a[:,6])/dt,rtol=0,atol=1e-12):raise AssertionError('jerk ledger')
    ij=float(np.sum(j*j*dt));mid=(a[:,2]+a[:,3])*.5
    windows={}
    assigned=np.zeros(len(a),dtype=bool)
    for name,(lo,hi) in WINDOWS.items():
        mask=(mid>=lo)&(mid<hi);assigned|=mask
        windows[name]=dict(Ij=float(np.sum(j[mask]**2*dt[mask])),
                           time_s=float(dt[mask].sum()),jerk_max=float(np.max(np.abs(j[mask]))) if mask.any() else None)
    windows['other']=dict(Ij=float(np.sum(j[~assigned]**2*dt[~assigned])),time_s=float(dt[~assigned].sum()),
                          jerk_max=float(np.max(np.abs(j[~assigned]))) if (~assigned).any() else None)
    if abs(sum(w['Ij'] for w in windows.values())-ij)>1e-8:raise AssertionError('window ledger')
    return dict(Ij=ij,jerk_rms=math.sqrt(ij/duration),jerk_p95=float(np.quantile(np.abs(j),.95)),
                jerk_max=float(np.max(np.abs(j))),a_min=float(np.min(a[:,8])),a_max=float(np.max(a[:,8])),
                time_s=duration,E_Wh=float(a[:,10].sum()/3600.),friction_Wh=float(a[:,11].sum()/3600.),
                R=float(a[:,12].sum()),nominal_limited_steps=int(a[:,15].sum()),
                safety_override_steps=int(a[:,16].sum()),jerk_override_steps=int(a[:,17].sum()),
                physical_violation_steps=int(a[:,23].sum()),stopped_time_s=float(dt[a[:,5]<.1].sum()),
                windows=windows,trace_steps=len(a))

def episode_metrics(env):
    m=trace_metrics(env.trace)
    expected=-env.e_batt/1e5-E.LAM_T*env.t-env.e_overspeed_pen-env.terminal_pen
    m['ledger_error']=m['R']-expected
    if abs(m['ledger_error'])>1e-7:raise AssertionError(('return ledger',m['ledger_error']))
    if abs(m['Ij']-env.jerk_sq*E.DT)>1e-7:raise AssertionError('independent jerk accumulation')
    if abs(m['E_Wh']-env.e_batt/3600.)>1e-7:raise AssertionError('energy ledger')
    m.update(arrived=bool(env.trace[-1][22]),deadline=bool(env.trace[-1][21]),
             violations=int(env.n_violation),overspeed_time_s=float(env.t_over),
             overspeed_max=float(env.v_over_max),safety_infeasible=int(env.safety_infeasible),
             end_x=float(env.x),end_v=float(env.v),stops=int(env.n_stop),
             soc_start=float(env.soc0),temperature_K=float(env.T),offset_s=float(env.offsets[0]),
             initial_speed=float(env.trace[0][4]),terminal_penalty=float(env.terminal_pen))
    # A descriptive latency only when stopped near the line at the start of green.
    arr=np.asarray(env.trace);eligible=np.flatnonzero((arr[:,2]>=2995.)&(arr[:,2]<3000.)&(arr[:,4]<.1)&(arr[:,19]>0)&(arr[:,14]>=0))
    latency=None
    if len(eligible):
        k=int(eligible[0]);moving=np.flatnonzero(arr[k:,5]>.1)
        if len(moving):latency=float(arr[k+int(moving[0]),1]-arr[k,0])
    m['green_start_latency_s']=latency
    return m

def save_trace(path,trace):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,values=np.asarray(trace,dtype=np.float64),columns=np.asarray(TRACE_COLS))
    os.replace(tmp,p)

class MLP(nn.Module):
    def __init__(self,inputs,tanh=False):
        super().__init__(); layers=[nn.Linear(inputs,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1)]
        if tanh:layers.append(nn.Tanh())
        self.f=nn.Sequential(*layers)
    def forward(self,x):return self.f(x)

class Replay:
    def __init__(self,capacity):
        self.capacity,self.ptr,self.size=capacity,0,0
        for name,dim in [('O',13),('U',1),('Y',1),('O2',13),('D',1)]:setattr(self,name,np.zeros((capacity,dim),np.float32))
        self.T=np.zeros(capacity,np.int64)
    def add(self,o,u,y,nxt,done,used):
        p=self.ptr
        self.O[p],self.U[p],self.Y[p],self.O2[p],self.D[p],self.T[p]=o,u,y,nxt,done,used
        self.ptr=(p+1)%self.capacity;self.size=min(self.size+1,self.capacity)
    def batch(self,rng,n):
        idx=rng.integers(0,self.size,n)
        return [torch.as_tensor(a[idx]) for a in [self.O,self.U,self.Y,self.O2,self.D]]

class Trainer:
    NET_NAMES=('actor','actor_t','q1','q2','q1t','q2t')
    def __init__(self,cfg):
        self.cfg=copy.deepcopy(cfg);self.code_hash=source_hash()
        seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        for name,code in [('env_rng',11),('explore_rng',22),('replay_rng',33)]:
            setattr(self,name,np.random.default_rng(np.random.SeedSequence([seed,code])))
        self.actor,self.actor_t=MLP(13,True),MLP(13,True)
        self.q1,self.q2,self.q1t,self.q2t=[MLP(14) for _ in range(4)]
        ck=torch.load(ROOT/'initial'/f"s{cfg['initial_seed']}.pt",map_location='cpu',weights_only=True)
        self.actor.load_state_dict(ck['actor'])
        for net,tgt in [(self.actor,self.actor_t),(self.q1,self.q1t),(self.q2,self.q2t)]:tgt.load_state_dict(net.state_dict())
        self.oa=torch.optim.Adam(self.actor.parameters(),lr=cfg['actor_lr'])
        self.oq=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=cfg['critic_lr'])
        self.replay=Replay(cfg['replay_capacity']);self.pending=deque()
        self.used=self.decisions=self.updates=self.actor_updates=self.episodes=0
        self.branch_origin=None;self.branch_update_origin=0;self.ou=0.
        self.train_arrivals=self.train_failures=0
        self.elapsed_train_s=self.elapsed_eval_s=self.elapsed_save_s=0.
        self.history=[];self.next_eval=None;self.completed_pending=[]
        self.diag=dict(last_qloss=None,last_actor_task_loss=None,last_comfort_loss=None,actor_grad_max=0.,critic_grad_max=0.)
        self.env=self.new_env();self.obs=self.env.obs()
    @property
    def branch_steps(self):return self.used-(self.branch_origin if self.branch_origin is not None else 0)
    def new_env(self):
        soc,temp=PACKS[int(self.env_rng.integers(0,3))]
        v=float(self.env_rng.uniform(12.,20.));off=float(self.env_rng.uniform(0.,90.))
        env=StudyEnv(soc,temp,off);env.reset(v);return env
    def step(self,max_substeps=None):
        start=time.monotonic();c=self.cfg
        with torch.no_grad():u=float(self.actor(torch.as_tensor(self.obs).unsqueeze(0))[0,0])
        self.ou+=-c['ou_theta']*self.ou+c['ou_sigma']*self.explore_rng.normal()
        u=float(np.clip(u+self.ou,-1.,1.));reward=0.
        for _ in range(min(c['repeat'],max_substeps or c['repeat'])):
            nxt,r,done,info=self.env.step(command(u));reward+=r;self.used+=1
            if done:break
        # The 600-second bound is genuine termination; no comfort term enters reward.
        self.pending.append((self.obs.copy(),u,reward,nxt.copy(),float(done)))
        if len(self.pending)>=c['nstep'] or done:
            while self.pending:
                first,last=self.pending[0],self.pending[-1]
                self.replay.add(first[0],first[1],sum(x[2] for x in self.pending),last[3],last[4],self.used)
                self.pending.popleft()
                if not done:break
        self.obs=nxt;self.decisions+=1
        if done:
            self.episodes+=1;self.train_arrivals+=int(info['arrived']);self.train_failures+=int(not info['arrived'])
            self.completed_pending.append((self.episodes,episode_metrics(self.env),self.env.trace))
            self.env=self.new_env();self.obs=self.env.obs();self.ou=0.
            if self.pending:raise AssertionError('terminal n-step queue not flushed')
        if self.used>=c['warmup'] and self.replay.size>=c['batch']:self.update()
        self.elapsed_train_s+=time.monotonic()-start
    def update(self):
        c=self.cfg;ob,ub,rb,nxt,done=self.replay.batch(self.replay_rng,c['batch'])
        with torch.no_grad():
            noise=(torch.randn_like(ub)*c['target_noise']).clamp(-c['target_clip'],c['target_clip'])
            nu=(self.actor_t(nxt)+noise).clamp(-1.,1.);x2=torch.cat([nxt,nu],1)
            y=rb+(1.-done)*torch.minimum(self.q1t(x2),self.q2t(x2))
        x=torch.cat([ob,ub],1);q1,q2=self.q1(x),self.q2(x)
        loss=(q1-y).square().mean()+(q2-y).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('critic loss')
        self.oq.zero_grad();loss.backward()
        qg=float(torch.sqrt(sum(p.grad.square().sum() for p in self.q1.parameters())))
        if not math.isfinite(qg):raise FloatingPointError('critic gradient')
        self.oq.step();self.updates+=1;self.diag['last_qloss']=float(loss.detach())
        self.diag['critic_grad_max']=max(self.diag['critic_grad_max'],qg)
        if self.updates%c['policy_delay']==0:
            if c['phase']=='branch':
                u=self.actor(ob);task=-self.q1(torch.cat([ob,u],1)).mean();smooth=comfort_loss(ob,u)
                aloss=task+c['lambda_c']*smooth
                self.oa.zero_grad();aloss.backward()
                ag=float(torch.sqrt(sum(p.grad.square().sum() for p in self.actor.parameters())))
                if not math.isfinite(ag):raise FloatingPointError('actor gradient')
                self.oa.step();self.actor_updates+=1
                self.diag.update(last_actor_task_loss=float(task.detach()),last_comfort_loss=float(smooth.detach()),
                                 actor_grad_max=max(ag,self.diag['actor_grad_max']))
            with torch.no_grad():
                for net,tgt in [(self.actor,self.actor_t),(self.q1,self.q1t),(self.q2,self.q2t)]:
                    for p,pt in zip(net.parameters(),tgt.parameters()):pt.mul_(1-c['tau']).add_(p,alpha=c['tau'])
    def state(self):
        excluded=set(self.NET_NAMES)|{'oa','oq','env_rng','explore_rng','replay_rng','env','replay','pending'}
        s={k:copy.deepcopy(v) for k,v in self.__dict__.items() if k not in excluded}
        s['networks']={k:getattr(self,k).state_dict() for k in self.NET_NAMES}
        s['optimizers']={'actor':self.oa.state_dict(),'critics':self.oq.state_dict()}
        s['rng']={'python':random.getstate(),'numpy_global':np.random.get_state(),'torch':torch.get_rng_state(),
                  **{k:copy.deepcopy(getattr(self,k).bit_generator.state) for k in ('env_rng','explore_rng','replay_rng')}}
        s['environment']=copy.deepcopy(self.env.__dict__);s['replay_state']=self.replay.__dict__
        s['pending_state']=list(self.pending);s['runtime']={'torch':torch.__version__,'numpy':np.__version__};s['schema']=1
        return s
    def save(self,path):
        start=time.monotonic();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
        with tmp.open('wb') as f:torch.save(self.state(),f);f.flush();os.fsync(f.fileno())
        if p.exists():os.replace(p,p.with_name(p.stem+'.previous'+p.suffix))
        os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        self.elapsed_save_s+=time.monotonic()-start
    @classmethod
    def load(cls,path):
        s=torch.load(path,map_location='cpu',weights_only=False)
        if s['schema']!=1 or s['code_hash']!=source_hash():raise ValueError('source/protocol mismatch')
        if s['runtime']!={'torch':torch.__version__,'numpy':np.__version__}:raise ValueError('runtime mismatch')
        t=cls(s['cfg']);special={'networks','optimizers','rng','environment','replay_state','pending_state','runtime','schema'}
        for k,v in s.items():
            if k not in special:setattr(t,k,v)
        for k,v in s['networks'].items():getattr(t,k).load_state_dict(v)
        t.oa.load_state_dict(s['optimizers']['actor']);t.oq.load_state_dict(s['optimizers']['critics'])
        t.env.__dict__=s['environment'];t.replay.__dict__=s['replay_state'];t.pending=deque(s['pending_state'])
        for k in ('env_rng','explore_rng','replay_rng'):getattr(t,k).bit_generator.state=s['rng'][k]
        random.setstate(s['rng']['python']);np.random.set_state(s['rng']['numpy_global']);torch.set_rng_state(s['rng']['torch'])
        return t

def rule_action(env):
    o=env.obs();target=min(18.,float(o[3])*R.V_FREE)
    if o[4]<1.:target=min(target,math.sqrt((float(o[5])*R.V_FREE)**2+2*1.4*max(float(o[4])*E.PREVIEW-E.MARGIN,0.)))
    if o[7]==0.:target=min(target,math.sqrt(2*1.4*max(float(o[6])*E.SPAT_RANGE-E.STOP_MARGIN,0.)))
    target=min(target,math.sqrt(2*1.4*max(4000.-env.x,0.)))
    # Reference only for feasibility; use the known 2 s decision interval.
    return float(np.clip((target-env.v)/2.,-1.4,1.))

def rollout(actor,soc,temp,v0,offset):
    env=StudyEnv(soc,temp,offset);o=env.reset(v0)
    while True:
        with torch.no_grad():a=rule_action(env) if actor is None else command(float(actor(torch.as_tensor(o).unsqueeze(0))[0,0]))
        for _ in range(4):
            o,r,done,info=env.step(a)
            if done:break
        if done:break
    return episode_metrics(env),env.trace

def conditions(split):
    if split=='development':speeds=[16.];offsets=[0.,30.,60.]
    elif split=='reporting':speeds=[14.,18.];offsets=[15.,45.,75.]
    else:raise ValueError(split)
    return [(s,t,v,o) for s,t in PACKS for v in speeds for o in offsets]
def summarize(rows):
    good=[r for r in rows if r['arrived']]
    fields=['Ij','jerk_rms','jerk_p95','jerk_max','time_s','E_Wh','R','overspeed_time_s','overspeed_max']
    return dict(n=len(rows),completed=len(good),failures=len(rows)-len(good),
                violations=sum(r['violations'] for r in rows),
                physical_violations=sum(r['physical_violation_steps'] for r in rows),
                completed_mean={k:float(np.mean([r[k] for r in good])) if good else None for k in fields},
                all_condition_mean={k:float(np.mean([r[k] for r in rows])) for k in fields},
                max_jerk=max(r['jerk_max'] for r in rows),max_overspeed=max(r['overspeed_max'] for r in rows))
def evaluate(trainer,out,split='development',label=None):
    start=time.monotonic();rows=[];tag=label or f'step_{trainer.branch_steps}'
    for i,(s,t,v,o) in enumerate(conditions(split)):
        row,trace=rollout(trainer.actor,s,t,v,o);row['condition_id']=i;rows.append(row)
        save_trace(Path(out)/'evaluation_traces'/f'{tag}_{split}_{i:02d}.npz',trace)
    rec=dict(step=trainer.branch_steps,total_steps=trainer.used,split=split,label=tag,
             rows=rows,summary=summarize(rows),source_hash=trainer.code_hash)
    json_save(Path(out)/'evaluations'/f'{tag}_{split}.json',rec)
    trainer.history.append({k:v for k,v in rec.items() if k!='rows'})
    trainer.elapsed_eval_s+=time.monotonic()-start;return rec

def flush_episodes(t,out):
    for idx,row,trace in t.completed_pending:
        name=f'episode_{idx:06d}'
        save_trace(Path(out)/'training_traces'/f'{name}.npz',trace)
        json_save(Path(out)/'episodes'/f'{name}.json',dict(episode=idx,**row))
    t.completed_pending=[]
def status(t,reason,wall):
    return dict(state=reason,seed=t.cfg['seed'],phase=t.cfg['phase'],lambda_c=t.cfg['lambda_c'],
                used=t.used,branch_steps=t.branch_steps,decisions=t.decisions,updates=t.updates,
                actor_updates=t.actor_updates,episodes=t.episodes,train_arrivals=t.train_arrivals,
                train_failures=t.train_failures,replay_size=t.replay.size,replay_pointer=t.replay.ptr,
                pending_nstep=len(t.pending),train_wall_s=t.elapsed_train_s,eval_wall_s=t.elapsed_eval_s,
                save_wall_s=t.elapsed_save_s,session_wall_s=wall,diagnostics=t.diag,source_hash=t.code_hash)

def train_cli(a):
    global STOP
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);checkpoint=out/'resume.pt'
    if a.resume:t=Trainer.load(a.resume)
    elif a.fork:
        if checkpoint.exists():raise FileExistsError('Use resume')
        t=Trainer.load(a.fork)
        if t.cfg['phase']!='prefix' or t.used!=t.cfg['prefix_steps']:raise ValueError('prefix incomplete')
        t.cfg['phase']='branch';t.cfg['lambda_c']=a.lambda_c;t.branch_origin=t.used;t.branch_update_origin=t.updates
        t.history=[];t.next_eval=None;t.elapsed_train_s=t.elapsed_eval_s=t.elapsed_save_s=0.
    else:
        if checkpoint.exists():raise FileExistsError('Use resume')
        t=Trainer(config(a.seed))
    if t.cfg['seed']!=a.seed:raise ValueError('seed mismatch')
    if t.cfg['phase']=='branch' and t.cfg['lambda_c']!=a.lambda_c:raise ValueError('lambda mismatch')
    start=last_save=last_status=time.monotonic()
    if getattr(t,'eval_interval',None)!=a.eval_every:
        t.next_eval=((t.branch_steps//a.eval_every)+1)*a.eval_every
        t.eval_interval=a.eval_every
    t.next_eval=t.next_eval or a.eval_every
    def request_stop(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    t.save(checkpoint);json_save(out/'config.json',dict(config=t.cfg,planned_steps=a.steps,eval_every=a.eval_every,
                   wall_budget_s=a.seconds,source_hash=t.code_hash,initial=digest(ROOT/'initial'/f's{a.seed}.pt')))
    reason='step_budget'
    try:
        while t.branch_steps<a.steps:
            if STOP or (ROOT/'STOP').exists() or (out/'STOP').exists():reason='requested_stop';break
            if time.monotonic()-start>=a.seconds:reason='wall_budget';break
            t.step(min(4,a.steps-t.branch_steps));flush_episodes(t,out)
            now=time.monotonic()
            if now-last_status>=15:
                json_save(out/'status.json',status(t,'training',now-start));last_status=now
            if now-last_save>=60:t.save(checkpoint);last_save=time.monotonic()
            if t.cfg['phase']=='branch' and t.branch_steps>=t.next_eval:
                t.save(checkpoint);ev=evaluate(t,out)
                t.next_eval=((t.branch_steps//a.eval_every)+1)*a.eval_every;t.save(checkpoint);last_save=time.monotonic()
                print(json.dumps(dict(event='evaluation',step=t.branch_steps,summary=ev['summary'])),flush=True)
        flush_episodes(t,out);t.save(checkpoint)
        if reason=='step_budget' and t.cfg['phase']=='branch' and (not t.history or t.history[-1]['step']!=t.branch_steps):
            evaluate(t,out,label=f'endpoint_{t.branch_steps}');t.save(checkpoint)
        s=status(t,reason,time.monotonic()-start);json_save(out/'status.json',s)
        print(json.dumps(s),flush=True)
    except Exception as ex:
        json_save(out/'failure.json',dict(type=type(ex).__name__,message=str(ex),used=t.used));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,required=True)
    ap.add_argument('--seconds',type=float,default=3600.);ap.add_argument('--lambda-c',type=float,default=0.)
    ap.add_argument('--eval-every',type=int,default=100000);ap.add_argument('--resume');ap.add_argument('--fork')
    train_cli(ap.parse_args())
