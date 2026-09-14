"""Bounded audit: exact discrete release, viable actions, and stop completion."""
import json,sys,math
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from curve_layer import reserve,brake_bound,backup_distance,interval,project,signal_project

ROOT=Path(__file__).resolve().parent.parent

def main():
    rng=np.random.default_rng(913);count=0;monotonic=0
    for v in np.r_[np.linspace(0,5,201),np.linspace(5,27,50)]:
        b=brake_bound(v)
        f=.5*sum(max(b-i,0.) for i in range(4))
        assert f<=v+1e-9
        if b<3.5-1e-8:
            f2=.5*sum(max(b+1e-6-i,0.) for i in range(4));assert f2>v
        count+=1
    for _ in range(400):
        v=float(rng.uniform(0,27));a=float(rng.uniform(-3.5,2.6))
        if reserve(v,a)<0:continue
        d=backup_distance(v,a)
        assert math.isfinite(d)
        lower=max(-3.5,a-1,-brake_bound(v));upper=min(2.6,a+1)
        vals=[]
        for z in np.linspace(lower,upper,15):
            vn=v+.5*z;vals.append((v+vn)*.25+backup_distance(vn,z))
        assert np.all(np.diff(vals)>=-1e-7),(v,a,vals)
        monotonic+=1
        remaining=d+float(rng.uniform(.001,50))
        bd,info=interval(v,a,[(remaining,0.)]);assert bd is not None
        acc,inf=project(v,a,float(rng.uniform(-3.5,2.6)),[(remaining,0.)],0.)
        vn=v+.5*acc;xn=(v+vn)*.25
        assert abs(acc-a)<=1+1e-8 and vn>=-1e-8
        assert xn+backup_distance(vn,acc)<=remaining+2e-7
        count+=1
    # Resolve solver failures in the offline witnesses as a linear feasibility task.
    p=ROOT/'reports/curve_witness.json';w=json.loads(p.read_text());fail=[]
    for item in w['witnesses']:
        if item['success']:continue
        n=round(item['duration_s']/.5);v0=item['start_v'];a0=item['start_a'];D=item['target_x']-item['start_x']
        L=.5*np.tril(np.ones((n,n)));diff=np.eye(n)-np.eye(n,k=-1);shift=np.zeros(n);shift[0]=a0
        eq=np.stack([.5*np.ones(n),.25*(n-.5-np.arange(n)),np.eye(n)[-1]])
        beq=np.array([-v0,D-v0*n*.5,0.]);A=np.concatenate([L,-L,diff,-diff]);b=np.r_[np.full(n,80/3.6-v0),np.full(n,v0),shift+1,1-shift]
        ans=linprog(np.zeros(n),A_ub=A,b_ub=b,A_eq=eq,b_eq=beq,bounds=[(-3.5,2.6)]*n,method='highs')
        fail.append(dict(arm=item['arm'],condition_id=item['condition_id'],stop_index=item['stop_index'],lp_status=int(ans.status),message=ans.message))
    result=dict(passed=True,release_and_action_checks=count,backup_monotonic_cases=monotonic,
                offline_failed_witness_lp=fail,scope='Numerical audit of the declared discrete implementation; not a proof for arbitrary plants.')
    # Close the positive-speed terminal condition and changing-green edge cases.
    positive_cases=0
    for _ in range(150):
        v=float(rng.uniform(9.73,27));a=float(rng.uniform(-3.5,2.6))
        lo=max(-3.5,a-1,-brake_bound(v));hi=min(2.6,a+1)
        vals=[]
        for acc in np.linspace(lo,hi,21):
            vn=v+.5*acc;vals.append((v+vn)*.25+backup_distance(vn,acc,35/3.6))
        assert np.all(np.diff(vals)>=-1e-7),(v,a,vals)
        positive_cases+=1
    signal_episodes=0;signal_steps=0
    for _ in range(60):
        v=float(rng.uniform(5,22));a=0.;distance=backup_distance(v,a)+float(rng.uniform(5,140))
        x=0.;end_green=float(rng.uniform(.1,20));t=0.
        for k in range(300):
            green=t<end_green
            targets=[] if green else [(max(distance-2-x,0.),0.)]
            vmax=80/3.6
            bounds,inf=interval(v,a,targets,vmax=vmax);assert bounds is not None,(v,a,distance-x,t,end_green)
            cmd=float(rng.uniform(-3.5,2.6))
            action=min(max(cmd,bounds[0]),bounds[1])
            if green:
                action,info=signal_project(v,a,cmd,bounds,distance-x,end_green-t,vmax)
                assert action is not None,(v,a,distance-x,t,end_green)
            vn=v+.5*action;xn=x+(v+vn)*.25
            assert vn>=-1e-7 and vn<=vmax+1e-7 and abs(action-a)<=1+1e-7
            if xn>=distance:
                cross=2*(distance-x)/max(v+np.sqrt(max(v*v+2*action*(distance-x),0)),1e-12)
                assert t+cross<end_green,(t,cross,end_green)
                break
            x,v,a,t=xn,max(vn,0.),action,t+.5;signal_steps+=1
            if not green and v<1e-8 and abs(a)<1e-8:break
        else:raise AssertionError('Signal audit did not stop or clear')
        signal_episodes+=1
    result.update(positive_speed_monotonic_cases=positive_cases,changing_signal_episodes=signal_episodes,changing_signal_steps=signal_steps)
    (ROOT/'audit/layer_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
