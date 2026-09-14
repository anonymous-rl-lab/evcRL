"""Fixed-time/fixed-distance kinematic witnesses; no RL training or model selection."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT.parent/'TIV_comfort_v1/code'))
import study as S

def profiles():
    v0,T=16.,12.;s=np.linspace(0,1,1201);t=s*T
    q=np.where(s<=.5,1-2*s*s,2*(1-s)**2)
    qa=np.where(s<=.5,-4*s,-4*(1-s))*v0/T
    qj=np.where(s<.5,-4,4)*v0/T**2
    c=1-3*s*s+2*s**3;ca=(-6*s+6*s*s)*v0/T;cj=(-6+12*s)*v0/T**2
    f=1-10*s**3+15*s**4-6*s**5;fa=(-30*s*s+60*s**3-30*s**4)*v0/T;fj=(-60*s+180*s*s-120*s**3)*v0/T**2
    data={'Piecewise quadratic':(v0*q,qa,qj,16*v0*v0/T**3),
          'Cubic':(v0*c,ca,cj,12*v0*v0/T**3),
          'Quintic':(v0*f,fa,fj,120/7*v0*v0/T**3)}
    rows=[]
    fig,axs=plt.subplots(3,1,figsize=(9,7.5),sharex=True,layout='constrained')
    for name,(v,a,j,ij) in data.items():
        assert abs(v[0]-v0)<1e-12 and abs(v[-1])<1e-12 and abs(a[0])+abs(a[-1])<1e-12
        assert abs(np.trapezoid(v,t)-96)<1e-8
        for ax,y in zip(axs,[v,a,j]):ax.plot(t,y,label=name,lw=1.7)
        rows.append(dict(profile=name,duration_s=T,distance_m=96.,peak_acceleration=float(abs(a).max()),
            peak_jerk=float(abs(j).max()),Ij_exact=ij,endpoint_jerks=[float(j[0]),float(j[-1])]))
    for ax,lab in zip(axs,['Speed (m/s)','Acceleration (m/s²)','Jerk (m/s³)']):
        ax.set_ylabel(lab);ax.grid(alpha=.2)
    axs[0].legend(ncol=3,fontsize=9);axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Same stop: 16 → 0 m/s, 12 s, 96 m, zero endpoint acceleration\nCurve shape trades peak jerk against integrated squared jerk',fontsize=11)
    fig.savefig(ROOT/'reports/velocity_profiles.png',dpi=180);fig.savefig(ROOT/'reports/velocity_profiles.pdf');plt.close(fig)
    return rows

def solve_clip(trace,end,seconds=12):
    dt=.5;n=int(seconds/dt);start=max(0,end+1-n);raw=trace[start:end+1];n=len(raw)
    v0=float(raw[0,4]);a0=float(raw[0,6]);x0=float(raw[0,2]);target=float(raw[-1,3]);D=target-x0
    # a[k] is interval acceleration, exactly matching the original simulator.
    L=np.tril(np.ones((n,n)));vel=dt*L
    dx=dt*dt*(n-.5-np.arange(n))
    diff=np.eye(n)-np.eye(n,k=-1);shift=np.zeros(n);shift[0]=a0
    eq=np.stack([dt*np.ones(n),dx,np.eye(n)[-1]])
    beq=np.array([-v0,D-v0*n*dt,0.])
    con=[LinearConstraint(eq,beq,beq),LinearConstraint(vel,-v0,80/3.6-v0),
         LinearConstraint(diff,shift-2*dt,shift+2*dt)]
    def fun(a):z=diff@a-shift;return float(z@z/dt)
    def jac(a):return 2*diff.T@(diff@a-shift)/dt
    began=time.monotonic()
    res=minimize(fun,raw[:,8],jac=jac,method='SLSQP',bounds=Bounds(-3.5,2.6),constraints=con,
                 options={'ftol':1e-9,'maxiter':300})
    acc=res.x;v=v0+vel@acc;x=x0+np.cumsum((np.r_[v0,v[:-1]]+v)*dt/2)
    eqerr=float(abs(eq@acc-beq).max());j=(diff@acc-shift)/dt
    valid=bool(res.success and eqerr<1e-6 and v.min()>-1e-6 and abs(j).max()<2+1e-6)
    result=dict(start_step=start,end_step=end,duration_s=n*dt,start_x=x0,target_x=target,start_v=v0,start_a=a0,
        success=valid,solver_message=str(res.message),old_Ij=float((raw[:,9]**2*dt).sum()),
        old_peak=float(abs(raw[:,9]).max()),witness_Ij=fun(acc),witness_peak=float(abs(j).max()),
        endpoint_max_error=eqerr,wall_s=time.monotonic()-began,
        interpretation='Fixed-time, fixed-position offline feasibility witness; not a learned or online controller.')
    return result,np.column_stack([np.arange(n)*dt,x,v,acc,j])

def main():
    rows=profiles();witnesses=[]
    for arm in ['A','B']:
        for p in sorted((ROOT/'reports/settled_traces').glob(f'{arm}_*.npz')):
            trace=np.load(p,allow_pickle=False)['values'];moving=False
            stops=[]
            for i,r in enumerate(trace):
                if r[5]>.1:moving=True
                if moving and r[5]<1e-9 and abs(r[8])<1e-9:
                    stops.append(i);moving=False
            for index,end in enumerate(stops):
                result,plan=solve_clip(trace,end);result.update(arm=arm,condition_id=int(p.stem.split('_')[1]),stop_index=index)
                witnesses.append(result)
                target=ROOT/'reports/fixed_time_witnesses';target.mkdir(exist_ok=True)
                np.savez_compressed(target/f'{p.stem}_stop{index}.npz',values=plan,
                    columns=np.array(['relative_t','x_end','v_end','a_interval','jerk']))
    obj=dict(profiles=rows,witnesses=witnesses,success=sum(w['success'] for w in witnesses),total=len(witnesses),
             equal_time_and_position=True,new_training_steps=0)
    (ROOT/'reports/curve_witness.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in obj.items() if k!='witnesses'},indent=2))

if __name__=='__main__':main()
