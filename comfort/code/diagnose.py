"""Read-only analysis of v1, including the unrecorded terminal settling step."""
import sys,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT   # archived v1 dependencies live inside this folder: deps/ (code), actor_evaluations/ (evaluations and traces)
sys.path.insert(0,str(BASE/'deps'))
import study as S

def velocity_reserve(v,a,dt=.5,j=2.):
    b=max(-a,0.)
    loss=dt*sum(max(b-k*j*dt,0.) for k in range(1,8))
    return float(v-loss)

def main():
    result={};windows={'launch':(0,800),'curve':(850,1850),'signal':(2200,3300),'terminal':(3500,5000)}
    for arm in ['A','B']:
        p=BASE/'actor_evaluations'
        saved=json.loads((p/f'{arm}_step_300000_development.json').read_text())
        rows=[]
        for rec in saved['rows']:
            i=rec['condition_id'];raw=np.load(p/'traces'/f'{arm}_step_300000_development_{i:02d}.npz')['values']
            dt=raw[:,1]-raw[:,0];mid=(raw[:,2]+raw[:,3])/2
            w={k:dict(time_s=float(dt[(mid>=lo)&(mid<hi)].sum()),Ij=float((raw[(mid>=lo)&(mid<hi),9]**2*.5).sum())) for k,(lo,hi) in windows.items()}
            overrides=[]
            for idx in np.flatnonzero(abs(raw[:,9])>2+1e-6):
                r=raw[idx]
                low=max(-3.5,-r[4]/.5);hi=min(2.6,r[14]);nominal_low=r[6]-1;nominal_hi=r[6]+1
                feasible=max(low,nominal_low)<=min(hi,nominal_hi)+1e-9
                overrides.append(dict(step=int(idx),x=float(r[2]),v=float(r[4]),a_prev=float(r[6]),a=float(r[8]),
                    jerk=float(r[9]),safe_ceiling=float(r[14]),physical_lower=float(low),
                    empty_current_interval=not bool(feasible),velocity_reserve=velocity_reserve(r[4],r[6]),
                    reason='zero_speed_boundary' if low>nominal_hi+1e-6 else ('safety_upper_bound' if hi<nominal_low-1e-6 else 'other')))
            # Restore actual final physical state; preserve the original trace unmodified.
            env=S.StudyEnv(rec['soc_start'],rec['temperature_K'],rec['offset_s']);env.reset(rec['initial_speed'])
            last=raw[-1];env.x=float(last[3]);env.v=float(last[5]);env.a=float(last[8]);env.t=float(last[1]);env.soc=float(last[13])
            tail=[]
            while (env.v>1e-9 or abs(env.a)>1e-9) and len(tail)<20:
                env.step(0.);tail.append(env.trace[-1])
            tail=np.asarray(tail,dtype=float).reshape(-1,24)
            assert len(tail)<20
            closed=np.concatenate([raw,tail]);S.save_trace(ROOT/'reports'/'settled_traces'/f'{arm}_{i:02d}.npz',closed)
            rows.append(dict(condition_id=i,old_Ij=rec['Ij'],old_peak=rec['jerk_max'],old_time=rec['time_s'],
                tail_steps=len(tail),final_a_before_tail=float(last[8]),final_v_before_tail=float(last[5]),
                settled_Ij=float((closed[:,9]**2*.5).sum()),settled_peak=float(abs(closed[:,9]).max()),
                settled_time=float(closed[-1,1]),settled_E_Wh=float(closed[:,10].sum()/3600),
                window_values=w,overrides=overrides,all_overrides_infeasible=all(x['empty_current_interval'] for x in overrides)))
        result[arm]=dict(rows=rows,mean_Ij=float(np.mean([r['settled_Ij'] for r in rows])),
            max_peak=max(r['settled_peak'] for r in rows),mean_time=float(np.mean([r['settled_time'] for r in rows])),
            total_overrides=sum(len(r['overrides']) for r in rows),
            infeasible_overrides=sum(x['empty_current_interval'] for r in rows for x in r['overrides']),
            zero_speed_overrides=sum(x['reason']=='zero_speed_boundary' for r in rows for x in r['overrides']),
            window_mean_time={w:float(np.mean([r['window_values'][w]['time_s'] for r in rows])) for w in windows})
    (ROOT/'reports/diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{k:v for k,v in d.items() if k!='rows'} for a,d in result.items()},indent=2))

if __name__=='__main__':main()
