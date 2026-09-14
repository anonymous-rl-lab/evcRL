"""Independently recalculate all three probe versions and report the final one."""
import json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT   # archived v1 dependencies live inside this folder: deps/ (code), actor_evaluations/ (evaluations and traces)

def metrics(a):
    dt=a[:,1]-a[:,0]
    assert np.allclose(dt,.5,atol=1e-12,rtol=0)
    assert np.allclose(a[1:,6],a[:-1,8],atol=1e-10,rtol=0)
    j=(a[:,8]-a[:,6])/dt
    assert np.allclose(j,a[:,9],atol=1e-10,rtol=0)
    ij=float(np.sum(j*j*dt))
    limit=np.where((a[:,3]>=1450-1e-9)&(a[:,3]<=1550+1e-9),35/3.6,80/3.6)
    excess=np.maximum(a[:,5]-limit,0.)
    return dict(Ij=ij,jerk_rms=float(np.sqrt(ij/dt.sum())),jerk_max=float(abs(j).max()),
                time_s=float(dt.sum()),E_Wh=float(a[:,10].sum()/3600),
                overspeed_time=float(dt[excess>1e-6].sum()),overspeed_max=float(excess.max()),
                red=int(a[:,18].sum()),physical=int(np.sum((a[:,8]<-3.5-1e-7)|(a[:,8]>2.6+1e-7))),
                settled=bool(a[-1,3]>=3999 and a[-1,5]<=1e-6 and abs(a[-1,8])<=1e-6))

def aggregate(rows):
    return dict(n=len(rows),settled=sum(r['settled'] for r in rows),red=sum(r['red'] for r in rows),
        physical=sum(r['physical'] for r in rows),mean={k:float(np.mean([r[k] for r in rows])) for k in ['Ij','jerk_rms','time_s','E_Wh','overspeed_time']},
        peak_jerk=max(r['jerk_max'] for r in rows),peak_overspeed=max(r['overspeed_max'] for r in rows))

def main():
    baseline={};versions={};checks=0;legacy_comparison={}
    for arm in ['A','B']:
        rows=[metrics(np.load(p,allow_pickle=False)['values']) for p in sorted((ROOT/'reports/settled_traces').glob(f'{arm}_*.npz'))]
        assert len(rows)==9;baseline[arm]=aggregate(rows)
    for name in ['frozen_s7','frozen_s7_r1','frozen_s7_r2']:
        run=ROOT/'runs'/name;progress=json.loads((run/'progress.json').read_text());v={}
        for arm in ['A','B']:
            rows=[];legacy=[];fallback=0
            for i in range(9):
                a=np.load(run/f'{arm}_{i:02d}.npz',allow_pickle=False)['values'];m=metrics(a)
                row=next(x for x in progress['rows'] if x['arm']==arm and x['condition_id']==i)
                for k in ['Ij','jerk_rms','jerk_max','time_s','E_Wh']:
                    assert np.isclose(m[k],row[k],atol=1e-9,rtol=1e-12),(name,arm,i,k);checks+=1
                assert m['settled']==row['settled'] and m['red']==row['violations']
                fallback+=row['layer_fallback_steps'];m['condition_id']=i;rows.append(m)
                arrived=np.flatnonzero(a[:,22])
                if len(arrived):legacy.append(metrics(a[:arrived[0]+1]))
            agg=aggregate(rows);agg['fallback_steps']=fallback;v[arm]=agg
            if name=='frozen_s7_r2':
                original=json.loads((BASE/'actor_evaluations'/f'{arm}_step_300000_development.json').read_text())['summary']
                legacy_comparison[arm]=dict(original=original['completed_mean'],new=aggregate(legacy),
                    note='Original arrival cut, prior to added settling tail; settled count not used for this secondary cut.')
        versions[name]=v
    final=versions['frozen_s7_r2'];effects={}
    for arm in ['A','B']:
        b,n=baseline[arm],final[arm];bm,nm=b['mean'],n['mean']
        changes={k:nm[k]/bm[k]-1 for k in ['Ij','jerk_rms','time_s','E_Wh']}
        guards=dict(complete=n['settled']==9 and b['settled']==9,no_red=n['red']==b['red']==0,
                    no_physical=n['physical']==0,lower_Ij=changes['Ij']<0,lower_RMS=changes['jerk_rms']<0,
                    jerk_peak=n['peak_jerk']<=b['peak_jerk']+.1,time=changes['time_s']<=.05,
                    energy=changes['E_Wh']<=.03,overspeed_time=nm['overspeed_time']<=bm['overspeed_time']+1e-8,
                    overspeed_peak=n['peak_overspeed']<=b['peak_overspeed']+.1,no_fallback=n['fallback_steps']==0)
        effects[arm]=dict(changes=changes,guards=guards,development_pass=all(guards.values()))
    result=dict(baseline=baseline,versions=versions,final_effects=effects,secondary_original_cut=legacy_comparison,
                new_training_steps=0,independent_training_seeds=1,source_actor_variants=2,
                formal_evaluation_started=False,evaluation_traces_checked=54,scalar_checks=checks)
    (ROOT/'reports/FINAL_ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(final=final,effects=effects,legacy=legacy_comparison),indent=2))

if __name__=='__main__':main()
