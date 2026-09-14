"""Recompute saved trace metrics and evaluate prespecified expansion gates."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import probe as P

FIELDS=['signal_Ij','Ij','jerk_rms','jerk_max','time_s','E_Wh','signal_time_s',
        'signal_energy_Wh','signal_friction_Wh','signal_strong_braking_s','strong_braking_s',
        'layer_fallback_steps','local_overspeed_time_s','violations','physical_violation_steps']

def independent_check(folder,row):
    stem=f"{row['mode']}_{row['condition_id']:02d}"
    z=np.load(folder/(stem+'.npz'))['values']
    dt=z[:,1]-z[:,0];j=(z[:,8]-z[:,6])/dt
    assert np.allclose(j,z[:,9],rtol=0,atol=1e-12)
    assert np.allclose(z[1:,6],z[:-1,8],rtol=0,atol=1e-12)
    mask=((z[:,2]+z[:,3])/2>=2200)&((z[:,2]+z[:,3])/2<3300)
    lim=np.where((z[:,3]>=1450-1e-9)&(z[:,3]<=1550+1e-9),35/3.6,80/3.6)
    over=np.maximum(z[:,5]-lim,0.)
    actual=dict(Ij=float(np.sum(j*j*dt)),signal_Ij=float(np.sum(j[mask]**2*dt[mask])),
        time_s=float(dt.sum()),E_Wh=float(z[:,10].sum()/3600),jerk_max=float(np.max(np.abs(j))),
        signal_strong_braking_s=float(dt[mask&(z[:,8]<-2)].sum()),
        local_overspeed_time_s=float(dt[over>1e-6].sum()),violations=int(z[:,18].sum()),
        physical_violation_steps=int(((z[:,8]<-3.5-1e-7)|(z[:,8]>2.6+1e-7)).sum()))
    for k,v in actual.items():assert abs(v-row[k])<1e-8,(stem,k,v,row[k])
    logs=json.loads((folder/(stem+'_layer.json')).read_text())
    assert len(logs)==len(z) and np.allclose([l['applied'] for l in logs],z[:,8],rtol=0,atol=1e-12)
    if row['mode']=='color':assert all(not l['countdown_visible'] and not l['timing_test'] for l in logs)
    assert all(0.<l['signal_distance']<=1000 for l in logs if l['countdown_visible'])
    assert sum(l['fallback'] for l in logs)==row['layer_fallback_steps']
    return z,logs

def analyze(stage):
    split='reporting' if stage=='reporting' else 'development'
    folder=P.ROOT/'runs'/split
    progress=json.loads((folder/'progress.json').read_text())
    assert progress['source_hash']==P.source_hash()
    rows=[r for r in progress['rows'] if stage!='smoke' or r['condition_id']==0]
    expected=2 if stage=='smoke' else (36 if stage=='reporting' else 18)
    assert len(rows)==expected,(len(rows),expected)
    traces={};logs={}
    for r in rows:traces[r['mode'],r['condition_id']],logs[r['mode'],r['condition_id']]=independent_check(folder,r)
    by={(r['mode'],r['condition_id']):r for r in rows}
    paired=[]
    for i in sorted({r['condition_id'] for r in rows}):
        a,b=by['color',i],by['timing',i]
        p=dict(condition_id=i,condition=a['condition'],both_settled=a['settled'] and b['settled'],
               color_settled=a['settled'],timing_settled=b['settled'])
        for k in FIELDS:p['color_'+k]=a[k];p['timing_'+k]=b[k];p['delta_'+k]=b[k]-a[k]
        p['signal_Ij_reduction_pct']=100*(a['signal_Ij']-b['signal_Ij'])/a['signal_Ij'] if a['signal_Ij'] else None
        z0,z1=traces['color',i],traces['timing',i];n=min(len(z0),len(z1))
        diff=np.flatnonzero(np.abs(z0[:n,8]-z1[:n,8])>1e-6)
        p['first_divergence']=None
        if len(diff):
            k=int(diff[0]);assert np.allclose(z0[:k,:],z1[:k,:],rtol=0,atol=1e-9)
            assert np.allclose(z0[k,[0,2,4,6,7]],z1[k,[0,2,4,6,7]],rtol=0,atol=1e-9)
            p['first_divergence']=dict(step=k,t=float(z0[k,0]),x=float(z0[k,2]),
                color_action=float(z0[k,8]),timing_action=float(z1[k,8]),
                timing_reason=logs['timing',i][k]['signal_reason'])
        paired.append(p)
    aggregate={}
    for mode in P.MODES:
        rr=[r for r in rows if r['mode']==mode]
        aggregate[mode]=dict(n=len(rr),settled=sum(r['settled'] for r in rr),
            mean={k:float(np.mean([r[k] for r in rr])) for k in FIELDS},
            totals={k:sum(r[k] for r in rr) for k in ('violations','physical_violation_steps','layer_fallback_steps','paired_action_difference_steps','timing_test_steps','countdown_visible_steps')},
            peak_jerk=max(r['jerk_max'] for r in rr))
    a,b=aggregate['color'],aggregate['timing'];am,bm=a['mean'],b['mean']
    gates=dict(all_settled=all(r['settled'] for r in rows),
        no_red_physical_or_overspeed=all(r['violations']==0 and r['physical_violation_steps']==0 and r['local_overspeed_time_s']==0 for r in rows),
        action_difference=any(p['first_divergence'] is not None for p in paired),
        signal_jerk_improves=bm['signal_Ij']<am['signal_Ij']-1e-6,
        full_jerk_not_worse=bm['Ij']<=am['Ij']+1e-6,
        mean_time_within_5pct=bm['time_s']<=1.05*am['time_s'],
        mean_energy_within_3pct=bm['E_Wh']<=1.03*am['E_Wh'],
        peak_jerk_not_worse=b['peak_jerk']<=a['peak_jerk']+.1,
        fallback_not_worse=b['totals']['layer_fallback_steps']<=a['totals']['layer_fallback_steps'])
    smoke=all(r['settled'] and r['violations']==0 and r['physical_violation_steps']==0 for r in rows)
    result=dict(stage=stage,source_hash=progress['source_hash'],independent_recomputation_passed=True,
        gate_passed=smoke if stage=='smoke' else all(gates.values()),gates=gates,
        aggregate=aggregate,paired=paired,independent_training_seeds=1,
        new_training_steps=0,rollout_count=len(rows),total_rollout_wall_s=sum(r['wall_s'] for r in rows),
        signal_Ij_reduction_pct=100*(am['signal_Ij']-bm['signal_Ij'])/am['signal_Ij'] if am['signal_Ij'] else None)
    P.S.json_save(P.ROOT/'reports'/f'{stage}.json',result)
    with (P.ROOT/'reports'/f'{stage}_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    print(json.dumps({k:result[k] for k in ('stage','gate_passed','gates','aggregate','signal_Ij_reduction_pct','total_rollout_wall_s')},indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('smoke','pilot','reporting'))
    analyze(ap.parse_args().stage)
