"""Independent arithmetic and provenance checks. No training or input mutation.
Usage: python audit_records.py EXTRACTED_ROOT OUTPUT_JSON
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]).resolve()
arms=['frozen','supervised','joint','joint_head']
runs=['v4r','v4r_s1','v4r_s2']
rows={}; errors=[]; maxerr={}; versions=[]; failures=[]; log_schema={}; hashes={}
checks=0; steps=0
def check(key,a,b,tol=1e-7):
    global checks
    checks+=1
    e=abs(float(a)-float(b));maxerr[key]=max(maxerr.get(key,0),e)
    if not e<=tol:errors.append(dict(key=key,actual=float(a),recorded=float(b),error=e))
for seed,run in enumerate(runs):
 for arm in arms:
    d=root/'runs'/run/'v4_pilot'/arm
    ev=json.loads((d/'evaluation.json').read_text()); prev=json.loads((d/'evaluation_prev_bypass.json').read_text())
    rows[seed,arm]=ev['rows']
    ver=dict(seed=seed,arm=arm,identical_rows=ev['rows']==prev['rows'],claimed_version=ev.get('evaluation_version'),bypass_trigger_steps=0,nonzero_commands_on_trigger=0,max_abs_changes={k:0. for k in ['Ij','time_s','end_x','R']},outcomes_changed=0)
    for row,old in zip(ev['rows'],prev['rows']):
        c=row['condition_id']; p=d/'evaluation_traces'/f'dev_{c:02d}.npz'
        z=np.load(p);v=z['values']; names=list(z['columns']); a={n:v[:,i] for i,n in enumerate(names)}
        lpath=p.with_name(p.stem+'_layer.json'); logs=json.loads(lpath.read_text())
        for path in [p,lpath]:hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(v)==len(logs) and np.isfinite(v).all()
        steps+=len(v);dt=a['t1']-a['t0']; jerk=(a['a']-a['a_prev'])/dt
        check('jerk_definition',np.max(abs(jerk-a['jerk'])),0)
        m=dict(Ij=float(np.sum(jerk**2*dt)),jerk_rms=float(np.sqrt(np.sum(jerk**2*dt)/dt.sum())),jerk_p95=float(np.quantile(abs(jerk),.95)),jerk_max=float(abs(jerk).max()),time_s=float(dt.sum()),E_Wh=float(a['dE_J'].sum()/3600),friction_Wh=float(a['dF_J'].sum()/3600),R=float(a['reward'].sum()),trace_steps=len(v),end_x=a['x1'][-1],end_v=a['v1'][-1],arrived=a['arrived'][-1],violations=a['red_crossing'].sum(),jerk_override_steps=np.sum(abs(jerk)>2+1e-7),physical_violation_steps=np.sum((a['a'] < -3.5-1e-7)|(a['a']>2.6+1e-7)),settled=bool(a['arrived'][-1] and a['v1'][-1]<=.05 and abs(a['a'][-1])<=.05),intervened_substeps=sum(abs(x['applied']-x['command'])>1e-9 for x in logs),fallback_substeps=sum(x['fallback'] for x in logs))
        for k,x in m.items():check(k,x,row[k])
        check('layer_command_vs_trace',max(abs(x['command']-y) for x,y in zip(logs,a['a_cmd'])),0)
        check('layer_applied_vs_trace',max(abs(x['applied']-y) for x,y in zip(logs,a['a'])),0)
        check('layer_time_vs_trace',max(abs(x['t']-y) for x,y in zip(logs,a['t0'])),0)
        check('signal_phase_is_t0',max(abs((a['t0']+row['offset_s'])%90-a['signal_phase'])),0)
        if len(v)>1:
            for k0,k1 in [('t0','t1'),('x0','x1'),('v0','v1'),('a_prev','a')]:check('continuity_'+k0,np.max(abs(a[k0][1:]-a[k1][:-1])),0)
        check('window_Ij_ledger',sum(w['Ij'] for w in row['windows'].values()),m['Ij'])
        b=(a['x0']>=3999)&(a['v0']<=.3)
        ver['bypass_trigger_steps']+=int(b.sum());ver['nonzero_commands_on_trigger']+=int(np.sum(abs(a['a_cmd'][b])>1e-9))
        for k in ver['max_abs_changes']:ver['max_abs_changes'][k]=max(ver['max_abs_changes'][k],abs(row[k]-old[k]))
        ver['outcomes_changed']+=int(any(row[k]!=old[k] for k in ['arrived','settled','violations']))
        for l in logs:
            schema=','.join(sorted(l['memory']['sig']));log_schema[schema]=log_schema.get(schema,0)+1
        if row['violations']:
            ix=int(np.flatnonzero(a['red_crossing'])[0]); f=dict(seed=seed,arm=arm,condition=c,crossing_index=ix,crossing_v0=float(a['v0'][ix]),crossing_v1=float(a['v1'][ix]),memory=logs[ix]['memory']['sig'])
            f['speed_at_actual_crossing']=float(np.sqrt(max(a['v0'][ix]**2+2*a['a'][ix]*(3000-a['x0'][ix]),0)))
            if a['v0'][ix]>=2:
                changes=np.flatnonzero((a['signal_green'][1:]==0)&(a['signal_green'][:-1]==1))+1
                eligible=changes[(changes<=ix)&(changes>=ix-40)]
                if len(eligible):
                    j=int(eligible[-1]);f.update(yellow_index=j,substeps_until_cross=ix-j,yellow_distance=3000-float(a['x0'][j]),yellow_speed=float(a['v0'][j]),ideal_brake_distance=float(a['v0'][j]**2/7),yellow_memory=logs[j]['memory']['sig'])
            failures.append(f)
        elif not row['settled']:
            ix=len(v)-1
            while ix>0 and a['v0'][ix-1]<.05:ix-=1
            failures.append(dict(seed=seed,arm=arm,condition=c,type='not_settled',end_x=float(a['x1'][-1]),tail_stationary_steps=len(v)-ix,truth_green_memory_nongreen=sum(x['truth_green'] is True and x['perceived']!='green' for x in logs[ix:])))
    versions.append(ver)
aggregate={};paired={}
for arm in arms:
 rr=sum([rows[s,arm] for s in range(3)],[])
 aggregate[arm]=dict(n=len(rr),arrived_settled=sum(r['arrived'] and r['settled'] for r in rr),signal_violations=sum(r['violations'] for r in rr),offroad_substeps=sum(r['offroad_substeps'] for r in rr),mean_R=float(np.mean([r['R'] for r in rr])),jerk_violation_episodes=sum(r['jerk_override_steps']>0 for r in rr),max_jerk=max(r['jerk_max'] for r in rr),intervention_ratio=sum(r['intervened_substeps'] for r in rr)/sum(r['trace_steps'] for r in rr),episode_equal_mean_intervention_ratio=float(np.mean([r['intervened_substeps']/r['trace_steps'] for r in rr])))
 if arm!='frozen':
    pp=[]
    for s in range(3):
        f={r['condition_id']:r for r in rows[s,'frozen']};b={r['condition_id']:r for r in rows[s,arm]}
        ids=[i for i in f if all(r['arrived'] and r['settled'] for r in [f[i],b[i]])]
        pp.append(dict(seed=s,n=len(ids),ids=ids,**{k:float(np.mean([b[i][k]-f[i][k] for i in ids])) for k in ['Ij','R','E_Wh','time_s']}))
    paired[arm]=dict(by_seed=pp,equal_seed_mean={k:float(np.mean([p[k] for p in pp])) for k in ['Ij','R','E_Wh','time_s']})
out=dict(root=str(root),checks=checks,episodes=108,total_substeps=steps,errors=errors,max_absolute_errors=maxerr,aggregate=aggregate,paired=paired,evaluation_versions=versions,signal_memory_log_schemas=log_schema,failures=failures,raw_sha256=hashes)
Path(sys.argv[2]).write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps({k:out[k] for k in ['checks','episodes','total_substeps','errors','aggregate','paired','failures']},ensure_ascii=False,indent=2))
