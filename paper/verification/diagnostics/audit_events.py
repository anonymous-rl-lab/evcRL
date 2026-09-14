"""Offline event reconstruction and static target checks. No policy or vehicle rollout.

Run from any directory: python verification/diagnostics/audit_events.py
Inputs are the packaged original-record excerpts and unchanged executor/LP code.
"""
import csv, gzip, hashlib, importlib.util, json, math
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
FOLDER=P/'verification/framework'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def save(name, data):
    (HERE/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvsave(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    C=module('curve',FOLDER/'source/curve_layer.py')
    A=module('old_audit',FOLDER/'audit_offline.py')
    LP=module('planning',FOLDER/'source/v6_theory_lp.py')
    data=json.loads(gzip.decompress((FOLDER/'log_excerpts.json.gz').read_bytes()))
    rows=[]
    def B(d):return max(.85*d-6.,0.)
    for r in data['v6']:
        if r['situation']!='stop':continue
        C.J=float(r['jerk']);C.Q=C.J*C.DT
        rec=r['recs'];i,z=A.first_color(rec);end=A.approach_end(rec)
        D=C.backup_distance(z['v'],z['a']);assert math.isfinite(D)
        event=next((k for k in range(i+1,end+1) if rec[k]['fallback'] or rec[k]['jerk_override']),None)
        row=dict(branch_id=r['branch_id'],tag=r['tag'],policy=r['policy'],rule=r['info'],jerk=r['jerk'],adoption_index=i,
                 adoption_target_m=r['d_adopt_target'],t_info=z['t'],v_adopt=z['v'],m_true_adopt=3000-z['x']-D,
                 m_est_adopt=z['mem_d_line']-D,m_target_adopt=B(z['mem_d_line'])-D,later_event=event is not None)
        if event is not None:
            before,w=rec[event-1],rec[event];dp=C.backup_distance(before['v'],before['a']);assert math.isfinite(dp)
            row.update(event_index=event,steps_after_adopt=event-i,v_event=w['v'],d_true_event=3000-w['x'],
                       event_type='fallback' if w['fallback'] else 'jerk_override',
                       m_true_pre=3000-before['x']-dp,m_est_pre=before['mem_d_line']-dp,m_target_pre=B(before['mem_d_line'])-dp)
            row['event_class']='terminal' if w['v']<5 else 'adoption_negative' if row['m_target_adopt']<0 else 'later_highspeed'
            # rec[event] contains memory updated AFTER the triggering action.
            innovations=[rec[k]['mem_d_line']-rec[k-1]['mem_d_line']+rec[k]['x']-rec[k-1]['x'] for k in range(i+1,event)]
            row['max_abs_innovation_m']=max(map(abs,innovations),default=0.)
            assert before['light_target']
            vmax=22.22222222222222 if r['tag'].endswith('v22') else 16.
            for mode,d in [('buffered',B(before['mem_d_line'])),('estimated_line',before['mem_d_line']),('true_line',3000-before['x'])]:
                interval,info=C.interval(before['v'],before['a'],[(max(d,0.),0.)],vmax=vmax)
                row[mode+'_reason']=info['reason'];row[mode+'_feasible']=interval is not None
        rows.append(row)
    events=[r for r in rows if r['later_event']]
    # Context fields come from the original visual layer logs, with hashes.
    ctx=json.loads(gzip.decompress((HERE/'visual_context.json.gz').read_bytes()))
    contexts={(r['seed'],r['condition']):r for r in ctx['episodes']}
    pv=[]
    for r in data['visual']:
        ids=[i for i,z in enumerate(r['recs']) if z['jerk_override']]
        i,z=A.first_color(r['recs'],True)
        out=dict(seed=r['seed'],condition=r['condition'],first_color=z['sig']['phase'],t_info=z['t'],
                 override_episode=bool(ids),n_override_substeps=len(ids),first_override_index=None,event_class='none')
        if ids:
            ix=ids[0];w=r['recs'][ix];raw=contexts[r['seed'],r['condition']]['first']
            assert abs(raw['x']-w['x'])<1e-8 and raw['memory']['sig']['phase']==w['sig']['phase']
            cls='endpoint' if w['x']>3900 else 'phase_flip' if w['v']>20 and raw['truth_green'] is True else 'range_update' if w['x']<2940 else 'signal_terminal'
            out.update(first_override_index=ix,x=w['x'],v=w['v'],phase=w['sig']['phase'],truth_green=raw['truth_green'],event_class=cls)
            if cls=='endpoint':
                out['end_distance_estimate']=raw['memory']['end']['d_est']
                out['end_raw_distance_error']=raw['memory']['end']['d_est']-(4000-w['x'])
                out['end_target_x']=max(t[0] for t in raw['targets'] if t[1]==0)
        pv.append(out)
    summary=dict(stop_branches=len(rows),later_events=len(events),negative_target_at_adoption=sum(r['m_target_adopt']<0 for r in rows),
                 below_5_at_adoption=sum(r['m_target_adopt']<5 for r in rows),event_classes=dict(Counter(r['event_class'] for r in events)),
                 pre_event_true_negative=sum(r['m_true_pre']<0 for r in events),pre_event_target_negative=sum(r['m_target_pre']<0 for r in events),
                 target_check_reasons={k:dict(Counter(r[k+'_reason'] for r in events)) for k in ('buffered','estimated_line','true_line')},
                 visual_classes=dict(Counter(r['event_class'] for r in pv)),innovation_quantiles=np.quantile([r['max_abs_innovation_m'] for r in events],[0,.5,1]).tolist(),
                 cohort_margins=[],class_details=[],policy_counts=[],new_training_steps=0,new_vehicle_rollouts=0)
    for tag in ('v6_formal_v22','v6_formal_v16'):
        rs=[r for r in rows if r['tag']==tag]
        summary['cohort_margins'].append(dict(tag=tag,n=len(rs),target_range=[min(r['m_target_adopt'] for r in rs),max(r['m_target_adopt'] for r in rs)],negative=sum(r['m_target_adopt']<0 for r in rs),below5=sum(r['m_target_adopt']<5 for r in rs),later=sum(r['later_event'] for r in rs)))
    for cls in ('adoption_negative','later_highspeed','terminal'):
        rs=[r for r in events if r['event_class']==cls]
        summary['class_details'].append(dict(event_class=cls,n=len(rs),adoption_margin_median=float(np.median([r['m_target_adopt'] for r in rs])),delay_substeps_median=float(np.median([r['steps_after_adopt'] for r in rs])),distance_median=float(np.median([r['d_true_event'] for r in rs]))))
    for learned in (False,True):
        rs=[r for r in rows if (r['policy']!='constant')==learned]
        summary['policy_counts'].append(dict(policy='S pooled' if learned else 'constant',n=len(rs),terminal=sum(r.get('event_class')=='terminal' for r in rs),highspeed=sum(r.get('event_class') in ('adoption_negative','later_highspeed') for r in rs)))
    assert summary['event_classes']=={'later_highspeed':40,'adoption_negative':31,'terminal':124}
    assert all(r['steps_after_adopt']==1 for r in events if r['event_class']=='adoption_negative')
    lp_all={}
    for tag,v0,T,N in [('22',22.2222,13.5,27),('16',16.,19.,38)]:
        scan=dict(v0=v0,d0=200.,Tm=T,p=.5,DT=.5,rows=[],boundaries={})
        for j in (2.,3.,4.):
            status,F0,loss0,*_=LP.solve(v0,200,T,j,0,.5);assert status=='ok'
            good=[];zero=[];bad=[];r_loss=[]
            for K in range(N+1):
                st,cost,loss,*_=LP.solve(v0,200,T,j,K,.5)
                item=dict(jerk=j,delta=K*.5,K=K,delayed=st)
                if st=='ok':
                    item.update(penalty_s=(cost-F0)/v0,lossR=loss[0],lossG=loss[1]);good.append(K*.5);r_loss.append(loss[0])
                    if abs(cost-F0)<1e-6:zero.append(K*.5)
                else:bad.append(K*.5)
                scan['rows'].append(item)
            scan['boundaries'][str(j)]=dict(zero_penalty_max_delta=max(zero),common_feasible_max_delta=max(good),first_infeasible=min(bad),stop_loss_span=max(r_loss)-min(r_loss))
            assert max(r_loss)-min(r_loss)<1e-6
        lp_all[tag]=scan
        (P/'reproduction/v6'/('theory_lp'+('_v16' if tag=='16' else '')+'_complete.json')).write_text(json.dumps(scan,indent=2)+'\n')
    summary['lp_complete_boundaries']={k:v['boundaries'] for k,v in lp_all.items()}
    pr=list(csv.DictReader((FOLDER/'lp_reference_decomposition.csv').open()))
    summary['floor_groups']=[]
    for tag in ('v6_formal_v22','v6_formal_v16'):
        for rule in ('min','cons'):
            rs=[r for r in pr if r['tag']==tag and r['info']==rule]
            vals=[float(r['residual_floor_s']) for r in rs]
            summary['floor_groups'].append(dict(tag=tag,rule=rule,n=len(rs),range=[min(vals),max(vals)]))
    summary['floor_all_nonnegative']=all(float(r['residual_floor_s'])>=-1e-7 for r in pr)
    # Independent algebra checks include clipped target distances and negative updates.
    errors=[];viol=[]
    for d in np.linspace(-10,200,43):
        for e in np.linspace(-20,20,41):
            for D in (0.,5.,40.):
                left=(B(d+e)-D)-(B(d)-D);right=B(d+e)-B(d)
                errors.append(abs(left-right));viol.append((B(d)-D)-.85*abs(e)-(B(d+e)-D))
    summary['target_update_checks']=dict(cases=len(errors),identity_max_error=max(errors),lower_bound_max_violation=max(viol))
    assert max(errors)<1e-10 and max(viol)<1e-10
    summary['source_sha256']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [FOLDER/'log_excerpts.json.gz',HERE/'visual_context.json.gz',FOLDER/'source/curve_layer.py',FOLDER/'source/v6_theory_lp.py']}
    csvsave('pb_event_rows.csv',rows);csvsave('pv_event_rows.csv',pv);save('event_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['stop_branches','later_events','negative_target_at_adoption','below_5_at_adoption','event_classes','visual_classes','floor_all_nonnegative','target_update_checks']},indent=2))

if __name__=='__main__':main()
