"""Offline-only framework audit. No PyTorch, environment, training, or vehicle rollouts.

First extraction: python audit_offline.py --workspace /path/to/workspace
Recompute from packaged, field-selected log excerpts: python audit_offline.py
Requires Python 3, NumPy and SciPy. Original source hashes accompany the excerpts.
"""
import argparse, collections, csv, gzip, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def load_module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def dump(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def extract(root):
    sources = {}; rows = []; visual = []
    def track(p): sources[str(p.relative_to(root))] = sha(p)
    for tag in ('v6_formal_v22', 'v6_formal_v16'):
        for p in sorted((root/'audit_v6_closure/runs/v5_timing'/tag).glob('*.json')):
            r = read(p); track(p)
            keys = ['t','x','v','a','fallback','jerk_override','mem_seen','mem_phase','mem_d_line','truth','light_target','hidden']
            out = {k:r[k] for k in ('tag','policy','jerk','info','d_adopt_target','situation','branch_id','timeline','outcome')}
            out['recs'] = [{k:z[k] for k in keys} for z in r['recs']]
            out['initial'] = r['decisions'][0]['state0']; rows.append(out)
    for seed, run in enumerate(('v4r','v4r_s1','v4r_s2')):
        base = root/'audit_v4r_full/runs'/run/'v4_pilot/supervised'
        evp = base/'evaluation.json'; track(evp); ev = read(evp)
        for cond in range(9):
            lp = base/f'evaluation_traces/dev_{cond:02d}_layer.json'
            tp = lp.with_name(f'dev_{cond:02d}.npz'); track(lp); track(tp)
            logs = read(lp); tr = np.load(tp); cols = list(tr['columns']); vals = tr['values']
            assert len(logs) == len(vals)
            compact=[]
            for z,t in zip(logs, vals):
                assert abs(z['t']-t[cols.index('t0')]) < 1e-8
                compact.append(dict(t=z['t'],x=z['x'],v=z['v'],a=float(t[cols.index('a_prev')]),
                    sig=z['memory']['sig'],targets=z['targets'],fallback=z['fallback'],
                    jerk_override=int(t[cols.index('jerk_override')]),red_crossing=int(t[cols.index('red_crossing')])))
            visual.append(dict(seed=seed,condition=cond,recs=compact,metrics=ev['rows'][cond]))
    source_dir=HERE/'source';source_dir.mkdir(exist_ok=True)
    for local,p in [('curve_layer.py',root/'TIV_comfort_v2/code/curve_layer.py'),
                    ('v6_theory_lp.py',root/'audit_v6_closure/visual_dev/v6_theory_lp.py'),
                    ('v4r_model.py',root/'audit_v4r_full/visual_z/model.py'),
                    ('v4r_learner.py',root/'audit_v4r_full/visual_z/learner.py'),
                    ('v4r_pipeline.py',root/'audit_v4r_full/visual_dev/pipeline.py'),
                    ('v4r_vision_state.py',root/'audit_v4r_full/visual_dev/vision_state.py'),
                    ('v6_timing.py',root/'audit_v6_closure/visual_dev/v5_timing_experiment.py')]:
        track(p);(source_dir/local).write_bytes(p.read_bytes())
    same = root/'audit_v4r_full/v19_deps/code/curve_layer.py';track(same)
    aux = {}
    for name,p in [('lp22',root/'audit_v6_closure/runs/v5_timing/theory_lp.json'),
                   ('lp16',root/'audit_v6_closure/runs/v5_timing/theory_lp_v16.json'),
                   ('visual_rows',root/'delivery/TIV_v23/verification/visual_rows.json')]:
        track(p);aux[name]=read(p)
    dump(HERE/'source_hashes.json',sources)
    data=dict(v6=rows,visual=visual,aux=aux,
              identical_base_executor=sha(same)==sha(root/'TIV_comfort_v2/code/curve_layer.py'))
    with gzip.open(HERE/'log_excerpts.json.gz','wt',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False)
    return data

def first_color(recs, visual=False):
    for i,z in enumerate(recs):
        sig=z['sig'] if visual else dict(seen=z['mem_seen'],phase=z['mem_phase'])
        # An unresolved off/unknown phase is NOT the adoption of an actionable color.
        if sig['seen'] and sig['phase'] in ('red','yellow','green'): return i,z
    return None,None

def crossing(r, x):
    prev=r['initial']
    for z in r['recs']:
        if prev['x'] <= x <= z['x']:
            d=x-prev['x'];v=prev['v'];a=z['a']
            tau=d/v if abs(a)<1e-10 else 2*d/(v+math.sqrt(max(v*v+2*a*d,0)))
            assert -.000001 <= tau <= .500001
            return prev['t']+tau
        prev=z
    raise ValueError(('no crossing',r['branch_id'],x))

def approach_end(recs):
    return next((i for i,z in enumerate(recs) if (z['v']<.05 and 2940<z['x']<3005) or z['x']>=3000),len(recs)-1)

def summarize_numbers(values):
    finite=[v for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values),finite=len(finite),missing=sum(v is None for v in values),
                infeasible=sum(v is not None and not math.isfinite(v) for v in values),
                negative=sum(v < -1e-7 for v in finite),
                quantiles=None if not finite else [float(x) for x in np.quantile(finite,[0,.25,.5,.75,1])])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path);args=ap.parse_args()
    if args.workspace: data=extract(args.workspace.resolve())
    else:
        with gzip.open(HERE/'log_excerpts.json.gz','rt',encoding='utf-8') as f:data=json.load(f)
    C=load_module('C',HERE/'source/curve_layer.py');LP=load_module('LP',HERE/'source/v6_theory_lp.py')
    def margins(v,a,j,d):
        C.J=float(j);C.Q=C.J*C.DT;D=C.backup_distance(v,a,0.);W=v-C.reserve(v,a)
        return D,v-W,None if d is None else d-D
    stops=[];visual=[];checkerr=[];adopt_diff=0;pretarget=0
    for r in data['v6']:
        if r['situation']!='stop':continue
        i,z=first_color(r['recs']);last=approach_end(r['recs'])
        row={k:r[k] for k in ('tag','policy','jerk','info','d_adopt_target','branch_id')}
        row.update(adoption_found=z is not None,adoption_before_approach_end=(i is not None and i<=last))
        if z is not None:
            D,release,mt=margins(z['v'],z['a'],r['jerk'],3000-z['x'])
            row.update(t_info=z['t'],v=z['v'],a=z['a'],d_true=3000-z['x'],d_est=z['mem_d_line'],
                       backup_distance=D,release_margin=release,m_true=mt,m_est=None if z['mem_d_line'] is None else z['mem_d_line']-D,
                       phase=z['mem_phase'],truth=z['truth'],
                       approach_override=any(w['fallback'] or w['jerk_override'] for w in r['recs'][:last+1]),
                       subsequent_approach_override=any(w['fallback'] or w['jerk_override'] for w in r['recs'][i+1:last+1]))
            recorded=r['timeline']['adopt'];adopt_diff+=int(recorded is None or abs(recorded['t']-z['t'])>1e-8)
            if recorded is not None and abs(recorded['t']-z['t'])<1e-8 and math.isfinite(mt):
                checkerr.append(abs(mt-recorded['M_branch']))
            pretarget+=int(r['timeline']['target'] is not None and r['timeline']['target']['t']<z['t'])
        stops.append(row)
    for r in data['visual']:
        i,z=first_color(r['recs'],True);row=dict(seed=r['seed'],condition=r['condition'],adoption_found=z is not None)
        row['episode_jerk_override']=r['metrics']['jerk_override_steps']>0
        if z is not None:
            sig=z['sig'];D,release,mt=margins(z['v'],z['a'],2,3000-z['x'])
            # Target distance is taken from the logged executor target, not reconstructed margins.
            candidates=[t[0]-z['x'] for t in z['targets'] if t[1]==0 and t[0]<3030]
            target=min(candidates) if candidates else None
            row.update(t_info=z['t'],v=z['v'],a=z['a'],phase=sig['phase'],d_true=3000-z['x'],d_est=sig['d_line'],
                       backup_distance=D,release_margin=release,m_true=mt,m_est=sig['d_line']-D,
                       target_distance=target,m_target=None if target is None else target-D)
        visual.append(row)
    # Solve the archived model at every actual-time bracketing grid point; LP only.
    lp_cache={};lp_checks=[]
    def solve(tag,j,K):
        cachekey=(tag,j,K)
        if cachekey not in lp_cache:
            par=data['aux']['lp22' if tag.endswith('v22') else 'lp16']
            st,F,loss,aR,aG=LP.solve(par['v0'],par['d0'],par['Tm'],j,K,par['p'],common=True)
            lp_cache[cachekey]=dict(status=st,F=F,loss=loss)
        return lp_cache[cachekey]
    for tag, name in [('v6_formal_v22','lp22'),('v6_formal_v16','lp16')]:
        for z in data['aux'][name]['rows']:
            if 'delta' not in z:continue
            ans=solve(tag,z['jerk'],round(z['delta']/.5))
            if z['delayed']=='ok':
                assert ans['status']=='ok';lp_checks.append(abs(ans['loss'][1]-z['lossG']))
            else: assert ans['status']=='infeasible'
    high_next_checks=[]
    for j in (2.,3.,4.):
        delta=data['aux']['lp22']['boundaries'][str(j)]['common_feasible_max_delta']+.5
        result=solve('v6_formal_v22',j,round(delta/.5))
        assert result['status']=='infeasible'
        high_next_checks.append(dict(jerk=j,delta=delta,status=result['status']))
    baseline={(r['tag'],r['jerk']):r for r in data['v6'] if r['policy']=='constant' and r['info']=='early' and r['situation']=='pass'}
    pass_rows=[]
    for r in data['v6']:
        if r['policy']!='constant' or r['info']=='early' or r['situation']!='pass':continue
        base=baseline[(r['tag'],r['jerk'])];i,z=first_color(r['recs']);assert z
        par=data['aux']['lp22' if r['tag'].endswith('v22') else 'lp16'];v0=par['v0']
        entry=crossing(r,2800);delay=z['t']-entry;distance_delay=(200-(3000-z['x']))/v0
        kf=max(0,math.floor(delay/.5+1e-8));kc=max(0,math.ceil(delay/.5-1e-8))
        e=solve(r['tag'],r['jerk'],0)
        def lpref(K):
            ans=solve(r['tag'],r['jerk'],K)
            return None if ans['status']!='ok' else (ans['loss'][1]-e['loss'][1])/v0
        flo,cei=lpref(kf),lpref(kc);frac=delay/.5-kf
        ref=flo if kf==kc else None if flo is None or cei is None else (1-frac)*flo+frac*cei
        dt=(crossing(r,3300)-r['timeline']['t0'])-(crossing(base,3300)-base['timeline']['t0'])
        row=dict(tag=r['tag'],info=r['info'],jerk=r['jerk'],branch_id=r['branch_id'],target_m=r['d_adopt_target'],
            t_entry_200=entry,t_info=z['t'],delay_actual_s=delay,delay_distance_s=distance_delay,
            observed_delta_time_s=dt,lp_reference_floor_s=flo,lp_reference_ceil_s=cei,lp_reference_interp_s=ref,
            residual_floor_s=None if flo is None else dt-flo,residual_ceil_s=None if cei is None else dt-cei,
            residual_interp_s=None if ref is None else dt-ref)
        pass_rows.append(row)
    def group_metrics(rows,groupkeys,keys):
        groups=collections.defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in groupkeys)].append(r)
        return [{**dict(zip(groupkeys,g)),**{k:summarize_numbers([r.get(k) for r in rs]) for k in keys}}
                for g,rs in sorted(groups.items())]
    # No probabilistic interpretation: these are deterministic cells of a development grid.
    cross=[]
    for tag in ('v6_formal_v22','v6_formal_v16'):
        for mname in ('m_true','m_est'):
            for category in ('negative','nonnegative','infeasible','missing'):
                rs=[]
                for r in stops:
                    if r['tag']!=tag:continue
                    m=r.get(mname);c='missing' if m is None else 'infeasible' if not math.isfinite(m) else 'negative' if m < -1e-7 else 'nonnegative'
                    if c==category:rs.append(r)
                cross.append(dict(tag=tag,margin=mname,category=category,n=len(rs),
                    later_override=sum(bool(r.get('subsequent_approach_override')) for r in rs),
                    full_approach_override=sum(bool(r.get('approach_override')) for r in rs)))
    vs=[z for z in data['aux']['visual_rows'] if z['mode']=='S']; sm=[z for r in vs for z in r['rows']]
    system_means={k:float(np.mean([r[k] for r in sm])) for k in ('R','time_s','E_Wh','Ij')}
    fm=[z for r in data['aux']['visual_rows'] if r['mode']=='F' for z in r['rows'] if z['settled']]
    assert len(fm)==24
    f_means={k:float(np.mean([r[k] for r in fm])) for k in ('time_s','E_Wh','Ij')}
    summ=dict(v6_branches=len(data['v6']),stop_branches=len(stops),pass_reference_rows=len(pass_rows),visual_episodes=len(visual),
        base_executor_identical=data['identical_base_executor'],
        adoption_diff_from_recorded_truth_correct_event=adopt_diff,stop_target_precedes_color_adoption=pretarget,
        backup_recompute_max_error=max(checkerr),lp_rerun_max_lossG_error=max(lp_checks),system_S_means=system_means,
        high_speed_next_grid_checks=high_next_checks,
        system_F_completed_means=f_means,
        stop_margin_groups=group_metrics(stops,['tag'],['m_true','m_est','release_margin']),
        visual_margin_summary={k:summarize_numbers([r.get(k) for r in visual]) for k in ('m_true','m_est','m_target','release_margin')},
        visual_nonpassable_first_colors=sum(r.get('phase') in ('red','yellow') for r in visual),
        visual_negative_est_episode_overrides=sum(r.get('m_est',0)<-1e-7 and r['episode_jerk_override'] for r in visual),
        stop_margin_cross=cross,
        pass_reference_groups=group_metrics(pass_rows,['tag','info'],['observed_delta_time_s','delay_actual_s','residual_interp_s','residual_floor_s','residual_ceil_s']),
        claims={'lp_reference_is_vehicle_lower_bound':False,'interpolation_is_LP_optimum':False,
                'new_training_or_vehicle_rollouts':False,'new_results_are_post_hoc_offline_diagnostics':True})
    def clean(v):
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not math.isfinite(v):return 'infeasible'
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    dump(HERE/'audit_results.json',clean(summ))
    for name,rows in [('v6_adoption_margins',stops),('visual_adoption_margins',visual),('lp_reference_decomposition',pass_rows),('margin_cross_table',cross)]:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (HERE/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(clean(rows))
    print(json.dumps(clean(summ),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
