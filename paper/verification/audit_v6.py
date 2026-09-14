"""Read-only audit of the supplied v6 closure archive; no training or simulator rollouts.

Usage: python audit_v6.py --root /path/to/extracted/evcRL_v6_closure --out audit_results.json
Requires NumPy/SciPy only for rerunning the supplied LP generator.
The LP rerun checks reproducibility, not an independent optimal-control model.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def approach(r):
    out = []
    for z in r['recs']:
        out.append(z)
        if (z['v'] < .05 and 2940 < z['x'] < 3005) or z['x'] >= 3000:
            break
    return out


def smooth(r, threshold):
    return not any((z['fallback'] or z['jerk_override']) and z['v'] >= threshold
                   for z in approach(r))


def key(r, situation=None):
    return (r['tag'], r['policy'], r['jerk'], r['info'], r['d_adopt_target'],
            situation or r['situation'])


def fixed_endpoint_time(r):
    """Exact last-substep crossing time when recorded constant-acceleration dynamics hold."""
    z, p = r['recs'][-1], r['recs'][-2]
    assert p['x'] < 3300 <= z['x']
    assert abs(z['x'] - p['x'] - .25 * (p['v'] + z['v'])) < 1e-7
    assert abs(z['v'] - p['v'] - .5 * z['a']) < 1e-7
    d, a = 3300 - p['x'], z['a']
    tau = d / p['v'] if abs(a) < 1e-9 else 2 * d / (p['v'] + math.sqrt(p['v']**2 + 2*a*d))
    assert 0 <= tau <= .5 + 1e-8
    return p['t'] - r['timeline']['t0'] + tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    rows = []
    hashes = {}
    for tag in ('v6_formal_v22', 'v6_formal_v16'):
        for f in sorted((root / 'runs/v5_timing' / tag).glob('*.json')):
            data = f.read_bytes()
            hashes[str(f.relative_to(root))] = hashlib.sha256(data).hexdigest()
            r = json.loads(data)
            r['_file'] = str(f.relative_to(root))
            rows.append(r)
    idx = {key(r): r for r in rows}
    assert len(idx) == len(rows), 'Duplicate factorial cells'
    problems = []
    numeric_max_errors = collections.defaultdict(float)
    for r in rows:
        recs, o = r['recs'], r['outcome']
        computed = dict(substeps=len(recs), time_s=.5*len(recs),
                        E_Wh=sum(d['dE_Wh'] for d in r['decisions']),
                        Ij=.5*sum(z['jerk']**2 for z in recs),
                        jerk_max=max(abs(z['jerk']) for z in recs),
                        jerk_override_substeps=sum(z['jerk_override'] for z in recs),
                        safe_override_substeps=sum(z['safe_override'] for z in recs),
                        fallback_substeps=sum(z['fallback'] for z in recs),
                        violation=sum(z['red_crossing'] for z in recs), x_end=recs[-1]['x'])
        for field, v in computed.items():
            err = abs(v-o[field])
            numeric_max_errors[field] = max(numeric_max_errors[field], err)
            if err > 1e-7:
                problems.append(dict(branch=r['branch_id'], field=field, recomputed=v, stored=o[field]))
        if r['approach']['smooth_highspeed'] != smooth(r, 5):
            problems.append(dict(branch=r['branch_id'], field='stored_smooth_highspeed_vs_5mps'))
        if any(z['hidden'] and (z['mem_phase'] != 'unknown' or z['det_color'] is not None) for z in recs):
            problems.append(dict(branch=r['branch_id'], field='phase_mask'))

    totals = dict(branches=len(rows), by_group=dict(collections.Counter(r['tag'] for r in rows)),
                  substeps=sum(len(r['recs']) for r in rows),
                  completed=sum(r['outcome']['reached_end'] for r in rows),
                  recorded_signal_violations=sum(r['outcome']['violation'] for r in rows),
                  timeouts=sum(r['outcome']['timeout'] for r in rows),
                  stop_branches=sum(r['situation']=='stop' for r in rows),
                  stop_branches_recorded_stopped=sum(r['situation']=='stop' and r['outcome']['stopped_before_line'] for r in rows),
                  episodes_with_jerk_override=sum(r['outcome']['jerk_override_substeps']>0 for r in rows),
                  peak_jerk=max(r['outcome']['jerk_max'] for r in rows),
                  approach_smooth_by_speed_threshold={str(v):sum(smooth(r,v) for r in rows) for v in (0,5,10)},
                  threshold_5_vs_10_disagreements=sum(smooth(r,5)!=smooth(r,10) for r in rows),
                  non_early_adopt_minus_cue_seconds=dict(collections.Counter(str(r['timeline']['adopt']['t']-r['timeline']['cue']['t']) for r in rows if r['info']!='early')))

    pairs = []
    for r in rows:
        if r['policy'] != 'constant' or r['situation'] != 'stop' or r['info'] == 'early':
            continue
        g = idx[key(r, 'pass')]
        both = [(s,t) for s,t in zip(r['recs'],g['recs']) if s['hidden'] and t['hidden']]
        assert all(s['t']==t['t'] for s,t in both)
        diffs = {}
        for field in ('cmd','applied','mem_seen','mem_d_line'):
            def differs(s,t):
                x,y = s[field],t[field]
                return x!=y if x is None or y is None else abs(x-y)>1e-8
            first = next(((s,t) for s,t in both if differs(s,t)),None)
            diffs[field] = None if first is None else dict(time_from_freeze=first[0]['t']-r['timeline']['t0'], stop=first[0][field], pass_=first[1][field])
        pairs.append(dict(group=r['tag'], info=r['info'], jerk=r['jerk'], target=r['d_adopt_target'],
                          stop_id=r['branch_id'], pass_id=g['branch_id'], differences=diffs))
    pair_counts = []
    for tag in ('v6_formal_v22','v6_formal_v16'):
        for info in ('min','cons'):
            ps = [p for p in pairs if p['group']==tag and p['info']==info]
            pair_counts.append(dict(group=tag,info=info,pairs=len(ps),
                               **{f+'_different':sum(p['differences'][f] is not None for p in ps) for f in ('cmd','applied','mem_seen','mem_d_line')}))

    # Rerun the supplied LP exactly at its saved horizon, including the v22 floating horizon.
    spec = importlib.util.spec_from_file_location('supplied_v6_lp', root/'visual_dev/v6_theory_lp.py')
    lp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lp)
    lp_checks, predictions = [], {}
    for tag,filename in (('v6_formal_v22','theory_lp.json'),('v6_formal_v16','theory_lp_v16.json')):
        f = root/'runs/v5_timing'/filename
        hashes[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
        q = json.loads(f.read_bytes()); predictions[tag] = q
        errs, status_mismatches = [], []
        for r in q['rows']:
            early=lp.solve(q['v0'],q['d0'],q['Tm'],r['jerk'],0,q['p'],common=False)
            cur=lp.solve(q['v0'],q['d0'],q['Tm'],r['jerk'],round(r['delta']/lp.DT),q['p'])
            if cur[0]!=r['delayed']:status_mismatches.append(r)
            if cur[0]=='ok':
                errs.extend((abs((cur[1]-early[1])/q['v0']-r['penalty_s']),abs(cur[2][0]-r['lossR']),abs(cur[2][1]-r['lossG'])))
        extension=[]
        for j in (2,3,4):
            last=max(r['delta'] for r in q['rows'])
            for D in (last+.5,last+1,last+1.5,last+2):
                z=lp.solve(q['v0'],q['d0'],q['Tm'],j,round(D/lp.DT),q['p'])
                extension.append(dict(jerk=j,delta=D,status=z[0]))
        lp_checks.append(dict(file=filename,rows=len(q['rows']),max_abs_numeric_error=max(errs),status_mismatches=status_mismatches,endpoint_extension=extension))

    boundaries, comfort, costs = [], [], []
    for tag in predictions:
        q=predictions[tag]
        for info in ('early','min','cons'):
            ss=[r for r in rows if r['tag']==tag and r['policy']=='constant' and r['situation']=='stop' and r['info']==info]
            comfort.append(dict(group=tag,info=info,n=len(ss),smooth={str(v):sum(smooth(r,v) for r in ss) for v in (0,5,10)}))
        for j in (2,3,4):
            ss=sorted([r for r in rows if r['tag']==tag and r['policy']=='constant' and r['situation']=='stop' and r['jerk']==j and r['info']=='min'],key=lambda r:-r['d_adopt_target'])
            fail=next(r for r in ss if not smooth(r,10));prev=ss[ss.index(fail)-1]
            d_lp=q['d0']-q['v0']*q['boundaries'][str(float(j))]['zero_penalty_max_delta']
            boundaries.append(dict(group=tag,jerk=j,lp_nominal_zero_penalty_distance=d_lp,
                                   first_failure_target=fail['d_adopt_target'], first_failure_actual_adoption=fail['timeline']['adopt']['d_line'],
                                   previous_pass_target=prev['d_adopt_target'],previous_pass_actual_adoption=prev['timeline']['adopt']['d_line'],
                                   numerical_distance_difference=abs(d_lp-fail['timeline']['adopt']['d_line']),branch=fail['branch_id'],
                                   warning='Different estimands and clocks; this numerical difference is not a valid model-prediction error.'))
            early=next(r for r in rows if r['tag']==tag and r['policy']=='constant' and r['situation']=='pass' and r['jerk']==j and r['info']=='early')
            for r in rows:
                if r['tag']==tag and r['policy']=='constant' and r['situation']=='pass' and r['jerk']==j and r['info']!='early':
                    costs.append(dict(group=tag,jerk=j,info=r['info'],target=r['d_adopt_target'],branch=r['branch_id'],
                                      raw_time_delta=r['outcome']['time_s']-early['outcome']['time_s'],
                                      time_delta_at_exact_3300m=fixed_endpoint_time(r)-fixed_endpoint_time(early),
                                      raw_energy_delta_Wh=r['outcome']['E_Wh']-early['outcome']['E_Wh'],
                                      endpoint_distance_delta=r['outcome']['x_end']-early['outcome']['x_end']))
    for f in sorted((root/'visual_dev').glob('*.py')):
        hashes[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
    branch_rows=[dict(branch=r['branch_id'],group=r['tag'],policy=r['policy'],situation=r['situation'],
                      completed=r['outcome']['reached_end'],violations=r['outcome']['violation'],
                      full_approach_no_recovery=smooth(r,0),jerk_override=r['outcome']['jerk_override_substeps']>0,
                      peak_jerk=r['outcome']['jerk_max']) for r in rows]
    result=dict(scope='Existing raw records and supplied LP rerun only; no retraining, no new closed-loop rollouts.',
                totals=totals, record_arithmetic_max_abs_errors=dict(numeric_max_errors), record_arithmetic_or_mask_problems=problems,
                branch_rows=branch_rows,
                lp_rerun=lp_checks, constant_pair_counts=pair_counts, constant_pairs=pairs,
                constant_stop_comfort=comfort, boundary_comparisons=boundaries,constant_pass_costs=costs, source_sha256=hashes)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(totals=totals,record_problems=len(problems),lp_rerun=[{k:v for k,v in r.items() if k!='endpoint_extension'} for r in lp_checks],pair_counts=pair_counts),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
