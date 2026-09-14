"""Independent recomputation of all metrics from the raw per-episode records, pair tables, event tables and figures.

Reads only results/{R1,R2}/<job_id>/{trace.npz,layer.json,substeps.json.gz,summary.json} and results/R3/*.
Writes episodes.csv, pairs_R1.csv, pairs_R2.csv, pairs_R3.csv, events_R1.csv, guard_events_R1.csv, tables.md, metrics_summary.json,
fig_R1_pairs.png, fig_R1_target_updates.png. NumPy/Matplotlib only; no vehicle stepping.
"""
import csv, gzip, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RES = HERE / 'results'; REPO = HERE.parent
J, DT, V_CURVE = 2.0, 0.5, 35 / 3.6
ARCHIVED_OVERRIDE = {('S0', 0), ('S0', 2), ('S0', 5), ('S0', 8), ('S2', 3), ('S2', 5), ('S2', 6), ('S2', 7), ('S2', 8)}   # nine archived S override episodes (paper Table VI / S8)


def load_jobs(): return list(csv.DictReader(open(HERE / 'jobs.csv')))


def recompute(tr):
    dt = tr[:, 1] - tr[:, 0]; j = (tr[:, 8] - tr[:, 6]) / dt
    assert np.allclose(j, tr[:, 9], rtol=0, atol=1e-12) and np.allclose(tr[1:, 6], tr[:-1, 8], rtol=0, atol=1e-12)
    curve = ((tr[:, 3] >= 1450 - 1e-9) & (tr[:, 3] <= 1550 + 1e-9) & (tr[:, 5] > V_CURVE + 1e-6))
    last = tr[-1]
    return dict(substeps=len(tr), Ij=float(np.sum(j * j * dt)), jerk_rms=float(np.sqrt(np.sum(j * j * dt) / dt.sum())), jerk_max=float(np.max(np.abs(j))), time_s=float(dt.sum()),
                E_Wh=float(tr[:, 10].sum() / 3600), R=float(tr[:, 12].sum()), override_steps=int(tr[:, 17].sum()), numeric_exceedance_steps=int(np.sum(np.abs(j) > J + 1e-7)),
                boundary_discrepancy_steps=int(np.sum((tr[:, 17] > 0) != (np.abs(j) > J + 1e-7))), red_crossings=int(tr[:, 18].sum()), curve_violation_steps=int(curve.sum()),
                arrived=bool(last[22]), settled=bool(last[22] and last[5] <= .05 and abs(last[8]) <= .05), overshoot=bool(last[3] > 4100.), end_x=float(last[3]), end_v=float(last[5]),
                physical_violation_steps=int(tr[:, 23].sum()), stopped_time_s=float(dt[tr[:, 5] < .1].sum()))


def segments(flags):
    n = 0; prev = False
    for f in flags:
        if f and not prev: n += 1
        prev = f
    return n


def classify(sub, k):
    s = sub[k]; x = s['pre']['x']; v = s['pre']['v']
    if x >= 3900: return 'endpoint'
    if 1300 <= x < 1600: return 'curve'
    if 2600 <= x < 3010:
        if v < 8.: return 'signal_terminal'
        if s['perceived_pre'] != 'green' and s['truth_green_pre'] is True: return 'near_line_color'
        for q in range(max(0, k - 2), k):
            u = sub[q]['update']
            if 'predicted' in u and u.get('predicted') is not None and abs(u['accepted'] - u['predicted']) >= 5.: return 'range_update'
        return 'signal_other'
    return 'other'


def episode_rows(exp):
    rows = []
    for job in [r for r in load_jobs() if r['experiment'] == exp]:
        d = RES / exp / job['job_id']
        if not (d / 'DONE').exists(): rows.append(dict(job_id=job['job_id'], status='not_run')); continue
        tr = np.load(d / 'trace.npz')['values']; summ = json.load(open(d / 'summary.json')); lay = json.load(open(d / 'layer.json'))
        with gzip.open(d / 'substeps.json.gz', 'rt') as f: sub = json.load(f)
        rc = recompute(tr); assert len(lay) == len(tr) == len(sub)
        for k, ref in (('Ij', 'Ij'), ('time_s', 'time_s'), ('E_Wh', 'E_Wh'), ('R', 'R'), ('jerk_max', 'jerk_max'), ('override_steps', 'jerk_override_steps'), ('red_crossings', 'violations')):
            assert abs(rc[k] - summ[ref]) < 1e-9, (job['job_id'], k, rc[k], summ[ref])
        assert rc['settled'] == summ['settled'] and rc['curve_violation_steps'] == summ['offroad_substeps']
        fb = [l['fallback'] for l in lay]; creep = [l['fallback'] and s['pre']['v'] < .5 for l, s in zip(lay, sub)]; move = [l['fallback'] and s['pre']['v'] >= .5 for l, s in zip(lay, sub)]
        ov = [k for k in range(len(sub)) if sub[k]['override_flag']]
        rows.append(dict(job_id=job['job_id'], status='complete', experiment=exp, policy=job['policy'], condition_id=int(job['condition_id']), profile=job['profile'], pair_id=job['pair_id'],
                         **rc, fallback_steps=int(sum(fb)), fallback_creeping_steps=int(sum(creep)), fallback_noncreeping_steps=int(sum(move)), fallback_segments=segments(fb), fallback_noncreeping_segments=segments(move),
                         intervened_steps=int(sum(l['intervened'] for l in lay)), override_episode=int(len(ov) > 0), override_segments=segments([s['override_flag'] for s in sub]),
                         first_override_k=(ov[0] if ov else ''), first_override_category=(classify(sub, ov[0]) if ov else ''), override_categories='|'.join(sorted({classify(sub, k) for k in ov})),
                         guard_rejections=int(sum(1 for s in sub if s['update'].get('range_guard'))), light_updates=int(sum(1 for s in sub if 'predicted' in s['update'])),
                         terminal='red_crossing' if rc['red_crossings'] else ('settled' if rc['settled'] else ('overshoot' if rc['overshoot'] else 'deadline')),
                         override_and_fallback_steps=int(sum(1 for l, s in zip(lay, sub) if l['fallback'] and s['override_flag'])), model_unchanged=summ['model_unchanged'], wall_s=summ['wall_s']))
    return rows


def override_events(exp='R1'):
    ev = []; guard = []
    for job in [r for r in load_jobs() if r['experiment'] == exp]:
        d = RES / exp / job['job_id']
        if not (d / 'DONE').exists(): continue
        with gzip.open(d / 'substeps.json.gz', 'rt') as f: sub = json.load(f)
        for k, s in enumerate(sub):
            if s['override_flag']:
                sig = s['memory']['sig']; upd = None
                for q in range(max(0, k - 2), k):
                    if 'predicted' in sub[q]['update']: upd = sub[q]['update']
                ev.append(dict(job_id=job['job_id'], policy=job['policy'], condition_id=job['condition_id'], profile=job['profile'], k=k, t=s['pre']['t'], x=round(s['pre']['x'], 2), v_pre=round(s['pre']['v'], 3), a_prev=round(s['pre']['a'], 4),
                               a=round(s['applied_a'], 4), jerk=round(s['jerk'], 4), command_a=round(s['command_a'], 4), fallback=s['fallback'], interval_lower=s['interval'][0], interval_upper=s['interval'][1],
                               perceived_pre=s['perceived_pre'], truth_green_pre=s['truth_green_pre'], sig_seen=sig['seen'], sig_phase=sig['phase'], sig_d_line=(None if sig['d_line'] is None else round(sig['d_line'], 2)), sig_hold=sig.get('hold'),
                               end_d_est=(None if s['memory']['end']['d_est'] is None else round(s['memory']['end']['d_est'], 2)), targets=json.dumps(s['targets']),
                               last_update_predicted=(None if not upd else upd.get('predicted')), last_update_accepted=(None if not upd else upd.get('accepted')), category=classify(sub, k)))
            u = s['update']
            if u.get('range_guard'):
                nxt = sub[k + 1:k + 5]
                guard.append(dict(job_id=job['job_id'], policy=job['policy'], condition_id=job['condition_id'], k=k, t=s['post']['t'], x=round(s['post']['x'], 2), v=round(s['post']['v'], 3), a=round(s['post']['a'], 4),
                                  predicted=u['predicted'], candidate=u['candidate'], accepted=u['accepted'], raw_line=u['raw_line'], phase=s['memory']['sig']['phase'],
                                  m_minus=s['margins']['m_minus'] if s['margins'] else None, m_cand=s['margins']['m_cand'] if s['margins'] else None, m_plus=s['margins']['m_plus'] if s['margins'] else None,
                                  backup_infinite=s['margins']['backup_infinite'] if s['margins'] else None, max_abs_jerk_next4=max([abs(q['jerk']) for q in nxt] or [0.]), fallback_next4=int(sum(q['fallback'] for q in nxt)), override_next4=int(sum(q['override_flag'] for q in nxt))))
    return ev, guard


def write_csv(path, rows):
    if not rows: path.write_text(''); return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def arm_summary(rows):
    n = len(rows); done = [r for r in rows if r['settled']]
    return dict(N=n, settled=sum(r['settled'] for r in rows), signal_violation_episodes=sum(r['red_crossings'] > 0 for r in rows), curve_violation_episodes=sum(r['curve_violation_steps'] > 0 for r in rows),
                override_episodes=sum(r['override_episode'] for r in rows), override_steps=sum(r['override_steps'] for r in rows), fallback_steps=sum(r['fallback_steps'] for r in rows),
                fallback_noncreeping_steps=sum(r['fallback_noncreeping_steps'] for r in rows), guard_rejections=sum(r['guard_rejections'] for r in rows),
                mean_Ij_all=float(np.mean([r['Ij'] for r in rows])), mean_time_all=float(np.mean([r['time_s'] for r in rows])), mean_E_all=float(np.mean([r['E_Wh'] for r in rows])), mean_R_all=float(np.mean([r['R'] for r in rows])),
                mean_Ij_settled=(float(np.mean([r['Ij'] for r in done])) if done else None), mean_time_settled=(float(np.mean([r['time_s'] for r in done])) if done else None), mean_E_settled=(float(np.mean([r['E_Wh'] for r in done])) if done else None),
                peak_jerk=max(r['jerk_max'] for r in rows))


def fmt(v, nd=3): return '' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def main():
    out = {}; md = []
    r1 = episode_rows('R1'); r2 = episode_rows('R2'); eps = [r for r in r1 + r2 if r.get('status') == 'complete']
    out['episodes'] = dict(R1_complete=sum(r.get('status') == 'complete' for r in r1), R1_not_run=sum(r.get('status') != 'complete' for r in r1), R2_complete=sum(r.get('status') == 'complete' for r in r2), R2_not_run=sum(r.get('status') != 'complete' for r in r2))
    write_csv(HERE / 'episodes.csv', eps)
    # ---------------- R1
    c1 = [r for r in r1 if r.get('status') == 'complete']; by = {(r['policy'], r['condition_id'], r['profile']): r for r in c1}; pairs = []
    keys = ['settled', 'red_crossings', 'curve_violation_steps', 'override_episode', 'override_steps', 'numeric_exceedance_steps', 'fallback_steps', 'fallback_noncreeping_steps', 'Ij', 'jerk_max', 'time_s', 'E_Wh', 'R', 'guard_rejections']
    for pol in ('S0', 'S1', 'S2'):
        for c in range(9):
            a, b = by.get((pol, c, 'paper_v25')), by.get((pol, c, 'guard_v26'))
            if not a or not b: continue
            p = dict(pair_id=f'R1_{pol}_c{c:02d}', policy=pol, condition_id=c, archived_override_episode=int((pol, c) in ARCHIVED_OVERRIDE), both_settled=int(a['settled'] and b['settled']))
            for k in keys: p['paper_' + k] = a[k]; p['guard_' + k] = b[k]; p['delta_' + k] = (b[k] - a[k]) if not isinstance(a[k], bool) else int(b[k]) - int(a[k])
            p['paper_first_override'] = a['first_override_category']; p['guard_first_override'] = b['first_override_category']; p['paper_terminal'] = a['terminal']; p['guard_terminal'] = b['terminal']
            for k in ('Ij', 'override_steps', 'time_s', 'E_Wh', 'R'):
                d = p['delta_' + k]; p['class_' + k] = 'same' if abs(d) <= 1e-9 else ('lower' if d < 0 else 'higher')
            pairs.append(p)
    write_csv(HERE / 'pairs_R1.csv', pairs)
    arms = {prof: arm_summary([r for r in c1 if r['profile'] == prof]) for prof in ('paper_v25', 'guard_v26') if any(r['profile'] == prof for r in c1)}
    per_seed = {f'{pol}/{prof}': arm_summary([r for r in c1 if r['policy'] == pol and r['profile'] == prof]) for pol in ('S0', 'S1', 'S2') for prof in ('paper_v25', 'guard_v26') if any(r['policy'] == pol and r['profile'] == prof for r in c1)}
    out['R1'] = dict(arms=arms, per_seed=per_seed, pairs=len(pairs), both_settled_pairs=sum(p['both_settled'] for p in pairs),
                     counts={k: {c: sum(p['class_' + k] == c for p in pairs) for c in ('lower', 'same', 'higher')} for k in ('Ij', 'override_steps', 'time_s', 'E_Wh', 'R')},
                     paired_delta_mean={k: float(np.mean([p['delta_' + k] for p in pairs])) for k in ('Ij', 'time_s', 'E_Wh', 'R', 'override_steps', 'fallback_steps')} if pairs else {},
                     paired_delta_median={k: float(np.median([p['delta_' + k] for p in pairs])) for k in ('Ij', 'time_s', 'E_Wh', 'R')} if pairs else {},
                     both_settled_delta_mean={k: float(np.mean([p['delta_' + k] for p in pairs if p['both_settled']])) for k in ('Ij', 'time_s', 'E_Wh', 'R')} if any(p['both_settled'] for p in pairs) else {})
    md.append('## R1 main table (all attempts per arm; means over all N and over settled trips)\n')
    md.append('| arm | N | settled | signal-violation ep. | curve-violation ep. | override ep. | override steps | fallback steps (non-creeping) | guard rejections | mean I_j all / settled | mean time all / settled | mean Wh all / settled | mean R | peak jerk |')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|')
    for name, s in list(arms.items()) + list(per_seed.items()):
        md.append(f"| {name} | {s['N']} | {s['settled']} | {s['signal_violation_episodes']} | {s['curve_violation_episodes']} | {s['override_episodes']} | {s['override_steps']} | {s['fallback_steps']} ({s['fallback_noncreeping_steps']}) | {s['guard_rejections']} | {fmt(s['mean_Ij_all'])} / {fmt(s['mean_Ij_settled'])} | {fmt(s['mean_time_all'])} / {fmt(s['mean_time_settled'])} | {fmt(s['mean_E_all'])} / {fmt(s['mean_E_settled'])} | {fmt(s['mean_R_all'])} | {fmt(s['peak_jerk'])} |")
    md.append('\n## R1 pairs (guard_v26 minus paper_v25)\n')
    md.append('| pair | archived override | paper: term/override steps/I_j/time/Wh/R | guard: term/override steps/I_j/time/Wh/R | ΔI_j | Δtime | ΔWh | ΔR | Δoverride steps | guard rej. | first override paper → guard |'); md.append('|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|')
    for p in pairs:
        md.append(f"| {p['pair_id']} | {p['archived_override_episode']} | {p['paper_terminal']}/{p['paper_override_steps']}/{fmt(p['paper_Ij'])}/{p['paper_time_s']}/{fmt(p['paper_E_Wh'],1)}/{fmt(p['paper_R'])} | {p['guard_terminal']}/{p['guard_override_steps']}/{fmt(p['guard_Ij'])}/{p['guard_time_s']}/{fmt(p['guard_E_Wh'],1)}/{fmt(p['guard_R'])} | {fmt(p['delta_Ij'])} | {fmt(p['delta_time_s'],1)} | {fmt(p['delta_E_Wh'],1)} | {fmt(p['delta_R'])} | {p['delta_override_steps']} | {p['guard_guard_rejections']} | {p['paper_first_override'] or '-'} → {p['guard_first_override'] or '-'} |")
    if pairs:
        md.append(f"\nCounts over {len(pairs)} pairs: " + '; '.join(f"{k}: lower {v['lower']}, same {v['same']}, higher {v['higher']}" for k, v in out['R1']['counts'].items()))
    ev, guard = override_events('R1'); write_csv(HERE / 'events_R1.csv', ev); write_csv(HERE / 'guard_events_R1.csv', guard)
    cat = {}
    for e in ev: cat.setdefault(e['profile'], {}).setdefault(e['category'], set()).add((e['policy'], e['condition_id']))
    out['R1']['override_events'] = dict(total_override_substeps={prof: sum(1 for e in ev if e['profile'] == prof) for prof in ('paper_v25', 'guard_v26')}, episodes_by_category={prof: {c: sorted(v) for c, v in d.items()} for prof, d in cat.items()}, guard_rejection_events=len(guard))
    md.append('\n## R1 mechanism: override episodes by first-event category (episode counted once per category it contains)\n')
    for prof, d in cat.items(): md.append(f"- {prof}: " + '; '.join(f"{c}: {len(v)} episodes {sorted(v)}" for c, v in sorted(d.items())))
    md.append(f"- guard rejection events (range update rejected by the inward-crossing guard): {len(guard)}")
    # ---------------- R2
    c2 = [r for r in r2 if r.get('status') == 'complete']; p2 = []
    for c in range(9):
        k = next((r for r in c2 if r['condition_id'] == c), None); s0 = by.get(('S0', c, 'paper_v25'))
        if not k or not s0: continue
        p = dict(pair_id=f'R2_c{c:02d}', condition_id=c)
        for key in ('settled', 'red_crossings', 'curve_violation_steps', 'override_episode', 'override_steps', 'fallback_steps', 'Ij', 'jerk_max', 'time_s', 'E_Wh', 'R'):
            p['constant_' + key] = k[key]; p['S0_' + key] = s0[key]; p['delta_S0_minus_constant_' + key] = (s0[key] - k[key]) if not isinstance(k[key], bool) else int(s0[key]) - int(k[key])
        for pol in ('S1', 'S2'):
            r = by.get((pol, c, 'paper_v25'))
            for key in ('Ij', 'time_s', 'E_Wh', 'R', 'settled'):
                p[f'delta_{pol}_minus_constant_' + key] = (None if not r else ((r[key] - k[key]) if not isinstance(k[key], bool) else int(r[key]) - int(k[key])))
        p2.append(p)
    write_csv(HERE / 'pairs_R2.csv', p2)
    if p2:
        out['R2'] = dict(constant=arm_summary(c2), S0=arm_summary([by[('S0', c, 'paper_v25')] for c in range(9) if ('S0', c, 'paper_v25') in by]), pairs=len(p2),
                         delta_S0_minus_constant_mean={k: float(np.mean([p['delta_S0_minus_constant_' + k] for p in p2])) for k in ('Ij', 'time_s', 'E_Wh', 'R')},
                         delta_S0_minus_constant_median={k: float(np.median([p['delta_S0_minus_constant_' + k] for p in p2])) for k in ('Ij', 'time_s', 'E_Wh', 'R')})
        md.append('\n## R2: constant u=+1 (paper_v25, same detector/memory/executor/protocol) versus S0/paper_v25 from R1\n')
        md.append('| condition | constant: term/override/I_j/time/Wh/R | S0: term/override/I_j/time/Wh/R | Δ(S0−const) I_j | Δtime | ΔWh | ΔR | Δ(S1−const) R | Δ(S2−const) R |'); md.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
        for p in p2:
            md.append(f"| {p['condition_id']} | {'settled' if p['constant_settled'] else 'fail'}/{p['constant_override_steps']}/{fmt(p['constant_Ij'])}/{p['constant_time_s']}/{fmt(p['constant_E_Wh'],1)}/{fmt(p['constant_R'])} | {'settled' if p['S0_settled'] else 'fail'}/{p['S0_override_steps']}/{fmt(p['S0_Ij'])}/{p['S0_time_s']}/{fmt(p['S0_E_Wh'],1)}/{fmt(p['S0_R'])} | {fmt(p['delta_S0_minus_constant_Ij'])} | {fmt(p['delta_S0_minus_constant_time_s'],1)} | {fmt(p['delta_S0_minus_constant_E_Wh'],1)} | {fmt(p['delta_S0_minus_constant_R'])} | {fmt(p['delta_S1_minus_constant_R'])} | {fmt(p['delta_S2_minus_constant_R'])} |")
        md.append('\nmean Δ(S0−constant): ' + ', '.join(f"{k} {v:.3f}" for k, v in out['R2']['delta_S0_minus_constant_mean'].items()) + '; median: ' + ', '.join(f"{k} {v:.3f}" for k, v in out['R2']['delta_S0_minus_constant_median'].items()))
    # ---------------- R3
    d3 = RES / 'R3'
    if (d3 / 'progress.json').exists():
        sys.path.insert(0, str(REPO / 'timing' / 'code')); import analyze as A, probe as PT
        prog = json.load(open(d3 / 'progress.json')); rows3 = prog['rows']; assert prog['source_hash'] == PT.source_hash()
        traces = {}; logs = {}
        for r in rows3: traces[r['mode'], r['condition_id']], logs[r['mode'], r['condition_id']] = A.independent_check(d3, r)
        byr = {(r['mode'], r['condition_id']): r for r in rows3}; p3 = []
        for i in sorted({r['condition_id'] for r in rows3}):
            if ('color', i) not in byr or ('timing', i) not in byr: continue
            a, b = byr['color', i], byr['timing', i]
            p = dict(condition_id=i, condition=json.dumps(a['condition']), both_settled=int(a['settled'] and b['settled']), color_settled=int(a['settled']), timing_settled=int(b['settled']),
                     color_window_complete=int(a['signal_window_complete']), timing_window_complete=int(b['signal_window_complete']))
            for k in A.FIELDS: p['color_' + k] = a[k]; p['timing_' + k] = b[k]; p['delta_' + k] = b[k] - a[k]
            p['window_Ij_partial'] = int(not (a['signal_window_complete'] and b['signal_window_complete']))
            z0, z1 = traces['color', i], traces['timing', i]; n = min(len(z0), len(z1)); diff = np.flatnonzero(np.abs(z0[:n, 8] - z1[:n, 8]) > 1e-6)
            p['identical_trajectories'] = int(len(diff) == 0 and len(z0) == len(z1)); p['first_divergence'] = ''
            if len(diff):
                k = int(diff[0]); assert np.allclose(z0[:k, :], z1[:k, :], rtol=0, atol=1e-9)
                lg = logs['timing', i][k]; p['first_divergence'] = json.dumps(dict(step=k, t=float(z0[k, 0]), x=float(z0[k, 2]), v=float(z0[k, 4]), color_action=float(z0[k, 8]), timing_action=float(z1[k, 8]), signal_distance=lg['signal_distance'], remaining=lg['remaining'], timing_reason=lg['signal_reason']))
            p3.append(p)
        write_csv(HERE / 'pairs_R3.csv', p3)
        agg = {m: dict(n=len([r for r in rows3 if r['mode'] == m]), settled=sum(r['settled'] for r in rows3 if r['mode'] == m), red=sum(r['violations'] for r in rows3 if r['mode'] == m),
                       overspeed_eps=sum(r['local_overspeed_time_s'] > 0 for r in rows3 if r['mode'] == m), physical=sum(r['physical_violation_steps'] for r in rows3 if r['mode'] == m), fallback=sum(r['layer_fallback_steps'] for r in rows3 if r['mode'] == m),
                       peak_jerk=max(r['jerk_max'] for r in rows3 if r['mode'] == m), timing_tests=sum(r['timing_test_steps'] for r in rows3 if r['mode'] == m), countdown_steps=sum(r['countdown_visible_steps'] for r in rows3 if r['mode'] == m)) for m in ('color', 'timing')}
        both = [p for p in p3 if p['both_settled']]
        out['R3'] = dict(rows=len(rows3), pairs=len(p3), aggregate=agg, identical_pairs=sum(p['identical_trajectories'] for p in p3), diverging_pairs=[p['condition_id'] for p in p3 if not p['identical_trajectories']],
                         partial_window_pairs=[p['condition_id'] for p in p3 if p['window_Ij_partial']], both_settled_pairs=len(both),
                         both_settled_delta_mean={k: float(np.mean([p['delta_' + k] for p in both])) for k in ('signal_Ij', 'Ij', 'time_s', 'E_Wh')} if both else {},
                         complete_window_delta_signal_Ij=[(p['condition_id'], p['delta_signal_Ij']) for p in p3 if not p['window_Ij_partial']])
        md.append('\n## R3: 18 extension conditions (same route, frozen actor A; current color vs color + timing)\n')
        md.append('| cond | (soc,T,v0,offset) | settled c/t | red c/t | overspeed s c/t | fallback c/t | window I_j c/t (partial?) | full I_j c/t | time c/t | Wh c/t | R c/t | trajectories | first difference |'); md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for p in p3:
            md.append(f"| {p['condition_id']} | {p['condition']} | {p['color_settled']}/{p['timing_settled']} | {p['color_violations']}/{p['timing_violations']} | {p['color_local_overspeed_time_s']}/{p['timing_local_overspeed_time_s']} | {p['color_layer_fallback_steps']}/{p['timing_layer_fallback_steps']} | {fmt(p['color_signal_Ij'])}/{fmt(p['timing_signal_Ij'])}{' (partial)' if p['window_Ij_partial'] else ''} | {fmt(p['color_Ij'])}/{fmt(p['timing_Ij'])} | {p['color_time_s']}/{p['timing_time_s']} | {fmt(p['color_E_Wh'],1)}/{fmt(p['timing_E_Wh'],1)} | {fmt(byr['color',p['condition_id']]['R'])}/{fmt(byr['timing',p['condition_id']]['R'])} | {'identical' if p['identical_trajectories'] else 'different'} | {p['first_divergence'] or '-'} |")
        md.append(f"\nAggregate: " + json.dumps(agg))
    (HERE / 'tables.md').write_text('\n'.join(md) + '\n'); (HERE / 'metrics_summary.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    figures(pairs, c1)
    print(json.dumps(out['episodes']), '\nR1 arms:', json.dumps({k: {kk: v[kk] for kk in ('N', 'settled', 'signal_violation_episodes', 'curve_violation_episodes', 'override_episodes', 'mean_Ij_all', 'mean_time_all', 'mean_E_all', 'mean_R_all')} for k, v in arms.items()}, indent=1))


def figures(pairs, c1):
    if not pairs: return
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, key, lab in zip(axs, ('Ij', 'time_s', 'E_Wh', 'override_steps'), ('ΔI_j (m²/s⁵)', 'Δtime (s)', 'ΔWh', 'Δoverride substeps')):
        for pol, col in (('S0', 'C0'), ('S1', 'C1'), ('S2', 'C2')):
            ps = [p for p in pairs if p['policy'] == pol]
            ax.scatter([p['condition_id'] for p in ps], [p['delta_' + key] for p in ps], color=col, label=pol, s=28, marker='o' if True else 'x')
            for p in ps:
                if not p['both_settled']: ax.scatter([p['condition_id']], [p['delta_' + key]], facecolors='none', edgecolors='k', s=90)
        ax.axhline(0, color='k', lw=.6); ax.set_xlabel('condition'); ax.set_title(lab + ' (guard − paper)'); ax.set_xticks(range(9))
    axs[0].legend(fontsize=8); fig.suptitle('R1 paired effects; hollow ring = a pair with an unsettled arm'); fig.tight_layout(); fig.savefig(HERE / 'fig_R1_pairs.png', dpi=150); plt.close(fig)
    # target-update trajectories: the pair with most guard rejections, else the pair with the largest |ΔI_j|
    best = max(pairs, key=lambda p: (p['guard_guard_rejections'], abs(p['delta_Ij'])))
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for prof, col in (('paper_v25', 'C0'), ('guard_v26', 'C3')):
        d = RES / 'R1' / f"R1_{best['policy']}_c{best['condition_id']:02d}_{prof}"
        with gzip.open(d / 'substeps.json.gz', 'rt') as f: sub = json.load(f)
        x = [s['post']['x'] for s in sub]; dl = [s['memory']['sig']['d_line'] if s['memory']['sig']['seen'] else np.nan for s in sub]
        axs[0].plot(x, dl, color=col, label=f'{prof}: memory stop-line range'); axs[1].plot(x, [s['post']['v'] for s in sub], color=col, label=f'{prof}: speed')
        cand = [(s['post']['x'], s['update']['candidate']) for s in sub if s['update'].get('range_guard')]
        if cand: axs[0].scatter([c[0] for c in cand], [c[1] for c in cand], color=col, marker='x', s=50, label=f'{prof}: rejected candidate')
        ov = [(s['post']['x'], s['post']['v']) for s in sub if s['override_flag']]
        if ov: axs[1].scatter([o[0] for o in ov], [o[1] for o in ov], color=col, marker='v', s=40, label=f'{prof}: jerk override')
    axs[0].axhline(15, color='gray', lw=.6, ls='--'); axs[0].set_ylabel('d_line estimate (m)'); axs[1].set_ylabel('v (m/s)'); axs[1].set_xlabel('x (m)'); axs[0].set_xlim(2700, 3100)
    for ax in axs: ax.legend(fontsize=7)
    fig.suptitle(f"Target-update trajectories, {best['policy']} condition {best['condition_id']}"); fig.tight_layout(); fig.savefig(HERE / 'fig_R1_target_updates.png', dpi=150); plt.close(fig)


if __name__ == '__main__': main()
