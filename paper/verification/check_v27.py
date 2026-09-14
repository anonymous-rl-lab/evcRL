"""Manuscript-to-record checks for v27; no new training or policy driving comparisons."""
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def read(n):return json.loads((P/'verification'/n).read_text())
def tables(s):
    return {k:[[c.strip() for c in l.strip().strip('|').split('|')] for l in b.splitlines()[2:]] for k,b in re.findall(r'\*\*Table ([A-Z0-9-]+) —[^\n]*\n\s*((?:\|[^\n]*\n)+)',s)}
def check(m,s):
    checks=[]
    def ck(k,b):
        assert b,k
        checks.append(k)
    def near(k,x,y,tol=.000501):ck(k,abs(float(x)-float(y))<=tol)
    mt,st=tables(m),tables(s);base=read('retained_outcome_tables.json')
    for k,bk in [('III','III'),('IV','IV'),('VII','VII')]:ck('retained main '+k,mt[k]==base['Manuscript'][bk])
    for k,v in base['Supplementary'].items():
        ck('retained SI '+k,st[k]==v)
    ck('main tables I-VIII',set(mt)==set(['I','II','III','IV','V','VI','VII','VIII']))
    ck('SI tables S1-S19',set(st)=={'S'+str(i) for i in range(1,20)})
    ck('protocol rows',len(mt['I'])==6);ck('contribution rows',len(mt['II'])==3)
    visual=read('visual_rows.json');vd={(r['seed'],r['arm']):r['rows'] for r in visual}
    ck('108 original visual rows',sum(len(r['rows']) for r in visual)==108)
    for row,arm in zip(mt['V'],('frozen','supervised')):
        rs=sum([vd[i,arm] for i in range(3)],[]);good=[r for r in rs if r['arrived'] and r['settled']]
        ck('V '+arm+' settled',row[1]==f'{len(good)}/27')
        ck('V '+arm+' violations',row[2]==f'{sum(r["violations"] for r in rs)} / {sum(r["offroad_substeps"] for r in rs)}')
        near('V '+arm+' return',row[3],np.mean([r['R'] for r in rs]))
        for col,k in [(4,'time_s'),(5,'E_Wh'),(6,'Ij')]:near('V '+arm+' '+k,row[col],np.mean([r[k] for r in good]))
        ck('V '+arm+' overrides',row[7]==f'{sum(r["jerk_override_steps"]>0 for r in rs)}/27')
    ck('probe remains descriptive',mt['V'][2][1:]==['9/9','Not aligned','-39.246','217.889','605.974','100.954','Not aligned'])
    for row,z in zip(st['S6'],visual):
        rs=z['rows'];good=[r for r in rs if r['arrived'] and r['settled']]
        ck('S13 identity',row[:2]==[str(z['seed']),z['mode']])
        for c,v in enumerate([len(good),np.mean([r['R'] for r in rs]),np.mean([r['Ij'] for r in good]),np.mean([r['time_s'] for r in good]),np.mean([r['E_Wh'] for r in good])],2):near('S13 metric',row[c],v)
    for row in st['S7']:
        arm={'S':'supervised','J':'joint'}[row[0].split()[0]];seed=int(row[1])
        a={r['condition_id']:r for r in vd[seed,arm]};f={r['condition_id']:r for r in vd[seed,'frozen']}
        ids=[i for i in f if all(r['arrived'] and r['settled'] for r in [a[i],f[i]])]
        ck('S14 IDs',row[2]==','.join(map(str,ids)));near('S14 n',row[3],len(ids),0)
        for col,k in enumerate(['Ij','time_s','E_Wh','R'],4):near('S14 '+k,row[col],np.mean([a[i][k]-f[i][k] for i in ids]))
    ev=read('diagnostics/event_summary.json');old=read('framework/audit_results.json')
    ck('event classes',ev['event_classes']=={'adoption_negative':31,'later_highspeed':40,'terminal':124})
    ck('195 event sources',ev['later_events']==195 and ev['pre_event_target_negative']==195 and ev['pre_event_true_negative']==0)
    for row,g in zip(mt['VIII'],ev['cohort_margins']):
        near('VIII n',row[1],g['n'],0)
        lo,hi=g['target_range'];ck('VIII actual range',row[2]==f'{lo:.2f}–{hi:.2f}')
        for col,k in [(3,'negative'),(4,'below5'),(5,'later')]:near('VIII '+k,row[col],g[k],0)
    for row,key in zip(mt['VI'],['endpoint','phase_flip','range_update','signal_terminal']):near('VII '+key,row[1],ev['visual_classes'][key],0)
    ck('paper 109 denominator','109/324' in m and '15/108' in m)
    ck('highspeed denominator','46/324' in m and '25/108' in m)
    for row,mode in zip(st['S15'],['buffered','estimated_line','true_line']):near('S22 target test',row[1],ev['target_check_reasons'][mode].get('feasible',0),0)
    pv=list(csv.DictReader((P/'verification/diagnostics/pv_event_rows.csv').open()))
    vs=[r for r in pv if r['override_episode']=='True'];ck('nine localized',len(vs)==9)
    for row,r in zip(st['S8'],vs):
        ck('S23 identity',row[0]==f'{r["seed"]} / {r["condition"]}');ck('S23 phase',row[1]==r['first_color'])
        near('S23 x',row[2],r['x'],.050001);near('S23 v',row[3],r['v'],.005001)
    for row,g,og in zip(st['S13'],ev['floor_groups'],sorted(old['pass_reference_groups'],key=lambda r:(r['tag']!='v6_formal_v22',r['info']!='min'))):
        near('S20 n',row[2],g['n'],0)
        for col,key in [(3,'observed_delta_time_s'),(5,'residual_interp_s')]:
            q=og[key]['quantiles'];ck('S20 '+key,row[col]==f'{q[0]:.3f}–{q[-1]:.3f}')
        lo,hi=g['range'];ck('S20 floor',row[4]==f'{lo:.3f}–{hi:.3f}')
        near('S20 finite interpolation',row[6],og['residual_interp_s']['finite'],0)
    for index,(tag,j) in enumerate((tag,j) for tag in ['22','16'] for j in [2.,3.,4.]):
        b=ev['lp_complete_boundaries'][tag][str(j)];r=st['S1'][index]
        for col,k in [(2,'zero_penalty_max_delta'),(3,'common_feasible_max_delta'),(4,'first_infeasible')]:near('S15 '+k,r[col],b[k],0)
        ck('LP stop component constant',b['stop_loss_span']<1e-6)
    for row,g in zip(st['S12'][:2],ev['cohort_margins']):
        lo,hi=g['target_range'];ck('S19 target range',row[4]==f'{lo:.2f}–{hi:.2f}; {g["negative"]} negative')
    ck('missing targets retained','missing for 9' in st['S12'][2][4])
    ck('floor nonnegative',ev['floor_all_nonnegative'] and 'All floor-reference residuals' in m)
    ck('no false LP censoring',all(x not in m+s for x in ['Feasible through 14.5','scan limit','not found in the original scan']))
    ck('no old localization denial',all(x not in m+s for x in ['Not localized here','without attribution to this particular signal event','Not substituted for line margin']))
    ck('target update identity',ev['target_update_checks']['identity_max_error']<1e-10)
    ck('target update bound',ev['target_update_checks']['lower_bound_max_violation']<1e-10)
    ck('prior event audit scope',ev['new_training_steps']==0 and ev['new_vehicle_rollouts']==0)

    v27=read('target_memory/summary.json');acc=v27['accounting'];replay=read('target_memory/memory_replay.json')
    classes=['adoption_step','before_adoption','highspeed_update','highspeed_already_negative','freeze_crossing','creep']
    details=read('target_memory/event_accounting.json')['rows']
    for row,c in zip(st['S16'],classes):
        group=[r for r in details if r['subclass']==c]
        near('S24 '+c+' n',row[1],len(group),0)
        near('S24 '+c+' propagated',row[2],sum(r['margin_propagated']>=0 for r in group),0)
        near('S24 '+c+' jerk',row[3],sum(r['actual_exceedance'] for r in group),0)
    for row,r in zip(st['S17'],replay['selected_events']):
        ck('S25 event identity',row[0]==f'{r["seed"]} / {r["condition"]}')
        for col,k in [(1,'original_margin'),(2,'recursive_odometry_margin'),(3,'recursive_guard_margin')]:near('S25 '+k,row[col],r[k])
    for row,tag in zip(mt['VIII'],['v6_formal_v22','v6_formal_v16']):
        t=acc['timing_by_speed'][tag]
        ck('VIII first timing',row[6]=='/'.join(str(t.get(x,0)) for x in ['before','at','after','none']))
    ck('191 update cases','191/195' in m+s and acc['positive_propagated']==191)
    ck('159 noncreep update cases','159/163' in m and acc['positive_noncreep']==159 and acc['noncreep']==163)
    ck('creep propagation','32 creeping cases' in s and acc['creep_propagation_residual']<1e-12)
    ck('precise old replay','12,081 timestamps' in m and replay['summary']['records']==12081 and replay['summary']['original_mismatch_records']==0)
    ck('candidate bounded outcome','two of three' in m and sum(r['recursive_guard_margin']>=0 for r in replay['selected_events'])==2)
    ck('phase changes disclosed','two phase-memory records' in m and replay['summary']['m2_phase_differences']==2)
    ck('two aligned jerk windows',acc['flag_count']==acc['actual_exceedances']==99 and acc['post_adoption_flags']==acc['post_adoption_actual_exceedances']==88 and '99 versus 88' in s)
    ck('nominal propagation decomposition',r'e^{\rm prop}=-a_{k+1}\delta^2/2' in m)
    ck('candidate actual condition',all(t in m for t in [r'$p\ge15$',r'$c<15$',r'$c<p$']))
    ck('standalone software scope','locally installable EvcRL 0.1.0' in m and 'Ten software tests' in s)
    ck('algorithm 3 bounded','Algorithm 3 — Experimental guarded memory update' in s)
    ck('short known biographies',all('**'+n+'** is with the School' in m for n in ['Ziran Peng','Zeyu Fan']))

    abstract=m.split('## ABSTRACT\n\n')[1].split('\n\n*Index Terms')[0]
    ck('abstract length',150<=len(abstract.split())<=250)
    ck('abstract expanded algorithm','TD3' not in abstract and 'twin delayed deep deterministic policy gradient' in abstract)
    for text in ['27/27','26.50%','0.11 s','195']:ck('abstract '+text,text in abstract)
    ck('main equations 1-19',re.findall(r'\\tag\{([^}]+)\}',m)==list(map(str,range(1,20))))
    tags=re.findall(r'\\tag\{([^}]+)\}',m+s);ck('unique equation tags',len(tags)==len(set(tags)))
    refs=re.findall(r'^- \*\*\[([^]]+)\]\*\*',m,re.M);cites=set(re.findall(r'\[((?:R-|S)[A-Za-z0-9-]+)\]',m+s))
    ck('unique complete references',len(refs)==len(set(refs)) and set(refs)==cites)
    ck('SI sections',re.findall(r'^## ([A-Z])\.',s,re.M)==list('ABCDEFGHIJ'))
    ck('main figure labels',re.findall(r'\*\*Fig\. (\d+)\.',m)==['1','2','3'])
    ck('SI figure labels',re.findall(r'\*\*Fig\. (S\d+)\.',s)==['S'+str(i) for i in range(1,4)])
    for path in re.findall(r'!\[[^]]*\]\(([^)]+)\)',m+s):ck('figure '+path,(P/path).exists() and (P/path).with_suffix('.pdf').exists())
    ck('Algorithm 1 explicit','Algorithm 1: Release-aware action execution' in m)
    ck('Algorithm 2 offline','Algorithm 2 — Offline target-trigger diagnosis' in s)
    ck('complete supervisor',all(r'\mathcal L_{\rm '+x+'}' in s for x in ['heat','box','range','presence','ROI']))
    ck('information access beside main comparison','expanded executor access to phase timing' in m)

    headings=dict(re.findall(r'^### ([A-Z]\.\d+)\. (.+)$',s,re.M))
    required={
        'B.3':'Visual-supervision objective',
        'B.4':'Implemented actor and critic objectives',
        'D.1':'Release inverse, backup recursion, and retained intervals',
        'D.2':'One-update margin relation and its scope',
        'E.1':'Sampled reference and complete delay scans',
        'E.2':'Continuous maneuver and terminal assumptions',
        'E.3':'Proof: feasible preparations',
        'E.4':'Proof: optimal arrival-time difference',
        'G.5':'Nine S override episodes',
        'H.9':'Trigger-aligned propagation, update, and response',
        'H.10':'Complete memory replay and experimental guard'}
    for dest,title in required.items():ck('semantic destination '+dest,headings.get(dest)==title)
    for dest in re.findall(r'(?:Section |in )([A-Z]\.\d+)',m+s):
        ck('existing subsection '+dest,dest in headings)
    for phrase in ['Supplementary Section B.4','Supplementary Section D.1',
                   'Supplementary Section E.1','proved in E.3–E.4',
                   'Supplementary Section G.5','Supplementary Section H.3',
                   'Supplementary Section I']:
        ck('main semantic link '+phrase,phrase in m)
    for letter in 'ABDEHI':
        tags=re.findall(r'\\tag\{('+letter+r'\d+)\}',s)
        ck('ordered SI equations '+letter,tags==[letter+str(i+1) for i in range(len(tags))])
    equation_tags=set(re.findall(r'\\tag\{([^}]+)\}',s))
    for dest in re.findall(r'\(([A-Z]\d+)\)',s):ck('equation destination '+dest,dest in equation_tags)
    for dest in re.findall(r'Table (S\d+)',m+s):ck('table destination '+dest,dest in st)
    forbidden=['A supplied review table','corrected here','This revision','revised source-identity',
       'v19 paper snapshot','The proof is in Section D below','newly supplied','398 numerical',
       '650 further assertions','bounds the benefit','all 92 plus three events are repaired']
    for phrase in forbidden:ck('editorial cleanup '+phrase,phrase not in m+s)
    ck('separate detector and innovation notation',r'\zeta_k&=D_\psi' in m and r'e_{k+1}&=' in m)
    ck('one-update limit explicit','not combined into a recursive safety guarantee' in s)
    ck('memory feedback limit explicit','fixed motion omits the feedback of changed actions' in m)
    ck('known action contract retained','first-applied-action label' in m and 'raw target command' in m)
    ck('no dropped seed failure','14.68' in s and '12.82 conditions on five completions' in s)
    ck('matched rule late means retained','**2 of 8**' in s and '**5 of 8**' in s)
    ck('prior unrelated theory removed',all(x not in s for x in ['EVENT-FREE SPEED TRANSFER','HISTORY-CONDITIONED INFORMATION','Proposition F2','Raw value slope']))

    ck('no placeholders','@@' not in m+s)
    ck('no illegal control bytes',not any(ord(c)<32 and c not in '\n\r\t' for c in m+s))
    return dict(status='PASS',check_count=len(checks),checks=checks,abstract_words=len(abstract.split()),new_training_steps=0,new_learned_policy_rollouts=0)
if __name__=='__main__':
    m=(P/'Manuscript_v27.md').read_text();s=(P/'Supplementary_v27.md').read_text();out=check(m,s);caught=[]
    for before,after,supp in [
        ('| S, P-V | 27/27 |','| S, P-V | 26/27 |',False),
        ('| 12 | 25 | 111 |','| 12 | 25 | 110 |',False),
        ('109/324','103/324',False),
        ('| 0.000–0.837 |','| 0.000–0.937 |',True),
        ('| 16 | 4 | 10.0 | 15.5 | 16.0 |','| 16 | 4 | 10.0 | 14.5 | 16.0 |',True),
        ('| +4.144 |','| -9.870 |',True),
        ('proved in E.3–E.4','proved in E.1–E.2',False),
        ('not combined into a recursive safety guarantee','a recursive safety guarantee',True),
        ('159/163','158/163',False),
        ('12,081 timestamps','12,080 timestamps',False),
        ('| Terminal distance-freeze crossing | 92 | 92 | 70 |','| Terminal distance-freeze crossing | 92 | 92 | 71 |',True),
        ('| 2 / 6 | -2.804 | -2.912 | -2.912 |','| 2 / 6 | -2.804 | -2.912 | +2.912 |',True)]:
        assert before in (s if supp else m),before
        try:check(m if supp else m.replace(before,after),s.replace(before,after) if supp else s)
        except AssertionError as e:caught.append(str(e))
        else:raise AssertionError('Injected error missed: '+before)
    out['injected_errors_caught']=caught
    out['canonical_sha256']={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['Manuscript_v27.md','Supplementary_v27.md']}
    (P/'verification/manuscript_checks_v27.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
