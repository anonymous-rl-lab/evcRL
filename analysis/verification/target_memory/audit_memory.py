"""Record-based target-update accounting and complete visual memory replay; no vehicle step."""
from pathlib import Path
import sys,copy,csv,gzip,json,math,hashlib
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parents[2]
for path in [P.parent/'software/EvcRL/src',P/'software/EvcRL/src']:
 if path.exists():sys.path.insert(0,str(path));break
from evcrl.memory import VisionMemory
from evcrl.execution import Executor
D=Path(__file__).resolve().parent
B=lambda d:max(.85*d-6.,0.)
def dump(name,data): (D/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvdump(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (D/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def recurse_compare(a,b,p=''):
 errors=[]
 for k,v in b.items():
  if k not in a:errors.append((p+k,'missing'));continue
  w=a[k]
  if isinstance(v,dict):errors+=recurse_compare(w,v,p+k+'.')
  elif v is None:
   if w is not None and not(isinstance(w,float) and not math.isfinite(w)):errors.append((p+k,(w,v)))
  elif isinstance(v,(bool,str)):
   if w!=v:errors.append((p+k,(w,v)))
  elif abs(float(w)-v)>1e-8:errors.append((p+k,(w,v)))
 return errors

def accounting():
 data=json.loads(gzip.decompress((P/'verification/framework/log_excerpts.json.gz').read_bytes()))
 old={r['branch_id']:r for r in csv.DictReader((P/'verification/diagnostics/pb_event_rows.csv').open())}
 output=[];timing=[]
 for branch in data['v6']:
  if branch['situation']!='stop':continue
  rr=branch['recs'];previous=old[branch['branch_id']];ad=int(previous['adoption_index']);C=Executor(branch['jerk'])
  end=next((i for i,r in enumerate(rr) if r['v']<.05 or r['x']>=3000),len(rr)-1)
  first=next((i for i in range(end+1) if rr[i]['fallback'] or rr[i]['jerk_override']),None)
  timing.append(dict(branch_id=branch['branch_id'],tag=branch['tag'],adoption=ad,first_event=first,relative=None if first is None else first-ad))
  if previous['later_event']!='True':continue
  event=int(previous['event_index'])
  trigger=event-1
  if previous['event_class']=='adoption_negative':
   trigger=next(i for i in range(1,ad+1) if rr[i]['mem_d_line'] is not None and math.isfinite(C.backup_distance(rr[i]['v'],rr[i]['a'])) and B(rr[i]['mem_d_line'])-C.backup_distance(rr[i]['v'],rr[i]['a'])<0)
  z,w=rr[trigger-1],rr[trigger];dx=w['x']-z['x'];prop=z['mem_d_line']-dx;e=w['mem_d_line']-prop
  Db=C.backup_distance(w['v'],w['a']);mt=B(w['mem_d_line'])-Db;mp=B(prop)-Db
  response=trigger+1
  eventjerk=abs(rr[response]['a']-rr[trigger]['a'])/.5
  cls=previous['event_class']
  if cls=='adoption_negative':sub='adoption_step' if trigger==ad else 'before_adoption'
  elif cls=='later_highspeed':sub='highspeed_update' if mp>=0 else 'highspeed_already_negative'
  else:sub='creep' if w['v']<.5 else 'freeze_crossing'
  eprop=dx-w['v']*.5
  o=dict(branch_id=branch['branch_id'],tag=branch['tag'],policy=branch['policy'],jerk=branch['jerk'],adoption=ad,event=event,trigger=trigger,subclass=sub,
   e=e,e_propagation=eprop,e_update=e-eprop,margin_propagated=mp,margin_updated=mt,innovation_identity_error=abs(mt-mp-.85*e),v_trigger=w['v'],a_trigger=w['a'],d_previous=z['mem_d_line'],d_after=w['mem_d_line'],
   crossed15=z['mem_d_line']>=15 and w['mem_d_line']<15,response=response,override_flag=int(rr[response]['jerk_override']),post_adoption_override_flag=int(rr[event]['jerk_override']),post_adoption_realized_jerk=abs(rr[event]['a']-rr[event-1]['a'])/.5,realized_jerk=eventjerk,actual_exceedance=eventjerk>branch['jerk']+1e-6)
  output.append(o)
 summary=dict(n=len(output),classes=dict(Counter(o['subclass'] for o in output)),positive_propagated=sum(o['margin_propagated']>=0 for o in output),
  positive_noncreep=sum(o['margin_propagated']>=0 and o['subclass']!='creep' for o in output),noncreep=sum(o['subclass']!='creep' for o in output),
  identity_max=max(o['innovation_identity_error'] for o in output),creep_propagation_residual=max(abs(o['e_update']) for o in output if o['subclass']=='creep'),
  flag_count=sum(o['override_flag'] for o in output),actual_exceedances=sum(o['actual_exceedance'] for o in output),post_adoption_flags=sum(o['post_adoption_override_flag'] for o in output),post_adoption_actual_exceedances=sum(o['post_adoption_realized_jerk']>o['jerk']+1e-6 for o in output),
  timing=dict(Counter('none' if r['relative'] is None else 'before' if r['relative']<0 else 'at' if r['relative']==0 else 'after' for r in timing)))
 summary['timing_by_speed']={tag:dict(Counter('none' if r['relative'] is None else 'before' if r['relative']<0 else 'at' if r['relative']==0 else 'after' for r in timing if r['tag']==tag)) for tag in sorted({r['tag'] for r in timing})}
 csvdump('event_accounting.csv',output);dump('event_accounting.json',dict(summary=summary,rows=output));dump('first_event_timing.json',timing)
 return summary

def memory_replay():
 episodes=json.loads(gzip.decompress((D/'pv_full_layer_logs.json.gz').read_bytes()))
 totals=Counter();epout=[];stepout=[];mismatches=[];phase_changes=[]
 event_ids={(int(r['seed']),int(r['condition'])):int(r['first_override_index']) for r in csv.DictReader((P/'verification/diagnostics/pv_event_rows.csv').open()) if r['event_class']=='signal_terminal'}
 for ep in episodes:
  opts=dict(det_thr={'traffic_light':.4,'curve_sign':.5,'end_marker':.55,'release_sign':.2})
  old=VisionMemory(**opts);m1=VisionMemory(**opts,propagation='consistent');m2=VisionMemory(**opts,propagation='consistent',guard=True)
  count=Counter();C=Executor(2.)
  for i,r in enumerate(ep['rows']):
   prev=ep['rows'][i-1] if i else r;dt=0. if i==0 else r['t']-prev['t'];ds=0. if i==0 else r['x']-prev['x'];detail=r['perceived_detail']
   if detail is None or 'dets' not in detail:raise ValueError(('incomplete log',ep['seed'],ep['condition'],i))
   dets=detail['dets'];probs=detail.get('probs')
   local=copy.deepcopy(old);local.propagation='consistent';local.guard=True
   old.update(dets,probs,r['v'],dt,odom_ds=ds)
   err=recurse_compare(old.snapshot(),r['memory'])
   if err:mismatches.append(dict(seed=ep['seed'],condition=ep['condition'],index=i,errors=err[:3]))
   local.update(dets,probs,r['v'],dt,odom_ds=ds)
   m1.update(dets,probs,r['v'],dt,odom_ds=ds);m2.update(dets,probs,r['v'],dt,odom_ds=ds)
   count['records']+=1;count['original_mismatch_records']+=bool(err)
   count['local_guard_rejections']+=bool(local.update_diagnostic.get('range_guard'))
   count['recursive_guard_rejections']+=bool(m2.update_diagnostic.get('range_guard'))
   count['m1_phase_differences']+=old.sig['phase']!=m1.sig['phase'];count['m2_phase_differences']+=old.sig['phase']!=m2.sig['phase']
   if old.sig['phase']!=m2.sig['phase']:
    phase_changes.append(dict(seed=ep['seed'],condition=ep['condition'],index=i,t=r['t'],original=old.sig['phase'],candidate=m2.sig['phase'],original_d=old.sig['d_line'],candidate_d=m2.sig['d_line']))
   if event_ids.get((ep['seed'],ep['condition']))==i:
    states={'original':old,'local_guard':local,'recursive_odometry':m1,'recursive_guard':m2}
    row=dict(seed=ep['seed'],condition=ep['condition'],index=i,v=r['v'],x=r['x'])
    for key,m in states.items():
     d=m.sig['d_line'];a=0. if i==0 else prev['applied'];Db=C.backup_distance(r['v'],a)
     row[key+'_distance']=d;row[key+'_margin']=None if d is None or not math.isfinite(Db) else B(d)-Db
     row[key+'_guard']=bool(m.update_diagnostic.get('range_guard'))
    stepout.append(row)
  totals.update(count);epout.append(dict(seed=ep['seed'],condition=ep['condition'],**dict(count)))
 summary=dict(episodes=len(episodes),**dict(totals),scientific_vehicle_rollouts=0,new_training_steps=0,
  gate_pass=totals['original_mismatch_records']==0,scope='recursive memory under fixed recorded motion/detections; not changed policy trajectories')
 dump('memory_replay.json',dict(summary=summary,episodes=epout,selected_events=stepout,phase_changes=phase_changes,mismatches=mismatches[:50]));return summary

if __name__=='__main__':
 a=accounting();b=memory_replay();dump('summary.json',dict(accounting=a,memory_replay=b));print(json.dumps(dict(accounting=a,memory_replay=b),indent=2))
