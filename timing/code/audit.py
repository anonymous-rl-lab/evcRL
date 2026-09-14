"""Bounded preflight: information isolation, physical ledgers and resumption."""
import copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import probe as P

def main():
    checks=[]
    actor=P.load_actor();fp=P.source_hash()
    e0=P.ProbeEnv(.85,288.15,0.,'color');e1=P.ProbeEnv(.85,288.15,0.,'timing')
    assert np.array_equal(e0.reset(16.),e1.reset(16.))
    for x in (0.,1999.,2000.,2450.,2990.,3010.):
        e0.x=e1.x=x;e0.t=e1.t=10.
        assert np.array_equal(e0.obs(),e1.obs())
        with torch.inference_mode():
            assert torch.equal(actor(torch.as_tensor(e0.obs())),actor(torch.as_tensor(e1.obs())))
    checks.append('same_actor_and_observation_at_matched_states')
    assert not any(p.requires_grad for p in actor.parameters())
    checks.append('actor_frozen_no_gradients')

    for x in (0.,1999.,2000.,2450.,2990.,3010.):
        e0.x=x;e0.t=10.
        with patch.object(P.S.R,'time_to_change',side_effect=AssertionError('forbidden countdown read')):
            assert e0.signal_view('color')['remaining'] is None
        if not 0.<3000-x<=1000:
            with patch.object(P.S.R,'time_to_change',side_effect=AssertionError('out of range')):
                assert e0.signal_view('timing')['remaining'] is None
    checks.append('color_and_out_of_range_sensor_cannot_read_countdown')
    e0.x=0.
    with patch.object(P.S.R,'signal_green',side_effect=AssertionError('out-of-range color')):
        e0._stop_targets()
    checks.append('shared_stop_targets_do_not_read_out_of_range_color')

    targets=[(200.,0.)]
    for green in (False,None):
        a,i=P.execute(10.,0.,2.6,targets,-3.5,dict(distance=30.,green=green,remaining=None))
        b,j=P.execute(10.,0.,2.6,targets,-3.5,dict(distance=30.,green=green,remaining=4.))
        assert a==b and not i['timing_test'] and not j['timing_test']
    checks.append('red_and_unobserved_execution_identical')
    color,ci=P.execute(10.,0.,2.6,targets,-3.5,dict(distance=30.,green=True,remaining=None))
    timing,ti=P.execute(10.,0.,2.6,targets,-3.5,dict(distance=30.,green=True,remaining=.5))
    assert timing<color-1e-6 and not ti['fallback'] and ti['signal_reason']=='stop'
    assert abs(timing)<=1.+1e-8 and -3.5<=timing<=2.6
    checks.append('late_green_changes_action_without_disabling_physical_constraints')

    # Ground-truth shadow diagnostics cannot influence the selected color action.
    e0.x=2970.;e0.v=10.;e0.a=0.;e0.t=10.
    with patch.object(P.S.R,'time_to_change',return_value=.5):a=e0.project(2.6)
    with patch.object(P.S.R,'time_to_change',return_value=25.):b=e0.project(2.6)
    assert a==b
    checks.append('shadow_countdown_does_not_change_actual_color_action')

    condition=P.S.conditions('development')[0]
    with tempfile.TemporaryDirectory() as td:
        cp=Path(td)/'resume.pkl'
        P.rollout(actor,condition,'timing',cp,fp,stop_after=16)
        _,resumed=P.rollout(actor,condition,'timing',cp,fp,stop_after=32)
        _,direct=P.rollout(actor,condition,'timing',stop_after=32)
        assert np.array_equal(np.asarray(resumed.trace),np.asarray(direct.trace))
        assert resumed.layer_log==direct.layer_log
    checks.append('interrupted_resumed_state_observation_and_trace_exactly_equal')
    z=np.asarray(direct.trace);dt=z[:,1]-z[:,0]
    m=P.extra_metrics(direct)
    ij=float(np.sum(((z[:,8]-z[:,6])/dt)**2*dt))
    assert abs(ij-m['Ij'])<1e-10 and abs(z[:,10].sum()/3600-m['E_Wh'])<1e-10
    corrupt=z.copy();corrupt[0,9]+=1.
    try:P.S.trace_metrics(corrupt)
    except AssertionError:pass
    else:raise AssertionError('injected jerk ledger error escaped')
    checks.append('independent_actual_jerk_energy_ledger_and_injected_error')
    assert P.S.E.DT==.5 and P.S.E.J_MAX==2. and direct.t_end==600.
    assert P.S.R.LENGTH==4000 and P.S.E.A_LO==-3.5 and P.S.E.A_HI==2.6
    checks.append('frozen_route_deadline_actuator_and_jerk_parameters')
    result=dict(passed=True,source_hash=fp,checks=checks,check_count=len(checks),
                late_green_witness=dict(color_action=color,timing_action=timing),
                scope='preflight and 32-step partial continuity checks; not a full-trip experiment')
    P.S.json_save(P.ROOT/'audit/audit.json',result)
    P.S.json_save(P.ROOT/'audit/freeze.json',dict(source_hash=fp,files={str(p.relative_to(P.ROOT)):P.S.digest(p) for p in P.frozen_files()}))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
