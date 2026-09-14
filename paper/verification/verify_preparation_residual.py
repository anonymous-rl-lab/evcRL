"""Check SI (E11)–(E12) using exact full-route integration, including excess and randomized preparations."""
from pathlib import Path
import json
from verify_preparation import complete_trip

errors=[];cases=0;randomized=0
for v in [4.,12.,22.]:
 for h in [2.,4.]:
  for p in [.2,.5,.8]:
   for rho in [2/3,.8,1.,1.5]:
    j=rho*v/h**2;s=v/h;r=j*h;lo=max(s-r,0.);hi=min(s,r/2)
    L,W=10*v*h,2*h
    early=p*complete_trip(v,h,j,lo,True,L,W)['time']+(1-p)*complete_trip(v,h,j,0.,False,L,W)['time']
    def gap(q):
     return p*complete_trip(v,h,j,q,True,L,W)['time']+(1-p)*complete_trip(v,h,j,q,False,L,W)['time']-early
    base=(1-p)*h*max(1-rho,0.)
    for f in [0.,.25,.5,1.]:
     q=lo+f*(hi-lo);errors.append(abs(gap(q)-base-h*h/v*(q-lo)));cases+=1
    mix=.3*gap(lo)+.7*gap(hi);eq=.3*lo+.7*hi
    errors.append(abs(mix-base-h*h/v*(eq-lo)));randomized+=1
assert max(errors)<1e-10
# An admissible but suboptimal early reference can reverse the sign of a policy contrast.
v,h,j,p,L,W=12.,4.,.6,.5,480.,8.
cost=lambda qR,qG:p*complete_trip(v,h,j,qR,True,L,W)['time']+(1-p)*complete_trip(v,h,j,qG,False,L,W)['time']
E_star,D_star=cost(.6,0.),cost(.6,.6);E,D=cost(1.,1.),D_star
assert D-E<0 and D_star-E_star>0
assert abs((D-E)-((D_star-E_star)+(D-D_star)-(E-E_star)))<1e-12
out={'status':'PASS','deterministic_cases':cases,'randomized_cases':randomized,'max_abs_identity_error_s':max(errors),
     'nonoptimal_reference_example':{'optimal_information_gap_s':D_star-E_star,'observed_policy_gap_s':D-E},
     'scope':'Exact integration in the declared three-ramp complete-route family; no vehicle training or rollout.'}
(Path(__file__).parent/'preparation_residual_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
