"""Stateless, jerk-aware acceleration interval for the existing discrete plant.

The backup policy ramps acceleration in bounded increments, giving sampled
piecewise-quadratic speed segments. It includes brake release before v=0.
This is a feasibility controller, not an original trajectory-generation law.
"""
import math

DT=.5
J=2.
LO=-3.5
HI=2.6
Q=J*DT

def reserve(v,a):
    """Exact discrete speed margin for releasing negative acceleration."""
    b=max(-a,0.)
    return v-DT*sum(max(b-i*Q,0.) for i in range(1,5))

def brake_bound(v):
    """Largest b with DT*sum_{i>=0} max(b-i*J*DT,0) <= v."""
    v=max(v,0.)
    for n in range(1,5):
        b=(v/DT+Q*n*(n-1)/2)/n
        if b<=n*Q+1e-12:return min(b,-LO)
    return -LO

def backup_distance(v,a,target_v=0.):
    """Distance of a specific feasible brake-and-release backup; not a DP oracle.

    A velocity cap above zero tolerates temporary undershoot during release.
    Zero-speed infeasibility returns infinity; no unlimited braking is used.
    """
    if reserve(v,a)<-1e-8:return math.inf
    d=0.
    for k in range(100):
        if v<=target_v+1e-9 and abs(a)<1e-9:return d
        if v<=target_v+1e-9:
            candidate=min(0.,a+Q) if a<0 else max(0.,a-Q)
        else:
            candidate=max(LO,a-Q,-brake_bound(v-target_v))
            if candidate>a+Q+1e-8:
                # Already inside the cap-release margin: going below target_v
                # is allowed, whereas negative physical velocity is not.
                candidate=max(LO,a-Q,-brake_bound(v))
        if candidate>a+Q+1e-7:return math.inf
        vn=v+DT*candidate
        if vn<-1e-8:return math.inf
        vn=max(vn,0.);d+=(v+vn)*DT/2;v,a=vn,candidate
    return math.inf

def interval(v,a,targets):
    """Return a verified nonempty interval or an explicit infeasibility marker.

    targets is (remaining distance, maximum terminal speed). Unexpected red
    transitions may make this interval empty. Caller retains emergency policy.
    """
    lower=max(LO,a-Q,-brake_bound(v))
    upper=min(HI,a+Q)
    if lower>upper+1e-9:return None,dict(reason='velocity_release_infeasible')
    def fits(acc,d,vg):
        vn=max(v+DT*acc,0.);travel=(v+vn)*DT/2
        if vg>0:
            rise=DT*sum(max(acc-i*Q,0.) for i in range(1,4))
            if vn+rise<=vg+1e-9:return True
        bdist=backup_distance(vn,acc,vg)
        return travel+bdist<=d+1e-7
    for d,vg in targets:
        d=max(d,0.)
        if not fits(lower,d,vg):return None,dict(reason='stop_distance_infeasible',distance=d,target_v=vg)
        if fits(upper,d,vg):continue
        low,high=lower,upper
        for _ in range(32):
            mid=(low+high)/2
            if fits(mid,d,vg):low=mid
            else:high=mid
        upper=low
    if lower>upper+1e-8:return None,dict(reason='empty_intersection')
    return (lower,upper),dict(reason='feasible',lower=lower,upper=upper)

def project(v,a,cmd,targets,fallback):
    bounds,info=interval(v,a,targets)
    if bounds is None:
        info['fallback']=True
        return fallback,info
    low,high=bounds
    result=min(max(cmd,low),high)
    info['fallback']=False
    return result,info
