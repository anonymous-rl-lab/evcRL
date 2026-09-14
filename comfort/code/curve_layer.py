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
        if v<=target_v+1e-9 and (a<=0 if target_v>0 else abs(a)<1e-9):return d
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

def interval(v,a,targets,vmax=None):
    """Return a verified nonempty interval or an explicit infeasibility marker.

    targets is (remaining distance, maximum terminal speed). Unexpected red
    transitions may make this interval empty. Caller retains emergency policy.
    """
    lower=max(LO,a-Q,-brake_bound(v))
    upper=min(HI,a+Q)
    if vmax is not None:upper=min(upper,brake_bound(max(vmax-v,0.)))
    if lower>upper+1e-9:return None,dict(reason='velocity_release_infeasible')
    def fits(acc,d,vg,tolerance=0.):
        vn=max(v+DT*acc,0.);travel=(v+vn)*DT/2
        if vg>0:
            rise=DT*sum(max(acc-i*Q,0.) for i in range(1,4))
            if vn+rise<=vg+1e-9:return True
        bdist=backup_distance(vn,acc,vg)
        return travel+bdist<=d+tolerance
    for d,vg in targets:
        d=max(d,0.)
        if not fits(lower,d,vg,1e-7):return None,dict(reason='stop_distance_infeasible',distance=d,target_v=vg)
        if not fits(lower,d,vg):
            upper=lower
            continue
        if fits(upper,d,vg):continue
        low,high=lower,upper
        for _ in range(32):
            mid=(low+high)/2
            if fits(mid,d,vg):low=mid
            else:high=mid
        upper=low
    if lower>upper+1e-8:return None,dict(reason='empty_intersection')
    return (lower,upper),dict(reason='feasible',lower=lower,upper=upper)

def project(v,a,cmd,targets,fallback,vmax=None):
    bounds,info=interval(v,a,targets,vmax)
    if bounds is None:
        info['fallback']=True
        return fallback,info
    low,high=bounds
    result=min(max(cmd,low),high)
    info['fallback']=False
    return result,info

def forward_distance(v,a,duration,vmax):
    """A reachable fastest-forward backup under the discrete actuator limits."""
    d=0.
    while duration>1e-12:
        acc=min(HI,a+Q,brake_bound(max(vmax-v,0.)))
        span=min(DT,duration);vn=v+acc*span
        d+=(v+vn)*span/2;v,a=vn,acc;duration-=span
    return d

def signal_project(v,a,cmd,bounds,distance,green_remaining,vmax):
    """Project onto a stop interval OR a clear-before-red interval.

    This anticipates the known end of green. Picking merely 'green now' can
    strand the vehicle outside its stop set when the light switches.
    """
    lower,upper=bounds
    def stop_ok(acc):
        vn=max(v+DT*acc,0.);travel=(v+vn)*DT/2
        return travel+backup_distance(vn,acc)<=max(distance-2.,0.)
    def go_ok(acc):
        span=min(DT,green_remaining)
        travel=v*span+.5*acc*span*span
        if travel>=distance+1e-7:return True
        if green_remaining<=DT:return False
        vn=max(v+DT*acc,0.);first=(v+vn)*DT/2
        return first+forward_distance(vn,acc,green_remaining-DT,vmax)>=distance+1e-7
    intervals=[]
    if stop_ok(lower):
        lo,hi=lower,upper
        if not stop_ok(hi):
            for _ in range(32):
                mid=(lo+hi)/2
                if stop_ok(mid):lo=mid
                else:hi=mid
            hi=lo
        intervals.append((lower,hi,'stop'))
    if go_ok(upper):
        lo,hi=lower,upper
        if not go_ok(lo):
            for _ in range(32):
                mid=(lo+hi)/2
                if go_ok(mid):hi=mid
                else:lo=mid
            lo=hi
        intervals.append((lo,upper,'clear'))
    if not intervals:return None,dict(signal_reason='no_stop_or_clear_backup')
    choices=[(min(max(cmd,lo),hi),mode) for lo,hi,mode in intervals]
    action,mode=min(choices,key=lambda p:abs(p[0]-cmd))
    return action,dict(signal_reason=mode,green_remaining=green_remaining,signal_distance=distance)
