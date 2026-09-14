"""Run in the ORIGINAL package to reproduce the three original defects.

Example: extract evidence/original/evsim_v8_original.tgz into a separate
folder, copy this script into that extracted evsim_v8 folder, then run it.
The output records observations, not assertions that the fixed version must fail.
"""
import json
from pathlib import Path
import env20 as E
import models as M
import route20 as R


def main():
    E.REWARD_MODE='L';E.CURVE_MODE='envelope'
    e=E.Route20(offsets=[0.]*len(R.SIGNALS));e.reset()
    e.x=R.SIGNALS[0]-10.;e.v=22.;e.t=30.;e.a=0.
    result={'red_stop_10m_projected_acceleration':e.project(2.6), 'traction_power':[]}
    for v in [0.5,1.5,1.999,2.,2.001]:
        pb,pf,pw=M.v1_power(v,2.6,0.,.85,288.15)
        result['traction_power'].append(dict(v=v,battery_W=pb,wheel_W=pw))
    refs=json.loads((Path(__file__).parent/'frozen'/'refs_v3.json').read_text())
    for label,offs in [('all',[0.,22.5,45.,67.5]),('reporting',[22.5,67.5])]:
        result[label]={}
        for name in ['attentive','normal','distracted']:
            scores=[-d['humans'][name]['E_Wh']*.036-.08*d['humans'][name]['t']
                    for key,d in refs['grid'].items() if float(key.split('_')[-1]) in offs]
            result[label][name]=sum(scores)/len(scores)
    print(json.dumps(result,indent=2,default=float))


if __name__=='__main__':main()
