"""CPU bounded experiment runner. Logs persist; child failures are not hidden."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=40000)
    ap.add_argument('--log',type=int,default=20000)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0])
    ap.add_argument('--jobs',type=int,default=1)
    ap.add_argument('--nstep',type=int,default=20)
    ap.add_argument('--buf',type=int,default=125000)
    ap.add_argument('--tag',default='run')
    ap.add_argument('--max-minutes',type=float,default=60.)
    a=ap.parse_args()
    if min(a.steps,a.log,a.jobs,a.nstep,a.buf,a.max_minutes)<=0:ap.error('budgets must be positive')
    if len(set(a.seeds))!=len(a.seeds):ap.error('seed list contains duplicates')
    if any((HERE/'out'/f'{a.tag}_s{s}.json').exists() for s in a.seeds):
        ap.error('output exists: choose a new --tag (existing runs are never overwritten)')
    env=os.environ.copy()
    for var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[var]='1'
    (HERE/'logs').mkdir(exist_ok=True)
    pending=list(a.seeds); active=[]; failures=[]; started=time.monotonic()
    print(f'{len(pending)} seeds, {a.steps:,} integration steps each; repeat=4; {a.jobs} CPU jobs; '
          f'wall-time cap {a.max_minutes:g} min. No epoch concept in online RL.',flush=True)
    try:
        while pending or active:
            if time.monotonic()-started>a.max_minutes*60:raise TimeoutError('batch time budget exhausted')
            while pending and len(active)<a.jobs:
                s=pending.pop(0); log=open(HERE/'logs'/f'{a.tag}_s{s}.log','x')
                cmd=[sys.executable,'-u',str(HERE/'td3_run.py'),'--seed',str(s),'--steps',str(a.steps),
                     '--log',str(a.log),'--nstep',str(a.nstep),'--buf',str(a.buf),'--tag',a.tag]
                proc=subprocess.Popen(cmd,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
                active.append((s,proc,log));print(f'start seed {s}',flush=True)
            for item in active[:]:
                s,proc,log=item; code=proc.poll()
                if code is not None:
                    log.close();active.remove(item)
                    if code:failures.append((s,code))
                    print(f'finish seed {s}: exit {code}',flush=True)
            time.sleep(.5)
    finally:
        for s,proc,log in active:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
    if failures:raise SystemExit(f'Failed runs: {failures}; inspect their logs.')
    subprocess.run([sys.executable,str(HERE/'analyze.py')],check=True,cwd=HERE)


if __name__=='__main__':main()
