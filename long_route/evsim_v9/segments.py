"""Inspect segment energies from a reproduced checkpoint evaluation.

First: python evaluate_checkpoint.py CHECKPOINT --output out/evaluation.json
Then:  python segments.py out/evaluation.json
Windows follow td3_run._seg; a repeat's energy is assigned by midpoint.
"""
import argparse
import json
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('evaluation');a=ap.parse_args()
    data=json.load(open(a.evaluation));grid=data['grid']
    print('Mean per trip; battery energy is signed net Wh, friction is Wh.')
    for segment in ('curve','signal','cruise'):
        e=np.mean([r[segment+'_E'] for r in grid])
        v=np.mean([r[segment+'_v'] for r in grid])
        f=[r.get(segment+'_F') for r in grid]
        fric=f'{np.mean(f):.2f}' if all(x is not None for x in f) else 'not recorded'
        print(f'{segment:8s} battery={e:.2f} Wh friction={fric} speed={v:.2f} m/s')
    residual=max(abs(sum(r[s+'_E'] for s in ('curve','signal','cruise'))-r['E_Wh']) for r in grid)
    print(f'Max segment/total energy discrepancy: {residual:.3g} Wh')


if __name__=='__main__':main()
