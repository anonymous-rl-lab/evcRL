"""Reproduce a saved policy's entire 12-condition evaluation grid."""
import argparse
import json
from pathlib import Path
import torch
import env20 as E
import td3_run as T
from refresh_refs import env_fingerprint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('--output',type=Path,default=Path('evaluation.json'))
    a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck.get('obs_dim')!=E.OBS_DIM:raise ValueError('Incompatible observation schema')
    args=ck['args']
    runfile=a.checkpoint.parent/f"{args['tag']}_s{args['seed']}.json"
    saved=json.loads(runfile.read_text())
    if saved['refs']['environment_sha256']!=env_fingerprint():
        raise ValueError('The checkpoint belongs to a different plant/environment')
    E.REWARD_MODE=args['reward'];E.CURVE_MODE=args['curve'];E.SHAPE_C=args['shape']
    E.CURVE_ENVELOPE=bool(args.get('curve_env') or args['curve']=='envelope')
    E.CLAMP_LIMIT=args['clamp'];E.K_CURVE=args['kcurve']
    actor=T.MLP(E.OBS_DIM,1,True);actor.load_state_dict(ck['actor']);actor.eval()
    grid=T.evaluate(actor,args['repeat'])
    matching=next((x for x in saved['curve'] if x['step']==ck['step']),None)
    error=None
    if matching:
        error=max(abs(x['R']-y['R']) for x,y in zip(grid,matching['grid']))
        if error>1e-6:raise AssertionError(f'Checkpoint replay mismatch: {error}')
    result=dict(checkpoint=str(a.checkpoint),step=ck['step'],summary=T.summ(grid),
                saved_R_max_error=error,grid=grid)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(f"step={ck['step']} R={result['summary']['R']:.6f} arrivals={sum(x['arr'] for x in grid)}/12; "
          f"max discrepancy from training log={error}")


if __name__=='__main__':main()
