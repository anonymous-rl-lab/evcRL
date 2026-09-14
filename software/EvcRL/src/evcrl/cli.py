"""``evcrl-smoke``: finite API smoke run (not a driving-performance evaluation)."""
import argparse
import json
import numpy as np
import gymnasium as gym
import evcrl


def main(argv=None):
    p = argparse.ArgumentParser(description='EvcRL smoke run')
    p.add_argument('--mode', choices=['camera', 'structured'], default='camera')
    p.add_argument('--profile', choices=sorted(evcrl.PROFILES), default='paper_v25')
    p.add_argument('--steps', type=int, default=20, help='maximum public steps (each holds the command for 2 s)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--command', type=float, default=0.3, help='constant normalized command in [-1, 1]')
    p.add_argument('--oracle', action='store_true', help='attach the truth-based OracleDetector (camera mode only)')
    p.add_argument('--episode', action='store_true', help='run until termination/truncation (ignores --steps)')
    a = p.parse_args(argv)
    kw = dict(profile=a.profile)
    if a.oracle and a.mode == 'camera': kw['detector'] = evcrl.OracleDetector(seed=a.seed)
    env = gym.make('EvcRL-Camera-v0' if a.mode == 'camera' else 'EvcRL-Structured-v0', **kw)
    obs, info = env.reset(seed=a.seed); n = 0; term = trunc = False
    while a.episode or n < a.steps:
        obs, r, term, trunc, info = env.step(np.array([a.command], np.float32)); n += 1
        if term or trunc: break
    out = {k: info[k] for k in ['time_s', 'energy_Wh', 'return_sum', 'settled', 'termination_reason', 'signal_violation', 'profile', 'profile_identity', 'detector_kind']}
    out.update(public_steps=n, physical_substeps=int(round(info['time_s'] / 0.5)), terminated=bool(term), truncated=bool(trunc), version=evcrl.__version__)
    print(json.dumps(out, indent=2)); env.close()


if __name__ == '__main__': main()
