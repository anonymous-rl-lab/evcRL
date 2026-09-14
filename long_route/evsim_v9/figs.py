"""Plot measured logs only. No machine-specific paths or hardcoded experimental values."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze import VAL_OFFSETS, score


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=Path(__file__).resolve().parent/'out')
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'figures')
    ap.add_argument('--prefixes',nargs='+',default=['pilot','fifo','fullbuf'])
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for path in sorted(a.input.glob('*.json')):
        if not any(path.stem.startswith(p) for p in a.prefixes):continue
        data=json.loads(path.read_text())
        if not isinstance(data,dict):continue
        curve=data.get('curve')
        if not curve:continue
        x=np.array([p['step'] for p in curve])/1e6
        axes[0,0].plot(x,[score(p,VAL_OFFSETS)[0] for p in curve],label=path.stem)
        axes[0,1].plot(x,[p['arr']*12 for p in curve])
        axes[1,0].plot(x,[p['Q_bias'] for p in curve])
        axes[1,1].plot(x,[p['replay_age_steps']/1e3 for p in curve])
    axes[0,0].set_ylabel('Validation return (higher is better)')
    axes[0,1].set_ylabel('Arrivals / 12 conditions');axes[0,1].set_ylim(-.2,12.2)
    axes[1,0].set_ylabel('Q minus deterministic MC return')
    axes[1,1].set_ylabel('Mean replay age (thousand steps)')
    for ax in axes.flat:ax.set_xlabel('Integration steps (million)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.savefig(a.output/'training_diagnostics.png',dpi=180)
    fig.savefig(a.output/'training_diagnostics.pdf')
    plt.close(fig)


if __name__=='__main__':main()
