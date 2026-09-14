"""Draw the prespecified signal event from saved traces; no simulation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':220})
fig,axes=plt.subplots(3,1,figsize=(7.2,7.0),sharex=True)
for mode,label,style in [('color','Current color','--'),('timing','Color + remaining time','-')]:
    z=np.load(ROOT/f'runs/development/{mode}_07.npz')['values']
    mask=(z[:,1]>=173.5)&(z[:,0]<=187.)
    a=z[mask]
    axes[0].plot(a[:,1],3000-a[:,3],style,color='black',linewidth=1.65,label=label)
    axes[1].step(a[:,1],a[:,8],where='pre',linestyle=style,color='black',linewidth=1.65)
    axes[2].step(a[:,1],a[:,9],where='pre',linestyle=style,color='black',linewidth=1.65)
for ax in axes:
    ax.axvspan(180.,187.,color='.94',zorder=-5)
    ax.axvline(180.,color='.4',linewidth=.9)
    ax.axvline(177.5,color='.45',linestyle=':',linewidth=.9)
    ax.grid(axis='y',color='.87',linewidth=.55)
    ax.set_xlim(174.,187.)
axes[0].axhline(0.,color='.5',linewidth=.8,linestyle=':')
decision=json.loads((ROOT/'reports/decision.json').read_text())
axes[0].plot(decision['red_crossing_time_s'],0,'o',mfc='white',mec='black',ms=5)
axes[0].set_ylim(-10,155)
axes[0].set_ylabel('Distance to line (m)')
axes[1].set_ylabel('Acceleration (m/s²)');axes[1].set_ylim(-4.,1.2)
axes[2].set_ylabel('Discrete jerk (m/s³)');axes[2].set_ylim(-8.5,2.5)
axes[2].axhline(-2,color='.5',linewidth=.8,linestyle=':')
axes[2].axhline(2,color='.5',linewidth=.8,linestyle=':')
axes[2].set_xlabel('Time from route start (s)')
axes[0].text(177.35,147,'First action change',ha='right',va='top',fontsize=9)
axes[0].text(180.15,147,'Nonpassable phase',ha='left',va='top',fontsize=9)
for ax,letter in zip(axes,'abc'):ax.text(.01,.06,f'({letter})',transform=ax.transAxes,fontweight='bold')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.52,.99))
fig.tight_layout(rect=(0,0,1,.955),h_pad=.6)
fig.savefig(ROOT/'reports/signal_event.png',bbox_inches='tight')
fig.savefig(ROOT/'reports/signal_event.svg',bbox_inches='tight')
