"""Compact rendering of the unchanged archived condition-7 timing event."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
d=json.loads((P/'verification/timing_event_rows.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
fig,axs=plt.subplots(1,3,figsize=(7.15,2.22),sharex=True)
for mode,label,style in [('color','Current color','--'),('timing','Color + remaining time','-')]:
 r=d[mode];t=[x['time'] for x in r]
 axs[0].plot(t,[x['distance'] for x in r],ls=style,color='black',lw=1.25,label=label)
 for ax,key in zip(axs[1:],['acceleration','jerk']):
  ax.step(t,[x[key] for x in r],where='pre',ls=style,color='black',lw=1.25)
for ax in axs:
 ax.axvspan(180,187,color='.94',zorder=-5);ax.axvline(180,color='.4',lw=.7);ax.axvline(177.5,color='.4',ls=':',lw=.8)
 ax.grid(axis='y',color='.87',lw=.5);ax.set_xlim(174,187);ax.set_xticks([175,180,185]);ax.set_xlabel('Route time (s)')
axs[0].axhline(0,color='.5',ls=':',lw=.7);axs[0].plot(d['crossing_time'],0,'o',ms=4,mfc='white',mec='black')
axs[0].set(ylim=(-10,155),ylabel='Distance to line (m)');axs[0].set_title('(a) Remaining distance',loc='left',fontsize=8.3)
axs[1].set(ylim=(-4,1.2),ylabel=r'Acceleration (m/s$^2$)');axs[1].set_title('(b) Executed acceleration',loc='left',fontsize=8.3)
axs[2].set(ylim=(-8.5,2.5),ylabel=r'Jerk (m/s$^3$)');axs[2].set_title('(c) Discrete jerk',loc='left',fontsize=8.3)
for y in [-2,2]:axs[2].axhline(y,color='.5',ls=':',lw=.7)
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.52,1.03),fontsize=8)
fig.subplots_adjust(left=.067,right=.99,top=.78,bottom=.23,wspace=.55)
for ext in ['pdf','png']:fig.savefig(P/'figs'/f'fig3_timing.{ext}',bbox_inches='tight',pad_inches=.04,dpi=260)
plt.close(fig)
