"""Plot recorded constant-command passing costs; no simulation and no LP overlay."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
rows=json.loads((P/'verification/v6_plot_rows.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':8})
fig,axs=plt.subplots(1,2,figsize=(7.15,2.65),sharey=True)
colors={2:'#193c5b',3:'#197a75',4:'#b66524'}
for ax,group,title in zip(axs,['v6_formal_v22','v6_formal_v16'],['(a) Initial speed 22.2 m/s','(b) Initial speed 16 m/s']):
 for j in (2,3,4):
  for info,style,marker in [('min','-','o'),('cons','--','s')]:
   rr=sorted([r for r in rows if r['group']==group and r['jerk']==j and r['info']==info],key=lambda r:r['actual_adoption_m'])
   ax.plot([r['actual_adoption_m'] for r in rr],[r['time_delta_at_exact_3300m'] for r in rr],color=colors[j],ls=style,marker=marker,ms=3,lw=1.1,label=('Triggered' if info=='min' else 'Conservative')+f', j={j}')
 ax.axhline(0,lw=.6,color='#999999');ax.grid(axis='y',lw=.5,color='#dddddd');ax.set_axisbelow(True)
 ax.set(xlim=(165,25),ylim=(-.09,3.4),xlabel='Actual phase-adoption distance (m)')
 ax.set_title(title,loc='left',fontsize=9)
axs[0].set_ylabel('Extra time to 3300 m (s)')
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.012),frameon=False,fontsize=7.5,handlelength=2.2,columnspacing=1.9)
fig.subplots_adjust(left=.09,right=.99,top=.89,bottom=.31,wspace=.14)
for ext in ('pdf','png'):
 fig.savefig(P/'figs'/f'fig_phase_cost.{ext}',bbox_inches='tight',pad_inches=.04,dpi=260)
plt.close(fig)
