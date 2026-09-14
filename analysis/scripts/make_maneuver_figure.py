"""Generate the retained continuous-maneuver SI figure. No experimental data are simulated."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9,
                     'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8})
NAVY='#193c5b'; TEAL='#197a75'; GRAY='#636b73'; ORANGE='#b66524'
def save(fig,name):
    fig.savefig(P/'figs'/f'{name}.pdf',bbox_inches='tight',pad_inches=.035)
    fig.savefig(P/'figs'/f'{name}.png',dpi=240,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)

def preparation():
    fig,axs=plt.subplots(1,3,figsize=(7.15,2.48),gridspec_kw={'wspace':.52})
    rho=np.linspace(0,1.4,841);lo=np.maximum(1-rho,0); hi=np.minimum(1,rho); hg=np.minimum(1,rho/2)
    ax=axs[0]
    ax.fill_between(rho,lo,hi,where=lo<=hi,color=ORANGE,alpha=.18,label='Stop feasible')
    ax.fill_between(rho,0,hg,color=TEAL,alpha=.16,label='Pass feasible')
    ax.fill_between(rho,lo,hg,where=lo<=hg,facecolor='none',edgecolor=NAVY,hatch='////',linewidth=0,label='Common')
    ax.plot(rho,np.where(rho>=.5,lo,np.nan),color=ORANGE,lw=1.3)
    ax.plot(rho,hg,color=TEAL,lw=1.3)
    ax.set(xlabel=r'Execution authority $\rho$',ylabel=r'Preparation $q/s$',xlim=(0,1.4),ylim=(0,1.05))
    ax.set_title('(a) Feasible preparations',loc='left',pad=9)
    ax.legend(fontsize=6.7,frameon=False,loc='upper left',handlelength=1.4)
    ax.set_xticks([0,2/3,1,1.4],['0',r'$2/3$','1','1.4'])
    ax=axs[1];xr=np.linspace(2/3,1.4,400)
    ax.axvspan(0,.5,color='#eeeeee');ax.axvspan(.5,2/3,color='#eee3d9')
    ax.plot(xr,np.maximum(1-xr,0),color=NAVY,lw=2)
    ax.plot(2/3,1/3,'o',ms=4,color=NAVY)
    ax.text(.33,.20,'No jointly feasible\ntrip comparison',rotation=90,ha='center',va='center',fontsize=7.1,color=GRAY)
    ax.set(xlabel=r'Execution authority $\rho$',ylabel=r'$\Delta T / [(1-p)\Delta]$',xlim=(0,1.4),ylim=(-.015,.38))
    ax.set_xticks([0,2/3,1,1.4],['0',r'$2/3$','1','1.4']);ax.set_title('(b) Exact trip penalty',loc='left',pad=9)
    ax=axs[2]; t=np.linspace(0,12,301);a=np.interp(t,[0,4,8,12],[0,-.6,.6,0])
    ax.plot(t,a,color=ORANGE,lw=1.8,label='Minimum common')
    over=np.interp(t,[0,4,8,12],[0,-1.,1.,0])
    ax.plot(t,over,color=GRAY,lw=1.3,ls=':',label='Excess preparation')
    ax.plot(t,np.zeros_like(t),color=TEAL,lw=1.4,ls='--',label='Early cue')
    ax.axvline(4,lw=.7,color=GRAY,ls=':')
    ax.text(4.3,-1.18,'cue resolved',fontsize=7.1,color=GRAY)
    ax.set(xlabel='Maneuver time (s)',ylabel=r'Acceleration (m/s$^2$)',xlim=(0,12),ylim=(-1.32,1.50))
    ax.set_xticks([0,4,8,12]);ax.set_title(r'(c) Pass branch: $\rho=0.8$',loc='left',pad=9)
    ax.legend(fontsize=6.0,loc='upper left',frameon=False,handlelength=1.4)
    for ax in axs:ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.065,right=.995,bottom=.20,top=.88)
    save(fig,'fig2_preparation')

if __name__=='__main__': preparation()
