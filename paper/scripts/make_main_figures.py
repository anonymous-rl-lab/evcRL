"""Vector figures from declared architecture and saved LP outputs; no rollouts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42,
                     "axes.spines.top":False, "axes.spines.right":False})
NAVY="#183c5b"; TEAL="#197a75"; ORANGE="#b66524"; GRAY="#606970"

def save(fig, name):
    fig.savefig(P/"figs"/(name+".pdf"), bbox_inches="tight", pad_inches=.03)
    fig.savefig(P/"figs"/(name+".png"), dpi=220, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(7.15,3.35))
ax.set(xlim=(0,10),ylim=(0,5));ax.axis("off")
def box(x,y,w,h,label,color=NAVY):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.05,rounding_size=.08",
                               fc="#f4f7f9",ec=color,lw=.9))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=8,color=color)
def arrow(p,q,color=GRAY,dashed=False):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=9,lw=1,
                                color=color,linestyle="--" if dashed else "-"))
box(.1,2.55,1.65,.85,"Camera frames\n96 × 160; 2 Hz")
box(2.35,3.65,2.05,.85,"Frozen detector\nEvents / range / color",GRAY)
box(4.95,3.65,2,.85,"Event memory\nTracking + rules",GRAY)
box(7.8,3.65,2,.85,"Release-aware\nexecution")
box(2.35,1.8,2.05,.85,"Policy encoder\n64-dimensional Z",TEAL)
box(4.95,1.8,2,.85,"Actor + twin critics\nEgo + memory + Z",TEAL)
box(7.8,1.8,2,.85,"Vehicle / reward\nApplied motion")
box(2.35,.35,2.05,.65,"Visual supervision",TEAL)
arrow((1.78,3.25),(2.30,3.96))
arrow((1.78,2.76),(2.30,2.27))
arrow((4.45,4.075),(4.90,4.075))
arrow((7.0,4.075),(7.75,4.075))
ax.text(7.37,4.3,"targets",ha="center",fontsize=7.4,color=GRAY)
arrow((5.95,3.60),(5.95,2.70))
ax.text(6.09,3.12,"memory",fontsize=7.3,color=GRAY)
arrow((4.45,2.225),(4.90,2.225),TEAL)
ax.text(4.67,2.88,"Z",ha="center",fontsize=8,color=TEAL)
ax.text(4.67,1.58,"RL gradients\nstopped here",ha="center",va="top",fontsize=6.9,color=GRAY)
arrow((7.0,2.55),(8.10,3.6),TEAL)
ax.text(7.42,3.08,"command",ha="center",fontsize=7.1,color=TEAL,
        bbox={"fc":"white","ec":"none","pad":1})
arrow((8.8,3.6),(8.8,2.7))
ax.text(8.96,3.08,"applied a",fontsize=7.4,color=GRAY)
arrow((7.75,2.1),(7.0,2.1))
ax.text(7.4,1.57,"replay",ha="center",fontsize=7.2,color=GRAY)
arrow((3.375,1.05),(3.375,1.75),TEAL,True)
ax.text(2.08,1.35,"gradient",ha="right",fontsize=7.1,color=TEAL)
ax.plot([8.8,8.8,.90,.90],[1.75,.12,.12,2.32],color=GRAY,lw=.8)
arrow((.9,2.32),(.9,2.5))
ax.text(6.6,.75,"Environment feedback:\nimages and ego state",fontsize=7.1,color=GRAY,ha="center")
ax.text(.1,4.8,"S configuration: separate event and policy-image paths",fontsize=10,weight="bold",color=NAVY)
save(fig,"fig1_interfaces")

fig,axes=plt.subplots(1,2,figsize=(7.15,2.65),sharey=True)
for ax,fn,title in zip(axes,["theory_lp_complete.json","theory_lp_v16_complete.json"],
                       ["(a) Initial speed 22.2 m/s","(b) Initial speed 16 m/s"]):
    data=json.loads((P/"reproduction/v6"/fn).read_text())
    for j,color,marker in [(2,NAVY,"o"),(3,TEAL,"s"),(4,ORANGE,"^")]:
        rows=[r for r in data["rows"] if r["jerk"]==j and r.get("delayed")=="ok"]
        ax.plot([0]+[r["delta"] for r in rows],
                [0]+[max(0,r["penalty_s"]) for r in rows],
                color=color,marker=marker,markersize=2.8,lw=1.25,
                label=rf"$j={j}$ m/s$^3$")
    ax.axhline(0,color=GRAY,lw=.6,ls="--")
    ax.grid(axis="y",lw=.5,color="#dddddd")
    ax.set_xlabel("Fixed revelation delay (s)",fontsize=8.4)
    ax.set_title(title,loc="left",fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(left=0)
axes[0].set_ylabel(r"$C_K^{\rm LP}$, time equivalent (s)",fontsize=8.5)
axes[0].legend(frameon=False,fontsize=7.5,loc="upper left")
fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.87,wspace=.18)
save(fig,"fig2_lp_reference")
