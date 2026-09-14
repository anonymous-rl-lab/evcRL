"""Figures 1-4 for the revision. Vector PDF plus a 600 dpi PNG for each."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import env20 as E, route20 as R, dp20L as D, td3_run as T

E.CURVE_ENVELOPE = True; E.REWARD_MODE = "L"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
OUT = "/root/ms-rev/figs"; os.makedirs(OUT, exist_ok=True)
refs = json.load(open("frozen/refs_v3.json")); NS = len(R.SIGNALS); LAM = E.LAM_T
RES = "/root/res/out"
C_DP, C_TD3, C_ATT, C_NOR, C_DIS = "#1b3a6b", "#c0392b", "#2e8b57", "#7f8c8d", "#b8b8b8"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=600); plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1: route
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    g, v = R.profile()
    ax.plot(g / 1000, v * 3.6, color="k", lw=1.0)
    for i, (a, b, vc) in enumerate(R.CURVES):
        ax.axvspan(a / 1000, b / 1000, color=C_ATT, alpha=0.22,
                   label="curve, 35 km/h entry limit" if i == 0 else None)
        ax.annotate(f"C{i+1}", ((a + b) / 2000, 84), ha="center", fontsize=7)
    for i, x in enumerate(R.SIGNALS):
        ax.axvline(x / 1000, color=C_DP, ls="--", lw=1.0,
                   label="signal, 90 s cycle / 30 s green" if i == 0 else None)
        ax.annotate(f"S{i+1}", (x / 1000, 84), ha="center", fontsize=7)
        ax.axvspan((x - E.SPAT_RANGE) / 1000, x / 1000, color=C_DP, alpha=0.10,
                   label="SPaT broadcast range (1000 m)" if i == 0 else None)
    ax.set_xlim(0, 20); ax.set_ylim(0, 92)
    ax.set_xlabel("distance (km)"); ax.set_ylabel("speed limit (km/h)")
    ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.62))
    save(fig, "fig1_route")


# ------------------------------------------------------- Fig 2: learning curves
def fig2():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for s, c in zip(range(4), ("#c0392b", "#1b3a6b", "#d98c00", "#2e8b57")):
        d = json.load(open(f"{RES}/main_s{s}.json"))["curve"]
        ax.plot([x["step"] / 1e6 for x in d], [x["R"] for x in d], color=c, label=f"seed {s}",
                lw=1.1, alpha=0.95)
    for y, lab, c, ls in ((refs_R("ref"), "full-information DP", C_DP, "-"),
                          (refs_R("info"), "1000 m information-matched DP", C_DP, "--"),
                          (hum_R("attentive"), "attentive driver", C_ATT, "-."),
                          (hum_R("normal"), "normal driver", C_NOR, ":")):
        ax.axhline(y, color=c, ls=ls, lw=0.9)
        ax.annotate(lab, (0.02, y), xycoords=("axes fraction", "data"), fontsize=6,
                    va="bottom", color=c)
    ax.set_xlabel("environment steps (millions)"); ax.set_ylabel(r"$R_L$ on the 12-condition grid")
    ax.set_ylim(-215, -160); ax.legend(frameon=False, loc="lower right", ncol=2)
    save(fig, "fig2_learning")


def refs_R(k): return float(np.mean([-v[k]["E_Wh"] * 3600 / 1e5 - LAM * v[k]["t"] for v in refs["grid"].values()]))
def hum_R(n): return float(np.mean([-v["humans"][n]["E_Wh"] * 3600 / 1e5 - LAM * v["humans"][n]["t"] for v in refs["grid"].values()]))


# ------------------------------------------- policies for the profile figures
def dp_pi(soc, Tk, off):
    s = D.solve(refs["grid"][f"{soc}_{Tk}_{off}"]["ref"]["lam"], soc, Tk, [off] * NS)
    xs, vs = s["xs"], s["vs"]
    def pi(e, o):
        v_t = float(np.interp(e.x, xs, vs))
        if v_t < 0.5 and e.x < R.LENGTH - 5.0: v_t = 0.5
        for ca, cb, cvc in R.CURVES:
            if cb > e.x - 1e-9:
                dd = max(ca - E.MARGIN - e.x, 0.0)
                v_t = min(v_t, np.sqrt(max(cvc ** 2 + 2.0 * D.A_DEC * dd, 0.0)))
        return (v_t - e.v) / E.DT
    return pi


def td3_pi():
    sd = torch.load(f"{RES}/actor_main_s1.pt")
    a = T.MLP(12, 1, True); a.load_state_dict(sd if "f.0.weight" in sd else sd["actor"]); a.eval()
    def pi(e, o):
        with torch.no_grad(): return T.act(float(a(torch.as_tensor(o).unsqueeze(0))[0, 0]))
    return pi


def profile(soc, Tk, off, pi):
    env = E.Route20(soc, Tk, offsets=[off] * NS); o = env.reset(); X, V = [], []
    while True:
        X.append(env.x); V.append(env.v)
        o, r, d, i = env.step(pi(env, o))
        if d: break
    return np.array(X), np.array(V)


# ------------------------------------------------- Fig 3: C1, onset vs pack state
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    packs = [(0.85, 288.15, "warm\n0.85, +15 °C"), (0.90, 263.15, "cold\n0.90, −10 °C"),
             (0.95, 263.15, "cold, full\n0.95, −10 °C")]
    onset = {"full DP": [377, 578, 475], "TD3 (seed 1)": [276, 319, 338],
             "attentive": [128, 128, 128], "normal": [106, 106, 106], "distracted": [106, 106, 106]}
    fric = {"full DP": [0.0, 0.0, 1.3], "TD3 (seed 1)": [0.0, 5.7, 3.4],
            "attentive": [0.0, 47.2, 71.4], "normal": [4.3, 60.6, 80.0], "distracted": [4.3, 60.6, 80.0]}
    cols = {"full DP": C_DP, "TD3 (seed 1)": C_TD3, "attentive": C_ATT, "normal": C_NOR, "distracted": C_DIS}
    x = np.arange(3)
    for ax, data, ylab in ((axes[0], onset, "braking onset before the curve (m)"),
                           (axes[1], fric, "friction-brake energy at the curve (Wh)")):
        for i, (k, ys) in enumerate(data.items()):
            ax.plot(x, ys, "o-", color=cols[k], label=k, ms=4, lw=1.2,
                    ls="--" if k == "distracted" else "-")
        ax.set_xticks(x); ax.set_xticklabels([p[2] for p in packs])
    axes[0].set_ylabel("braking onset before curve 1 (m)")
    axes[1].set_ylabel("friction-brake energy at curve 1 (Wh)")
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    save(fig, "fig3_c1_onset")


# ----------------------------------- Fig 4: speed profiles at a curve and a signal
def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex="col")
    td3 = td3_pi()
    CA = R.CURVES[0][0]; XS = R.SIGNALS[0]
    for row, (soc, Tk, lab) in enumerate([(0.85, 288.15, "warm pack, SOC 0.85, +15 °C"),
                                          (0.95, 263.15, "cold pack, SOC 0.95, −10 °C")]):
        for col, (x0, x1, centre, title) in enumerate(
                [(CA - 700, CA + 200, CA, "curve 1"), (XS - 1100, XS + 200, XS, "signal 1")]):
            ax = axes[row][col]
            for nm, pi, c, ls in (("full DP", dp_pi(soc, Tk, 0.0), C_DP, "-"),
                                  ("TD3 (seed 1)", td3, C_TD3, "-"),
                                  ("normal driver", E.make_pi_human(*E.HUMANS["normal"]), C_NOR, "--")):
                X, V = profile(soc, Tk, 0.0, pi)
                m = (X >= x0) & (X <= x1)
                ax.plot(X[m] - centre, V[m] * 3.6, color=c, ls=ls, label=nm if (row == 0 and col == 0) else None)
            if col == 0:
                ax.axvspan(0, R.CURVES[0][1] - CA, color=C_ATT, alpha=0.2)
                ax.axhline(R.V_CURVE * 3.6, color="k", lw=0.6, ls=":")
            else:
                ax.axvline(0, color="k", lw=0.8)
                ax.axvspan(-E.SPAT_RANGE, 0, color=C_DP, alpha=0.07)
            ax.set_ylim(0, 88)
            if row == 1: ax.set_xlabel(f"distance to {title} (m)")
            if col == 0: ax.set_ylabel(f"{lab}\nspeed (km/h)", fontsize=7)
    axes[0][0].legend(frameon=False, loc="lower left", fontsize=6.5)
    save(fig, "fig4_profiles")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
