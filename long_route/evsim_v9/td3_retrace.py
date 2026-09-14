"""TD3 on the physical 20 km arterial; see README.md for validated status.

Default: legacy 20-step recipe with finite FIFO replay and an undiscounted finite horizon.
--nstep 1 enables the one-step diagnostic. The default uncorrected behavior return is not
claimed to be an unbiased deterministic-policy target. Target smoothing also
means Q versus a noise-free rollout is a diagnostic, not an identity test.

The actor sees 13 channels (the old 12 plus elapsed time). Cold evaluation
uses a fixed 1312.5 s deadline. Arrival and the separate 1050 s budget flag
must not be conflated. Checkpoints are selected on validation offsets only.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
import env20 as E, route20 as R

torch.set_num_threads(1)

OBS, HID, CLR, TAU, BATCH, WARMUP = E.OBS_DIM, 64, 3e-4, 0.005, 256, 4_000
POLICY_DELAY, TARGET_NOISE, TARGET_CLIP = 2, 0.2, 0.5
import os as _os
HERE = os.path.dirname(os.path.abspath(__file__))
_RF = os.path.join(HERE, "frozen", "refs_mini_fixed.json" if R.MINI else "refs_fixed.json")
REFS = json.load(open(_RF)) if os.path.exists(_RF) else None



def save_json(data, path, **kwargs):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(data, handle, **kwargs)
    os.replace(temporary, path)


def use_lagrangian_refs():
    """Under REWARD_MODE 'L' the objective is energy + LAM_T * time, so every
    reference must be re-scored with the same formula before it can be compared."""
    lam = E.LAM_T
    def rl(d): return -d["E_Wh"] * 3600 / 1e5 - lam * d["t"]
    for k, v in REFS["grid"].items():
        # Fixed references already contain actual rollout rewards, including penalties.
        if REFS.get("schema") == 2: return
        v["ref"]["R"] = rl(v["ref"]); v["info"]["R"] = rl(v["info"])
        for n in v["humans"]: v["humans"][n]["R"] = rl(v["humans"][n])
        if "humans_paced" in v:
            for n in v["humans_paced"]: v["humans_paced"][n]["R"] = rl(v["humans_paced"][n])
    g = REFS["grid"].values()
    REFS["R_dp"] = float(np.mean([v["ref"]["R"] for v in g]))
    REFS["R_dp_info"] = float(np.mean([v["info"]["R"] for v in g]))
    for n in E.HUMANS:
        REFS[f"R_{n}"] = float(np.mean([v["humans"][n]["R"] for v in g]))
        if "humans_paced" in list(g)[0]:
            REFS[f"R_{n}_paced"] = float(np.mean([v["humans_paced"][n]["R"] for v in g]))
PACK = [(0.85, 288.15), (0.90, 263.15), (0.95, 263.15)]
OFFS = [0., 22.5, 45., 67.5]
NSIG = len(R.SIGNALS)


def act(u):
    u = float(np.clip(u, -1, 1)); return u * (E.A_HI if u > 0 else -E.A_LO)


class MLP(nn.Module):
    def __init__(s, nin, nout, tanh=False):
        super().__init__()
        L = [nn.Linear(nin, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, nout)]
        if tanh: L.append(nn.Tanh())
        s.f = nn.Sequential(*L)
    def forward(s, x): return s.f(x)


def env_step(env, u, k):
    Rr = 0.
    elapsed = 0
    for _ in range(k):
        o, r, d, i = env.step(act(u)); Rr += r; elapsed += 1
        if d: break
    i["substeps"] = elapsed
    return o, Rr, d, i


CURVE_W = [(a - 600.0, b + 300.0) for a, b, _ in R.CURVES]
SIG_W = [(x - E.SPAT_RANGE, x + 300.0) for x in R.SIGNALS]


def _seg(x):
    if any(p <= x < q for p, q in CURVE_W): return "curve"
    if any(p <= x < q for p, q in SIG_W): return "signal"
    return "cruise"


def run_policy(actor, env, k):
    """Rollout with the probes the diagnosis needs: A1 (is the action alive?),
    actor saturation, and the energy/friction split over the curve, signal and
    cruise windows separately."""
    o = env.reset(); Rr = 0.; jerks = []
    dead = sat = nstep = 0
    seg = {s: dict(E=0.0, F=0.0, v=[], t=0.0) for s in ("curve", "signal", "cruise")}
    while True:
        with torch.no_grad(): u = float(actor(torch.as_tensor(o).unsqueeze(0))[0, 0])
        a_prev, eb, ef, x0, v0 = env.a, env.e_batt, env.e_fric, env.x, env.v
        lo = env.project(act(max(u - 0.2, -1.0))); hi = env.project(act(min(u + 0.2, 1.0)))
        dead += abs(hi - lo) < 1e-6; sat += abs(u) > 0.99; nstep += 1
        o, r, d, i = env_step(env, u, k); Rr += r
        jerks.append(abs(env.a - a_prev) / (k * E.DT))
        s = seg[_seg(0.5 * (x0 + env.x))]
        s["E"] += (env.e_batt - eb) / 3600.; s["F"] += (env.e_fric - ef) / 3600.
        s["v"].append(0.5 * (v0 + env.v)); s["t"] += i["substeps"] * E.DT
        if d: break
    # report the UNSHAPED return so every configuration is comparable with the references.
    # With Phi(absorbing) = 0 the total shaping over an episode is the CONSTANT
    # SHAPE_C * (x_T - x_0) + SHAPE_C * (L - x_T) = SHAPE_C * L, for every episode.
    Rr = Rr / E.REWARD_SCALE - E.SHAPE_C * R.LENGTH
    out = dict(R=Rr, t=env.t, E_Wh=env.e_batt / 3600., fric=env.e_fric / 3600., reg=env.e_reg / 3600.,
               stops=env.n_stop, off=env.n_offroad, viol=env.n_violation, arr=i["arrived"], x=env.x,
               jerk_rms=float(np.sqrt(env.jerk_sq / max(env.steps, 1))),
               jerk_max=env.jerk_max, jerk_overrides=env.jerk_overrides,
               safety_infeasible=env.safety_infeasible, a_min=env.a_min, a_max=env.a_max,
               dead=dead / max(nstep, 1), sat=sat / max(nstep, 1),
               v_max=float(env.v_over_max), t_over=float(env.t_over), curve_pen=float(env.e_curve_pen))
    for s in seg:
        out[f"{s}_E"] = seg[s]["E"]; out[f"{s}_F"] = seg[s]["F"]
        out[f"{s}_v"] = float(np.mean(seg[s]["v"])) if seg[s]["v"] else 0.0
    return out


def evaluate(actor, k):
    out = []
    for soc, T in PACK:
        for off in OFFS:
            e = run_policy(actor, E.Route20(soc, T, offsets=[off] * NSIG), k)
            e.update(soc=soc, T=T, offset=off); out.append(e)
    return out


def summ(ev):
    keys = ("R", "E_Wh", "fric", "reg", "t", "stops", "off", "viol", "arr", "x", "jerk_rms",
            "dead", "sat", "v_max", "t_over", "curve_E", "curve_F", "curve_v", "signal_E", "signal_F",
            "signal_v", "cruise_E", "cruise_v", "curve_pen",
            "jerk_max", "jerk_overrides", "safety_infeasible", "a_min", "a_max")
    s = {kk: float(np.mean([e[kk] for e in ev])) for kk in keys}
    s["closed"] = 100 * (s["R"] - REFS["R_normal"]) / (REFS["R_dp"] - REFS["R_normal"])
    def key(e): return f"{e['soc']}_{e['T']}_{e['offset']}"
    def hum(e, n, paced=False):
        h = REFS["grid"][key(e)]["humans_paced" if paced else "humans"][n]
        return h["R"] if isinstance(h, dict) else h
    s["closed_paced"] = 100 * (s["R"] - REFS["R_normal_paced"]) / (REFS["R_dp"] - REFS["R_normal_paced"])
    s["beats_normal"] = int(sum(e["R"] > hum(e, "normal") for e in ev))
    s["beats_expert"] = int(sum(e["R"] > hum(e, list(E.HUMANS)[-1]) for e in ev))
    s["beats_normal_paced"] = int(sum(e["R"] > hum(e, "normal", True) for e in ev))
    s["beats_expert_paced"] = int(sum(e["R"] > hum(e, list(E.HUMANS)[-1], True) for e in ev))
    return s


def train(a):
    seed, steps, repeat, sigma = a.seed, a.steps, a.repeat, a.sigma
    WARM = a.warmup
    tag = f"{a.tag}_s{seed}"
    outdir = a.output
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(os.path.join(outdir, tag + ".json")):
        raise FileExistsError(f"Run {tag} already exists in {outdir}; choose a new tag.")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    actor, actor_t = MLP(OBS, 1, True), MLP(OBS, 1, True)
    q1, q2, q1t, q2t = (MLP(OBS + 1, 1) for _ in range(4))
    for x, y in ((actor_t, actor), (q1t, q1), (q2t, q2)): x.load_state_dict(y.state_dict())
    if a.init:
        # ACTOR only. The stage-1 critics were fitted to a different route length and
        # therefore a different return scale (-12 vs -85); loading them makes the first
        # actor updates destroy the transferred policy.
        ck = torch.load(a.init, map_location="cpu", weights_only=True)
        if ck.get("obs_dim", 12) != OBS:
            raise ValueError(f"Checkpoint observation size is incompatible with fixed environment ({OBS}).")
        actor.load_state_dict(ck["actor"]); actor_t.load_state_dict(ck["actor"])
        print(f"  actor warm-started from {a.init} (stage-1 step {ck.get('step')}); actor updates delayed {a.actor_delay} steps", flush=True)
    oa = torch.optim.Adam(actor.parameters(), lr=a.alr)
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=CLR)
    BUF = a.buf
    O = np.zeros((BUF, OBS), np.float32); U = np.zeros((BUF, 1), np.float32); Rb = np.zeros((BUF, 1), np.float32)
    O2 = np.zeros((BUF, OBS), np.float32); D = np.zeros((BUF, 1), np.float32); ARR = np.zeros(BUF, np.int8)
    R1 = np.zeros((BUF, 1), np.float32)      # RETRACE: single-step reward
    O21 = np.zeros((BUF, OBS), np.float32)   # RETRACE: ONE-step next obs (O2 holds the n-step one)
    D1 = np.zeros((BUF, 1), np.float32)      # RETRACE: ONE-step done  (D  holds the n-step one)
    EPI = np.full(BUF, -1, np.int64)          # RETRACE: episode id, to stop a window at a boundary
    n = ptr = g = inserted = 0
    TGEN = np.zeros(BUF, np.int64)

    def mkenv():
        # Pack state is drawn from the three regimes that are evaluated, not from a
        # continuous range: with SOC ~ U(0.75,0.97) and T ~ U(-15,+30 C) the return
        # spread from the pack state alone was +-3.05 units, the same size as the whole
        # value of the signal decision, and the critic never got enough samples per
        # regime to separate them.
        soc, Tk = PACK[int(rng.integers(0, len(PACK)))]
        env = E.Route20(soc, Tk, rng=rng)
        env.reset()
        if rng.random() < 0.85:
            x0 = float((R.LENGTH - 300.0) * (1.0 - rng.random() ** a.startbias))
            pace = 19.0 if a.pace == "fixed" else rng.uniform(a.pace_lo, a.pace_hi)
            t0 = max(float(x0 / pace + (rng.normal(0.0, 10.0) if a.pace == "fixed" else 0.0)), 0.0)
            env.reset_from(x0, float(rng.uniform(0.0, E.v_limit(x0))), t0)
            # Do not initialize the vehicle in an already impossible stopping state.
            if env.a_safe() < E.A_LO:
                env.reset()
        return env

    ou = 0.0
    from collections import deque
    nq_ = deque(maxlen=max(a.nstep, 1))
    dg = dict(qloss=0., ynorm=0., twin=0., qgrad=0., agrad=0., aloss=0., nq=0, na=0, trace=0., ntr=0)
    # fixed start states of the 12 evaluation conditions: Q(s0, pi(s0)) against the
    # Monte-Carlo return actually collected from those same states is the calibration
    # test - a critic that is not converging shows a growing gap here.
    S0 = np.stack([E.Route20(soc, Tk, offsets=[off] * NSIG).reset() for soc, Tk in PACK for off in OFFS])
    env = mkenv(); o = env.obs(); curve = []; t0 = time.time(); used = ep = ep_off = ep_arr = 0
    ep_start = inserted
    while used < steps:
        if used < WARM:
            if a.prime > 0.0:
                # THE BOOTSTRAP HOLE.  The episode cut-off is t_end = t + 1.25 (L-x)/pace,
                # which is SCALE-FREE: arriving requires an average of pace/1.25 = 15.24 m/s
                # from ANY start, so exploring starts near the finish are no easier than a
                # cold start.  A run whose early policy cruises below that threshold completes
                # nothing, its replay buffer holds no evidence of what a finished trip is worth,
                # and the critic cannot price speed.  Measured: a seed on the wrong side of the
                # threshold sat at buf-arrived = 0.03 for 600k steps and needed 1.35M steps to
                # recover.  A scripted cruise during warmup removes the hole without changing the
                # reward, the criterion or the evaluation: it shows the learner what FINISHING
                # looks like, not how to save energy.
                if not hasattr(env, "_vt"): env._vt = float(rng.uniform(16.0, 20.0))
                u = float(np.clip((env._vt - env.v) / (E.DT * E.A_HI), -1, 1) + rng.normal(0, a.prime))
            else:
                u = float(rng.normal(0, 0.6))
        else:
            with torch.no_grad(): u = float(actor(torch.as_tensor(o).unsqueeze(0))[0, 0])
            if a.noise == "ou":
                # i.i.d. Gaussian noise cannot explore a SUSTAINED speed change: at
                # sigma 0.05 it perturbs each 2 s decision by +-0.15 m/s^2 and averages
                # out over the tens of decisions needed to move the cruise speed. The
                # OU process is correlated over ~1/theta decisions, which is what lets
                # the policy leave the speed-limit boundary it otherwise saturates on.
                ou += -a.theta * ou + sigma * rng.normal()
                u += ou
            else:
                u += rng.normal(0, sigma)
        u = float(np.clip(u, -1, 1))
        o2, r, done, info = env_step(env, u, min(repeat, steps - used)); used += info["substeps"]
        # n-step transition: (o_t, u_t, sum of the next n rewards, o_{t+n}). With
        # gamma = 1 the discounting is trivial and the terminal cost of an episode
        # reaches the states n decisions back in ONE critic sweep instead of n.
        nq_.append((o, u, r, o2, done))
        if len(nq_) >= a.nstep or done:
            while nq_:
                o_0, u_0 = nq_[0][0], nq_[0][1]
                Rn = sum(z[2] for z in nq_)
                o_n, d_n = nq_[-1][3], nq_[-1][4]
                O[ptr], U[ptr], Rb[ptr], O2[ptr], D[ptr] = o_0, u_0, Rn, o_n, float(d_n)
                R1[ptr] = nq_[0][2]; O21[ptr] = nq_[0][3]; D1[ptr] = float(nq_[0][4]); EPI[ptr] = ep
                ARR[ptr] = 0  # clear an overwritten episode outcome immediately
                TGEN[ptr] = used
                inserted += 1
                ptr = (ptr + 1) % BUF; n = min(n + 1, BUF)
                nq_.popleft()
                if not done: break
        o = o2
        if done:
            ep += 1; ep_off += bool(info.get("offroad")); ep_arr += bool(info.get("arrived"))
            # tag the finished episode's transitions with its outcome (buffer composition diagnostic)
            L = min(inserted - ep_start, BUF)
            idxs = (ptr - L + np.arange(L)) % BUF
            ARR[idxs] = (1 if info.get("arrived") else
                         4 if info.get("red_crossing") else
                         2 if info.get("offroad") else 3)
            env = mkenv(); o = env.obs(); ep_start = inserted; ou = 0.0; nq_.clear()
        if n >= min(BATCH, BUF) and used >= WARM:
            if a.balance > 0.0:
                # stratified replay: half the batch from completed trips when the buffer has any.
                # Without it the buffer of a slow policy is almost all timed-out episodes, the
                # critic never sees the value structure of a finished trip, and it settles on a
                # cruise set-point far below the optimum (Section VII-D).
                arr_idx = np.flatnonzero(ARR[:n] == 1)
                k_arr = int(BATCH * a.balance)
                # only once completed trips are a non-trivial share of the buffer: at 1-2 %
                # the same few transitions would be drawn into half of every batch and the
                # critic overfits them.
                if len(arr_idx) >= 0.05 * n and k_arr > 0:
                    idx = np.concatenate([rng.choice(arr_idx, k_arr),
                                          rng.integers(0, n, BATCH - k_arr)])
                else:
                    idx = rng.integers(0, n, BATCH)
            else:
                idx = rng.integers(0, n, BATCH)
            ob = torch.as_tensor(O[idx]); ub = torch.as_tensor(U[idx]); rb = torch.as_tensor(Rb[idx])
            o2b = torch.as_tensor(O2[idx]); db = torch.as_tensor(D[idx])
            with torch.no_grad():
                if a.retrace > 0:
                    # ---------------- RETRACE(lambda) ----------------
                    # The uncorrected 20-step return sums rewards collected under the BEHAVIOUR
                    # policy (target policy + OU noise).  Measured contamination: freezing the
                    # policy and re-noising only the following 19 decisions shifts the target by
                    # about -9 units at early positions.  Retrace re-weights each step by how much
                    # the collected action still looks like what the current target policy would
                    # do, so the trace is cut where the data has gone off-policy:
                    #     y = Q(s0,a0) + sum_k (prod_{j<=k} c_j) * delta_k
                    #     delta_k = r_k + (1-d_k) Q_t(s_{k+1}, pi_t(s_{k+1})) - Q_t(s_k, a_k)
                    #     c_j     = min(1, exp(-(a_j - pi_t(s_j))^2 / (2 sigma_c^2)))
                    # c_j is the Gaussian ratio pi/mu under target-policy smoothing; it is 1 where
                    # the behaviour action matches the current policy and decays as it diverges.
                    K = a.retrace
                    off = np.arange(K)[None, :]
                    ii = (idx[:, None] + off) % BUF                       # (B,K) contiguous window
                    same = (EPI[ii] == EPI[idx][:, None]) & (EPI[ii] >= 0)
                    valid = torch.as_tensor(np.cumprod(same, axis=1).astype(np.float32))
                    Ow  = torch.as_tensor(O[ii].reshape(-1, OBS))
                    Uw  = torch.as_tensor(U[ii].reshape(-1, 1))
                    Rw  = torch.as_tensor(R1[ii].reshape(-1, 1))
                    O2w = torch.as_tensor(O21[ii].reshape(-1, OBS))   # ONE-step successor
                    Dw  = torch.as_tensor(D1[ii].reshape(-1, 1))      # ONE-step done
                    pi_w = actor_t(Ow)                                     # what the policy would do
                    e2 = (torch.randn_like(Uw) * a.target_noise).clamp(-TARGET_CLIP, TARGET_CLIP)
                    u2w = (actor_t(O2w) + e2).clamp(-1, 1)
                    q_next = torch.min(q1t(torch.cat([O2w, u2w], 1)), q2t(torch.cat([O2w, u2w], 1)))
                    q_cur  = torch.min(q1t(torch.cat([Ow, Uw], 1)),  q2t(torch.cat([Ow, Uw], 1)))
                    delta = (Rw + (1 - Dw) * q_next - q_cur).reshape(-1, K)
                    c = torch.exp(-((Uw - pi_w) ** 2) / (2 * a.sigma_c ** 2)).clamp(max=1.0).reshape(-1, K)
                    c = torch.cat([torch.ones_like(c[:, :1]), c[:, 1:]], 1)   # c_0 = 1 by convention
                    w = torch.cumprod(c, 1) * valid
                    q0 = q_cur.reshape(-1, K)[:, :1]
                    y = q0 + (w * delta).sum(1, keepdim=True)
                    dg["trace"] += float(w[:, 1:].mean()); dg["ntr"] += 1
                else:
                    eps = (torch.randn_like(ub) * a.target_noise).clamp(-TARGET_CLIP, TARGET_CLIP)
                    u2 = (actor_t(o2b) + eps).clamp(-1, 1)
                    y = rb + (1 - db) * torch.min(q1t(torch.cat([o2b, u2], 1)), q2t(torch.cat([o2b, u2], 1)))
            x = torch.cat([ob, ub], 1)
            lq = ((q1(x) - y) ** 2).mean() + ((q2(x) - y) ** 2).mean()
            oq.zero_grad(); lq.backward()
            dg["qloss"] += float(lq.detach()); dg["ynorm"] += float(y.abs().mean())
            dg["twin"] += float((q1(x) - q2(x)).abs().mean().detach())
            dg["qgrad"] += float(torch.sqrt(sum((p.grad ** 2).sum() for p in q1.parameters())))
            dg["nq"] += 1
            oq.step(); g += 1
            if g % POLICY_DELAY == 0 and used >= a.actor_delay:
                la = -q1(torch.cat([ob, actor(ob)], 1)).mean(); oa.zero_grad(); la.backward()
                dg["aloss"] += float(la.detach())
                dg["agrad"] += float(torch.sqrt(sum((p.grad ** 2).sum() for p in actor.parameters())))
                dg["na"] += 1
                oa.step()
                with torch.no_grad():
                    for net, tgt in ((actor, actor_t), (q1, q1t), (q2, q2t)):
                        for p, pt in zip(net.parameters(), tgt.parameters()): pt.mul_(1 - TAU).add_(TAU * p)
        next_log = ((curve[-1]["step"] // a.log + 1) * a.log) if curve else a.log
        if used >= next_log or used >= steps:
            ev = evaluate(actor, repeat); s = summ(ev)
            with torch.no_grad():
                ob0 = torch.as_tensor(S0); u0 = actor(ob0)
                q0 = q1(torch.cat([ob0, u0], 1)).squeeze(1).numpy()
            mc = E.REWARD_SCALE * (np.array([e["R"] for e in ev]) + E.SHAPE_C * R.LENGTH)
            s["Q_s0"] = float(q0.mean()); s["MC_s0"] = float(mc.mean()); s["Q_bias"] = float((q0 - mc).mean())
            nq, na = max(dg["nq"], 1), max(dg["na"], 1)
            s["trace"] = dg["trace"] / max(dg["ntr"], 1)
            s.update(qloss=dg["qloss"] / nq, ynorm=dg["ynorm"] / nq, twin=dg["twin"] / nq,
                     qgrad=dg["qgrad"] / nq, agrad=dg["agrad"] / na, aloss=dg["aloss"] / na,
                     u_mean=float(u0.mean()), u_std=float(u0.std()))
            dg = dict(qloss=0., ynorm=0., twin=0., qgrad=0., agrad=0., aloss=0., nq=0, na=0, trace=0., ntr=0)
            comp = np.bincount(ARR[:n], minlength=5) / max(n, 1)     # 0 open, 1 arrived, 2 offroad, 3 timeout, 4 red
            s.update(step=used, episodes=ep, train_offroad=ep_off / max(ep, 1), train_arrived=ep_arr / max(ep, 1),
                     buf_arrived=float(comp[1]), buf_offroad=float(comp[2]), buf_timeout=float(comp[3]),
                     buf_red=float(comp[4]), buf_size=n,
                     replay_age_steps=float(np.mean(used - TGEN[:n])) if n else 0.0, updates=g,
                     wall_s=time.time() - t0, grid=ev)
            curve.append(s)
            torch.save(dict(actor=actor.state_dict(), q1=q1.state_dict(), q2=q2.state_dict(), step=used, obs_dim=OBS, args=vars(a)), os.path.join(outdir, f"ckpt_{tag}_{used}steps.pt"))
            save_json(dict(args=vars(a), refs=REFS,
                           curve=curve), os.path.join(outdir, f"{tag}.json"), indent=1, default=float, allow_nan=False)
            print(f"  {tag:<12} step {used:9d} ep {ep:5d} (arr {100*s['train_arrived']:3.0f}% off {100*s['train_offroad']:3.0f}% | buf arr {100*comp[1]:2.0f}% off {100*comp[2]:2.0f}% to {100*comp[3]:2.0f}%) "
                  f"| 12-grid R {s['R']:8.3f} closed {s['closed_paced']:+6.1f}% E {s['E_Wh']:5.0f} fric {s['fric']:4.0f} t {s['t']:5.0f} stops {s['stops']:.2f} off {s['off']:.2f} arr {s['arr']:.2f} "
                  f">nrm {s['beats_normal']:2d} >att {s['beats_expert']:2d} /12 | dead {100*s['dead']:3.0f}% sat {100*s['sat']:3.0f}% vmax {s['v_max']:4.1f} "
                  f"| CRV {s['curve_E']:4.0f}/{s['curve_F']:3.0f}@{s['curve_v']:4.1f} SIG {s['signal_E']:4.0f}/{s['signal_F']:3.0f}@{s['signal_v']:4.1f} CRU {s['cruise_E']:4.0f}@{s['cruise_v']:4.1f}\n"
                  f"    {'':12s} learner: Qloss {s['qloss']:9.1f} |y| {s['ynorm']:7.1f} twin {s['twin']:6.2f} |gQ| {s['qgrad']:7.1f} |gA| {s['agrad']:6.3f} "
                  f"| Q(s0) {s['Q_s0']:8.1f} vs MC {s['MC_s0']:8.1f} bias {s['Q_bias']:+7.1f} | u {s['u_mean']:+5.2f}+-{s['u_std']:.2f} [{s['wall_s']/60:5.0f} min, ETA {(steps-used)*s['wall_s']/used/60:5.0f} min]", flush=True)
    last3 = curve[-3:]
    fin = dict(last=curve[-1], last3_mean_R=float(np.mean([c["R"] for c in last3])), last3_mean_closed=float(np.mean([c["closed"] for c in last3])))
    save_json(dict(args=vars(a), refs=REFS,
                   curve=curve, final=fin), os.path.join(outdir, f"{tag}.json"), indent=1, default=float, allow_nan=False)
    torch.save(actor.state_dict(), os.path.join(outdir, f"actor_{tag}.pt"))
    print(f"FINAL {tag}: last ckpt R {curve[-1]['R']:.3f} (closed {curve[-1]['closed']:+.1f}%, >normal {curve[-1]['beats_normal']}/12, >expert {curve[-1]['beats_expert']}/12); "
          f"last-3 mean R {fin['last3_mean_R']:.3f} (closed {fin['last3_mean_closed']:+.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=12_000_000)
    ap.add_argument("--tag", default="main"); ap.add_argument("--log", type=int, default=500_000)
    ap.add_argument("--buf", type=int, default=125_000, help="FIFO capacity in DECISION transitions; repeat=4 means about 500k integration steps")
    ap.add_argument("--pace", choices=["uniform", "fixed"], default="uniform", help="exploring-start clock (ablation: fixed 19 m/s + N(0,10))")
    ap.add_argument("--pace_lo", type=float, default=12.0); ap.add_argument("--pace_hi", type=float, default=22.0)
    ap.add_argument("--acct", choices=["ctg", "flat"], default="ctg", help="terminal accounting (ablation: flat penalty, no cost-to-go)")
    ap.add_argument("--alr", type=float, default=3e-5)
    ap.add_argument("--curve", choices=["penalty", "envelope", "terminate"], default="envelope",
                    help="how the 35 km/h curve entry limit is enforced")
    ap.add_argument("--kcurve", type=float, default=0.05, help="curve excess-speed cost, per (m/s)^2 per second")
    ap.add_argument("--curve_env", action="store_true", help="curves enforced by the safety envelope (no off-road terminations)")
    ap.add_argument("--repeat", type=int, default=4, help="integration steps per decision: 4 = 2 s (fine), 10 = 5 s (coarse)")
    ap.add_argument("--clamp", action="store_true", help="ablation: clamp the legal speed limit inside the envelope (v1 behaviour)")
    ap.add_argument("--shape", type=float, default=0.0,
                    help="potential-based shaping on distance, units of 100 kJ per metre; "
                         "0.0092 is the nominal cost per metre of a good trip on this route")
    ap.add_argument("--startbias", type=float, default=1.0,
                    help="exponent p in x0 = L*(1-U^p): p>1 concentrates exploring starts near the "
                         "finish, which puts completed trips into the buffer whatever the policy does")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="fraction of each minibatch drawn from transitions of episodes that ARRIVED. "
                         "0 = uniform replay. The two seeds that failed had 63-72 %% of the buffer in "
                         "timed-out episodes and their critics preferred to brake at every cruise speed.")
    ap.add_argument("--prime", type=float, default=0.0,
                    help="during warmup, act with a scripted constant-cruise policy at U(16,20) m/s plus this much noise, "
                         "so the replay buffer contains COMPLETED trips from step 0. 0 = the old random-action warmup.")
    ap.add_argument("--retrace", type=int, default=0,
                    help="Retrace(lambda) window length in decisions (0 = uncorrected n-step return)")
    ap.add_argument("--sigma_c", type=float, default=0.25,
                    help="width of the Gaussian policy ratio in the Retrace trace coefficient")
    ap.add_argument("--nstep", type=int, default=20, help="n-step returns: with gamma=1 and a 500-decision trip, 1-step TD needs ~500 sweeps to propagate the terminal cost")
    ap.add_argument("--noise", choices=["gauss", "ou"], default="ou", help="ou = temporally correlated (escapes a boundary optimum)")
    ap.add_argument("--theta", type=float, default=0.15, help="OU mean reversion per decision")
    ap.add_argument("--sigma", type=float, default=0.2, help="exploration noise on the action")
    ap.add_argument("--warmup", type=int, default=4_000, help="steps of purely random actions at the start")
    ap.add_argument("--init", default="", help="checkpoint whose ACTOR warm-starts this run (curriculum stage 1)")
    ap.add_argument("--reward", choices=["P", "L"], default="L", help="P: energy + shaping + deadline price; L: energy + Lagrangian time price (Sec. IV)")
    ap.add_argument("--actor_delay", type=int, default=0, help="env steps before actor updates begin (lets the critic fit the new route's value scale first)")
    ap.add_argument("--target_noise", type=float, default=0.2, help="TD3 target smoothing in normalized action units")
    ap.add_argument("--output", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    for name in ("steps", "log", "buf", "repeat", "nstep"):
        if getattr(a, name) <= 0: ap.error(f"--{name} must be positive")
    if a.warmup < 0 or a.sigma < 0 or a.target_noise < 0: ap.error("warmup/noise must be nonnegative")
    if not 0 <= a.balance <= 1: ap.error("--balance must be in [0, 1]")
    if a.pace_lo <= 0 or a.pace_hi < a.pace_lo or a.startbias <= 0: ap.error("invalid start distribution")
    if REFS is None: ap.error("Run python refresh_refs.py for this route first.")
    from refresh_refs import env_fingerprint
    if REFS.get("environment_sha256") != env_fingerprint():
        ap.error("References do not match the environment; run python refresh_refs.py.")
    if a.reward != "L" or a.curve != "envelope" or a.acct != "ctg" or a.clamp:
        ap.error("Fixed references require --reward L --curve envelope --acct ctg, without --clamp.")
    if a.nstep > 1:
        print("NOTE: uncorrected off-policy n-step target; this is an explicit diagnostic ablation, not plain TD3.", flush=True)
    E.REWARD_MODE = a.reward
    E.CLAMP_LIMIT = a.clamp
    if a.reward == "L": use_lagrangian_refs()
    E.CURVE_MODE = "envelope" if a.curve_env else a.curve
    E.K_CURVE = a.kcurve
    E.SHAPE_C = a.shape
    E.CURVE_ENVELOPE = (E.CURVE_MODE == "envelope")   # the Lagrangian objective assumes
                                                       # a curve can be paid for, not fatal
    if a.acct == "flat": E.CTG_PER_KM, E.FLAT_PEN = 0.0, 20.0
    os.makedirs(a.output, exist_ok=True)
    print(f"TD3 {'mini' if R.MINI else '20km'} seed {a.seed} steps {a.steps} buf {a.buf} pace {a.pace} acct {a.acct} reward {a.reward} | refs DP {REFS['R_dp']:.3f} normal {REFS['R_normal']:.3f} attentive {REFS.get('R_attentive', 0):.3f} paced-normal {REFS.get('R_normal_paced', 0):.3f}", flush=True)
    train(a)
