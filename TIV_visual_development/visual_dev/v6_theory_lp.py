"""把论文定理 1 的结构（早知/延迟知情、停车或通行两种情境、共同准备、jerk 上限）搬到已部署的离散车辆模型上，用两阶段线性规划求
早知与延迟知情的最优期望"损失距离"（= 损失时间 × v0），得到三个边界（各情境可行 / 共同准备可行 / 无额外代价）随分辨延迟 Δ 与 jerk 上限 j 的位置。
离散动力学与执行层一致：DT=0.5 s，a∈[−3.5, 2.6]，|Δa|≤j·DT，0≤v≤v0，梯形积分。灯位置在 d0 处已知（t=0），灯色在 Δ 后可知；
机动窗口 T_m 固定（默认 1.5·d0/v0，对应定理的 T_m=3Δ 与 d=2v0Δ 的比例），终端：R 停在线前（v=0, x≤d0），G 过线且恢复 v0（v=v0, x≥d0），终端加速度 0。
目标：最小化期望损失距离 p·(v0·T_m − x_R(T_m)) + (1−p)·(v0·T_m − x_G(T_m))；延迟观察者在 k<Δ/DT 的动作对两情境相同（非预期约束）。
用法：python visual_dev/v6_theory_lp.py [--v0 22.2 --d0 200 --jerks 2,3,4 --deltas 0.5:9:0.5 --p 0.5]"""
import argparse, json, sys
import numpy as np
from scipy.optimize import linprog
DT = .5; A_LO = -3.5; A_HI = 2.6


def solve(v0, d0, Tm, j, K, p, scenes=('R', 'G'), common=True):
    """两情境联合 LP；common=True 时前 K 步动作相同（延迟观察者），False 时各自独立（早知观察者，等价于分别求解）。返回 (状态, 期望损失距离, 各情境损失, a_R, a_G)。"""
    N = int(round(Tm / DT)); n = N  # 每情境 N 个动作
    nv = 2 * n; c = np.zeros(nv)
    # x_N = sum_k [DT*v_k + 0.5*DT^2*a_k]，v_k = v0 + DT*sum_{i<k} a_i → x_N = N*DT*v0 + sum_k a_k * (DT^2*(N-k-1) + 0.5*DT^2) → 损失 = v0*Tm − x_N 线性于 a
    w = np.array([DT * DT * (N - k - 1) + .5 * DT * DT for k in range(N)])
    c[:n] = -p * w; c[n:] = -(1 - p) * w   # 最小化 −x（即最小化损失）
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for s, off in (('R', 0), ('G', n)):
        for k in range(N):   # 速度上下界：v_{k+1} = v0 + DT*sum_{i<=k} a_i ∈ [0, v0]
            row = np.zeros(nv); row[off:off + k + 1] = DT; A_ub.append(row); b_ub.append(0.)          # v_{k+1} ≤ v0
            A_ub.append(-row); b_ub.append(v0)                                                       # v_{k+1} ≥ 0
        for k in range(N):   # jerk：|a_k − a_{k−1}| ≤ j·DT，a_{−1}=0
            row = np.zeros(nv); row[off + k] = 1.
            if k > 0: row[off + k - 1] = -1.
            A_ub.append(row); b_ub.append(j * DT); A_ub.append(-row); b_ub.append(j * DT)
        row = np.zeros(nv); row[off + N - 1] = 1.; A_eq.append(row); b_eq.append(0.)   # 终端加速度 0
        row = np.zeros(nv); row[off:off + N] = DT   # v_N − v0
        if s == 'R': A_eq.append(row); b_eq.append(-v0)      # v_N = 0
        else: A_eq.append(row); b_eq.append(0.)              # v_N = v0
        rowx = np.zeros(nv); rowx[off:off + N] = w            # x_N − N*DT*v0
        if s == 'R': A_ub.append(rowx); b_ub.append(d0 - N * DT * v0)     # x_N ≤ d0
        else: A_ub.append(-rowx); b_ub.append(-(d0 - N * DT * v0))       # x_N ≥ d0
    if common:
        for k in range(min(K, N)):
            row = np.zeros(nv); row[k] = 1.; row[n + k] = -1.; A_eq.append(row); b_eq.append(0.)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(A_LO, A_HI)] * nv, method='highs')
    if res.status != 0: return 'infeasible', None, None, None, None
    aR, aG = res.x[:n], res.x[n:]; lossR = v0 * Tm - (N * DT * v0 + w @ aR); lossG = v0 * Tm - (N * DT * v0 + w @ aG)
    return 'ok', p * lossR + (1 - p) * lossG, (lossR, lossG), aR, aG


def scene_feasible(v0, d0, Tm, j, scene):
    st, *_ = solve(v0, d0, Tm, j, 0, .5, common=False)
    return st == 'ok'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--v0', type=float, default=22.2222); ap.add_argument('--d0', type=float, default=200.); ap.add_argument('--Tm', type=float, default=None); ap.add_argument('--jerks', default='2,3,4'); ap.add_argument('--deltas', default='0.5:9:0.5'); ap.add_argument('--p', type=float, default=.5); ap.add_argument('--out', default='runs/v5_timing/theory_lp.json')
    a = ap.parse_args(); Tm = a.Tm or 1.5 * a.d0 / a.v0; lo, hi, st = [float(x) for x in a.deltas.split(':')]; deltas = np.arange(lo, hi + 1e-9, st)
    out = dict(v0=a.v0, d0=a.d0, Tm=Tm, p=a.p, DT=DT, rows=[])
    print(f"v0={a.v0:.1f} m/s  d0={a.d0:.0f} m  T_m={Tm:.1f} s（N={int(round(Tm/DT))} 步）  p={a.p}")
    for j in [float(x) for x in a.jerks.split(',')]:
        st_e, E_e, l_e, aRe, aGe = solve(a.v0, a.d0, Tm, j, 0, a.p, common=False)
        if st_e != 'ok': print(f"jerk {j:.0f}: 早知观察者不可行（各情境无法在 T_m 内完成）"); out['rows'].append(dict(jerk=j, early='infeasible')); continue
        print(f"jerk {j:.0f}: 早知 期望损失 {E_e:.2f} m（R {l_e[0]:.2f}, G {l_e[1]:.2f}）→ 时间 {E_e/a.v0:.3f} s")
        d_common = None; d_free = None
        for D in deltas:
            K = int(round(D / DT)); st_d, E_d, l_d, aRd, aGd = solve(a.v0, a.d0, Tm, j, K, a.p, common=True)
            if st_d != 'ok': row = dict(jerk=j, delta=float(D), delayed='infeasible'); print(f"   Δ={D:4.1f} s（分辨时 d≈{a.d0 - a.v0*D:5.1f} m）: 共同准备不可行"); out['rows'].append(row); continue
            pen = (E_d - E_e) / a.v0; q = -min(aRd[:K]) if K > 0 else 0.
            row = dict(jerk=j, delta=float(D), delayed='ok', penalty_s=float(pen), lossR=float(l_d[0]), lossG=float(l_d[1]), prep_max_decel=float(q), a_common=[float(x) for x in aRd[:K]])
            print(f"   Δ={D:4.1f} s（分辨时 d≈{a.d0 - a.v0*D:5.1f} m）: 可行，期望损失时间惩罚 {pen:.3f} s（G 支 {(l_d[1]-l_e[1])/a.v0:+.3f} s, R 支 {(l_d[0]-l_e[0])/a.v0:+.3f} s），共同准备最大减速 {q:.2f} m/s²")
            d_common = float(D)
            if pen < 1e-6 and d_free is None: d_free = float(D)
            elif pen < 1e-6: d_free = float(D)
            out['rows'].append(row)
        out.setdefault('boundaries', {})[str(j)] = dict(common_feasible_max_delta=d_common, zero_penalty_max_delta=d_free)
        print(f"   → jerk {j:.0f}: 共同准备可行的最大 Δ = {d_common} s；无额外代价的最大 Δ = {d_free} s")
    json.dump(out, open(a.out, 'w'), indent=1); print('写入', a.out)


if __name__ == '__main__':
    main()
