"""v4：无地图、靠视觉获取道路事件信息的纵向驾驶——视觉短期记忆与感知执行层。

VisionMemory 由检测器输出维护三条轨迹（弯道、信号灯、终点），是 actor、critic 与执行层共同看到的唯一道路事件信息：
  弯道：看到警示牌（距离估计 d_sign）→ 入弯距离估计 = d_sign + 400 m → 按车速积分推算 → 到达后限速 35 km/h 生效 →
        看到解除牌并驶过其估计位置后解除（安全阀：入弯估计后行驶 >1000 m 自动解除）。
  信号灯：200 m 内检测到灯 → 停止线距离估计 = 灯箱距离 − 14 m，相位/置信度/信息年龄/当前相位持续时间；灯离开画面后
        距离按车速推算，相位保持 hold_s 秒后变为 unknown；驶过停止线估计位置 5 m 后清除。
  终点：看到终点标志 → 距离估计并推算。
真值（仿真的道路与信号位置）只用于渲染画面、物理过程与裁判，不进入本模块。
"""
import math
import numpy as np
import study as S

PHASES = ('red', 'yellow', 'green', 'off', 'unknown')
V_FREE, V_CURVE = S.R.V_FREE, S.R.V_CURVE
SIGN_AHEAD, LIGHT_AHEAD, LIGHT_RANGE = 400., 14., 200.


class VisionMemory:
    def __init__(self, det_thr=0.5, hold_s=3.0, curve_len_max=1000., release_grace_m=5.):
        self.det_thr = det_thr; self.hold_s = hold_s; self.curve_len_max = curve_len_max; self.release_grace_m = release_grace_m
        self.reset()

    def reset(self):
        self.curve = dict(d_est=None, announced=False, active=False, age=math.inf, release_d=None, traveled_since_entry=0.)
        self.sig = dict(d_line=None, phase='unknown', conf=0., age=math.inf, dur=0., seen=False)
        self.end = dict(d_est=None, age=math.inf)
        self.last_dets = None

    # ---------------------------------------------------------------- 观测融合
    def update(self, dets, color_probs, v, dt):
        """dets: dict 类别名 -> dict(score, dist_m)；color_probs: 5 类概率（灯未检出时可为 None）；v: 车速 m/s；dt: 子步长。"""
        self.last_dets = dets; v = float(max(v, 0.)); ds = v * dt
        for tr, key in ((self.curve, 'd_est'), (self.sig, 'd_line'), (self.end, 'd_est')):
            if tr[key] is not None: tr[key] -= ds
            tr['age'] += dt
        if self.curve['release_d'] is not None: self.curve['release_d'] -= ds
        if self.curve['active']: self.curve['traveled_since_entry'] += ds
        # 弯道警示牌
        cs = dets.get('curve_sign')
        if cs and cs['score'] >= self.det_thr and not self.curve['active']:
            self.curve['d_est'] = float(cs['dist_m']) + SIGN_AHEAD; self.curve['announced'] = True; self.curve['age'] = 0.
        if self.curve['announced'] and not self.curve['active'] and self.curve['d_est'] is not None and self.curve['d_est'] <= 0.:
            self.curve['active'] = True; self.curve['traveled_since_entry'] = 0.
        rs = dets.get('release_sign')
        if rs and rs['score'] >= self.det_thr and (self.curve['announced'] or self.curve['active']):
            self.curve['release_d'] = float(rs['dist_m'])
        if self.curve['active'] and ((self.curve['release_d'] is not None and self.curve['release_d'] <= -self.release_grace_m) or self.curve['traveled_since_entry'] > self.curve_len_max):
            self.curve.update(d_est=None, announced=False, active=False, age=math.inf, release_d=None, traveled_since_entry=0.)
        # 信号灯
        lt = dets.get('traffic_light')
        if lt and lt['score'] >= self.det_thr and lt['dist_m'] <= LIGHT_RANGE + LIGHT_AHEAD and color_probs is not None:
            phase = PHASES[int(np.argmax(color_probs))]; conf = float(np.max(color_probs))
            self.sig['dur'] = self.sig['dur'] + dt if (self.sig['seen'] and phase == self.sig['phase']) else 0.
            self.sig.update(d_line=float(lt['dist_m']) - LIGHT_AHEAD, phase=phase, conf=conf, age=0., seen=True)
        elif self.sig['seen'] and self.sig['age'] > self.hold_s and self.sig['phase'] != 'unknown':
            self.sig['phase'] = 'unknown'; self.sig['conf'] = 0.
        if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['d_line'] < -5.:
            self.sig.update(d_line=None, phase='unknown', conf=0., age=math.inf, dur=0., seen=False)
        # 终点
        em = dets.get('end_marker')
        if em and em['score'] >= self.det_thr:
            self.end['d_est'] = float(em['dist_m']); self.end['age'] = 0.

    # ---------------------------------------------------------------- 三方共享的接口
    def v_limit(self):
        return V_CURVE if self.curve['active'] else V_FREE

    def executor_targets(self, x):
        """(绝对位置估计, 目标速度[, 2.0]) 列表，全部来自记忆。"""
        out = []
        if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['phase'] != 'green':
            out.append((x + max(self.sig['d_line'], 0.) - S.E.STOP_MARGIN, 0.0))
        if self.curve['announced'] and not self.curve['active'] and self.curve['d_est'] is not None:
            out.append((x + max(self.curve['d_est'] - S.E.MARGIN, 0.), V_CURVE, 2.0))
        if self.end['d_est'] is not None:
            out.append((x + max(self.end['d_est'], 0.), 0.0))
        return out

    def executor_signal(self, assumed_green_remaining=30.):
        """(前方信号是否可通行, 假定剩余绿灯时间, 停止线距离估计)。"""
        if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['d_line'] > 0.:
            return self.sig['phase'] == 'green', assumed_green_remaining, self.sig['d_line']
        return None, None, None

    def legacy(self, v, a, soc, T, t_end, t, t_budget):
        """与原 13 维观测同槽位、同归一化，但道路事件信息全部来自记忆（未知 → 1.0 表示“远/未知”）。"""
        d_end = self.end['d_est'] if self.end['d_est'] is not None else S.R.LENGTH
        dc = self.curve['d_est'] if (self.curve['announced'] and not self.curve['active'] and self.curve['d_est'] is not None) else S.E.PREVIEW
        vc = V_CURVE if self.curve['announced'] or self.curve['active'] else V_FREE
        ds = self.sig['d_line'] if (self.sig['seen'] and self.sig['d_line'] is not None) else S.E.SPAT_RANGE
        phase = {'green': 1., 'red': 0., 'yellow': 0., 'off': 0.}.get(self.sig['phase'], -1.) if self.sig['seen'] else -1.
        return np.array([v / V_FREE, a / 3.5, max(d_end, 0.) / S.R.LENGTH, self.v_limit() / V_FREE, min(max(dc, 0.), S.E.PREVIEW) / S.E.PREVIEW, vc / V_FREE,
                         min(max(ds, 0.), S.E.SPAT_RANGE) / S.E.SPAT_RANGE, phase, 1.0, (soc - .5) / .5, (T - 283.15) / 30., (t_end - t) / t_budget, t / t_budget], np.float32)

    def extra(self):
        """附加 6 维：灯置信度、灯信息年龄/10、当前相位持续/90、弯道已宣告、弯道限速激活、终点已见。"""
        return np.array([self.sig['conf'] if self.sig['seen'] else 0., min(self.sig['age'], 10.) / 10. if self.sig['seen'] else 1.,
                         min(self.sig['dur'], 90.) / 90. if self.sig['seen'] else 0., float(self.curve['announced']), float(self.curve['active']),
                         float(self.end['d_est'] is not None)], np.float32)

    def light_detected(self):
        return int(self.sig['seen'] and self.sig['age'] <= 0.)

    def snapshot(self):
        return dict(curve=dict(self.curve), sig=dict(self.sig), end=dict(self.end))
