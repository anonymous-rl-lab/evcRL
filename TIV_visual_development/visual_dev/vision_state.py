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
    def __init__(self, det_thr=0.5, hold_s=3.0, curve_len_max=1000., release_grace_m=5., init_votes=2, vote_window=3, consistency_m=40., expire_s=4.0, light_expire_s=10.0,
                 gain=0.4, margin_rel=0.15, margin_abs=3.0, color_freeze_m=8.0, dist_freeze_m=15.0, expire_far_m=60.0, maintain_ratio=0.5):
        """gain：已有轨迹的距离更新增益（推算值与新测量的加权），抑制逐帧抖动；margin_rel/abs：执行层目标的感知不确定性余量，
        目标距离 = 估计 − (margin_rel·估计 + margin_abs)（红灯停车与入弯限速都提前，保守）；color_freeze_m：停止线估计小于该距离时不再更新灯色（近距离灯色不可靠，且已无法改变决策）。"""
        """det_thr：检测门控阈值（float 或 dict 类别→阈值）；init_votes/vote_window：新建轨迹需最近 vote_window 帧中 ≥init_votes 帧过阈值；
        consistency_m：已有轨迹只接受与推算值相差 ≤consistency_m 的新距离估计（否则记一次不一致，连续 3 次后重置轨迹）。"""
        self.det_thr = det_thr; self.hold_s = hold_s; self.curve_len_max = curve_len_max; self.release_grace_m = release_grace_m
        self.init_votes = init_votes; self.vote_window = vote_window; self.consistency_m = consistency_m; self.expire_s = expire_s; self.light_expire_s = light_expire_s
        self.gain = gain; self.margin_rel = margin_rel; self.margin_abs = margin_abs; self.color_freeze_m = color_freeze_m
        self.dist_freeze_m = dist_freeze_m; self.expire_far_m = expire_far_m   # 近线（<15 m）距离只按车速推算不再用视觉更新；轨迹只在目标仍远（>60 m）且长时间未见时过期
        self.maintain_ratio = maintain_ratio   # 迟滞：已有轨迹的维持阈值 = 建轨阈值 × maintain_ratio（近距离/绿灯等弱响应下不丢轨迹）
        self.reset()

    def _fuse(self, tracked, new):
        return new if tracked is None else (1. - self.gain) * tracked + self.gain * new

    def _margin(self, d):
        return max(d - (self.margin_rel * max(d, 0.) + self.margin_abs), 0.)

    def thr(self, name):
        return self.det_thr.get(name, 0.5) if isinstance(self.det_thr, dict) else self.det_thr

    def reset(self):
        self.curve = dict(d_est=None, announced=False, active=False, age=math.inf, release_d=None, traveled_since_entry=0.)
        self.sig = dict(d_line=None, phase='unknown', conf=0., age=math.inf, dur=0., seen=False)
        self.end = dict(d_est=None, age=math.inf)
        self.last_dets = None; self.votes = {n: [] for n in ('traffic_light', 'curve_sign', 'end_marker', 'release_sign')}; self.mismatch = {n: 0 for n in self.votes}

    def _tracked(self, name):
        return {'traffic_light': self.sig['seen'], 'curve_sign': self.curve['announced'] and not self.curve['active'], 'end_marker': self.end['d_est'] is not None,
                'release_sign': self.curve['release_d'] is not None}[name]

    def _vote(self, name, dets):
        d = dets.get(name); t = self.thr(name) * (self.maintain_ratio if self._tracked(name) else 1.)
        hit = bool(d and d['score'] >= t); v = self.votes[name]; v.append(bool(d and d['score'] >= self.thr(name))); del v[:-self.vote_window]
        return hit, sum(v) >= self.init_votes

    def _accept(self, name, tracked_d, new_d):
        """已有轨迹的距离更新是否与推算一致。"""
        if tracked_d is None: self.mismatch[name] = 0; return True
        if abs(new_d - tracked_d) <= self.consistency_m: self.mismatch[name] = 0; return True
        self.mismatch[name] += 1; return False

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
        hit, ok = self._vote('curve_sign', dets); cs = dets.get('curve_sign')
        if hit and not self.curve['active'] and (self.curve['announced'] or ok):
            new_d = float(cs['dist_m']) + SIGN_AHEAD
            if self._accept('curve_sign', self.curve['d_est'] if self.curve['announced'] else None, new_d):
                self.curve['d_est'] = self._fuse(self.curve['d_est'] if self.curve['announced'] else None, new_d); self.curve['announced'] = True; self.curve['age'] = 0.
            elif self.mismatch['curve_sign'] >= 3 and cs['score'] >= self.thr('curve_sign') and ok:   # 重置只接受建轨强度（初始阈值 + 2/3 票）的一致新证据
                self.curve.update(d_est=new_d, announced=True, age=0.); self.mismatch['curve_sign'] = 0
            else: self.curve['age'] = 0. if self.curve['announced'] else self.curve['age']   # 弱一致检测：只刷新年龄，不改距离
        if self.curve['announced'] and not self.curve['active'] and self.curve['d_est'] is not None and self.curve['d_est'] <= 0.:
            self.curve['active'] = True; self.curve['traveled_since_entry'] = 0.
        hit, ok = self._vote('release_sign', dets); rs = dets.get('release_sign')
        if hit and ok and self.curve['active']:   # 只在弯道限速生效期间采纳解除牌（出口处），避免入弯前的误检提前解除
            self.curve['release_d'] = float(rs['dist_m']) if self.curve['release_d'] is None else self._fuse(self.curve['release_d'], float(rs['dist_m']))
        if self.curve['active'] and ((self.curve['release_d'] is not None and self.curve['release_d'] <= -self.release_grace_m) or self.curve['traveled_since_entry'] > self.curve_len_max):
            self.curve.update(d_est=None, announced=False, active=False, age=math.inf, release_d=None, traveled_since_entry=0.)
        # 信号灯
        hit, ok = self._vote('traffic_light', dets); lt = dets.get('traffic_light')
        if hit and (self.sig['seen'] or ok) and lt['dist_m'] <= LIGHT_RANGE + LIGHT_AHEAD and color_probs is not None:
            new_d = float(lt['dist_m']) - LIGHT_AHEAD
            if self._accept('traffic_light', self.sig['d_line'] if self.sig['seen'] else None, new_d) or (self.mismatch['traffic_light'] >= 3 and lt['score'] >= self.thr('traffic_light') and ok):
                near = self.sig['seen'] and self.sig['d_line'] is not None and self.sig['d_line'] < self.dist_freeze_m
                fused = self.sig['d_line'] if near else self._fuse(self.sig['d_line'] if self.sig['seen'] else None, new_d)   # 近线：距离冻结（推算更准）
                if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['d_line'] < self.color_freeze_m and v > 3.:
                    phase, conf = self.sig['phase'], self.sig['conf']   # 近距离且仍在行驶（已无法改变决策）：灯色冻结；停车等待时照常更新
                else:
                    phase = PHASES[int(np.argmax(color_probs))]; conf = float(np.max(color_probs))
                self.sig['dur'] = self.sig['dur'] + dt if (self.sig['seen'] and phase == self.sig['phase']) else 0.
                self.sig.update(d_line=fused, phase=phase, conf=conf, age=0., seen=True); self.mismatch['traffic_light'] = 0
        elif self.sig['seen'] and self.sig['age'] > self.hold_s and self.sig['phase'] != 'unknown':
            self.sig['phase'] = 'unknown'; self.sig['conf'] = 0.
        if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['d_line'] < -5.:
            self.sig.update(d_line=None, phase='unknown', conf=0., age=math.inf, dur=0., seen=False)
        # 终点
        hit, ok = self._vote('end_marker', dets); em = dets.get('end_marker')
        if hit and (self.end['d_est'] is not None or ok):
            if self._accept('end_marker', self.end['d_est'], float(em['dist_m'])) or (self.mismatch['end_marker'] >= 3 and em['score'] >= self.thr('end_marker') and ok):
                if not (self.end['d_est'] is not None and self.end['d_est'] < self.dist_freeze_m):
                    self.end['d_est'] = self._fuse(self.end['d_est'], float(em['dist_m']))
                self.end['age'] = 0.; self.mismatch['end_marker'] = 0
        # 轨迹失效（误检保护）：目标按估计仍应在视野内却长时间未再检出 → 丢弃
        if self.end['d_est'] is not None and self.end['d_est'] > self.expire_far_m and self.end['age'] > self.expire_s:
            self.end.update(d_est=None, age=math.inf)
        if self.curve['announced'] and not self.curve['active'] and self.curve['d_est'] is not None and (self.curve['d_est'] - SIGN_AHEAD) > self.expire_far_m and self.curve['age'] > self.expire_s:
            self.curve.update(d_est=None, announced=False, age=math.inf)
        if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['d_line'] > self.expire_far_m and self.sig['age'] > self.light_expire_s:
            self.sig.update(d_line=None, phase='unknown', conf=0., age=math.inf, dur=0., seen=False)

    # ---------------------------------------------------------------- 三方共享的接口
    def v_limit(self):
        return V_CURVE if self.curve['active'] else V_FREE

    def executor_targets(self, x):
        """(绝对位置估计, 目标速度[, 2.0]) 列表，全部来自记忆。"""
        out = []
        if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['phase'] != 'green':
            out.append((x + self._margin(self.sig['d_line']), 0.0))                       # 保守：按不确定性余量提前停
        if self.curve['announced'] and not self.curve['active'] and self.curve['d_est'] is not None:
            out.append((x + max(self._margin(self.curve['d_est']) - S.E.MARGIN, 0.), V_CURVE, 2.0))
        if self.end['d_est'] is not None:
            out.append((x + self.end['d_est'] + 2.0, 0.0))   # 终点：固定锚点（估计可为负 → 立即停），偏后 2 m 保证过线到达；不随车移动
        return out

    def executor_signal(self, assumed_green_remaining=30.):
        """(前方信号是否可通行, 假定剩余绿灯时间, 停止线距离估计)。"""
        if self.sig['seen'] and self.sig['d_line'] is not None and self.sig['d_line'] > 0.:
            return self.sig['phase'] == 'green', assumed_green_remaining, self._margin(self.sig['d_line'])
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
