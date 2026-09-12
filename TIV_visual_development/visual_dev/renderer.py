"""同步只读程序化渲染器（CameraProvider 的最小接入实现），v2。

由原 4 km 环境的位姿 x、时刻 t、道路几何与同一绝对时刻的信号相位驱动，生成 RGB 帧；
图像随策略动作变化，不是固定录像。真值只进入标签与裁判，不画 HUD/彩条/编码图案。
逐 episode 随机：光照、天空、雾（大气衰减）、传感器噪声、遮挡物（树冠）位置与有无。
分辨率为 CPU 烟测用的 96×160，是设计文档 384×640 候选的 1/4，须如实记录。

v2 相对 v1 的修正（对应独立审计 v1 第 3–5 节）：
1. 灯箱安装在停止线远侧 LIGHT_AHEAD=14 m（路口对面的常规安装位置），停车等待时灯箱仍在画面内。
2. 检测框标签改为“目标中心所在特征格内的归一化偏移 + 归一化尺寸”，四个分量都在 [0,1]，与框头 sigmoid 匹配；
   框先裁剪到画面，中心出画面或裁剪后面积为 0 的目标不打标签；解码见 decode_box()。
3. association_valid = 地图关联存在（450 m 内有控制灯）且由地图投影的 ROI 与画面相交（像素数 > 0）；
   另记 map_association、roi_pixels、visible、occluded。
4. 同时记录名义灯箱像素高 housing_px_h 与实际渲染的亮灯光斑直径 lamp_px_d（含 bloom 下限），分层评估两者都报。
5. capture(..., force_color=...) 可在同一位姿/外观下强制灯色，用于“同状态换图”的策略响应检查；不用于训练。
6. v2c：格分配按最近卷积采样中心 (4i,4j)（审查发现原 floor 分配与采样中心有半格错位）；遮挡只清灯所在格的框监督；评估真值框为裁剪后的框。
"""
import math
import numpy as np
from PIL import Image, ImageDraw

W, H = 160, 96
HORIZON = 0.44 * H
CAM_H = 1.4                     # 相机高度 m
FOCAL = 0.9 * W                 # 像素焦距 ≈ 144 px，水平视场约 58°
LAMP_R = 0.15                   # 灯面半径 m
HOUSING = (0.45, 1.2)           # 灯箱宽、高 m
LIGHT_LATERAL, LIGHT_BASE = 3.5, 4.0   # 灯箱位于右侧 3.5 m、离地 4.0 m
LIGHT_AHEAD = 14.0              # 灯箱在停止线远侧 14 m（v2）
SIGN_LATERAL, SIGN_H, SIGN_SIZE = 4.0, 2.5, 0.9
VISIBLE_LAMP_M = 450.0          # 超过此距离灯面低于可分辨像素，不渲染，标签 unknown
MIN_GLOW_PX = 1.3               # 亮灯的最小发光半径（相机 bloom）；这是仿真假设，评估时按实际光斑直径分层披露
CLASSES = ('traffic_light', 'curve_sign', 'end_marker')   # 热图前三类，其余五类保留为 0
COLOR_CLASSES = ('red', 'yellow', 'green', 'off', 'unknown')
LAMP_RGB = {'red': (255, 40, 30), 'yellow': (255, 200, 40), 'green': (40, 230, 90)}
CELL = 4
P2 = (H // CELL, W // CELL)
import os as _os
CELL_ASSIGN = _os.environ.get('RENDER_CELL_ASSIGN', 'floor')   # 'floor'：按 floor(c/4)（默认）；'nearest'：按最近卷积采样中心 (4i,4j)。受控对比（cell_assign_probe，2 种子×600 步）floor 格一致 0.78/0.77、阈值召回 0.48/0.36 均优于 nearest 0.65/0.71、0.09/0.14，故固定为 floor


def signal_color(t, offset, cycle=90., green=30., yellow=4.):
    phase = (t + offset) % cycle
    return 'green' if phase < green else ('yellow' if phase < green + yellow else 'red')


def project(lateral, height, d):
    """地面坐标（横向 m，高度 m，前向距离 m）→ 像素 (u, v)。"""
    u = W / 2 + FOCAL * lateral / d
    v = HORIZON + FOCAL * (CAM_H - height) / d
    return u, v


def encode_box(box):
    """框 (l,t,r,b) 像素 → (格 i, 格 j, 目标向量 [dx, dy, w/W, h/H])，全部在 [0,1]；不可表示时返回 None。"""
    l, t, r, b = box
    l, r = max(l, 0.), min(r, W - 1e-6); t, b = max(t, 0.), min(b, H - 1e-6)
    if r - l <= 1e-6 or b - t <= 1e-6: return None
    cx, cy = (l + r) / 2, (t + b) / 2
    # v2c：特征格 (i,j) 的卷积采样中心在像素 (4i,4j)（stem 与 c2 均为 3×3/stride 2/pad 1），
    # 因此按最近采样中心分配格：i = floor(cy/4 + 0.5)，格内偏移 dx = (cx − 4j)/4 + 0.5 ∈ [0,1)。
    if CELL_ASSIGN == 'nearest':
        i, j = int(math.floor(cy / CELL + .5)), int(math.floor(cx / CELL + .5)); off = .5
    else:   # floor：格 [4i,4i+4) 内的目标归该格，格内偏移 (c−4j)/4 ∈ [0,1)
        i, j = int(math.floor(cy / CELL)), int(math.floor(cx / CELL)); off = 0.
    if not (0 <= i < P2[0] and 0 <= j < P2[1]): return None
    target = np.array([(cx - j * CELL) / CELL + off, (cy - i * CELL) / CELL + off, min((r - l) / W, 1.), min((b - t) / H, 1.)], np.float32)
    if not ((target >= 0).all() and (target <= 1).all()): return None
    return i, j, target, (l, t, r, b)


def decode_box(i, j, target):
    """encode_box 的逆：目标向量 → 像素框 (l,t,r,b)。"""
    dx, dy, w, h = [float(x) for x in target]; off = .5 if CELL_ASSIGN == 'nearest' else 0.
    cx, cy = (j + dx - off) * CELL, (i + dy - off) * CELL; w, h = w * W, h * H
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class SceneCamera:
    def __init__(self, curves, signals, length, seed=0):
        self.curves, self.signals, self.length = list(curves), list(signals), float(length)
        self.rng = np.random.default_rng(seed)
        self.appearance = None
        self.episode_id = None
        self.frames_captured = 0

    # ------------------------------------------------------------ episode
    def new_episode(self, episode_id):
        r = self.rng
        self.episode_id = str(episode_id)
        self.appearance = dict(
            brightness=float(r.uniform(0.6, 1.15)),
            sky=tuple(int(v) for v in r.choice([(150, 190, 235), (200, 205, 215), (230, 170, 120), (90, 100, 130)])),
            road=int(r.integers(70, 105)),
            fog_m=float(r.uniform(250., 2500.)),
            noise=float(r.uniform(1.5, 8.)),
            occluder=(bool(r.random() < 0.35), float(r.uniform(90., 220.)), float(r.uniform(3.5, 6.5)), float(r.uniform(2., 4.))),
        )
        self.frames_captured = 0

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state, appearance=self.appearance, episode_id=self.episode_id,
                    frames_captured=self.frames_captured)

    def load_state_dict(self, s):
        self.rng.bit_generator.state = s['rng']; self.appearance = s['appearance']
        self.episode_id = s['episode_id']; self.frames_captured = s['frames_captured']

    # ------------------------------------------------------------ helpers
    def _fog(self, rgb, d):
        a = 1.0 - math.exp(-d / self.appearance['fog_m'])
        sky = self.appearance['sky']
        return tuple(int(round(c * (1 - a) + s * a)) for c, s in zip(rgb, sky))

    def _bright(self, rgb):
        b = self.appearance['brightness']
        return tuple(int(min(255, c * b)) for c in rgb)

    # ------------------------------------------------------------ capture
    def capture(self, *, episode_id, sim_time, pose, force_color=None):
        """pose = dict(x=..., v=..., offsets=[...]). 返回帧、标签与元信息；capture_time = sim_time。"""
        assert self.appearance is not None and str(episode_id) == self.episode_id, '必须先 new_episode'
        x = float(pose['x']); ap = self.appearance
        img = Image.new('RGB', (W, H), self._bright(ap['sky']))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, HORIZON, W, H], fill=self._bright((110, 125, 80)))
        def lat_off(d):
            for a, b, _ in self.curves:
                da = a - x
                if d > da and da > -100:
                    return min(0.5 * (d - max(da, 0.)) ** 2 / 150., 60.)
            return 0.
        left, right = [], []
        for d in np.geomspace(2.5, 700., 60):
            o = lat_off(d); left.append(project(-5.25 + o, 0., d)); right.append(project(1.75 + o, 0., d))
        dr.polygon(left + right[::-1], fill=self._bright((ap['road'],) * 3))
        for d0 in np.arange(6., 160., 12.):
            p1 = project(-1.75 + lat_off(d0), 0., d0); p2 = project(-1.75 + lat_off(d0 + 4), 0., d0 + 4)
            dr.line([p1, p2], fill=self._bright((225, 225, 210)), width=1)
        labels = dict(heat=np.zeros((8, *P2), np.float32), heat_valid=np.ones((1, *P2), np.float32),
                      boxes=np.zeros((4, *P2), np.float32), box_valid=np.zeros((1, *P2), np.float32),
                      signal=4, visible=0, occluded=0, housing_px_h=0., lamp_px_d=0., light_px_h=0.,
                      light_distance_m=None, color_truth=None, light_box_px=None)
        objects = []
        d_end = self.length - x
        if 0.5 < d_end < 400.:
            objects.append((d_end, lambda: self._draw_marker(dr, d_end, labels)))
        for a, b, _ in self.curves:
            d = a - x
            if 0.5 < d < 320.:
                objects.append((d, lambda d=d: self._draw_sign(dr, d, labels)))
        light = None
        for k, xs in enumerate(self.signals):
            d = xs + LIGHT_AHEAD - x
            if 0.5 < d < 700.:
                color = force_color or signal_color(sim_time, pose['offsets'][k])
                labels['color_truth'] = color; labels['light_distance_m'] = float(d)
                light = (d, color)
                objects.append((d, lambda d=d, c=color: self._draw_light(dr, d, c, labels)))
        occ_on, occ_ahead, occ_h, occ_w = ap['occluder']; occ = None
        if occ_on and light is not None:
            d_occ = light[0] - occ_ahead
            if 0.5 < d_occ < 500.:
                occ = d_occ; objects.append((d_occ, lambda: self._draw_occluder(dr, d_occ, occ_h, occ_w, labels)))
        for d, fn in sorted(objects, key=lambda z: -z[0]):
            fn()
        arr = np.asarray(img).astype(np.float32)
        arr += self.rng.normal(0., ap['noise'], arr.shape)
        rgb = np.clip(arr, 0, 255).astype(np.uint8)
        if light is not None and occ is not None and occ < light[0] and labels.get('_light_uv') and labels.get('_occ_box'):
            u, v = labels['_light_uv']; l, t, r, b = labels['_occ_box']
            if l <= u <= r and t <= v <= b:
                labels['occluded'] = 1; labels['visible'] = 0; labels['signal'] = 4; labels['lamp_px_d'] = 0.
                labels['heat'][0] = 0.; labels['light_box_px'] = None
                if labels.get('_light_cell'):   # v2c：只清灯所在格的框监督，不影响同帧其它类别
                    ci, cj = labels['_light_cell']; labels['box_valid'][0, ci, cj] = 0.; labels['boxes'][:, ci, cj] = 0.
        # 由地图距离投影的 ROI（不用真值框），裁剪到画面；association_valid 要求 ROI 与画面相交
        roi = np.zeros((1, H, W), np.float32); d_map = None
        for xs in self.signals:
            if 0.5 < xs + LIGHT_AHEAD - x <= VISIBLE_LAMP_M: d_map = xs + LIGHT_AHEAD - x
        if d_map is not None:
            u, v = project(LIGHT_LATERAL, LIGHT_BASE + HOUSING[1] / 2, d_map)
            hh = max(FOCAL * HOUSING[1] / d_map, 6.); ww = max(FOCAL * HOUSING[0] / d_map, 6.)
            l, r = int(max(u - ww, 0)), int(min(u + ww, W - 1)); t, b = int(max(v - hh, 0)), int(min(v + hh, H - 1))
            if r >= l and b >= t and u + ww >= 0 and u - ww <= W - 1 and v + hh >= 0 and v - hh <= H - 1:
                roi[0, t:b + 1, l:r + 1] = 1.
        roi_pixels = int(roi.sum())
        meta = dict(capture_time=float(sim_time), episode_id=self.episode_id, frame_index=self.frames_captured,
                    map_association=int(d_map is not None), roi_pixels=roi_pixels,
                    association_valid=int(d_map is not None and roi_pixels > 0), map_distance_m=d_map,
                    forced_color=force_color)
        for k in ('_light_uv', '_occ_box', '_light_cell'):
            labels.pop(k, None)
        self.frames_captured += 1
        return dict(rgb=rgb, labels=labels, roi=roi, meta=meta)

    # ------------------------------------------------------------ objects
    def _mark(self, labels, cls, box, extra=None):
        enc = encode_box(box)
        if enc is None: return False
        i, j, target, clipped = enc
        labels['heat'][cls, i, j] = 1.; labels['boxes'][:, i, j] = target; labels['box_valid'][0, i, j] = 1.
        if extra: labels.update(extra)
        if cls == 0: labels['_light_cell'] = (i, j); labels['light_box_px'] = clipped   # 评估真值框用裁剪后的框（与训练目标一致）
        return True

    def _draw_light(self, dr, d, color, labels):
        pole_rgb = self._fog(self._bright((60, 60, 60)), d)
        base_u, base_v = project(LIGHT_LATERAL, 0., d); top_u, top_v = project(LIGHT_LATERAL, LIGHT_BASE + HOUSING[1], d)
        dr.line([(base_u, base_v), (top_u, top_v)], fill=pole_rgb, width=max(1, int(FOCAL * 0.12 / d)))
        hw, hh = FOCAL * HOUSING[0] / (2 * d), FOCAL * HOUSING[1] / (2 * d)
        cu, cv = project(LIGHT_LATERAL, LIGHT_BASE + HOUSING[1] / 2, d)
        box = (cu - max(hw, .5), cv - max(hh, .5), cu + max(hw, .5), cv + max(hh, .5))
        dr.rectangle(box, fill=self._fog(self._bright((25, 25, 25)), d))
        lamp_px = FOCAL * LAMP_R / d
        in_frame = 0. <= cv <= H - 1 and 0. <= cu <= W - 1
        visible = d <= VISIBLE_LAMP_M and in_frame
        glow_r = max(lamp_px, MIN_GLOW_PX)
        for k, name in enumerate(('red', 'yellow', 'green')):
            lu, lv = project(LIGHT_LATERAL, LIGHT_BASE + HOUSING[1] * (0.83 - 0.33 * k), d)
            if name == color and d <= VISIBLE_LAMP_M:
                dr.ellipse([lu - glow_r, lv - glow_r, lu + glow_r, lv + glow_r], fill=self._fog(self._bright(LAMP_RGB[name]), d))
            elif lamp_px >= 0.8:
                dr.ellipse([lu - lamp_px, lv - lamp_px, lu + lamp_px, lv + lamp_px], fill=self._fog(self._bright((45, 40, 40)), d))
        labels['_light_uv'] = (cu, cv)
        labels['housing_px_h'] = labels['light_px_h'] = float(2 * hh)
        labels['lamp_px_d'] = float(2 * glow_r) if visible else 0.
        labels['light_box_px'] = None
        if visible and self._mark(labels, 0, box, dict(signal=COLOR_CLASSES.index(color), visible=1)):
            return
        labels['signal'] = 4; labels['visible'] = 0

    def _draw_sign(self, dr, d, labels):
        cu, cv = project(SIGN_LATERAL, SIGN_H, d); s = max(FOCAL * SIGN_SIZE / (2 * d), .6)
        pu, pv = project(SIGN_LATERAL, 0., d)
        dr.line([(pu, pv), (cu, cv)], fill=self._fog(self._bright((70, 70, 70)), d), width=1)
        dr.polygon([(cu, cv - s), (cu + s, cv), (cu, cv + s), (cu - s, cv)], fill=self._fog(self._bright((245, 200, 30)), d))
        if s >= 1.2:
            dr.line([(cu - s * .5, cv + s * .3), (cu, cv - s * .2), (cu + s * .5, cv + s * .3)], fill=self._fog(self._bright((20, 20, 20)), d), width=1)
        self._mark(labels, 1, (cu - s, cv - s, cu + s, cv + s))

    def _draw_marker(self, dr, d, labels):
        lu, lv = project(-5.25, 0., d); ru, rv = project(1.75, 0., d)
        dr.line([(lu, lv), (ru, rv)], fill=self._fog(self._bright((240, 240, 240)), d), width=max(1, int(FOCAL * 0.5 / d)))
        for lat, col in ((-5.25, (200, 30, 30)), (1.75, (200, 30, 30))):
            bu, bv = project(lat, 0., d); tu, tv = project(lat, 1.6, d)
            dr.line([(bu, bv), (tu, tv)], fill=self._fog(self._bright(col), d), width=max(1, int(FOCAL * 0.2 / d)))
        self._mark(labels, 2, (lu, min(lv, rv) - max(FOCAL * 1.6 / d, 1), ru, max(lv, rv)))

    def _draw_occluder(self, dr, d, h, w, labels):
        cu, cv = project(LIGHT_LATERAL - 0.5, h, d); rw, rh = max(FOCAL * w / (2 * d), 1.), max(FOCAL * (h * 0.6) / (2 * d), 1.)
        bu, bv = project(LIGHT_LATERAL - 0.5, 0., d)
        dr.line([(bu, bv), (cu, cv)], fill=self._fog(self._bright((80, 55, 30)), d), width=max(1, int(FOCAL * 0.3 / d)))
        dr.ellipse([cu - rw, cv - rh, cu + rw, cv + rh], fill=self._fog(self._bright((40, 95, 40)), d))
        labels['_occ_box'] = (cu - rw, cv - rh, cu + rw, cv + rh)
