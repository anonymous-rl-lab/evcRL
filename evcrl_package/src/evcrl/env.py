"""Gymnasium adapter of the archived eco-driving environment.

Action contract
    ``step(u)`` receives ONE normalized command ``u`` in [-1, 1] (shape ``(1,)``; a
    scalar is accepted). It is mapped to a raw acceleration command
    ``a_cmd = 2.6*u`` for ``u > 0`` and ``3.5*u`` otherwise (``normalized_to_command``)
    and HELD for ``repeat`` physical substeps of 0.5 s. The executor decides, in
    every substep, the acceleration actually applied. ``info['substeps']`` is the
    execution record: for each substep the command, the applied acceleration, the
    speeds before/after, the odometric displacement, the jerk and the reward
    components. ``info['applied_a']`` lists the ``repeat`` applied accelerations.
    The command is the environment action; no applied acceleration (first or
    otherwise) is offered as a learning label.

Termination
    ``terminated`` = settled at the destination, red-light crossing, the intrinsic
    task deadline (``task_deadline_s``), or overshooting the destination by more
    than ``overshoot_m`` (camera mode). ``truncated`` = external cap
    ``max_episode_seconds`` only. The intrinsic deadline is a task outcome and
    carries the archived terminal cost; the external cap carries none.

Truth
    Road geometry and signal phases are used for rendering, the physics and the
    judge only. Under ``mode='camera'`` the executor sees road events through the
    attached detector and the memory; without a detector it is blind.
"""
import copy
import hashlib
import json
import math
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .backend import Backend, parameters
from .route import Route
from .memory import VisionMemory
from .execution import Executor
from .camera import SceneCamera, W, H
from .profiles import get_profile, profile_identity, source_identity, CONTRACT_VERSION
from ._version import __version__

SNAPSHOT_SCHEMA = 2
DETECTION_CLASSES = ('traffic_light', 'curve_sign', 'end_marker', 'release_sign')
HISTORY_FRAMES = 4


def normalized_to_command(u):
    """Normalized command in [-1, 1] -> raw acceleration command in m/s² (archived ``study.command``)."""
    u = float(min(max(float(u), -1.), 1.))
    return u * (2.6 if u > 0. else 3.5)


def command_to_normalized(a):
    """Inverse of ``normalized_to_command`` (clipped to [-1, 1])."""
    a = float(a)
    return float(min(max(a / 2.6 if a > 0. else a / 3.5, -1.), 1.))


class PerceivedBackend(Backend):
    """Backend whose executor targets come from the vision memory instead of the map."""

    def __init__(self, *args, memory, executor, a_comfort, assumed_green_remaining, **kwargs):
        self.memory = memory; self.executor = executor; self.a_comfort = float(a_comfort)
        self.assumed_green_remaining = float(assumed_green_remaining); self.last_execution = {}
        super().__init__(*args, **kwargs)

    def _stop_targets(self): return self.memory.executor_targets(self.x)

    def project(self, cmd):
        fallback = super().project(cmd)   # archived envelope on the memory targets: emergency policy only
        targets = [(max(t[0] - self.x, 0.), t[1]) for t in self._stop_targets()]
        limit = self.memory.v_limit()
        for d, vt in targets: limit = min(limit, math.sqrt(max(vt * vt + 2. * self.a_comfort * d, 0.)))
        C = self.executor
        result, info = C.project(self.v, self.a, cmd, targets, fallback, vmax=limit)
        result = min(result, C.brake_bound(max(limit - self.v, 0.)))
        lower = info.get('lower'); upper = info.get('upper')
        if upper is not None: upper = min(upper, C.brake_bound(max(limit - self.v, 0.)))
        info['envelope_braking'] = False
        if not info['fallback'] and self.v > limit + 1e-6:
            a_env = max((limit - self.v) / self.cfg.DT, lower)
            result = min(result, a_env); upper = min(upper, a_env)
            info['envelope_braking'] = True
        green, remaining, d = self.memory.executor_signal(self.assumed_green_remaining)
        if not info['fallback'] and green:
            action, extra = C.signal_project(self.v, self.a, cmd, (lower, upper), d, remaining, limit)
            info.update(extra)
            if action is None: info['fallback'] = True; result = fallback
            else: result = action
        info.update(applied=float(result), command=float(cmd), lower=lower, upper=upper, v_limit=float(limit),
                    target_distances=[(float(a), float(b)) for a, b in targets])
        self.last_execution = info
        return result


class EcoDriveEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 2}

    def __init__(self, mode='camera', route_length=4000., profile='paper_v25', render_mode=None, detector=None,
                 jerk=None, repeat=4, task_deadline_s='historical', max_episode_seconds=None, clip_actions=True):
        if mode not in ('camera', 'structured'): raise ValueError("mode must be 'camera' or 'structured', got %r" % (mode,))
        if render_mode not in (None, 'rgb_array'): raise ValueError("render_mode must be None or 'rgb_array'")
        if not isinstance(repeat, (int, np.integer)) or isinstance(repeat, bool) or repeat < 1: raise ValueError('repeat must be a positive integer')
        if detector is not None and not callable(detector): raise TypeError('detector must be callable or None')
        self.rules = get_profile(profile); self.profile = profile; self.profile_identity = profile_identity(profile)
        self.mode = mode; self.route = Route(float(route_length))
        if mode == 'camera' and not self.route.MINI: raise ValueError('Only the 4 km route is packaged for the camera task')
        ep = self.rules['episode']; self._route_rules = ep['routes'][str(self.route.length)]
        self.render_mode = render_mode; self.detector = detector; self.repeat = int(repeat); self.clip_actions = bool(clip_actions)
        ex = self.rules['execution']; self.jerk = float(ex['jerk'] if jerk is None else jerk); self.dt = float(ex['dt'])
        Executor(self.jerk, self.dt)   # validates the supported release-bound family early
        if task_deadline_s == 'historical': self.deadline = float(self._route_rules['deadline_s'])
        elif task_deadline_s is None: self.deadline = None
        else:
            self.deadline = float(task_deadline_s)
            if not math.isfinite(self.deadline) or self.deadline <= 0: raise ValueError('task_deadline_s must be positive, None or "historical"')
        if max_episode_seconds is not None and (not math.isfinite(max_episode_seconds) or max_episode_seconds <= 0): raise ValueError('max_episode_seconds must be positive or None')
        self.max_episode_seconds = None if max_episode_seconds is None else float(max_episode_seconds)
        self.action_space = spaces.Box(-1., 1., (1,), dtype=np.float32)
        if mode == 'camera':
            self.observation_space = spaces.Dict({'rgb': spaces.Box(0, 255, (H, W, 3), dtype=np.uint8),
                                                  'ego': spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)})
        else: self.observation_space = spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32)
        config = dict(package='evcrl', contract=CONTRACT_VERSION, profile=profile, profile_identity=self.profile_identity, mode=mode,
                      route_length=self.route.length, jerk=self.jerk, repeat=self.repeat, deadline=self.deadline, external_cap=self.max_episode_seconds)
        self.identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.source_identity = source_identity()
        self._needs_camera = mode == 'camera' or render_mode == 'rgb_array'
        self.camera = None; self.rgb = None; self.history = []
        self._ended = True; self._initialized = False

    # ------------------------------------------------------------------ lifecycle
    @property
    def detector_kind(self):
        if self.detector is None: return None
        return getattr(self.detector, 'kind', 'callback')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); options = dict(options or {})
        init = self.rules['episode']['initial']
        soc = float(options.pop('soc', init['soc'])); temp = float(options.pop('temperature', init['temperature'])); v = float(options.pop('speed', init['speed']))
        if not (0 < soc <= 1 and math.isfinite(temp) and temp > 0 and math.isfinite(v) and v >= 0): raise ValueError('invalid initial state (soc in (0,1], temperature > 0 K, speed >= 0)')
        cam_seed = int(options.pop('camera_seed', self.np_random.integers(0, 2 ** 31)))
        if 'signal_offsets' in options: offsets = [float(o) for o in options.pop('signal_offsets')]
        elif 'signal_offset' in options: offsets = [float(options.pop('signal_offset'))] * len(self.route.SIGNALS)
        else: offsets = [float(self.np_random.uniform(0., self.route.CYCLE)) for _ in self.route.SIGNALS]
        if len(offsets) != len(self.route.SIGNALS) or not np.isfinite(offsets).all(): raise ValueError('signal offsets do not match the route (%d signals)' % len(self.route.SIGNALS))
        if options: raise ValueError('unknown reset options: %s' % sorted(options))
        cfg = parameters(J_MAX=self.jerk, CURVE_MODE=self._route_rules['curve_mode'])
        self.memory = VisionMemory(**self.rules['memory'], route_length=self.route.length)
        self.executor = Executor(self.jerk, self.dt)
        kw = dict(soc0=soc, T=temp, offsets=offsets, t_budget=float(self._route_rules['t_budget']), route=self.route, cfg=cfg)
        ex = self.rules['execution']
        self.vehicle = (PerceivedBackend(**kw, memory=self.memory, executor=self.executor, a_comfort=ex['a_comfort'], assumed_green_remaining=ex['assumed_green_remaining'])
                        if self.mode == 'camera' else Backend(**kw))
        self.vehicle.reset(); self.vehicle.v = v; self.vehicle._was_moving = v >= .1
        self.vehicle.t_end = self.deadline if self.deadline is not None else math.inf
        self.camera = None; self.rgb = None; self.history = []
        if self._needs_camera:
            self.camera = SceneCamera(self.route.CURVES, self.route.SIGNALS, self.route.length, seed=cam_seed, world='v4')
            self.camera.new_episode('episode')
        self._ended = False; self._initialized = True; self.decision = 0; self.total_reward = 0.; self.last_substeps = []
        if self.detector is not None and hasattr(self.detector, 'reset'): self.detector.reset()
        self._capture(0., 0.)
        return self._observation(), dict(profile=self.profile, profile_identity=self.profile_identity, detector_attached=self.detector is not None,
                                         detector_kind=self.detector_kind, route_length=self.route.length)

    # ------------------------------------------------------------------ perception
    @staticmethod
    def _validate_detection(out):
        if not (isinstance(out, (tuple, list)) and len(out) == 2): raise ValueError('detector must return (detections, color_probs)')
        dets, probs = out
        if dets is None: dets = {}
        if not isinstance(dets, dict): raise ValueError('detections must be a dict class name -> {score, dist_m}')
        for name, d in dets.items():
            if name not in DETECTION_CLASSES: raise ValueError('unknown detection class %r (expected one of %s)' % (name, DETECTION_CLASSES))
            if d is None: continue
            if not isinstance(d, dict) or 'score' not in d or 'dist_m' not in d: raise ValueError('detection %r must provide score and dist_m' % name)
            if not (math.isfinite(float(d['score'])) and math.isfinite(float(d['dist_m']))): raise ValueError('detection %r has non-finite score/dist_m' % name)
        if probs is not None:
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            if probs.shape != (5,) or not np.isfinite(probs).all() or (probs < 0).any(): raise ValueError('color_probs must be 5 finite nonnegative values (red/yellow/green/off/unknown) or None')
        return {k: v for k, v in dets.items() if v is not None}, probs

    def _capture(self, ds, dt):
        if self.camera is None: return
        b = self.vehicle
        frame = self.camera.capture(episode_id='episode', sim_time=b.t, pose={'x': b.x, 'v': b.v, 'offsets': b.offsets})
        self.rgb = frame['rgb']; self.history = (self.history + [self.rgb.copy()])[-HISTORY_FRAMES:]
        if self.mode != 'camera': return
        if self.detector is None: dets, probs = {}, None
        else:
            if getattr(self.detector, 'requires_truth', False):
                L = frame['labels']; truth = dict(L['truth'], color=L['color_truth'], visible=int(L['visible']), occluded=int(L['occluded']))
                out = self.detector(self.rgb.copy(), truth=truth)
            else: out = self.detector(self.rgb.copy())
            dets, probs = self._validate_detection(out)
        self.memory.update(dets, probs, b.v, dt, odom_ds=ds)

    def _observation(self):
        b = self.vehicle
        if self.mode == 'structured':
            o = b.obs()
            if self.deadline is None: o[11] = 0.
            return o
        remaining = 0. if self.deadline is None else self.deadline - b.t
        return {'rgb': self.rgb.copy(), 'ego': np.array([b.v, b.a, b.soc, b.T, b.t, remaining], np.float32)}

    def memory_observation(self, extended=False):
        """Legacy 13-dim observation generated from the memory plus its 6 (or 10) extra entries, for frozen-policy adapters (camera mode only)."""
        if self.mode != 'camera': raise RuntimeError('memory_observation is defined for the camera task only')
        if not self._initialized: raise RuntimeError('reset first')
        b = self.vehicle
        out = self.memory.legacy(b.v, b.a, b.soc, b.T, b.t if self.deadline is None else b.t_end, b.t, b.t_budget)
        return out, self.memory.extra(extended=extended)

    # ------------------------------------------------------------------ step
    def _parse_action(self, action):
        u = np.asarray(action, dtype=np.float64).reshape(-1)
        if u.size != 1 or not np.isfinite(u).all(): raise ValueError('expected one finite normalized command in [-1, 1], got %r' % (action,))
        raw = float(u[0]); clipped = raw < -1. or raw > 1.
        if clipped and not self.clip_actions: raise ValueError('command %r outside [-1, 1] (clip_actions=False)' % raw)
        return min(max(raw, -1.), 1.), clipped

    def step(self, action):
        if not self._initialized: raise RuntimeError('reset is required before stepping')
        if self._ended: raise RuntimeError('the episode has ended (terminated or truncated); call reset')
        u, clipped = self._parse_action(action); cmd = normalized_to_command(u)
        b = self.vehicle; self.last_substeps = []; reward = 0.; terminated = truncated = False; reason = None
        settle = self.rules['episode']['settle']; overshoot_m = self.rules['episode']['overshoot_m']
        for k in range(self.repeat):
            t0, x0, v0, a0, eb = b.t, b.x, b.v, b.a, b.e_batt
            _, r, _, flags = b.step(cmd)
            energy = -(b.e_batt - eb) / 1e5; time_cost = -b.cfg.LAM_T * b.cfg.DT
            jerk = (b.a - a0) / b.cfg.DT
            ex = getattr(b, 'last_execution', {})
            row = dict(decision=self.decision, substep=k, t0=t0, t1=b.t, command_u=u, command_a=cmd, applied_a=float(b.a),
                       v0=v0, v1=b.v, odom_ds=b.x - x0, jerk=jerk, jerk_exceeded=bool(abs(jerk) > self.jerk + 1e-7),
                       fallback=bool(ex.get('fallback', False)), reason=ex.get('reason', 'archived_envelope'),
                       envelope_braking=bool(ex.get('envelope_braking', False)), signal_reason=ex.get('signal_reason'),
                       interval=(ex.get('lower'), ex.get('upper')), v_limit=ex.get('v_limit'), target_distances=ex.get('target_distances'),
                       reward=float(r), reward_components={'energy': energy, 'time': time_cost, 'events_and_other': float(r) - energy - time_cost})
            self.last_substeps.append(row); reward += r
            self._capture(b.x - x0, b.cfg.DT)
            settled = bool(flags['arrived'] and b.v <= settle['v'] and abs(b.a) <= settle['a']) if self.mode == 'camera' else bool(flags['arrived'])
            deadline = bool(self.deadline is not None and b.t >= self.deadline)
            overshoot = bool(self.mode == 'camera' and b.x > self.route.length + overshoot_m)
            if settled: reason = 'settled'
            elif flags['red_crossing']: reason = 'red_crossing'
            elif deadline: reason = 'task_deadline'
            elif overshoot: reason = 'overshoot'
            terminated = reason is not None
            truncated = bool(not terminated and self.max_episode_seconds is not None and b.t >= self.max_episode_seconds)
            if terminated or truncated: break
        self.decision += 1; self.total_reward += reward; self._ended = terminated or truncated
        info = dict(profile=self.profile, profile_identity=self.profile_identity, command_u=u, command_a=cmd, command_clipped=bool(clipped),
                    applied_a=[q['applied_a'] for q in self.last_substeps], elapsed_dt=sum(q['t1'] - q['t0'] for q in self.last_substeps),
                    substeps=copy.deepcopy(self.last_substeps), settled=settled, termination_reason=reason,
                    signal_violation=bool(flags['red_crossing']), curve_violation=bool(flags['offroad']), task_deadline=deadline, external_cap=truncated,
                    energy_Wh=b.e_batt / 3600., time_s=b.t, return_sum=self.total_reward,
                    fallback_substeps=sum(q['fallback'] for q in self.last_substeps), detector_kind=self.detector_kind)
        if self.mode == 'camera':
            info['memory'] = self.memory.snapshot()
            if self.memory.update_diagnostic: info['memory_update'] = dict(self.memory.update_diagnostic)
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ render / snapshot
    def render(self):
        if self.render_mode is None:
            warnings.warn('render() called without render_mode="rgb_array"; returning None'); return None
        if not self._initialized: raise RuntimeError('reset first')
        return self.rgb.copy()

    def get_state(self):
        """Complete episode snapshot (vehicle, memory, camera and RNG state, frame history, command boundary). Take it between public steps."""
        if not self._initialized: raise RuntimeError('reset first')
        if self.detector is not None and not all(hasattr(self.detector, k) for k in ('get_state', 'set_state')):
            raise ValueError('an attached detector must provide get_state/set_state for snapshot support')
        vehicle = {k: v for k, v in self.vehicle.__dict__.items() if k not in ('memory', 'executor')}
        return copy.deepcopy({'schema': SNAPSHOT_SCHEMA, 'package_version': __version__, 'identity': self.identity, 'profile': self.profile,
                              'profile_identity': self.profile_identity, 'source_identity': self.source_identity,
                              'vehicle': vehicle, 'memory': self.memory.get_state(), 'camera': None if self.camera is None else self.camera.__dict__,
                              'rng': self.np_random.bit_generator.state, 'rgb': self.rgb, 'history': self.history, 'decision': self.decision,
                              'total_reward': self.total_reward, 'ended': self._ended, 'last_substeps': self.last_substeps,
                              'detector': None if self.detector is None else self.detector.get_state()})

    def set_state(self, state, strict_source=True):
        """Restore a snapshot taken by ``get_state`` from an environment with the same configuration identity.
        ``strict_source=False`` allows a snapshot taken with a different installed rule-module digest (logged, not recommended)."""
        if not isinstance(state, dict) or state.get('schema') != SNAPSHOT_SCHEMA: raise ValueError('unsupported snapshot schema (expected %d)' % SNAPSHOT_SCHEMA)
        if state.get('identity') != self.identity: raise ValueError('snapshot configuration identity mismatch (profile/mode/route/jerk/repeat/deadline differ)')
        if state.get('source_identity') != self.source_identity:
            if strict_source: raise ValueError('snapshot was taken with different rule-module sources; pass strict_source=False to force')
            warnings.warn('restoring a snapshot taken with different rule-module sources')
        if (state.get('detector') is None) != (self.detector is None): raise ValueError('snapshot detector attachment does not match this environment')
        if not self._initialized: self.reset(seed=0)
        s = copy.deepcopy(state); self.memory.set_state(s['memory'])
        if self.mode == 'camera': self.vehicle.__dict__ = dict(s['vehicle'], memory=self.memory, executor=self.executor)
        else: self.vehicle.__dict__ = s['vehicle']
        if s['camera'] is None: self.camera = None
        else:
            if self.camera is None: self.camera = SceneCamera(self.route.CURVES, self.route.SIGNALS, self.route.length, seed=0, world='v4')
            self.camera.__dict__ = s['camera']
        self.np_random.bit_generator.state = s['rng']
        self.rgb = s['rgb']; self.history = s['history']; self.decision = s['decision']; self.total_reward = s['total_reward']
        self._ended = s['ended']; self.last_substeps = s['last_substeps']
        if self.detector is not None: self.detector.set_state(s['detector'])

    def close(self): pass
