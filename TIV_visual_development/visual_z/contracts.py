from dataclasses import dataclass, fields
from typing import Protocol, Any
import torch

LEGACY_DIM = 13
Z_DIM = 64
META_DIM = 4
META_EXTRA_DIM = 6   # v4 vision_memory 模式的附加元信息维数
POLICY_DIM = LEGACY_DIM + Z_DIM + META_DIM


def policy_dim(information_mode='camera_map', extra_dim=None):
    """extra_dim：vision_memory 模式附加元信息维数（默认 META_EXTRA_DIM=6；审计整改后的扩展接口为 10，含锁存/承诺/观测灯色/观测置信）。"""
    return POLICY_DIM + ((META_EXTRA_DIM if extra_dim is None else int(extra_dim)) if information_mode == 'vision_memory' else 0)
# Preserve measured acceleration at index 1; original comfort_loss depends on it.
CHANNELS = ('v','a','distance_end','local_limit','distance_curve','curve_limit',
            'map_distance_signal','signal_green','countdown','soc','temperature',
            'remaining_budget','absolute_time')

@dataclass
class VisualObservation:
    legacy: torch.Tensor       # [B,13], ego/map; signal channels sanitized below
    frames: torch.Tensor       # uint8 [B,T,3,H,W], oldest to newest, RGB
    valid: torch.Tensor        # bool [B,T]; padding is invalid, never next episode
    age_s: torch.Tensor        # [B,T]; decision_time - capture_time, nonnegative
    signal_roi: torch.Tensor   # [B,T,1,H,W], projected map region, NOT truth color/box
    association_valid: torch.Tensor # [B,1], controlling-light association known
    v2x_valid: torch.Tensor    # [B,1], explicit protocol flag
    extra: torch.Tensor | None = None   # v4：视觉记忆的附加元信息 [B,META_EXTRA_DIM]（灯置信/年龄/相位持续、弯道已宣告/激活、终点已见）

    def to(self,device):
        return VisualObservation(**{f.name:(None if getattr(self,f.name) is None else getattr(self,f.name).to(device)) for f in fields(self)})

    def validate(self, stack=4):
        b,t,c,h,w=self.frames.shape
        assert self.frames.dtype==torch.uint8 and t==stack and c==3
        assert self.legacy.shape==(b,13) and self.valid.shape==(b,t)
        assert self.age_s.shape==(b,t) and self.signal_roi.shape==(b,t,1,h,w)
        assert self.association_valid.shape==self.v2x_valid.shape==(b,1)
        assert torch.isfinite(self.legacy).all() and torch.isfinite(self.age_s).all()
        assert (self.age_s>=0).all(), 'future frame leakage'
        assert not (self.valid[:,1:] & self.valid[:,:-1] & (self.age_s[:,1:]>self.age_s[:,:-1])).any()

@dataclass
class TransitionBatch:
    obs: VisualObservation
    nxt: VisualObservation
    u_command: torch.Tensor     # [B,1], BEFORE execution projection, normalized [-1,1]
    return_n: torch.Tensor      # actual accumulated environment return, no aux loss
    bootstrap_discount: torch.Tensor # (1-true_terminal)*gamma**actual_n
    labels: dict[str,torch.Tensor]   # labels for visual_obs (or obs when absent); never actor inputs
    visual_obs: VisualObservation | None = None # common offline supervised batch for S/J
    u_applied: torch.Tensor | None = None   # v3：执行层投影后的实际动作（归一化），critic_action='applied' 时作 critic 的动作输入
    u_lo: torch.Tensor | None = None        # v3：决策时刻执行层可行区间下界（归一化），actor 直通裁剪用
    u_hi: torch.Tensor | None = None

    def to(self,device):
        f=lambda x:None if x is None else x.to(device)
        return TransitionBatch(self.obs.to(device),self.nxt.to(device),self.u_command.to(device),
            self.return_n.to(device),self.bootstrap_discount.to(device),
            {k:v.to(device) for k,v in self.labels.items()},
            self.visual_obs.to(device) if self.visual_obs is not None else None,f(self.u_applied),f(self.u_lo),f(self.u_hi))

class CameraProvider(Protocol):
    """Policy-dependent synchronized rendering/capture; fixed driving videos are insufficient."""
    def capture(self, *, episode_id: str, sim_time: float, pose: Any) -> dict: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...

class SceneCameraNotConnected:
    def capture(self, **kwargs):
        raise NotImplementedError('Connect synchronized camera renderer; v19 has no RGB sensor.')
