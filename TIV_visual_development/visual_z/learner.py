"""TD3 gradient routing over image batches. Collector and RGB data are external."""
from dataclasses import dataclass,asdict
import copy
import torch
from .model import *

@dataclass
class Config:
    mode: str='joint'  # frozen / supervised / joint
    information_mode: str='camera_map'
    stack: int=4
    actor_lr: float=3e-5
    critic_lr: float=3e-4
    encoder_lr: float=1e-5
    vision_weight: float=1.
    lambda_c: float=0.  # principal original A; use same coefficient in every comparison
    tau: float=.005
    policy_delay: int=2
    target_noise: float=.2
    target_clip: float=.5

def grad_norm(module):
    return sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5

class VisualTD3:
    def __init__(self,cfg=None,device='cpu'):
        self.cfg=cfg or Config();self.device=torch.device(device);self.updates=0
        if self.cfg.mode not in ('frozen','supervised','joint'):raise ValueError(self.cfg.mode)
        self.encoder=VisualEncoder(self.cfg.stack).to(device)
        self.adapter=InputAdapter(self.cfg.information_mode).to(device)
        self.actor=MLP(POLICY_DIM,True).to(device)
        self.q1=MLP(POLICY_DIM+1).to(device);self.q2=MLP(POLICY_DIM+1).to(device)
        self.actor_t=copy.deepcopy(self.actor);self.q1t=copy.deepcopy(self.q1);self.q2t=copy.deepcopy(self.q2)
        self.encoder_t=copy.deepcopy(self.encoder)
        for m in self.targets():m.requires_grad_(False)
        if self.cfg.mode=='frozen':self.encoder.requires_grad_(False)
        self.oa=torch.optim.Adam(self.actor.parameters(),lr=self.cfg.actor_lr)
        self.oq=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=self.cfg.critic_lr)
        self.oe=None if self.cfg.mode=='frozen' else torch.optim.Adam(self.encoder.parameters(),lr=self.cfg.encoder_lr)
    def targets(self):return (self.actor_t,self.q1t,self.q2t,self.encoder_t)
    def named_nets(self):
        return {k:getattr(self,k) for k in ('actor','q1','q2','encoder','actor_t','q1t','q2t','encoder_t')}
    @torch.no_grad()
    def synchronize_targets(self):
        for src,dst in ((self.actor,self.actor_t),(self.q1,self.q1t),(self.q2,self.q2t),(self.encoder,self.encoder_t)):
            dst.load_state_dict(src.state_dict())
    def losses(self,batch):
        if self.cfg.mode=='supervised' and not batch.labels:raise ValueError('S arm requires visual labels')
        out=self.encoder(batch.obs)
        state=self.adapter(batch.obs,out['z'])
        qstate=state if self.cfg.mode=='joint' else state.detach()
        with torch.no_grad():
            nxt=self.adapter(batch.nxt,self.encoder_t(batch.nxt)['z'])
            noise=(torch.randn_like(batch.u_command)*self.cfg.target_noise).clamp(-self.cfg.target_clip,self.cfg.target_clip)
            u2=(self.actor_t(nxt)+noise).clamp(-1,1)
            target=batch.return_n+batch.bootstrap_discount*torch.minimum(
                self.q1t(torch.cat([nxt,u2],1)),self.q2t(torch.cat([nxt,u2],1)))
        x=torch.cat([qstate,batch.u_command],1)
        qloss=(self.q1(x)-target).square().mean()+(self.q2(x)-target).square().mean()
        aux_obs=batch.visual_obs if batch.visual_obs is not None else batch.obs
        aux_out=self.encoder(aux_obs) if batch.visual_obs is not None else out
        vis=perception_loss(aux_out,batch.labels,aux_obs)
        return qloss,vis
    def actor_step(self,obs):
        # Critic parameters frozen, but dQ/du remains live. Actor never moves the encoder.
        with torch.no_grad():state=self.adapter(obs,self.encoder(obs)['z'])
        self.q1.requires_grad_(False)
        try:
            u=self.actor(state);task=-self.q1(torch.cat([state,u],1)).mean()
            a=torch.where(u>0,2.6*u,3.5*u)
            smooth=(a-3.5*obs.legacy[:,1:2]).square().mean()
            loss=task+self.cfg.lambda_c*smooth
            self.oa.zero_grad(set_to_none=True);loss.backward();self.oa.step()
        finally:self.q1.requires_grad_(True)
        return float(loss.detach())
    def update(self,batch):
        # Production integrator: microbatch/accumulate when required; retain effective batch.
        self.oq.zero_grad(set_to_none=True)
        if self.oe:self.oe.zero_grad(set_to_none=True)
        qloss,vis=self.losses(batch)
        loss=qloss+(self.cfg.vision_weight*vis if self.oe else 0)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite learner loss')
        loss.backward();gn=grad_norm(self.encoder)
        self.oq.step()
        if self.oe:self.oe.step()
        self.updates+=1;actor_loss=None
        if self.updates%self.cfg.policy_delay==0:
            actor_loss=self.actor_step(batch.obs)
            with torch.no_grad():
                for src,dst in ((self.actor,self.actor_t),(self.q1,self.q1t),(self.q2,self.q2t),(self.encoder,self.encoder_t)):
                    for p,t in zip(src.parameters(),dst.parameters()):t.lerp_(p,self.cfg.tau)
        return {'q_loss':float(qloss.detach()),'vision_loss':float(vis.detach()),
                'encoder_grad':gn,'actor_loss':actor_loss,'updates':self.updates}
    def state_dict(self):
        return {'cfg':asdict(self.cfg),'updates':self.updates,
          'nets':{k:m.state_dict() for k,m in self.named_nets().items()},
          'optimizers':{'actor':self.oa.state_dict(),'critic':self.oq.state_dict(),
                        'encoder':self.oe.state_dict() if self.oe else None}}
    def load_state_dict(self,state):
        if state['cfg']!=asdict(self.cfg):raise ValueError('configuration mismatch')
        for k,v in state['nets'].items():self.named_nets()[k].load_state_dict(v)
        self.oa.load_state_dict(state['optimizers']['actor']);self.oq.load_state_dict(state['optimizers']['critic'])
        if self.oe:self.oe.load_state_dict(state['optimizers']['encoder'])
        self.updates=state['updates']
