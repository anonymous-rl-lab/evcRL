"""Runnable lightweight FPN prototype, not an evaluated/pretrained vision model."""
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import *

def block(cin,cout,stride):
    return nn.Sequential(nn.Conv2d(cin,cin,3,stride,1,groups=cin,bias=False),
        nn.GroupNorm(1,cin),nn.SiLU(),nn.Conv2d(cin,cout,1,bias=False),
        nn.GroupNorm(4,cout),nn.SiLU())

class VisualEncoder(nn.Module):
    def __init__(self,stack=4,zdim=64,classes=8):
        super().__init__();self.stack=stack
        self.stem=nn.Sequential(nn.Conv2d(3,16,3,2,1,bias=False),nn.GroupNorm(4,16),nn.SiLU())
        self.c2=block(16,32,2);self.c3=block(32,48,2);self.c4=block(48,64,2)
        self.lat2=nn.Conv2d(32,32,1);self.lat3=nn.Conv2d(48,32,1);self.lat4=nn.Conv2d(64,32,1)
        self.heat=nn.Conv2d(32,classes,1);self.box=nn.Conv2d(32,4,1)
        self.signal=nn.Linear(32,5)  # red/yellow/green/off/unknown, controlling light only
        self.temporal=nn.Sequential(nn.Linear(stack*(64+2),128),nn.SiLU(),nn.Linear(128,zdim),nn.LayerNorm(zdim))

    def forward(self,obs):
        obs.validate(self.stack);b,t,c,h,w=obs.frames.shape
        x=obs.frames.float().reshape(b*t,c,h,w)/255.
        c2=self.c2(self.stem(x));c3=self.c3(c2);c4=self.c4(c3)
        p4=self.lat4(c4);p3=self.lat3(c3)+F.interpolate(p4,size=c3.shape[-2:],mode='nearest')
        p2=self.lat2(c2)+F.interpolate(p3,size=c2.shape[-2:],mode='nearest')
        logits=self.heat(p2);boxes=self.box(p2).sigmoid()
        # All high-resolution cells participate: no top-k/NMS/argmax on the RL gradient path.
        att=logits.sigmoid().sum(1,keepdim=True)+1e-6
        event=(p2*att).sum((2,3))/att.sum((2,3))
        scene=p4.mean((2,3));f=torch.cat([scene,event],1).reshape(b,t,64)
        f=f*obs.valid.unsqueeze(-1)
        temporal=torch.cat([f,obs.valid.float().unsqueeze(-1),obs.age_s.clamp(0,10).unsqueeze(-1)],-1)
        z=self.temporal(temporal.flatten(1))
        # Map geometry may specify an ROI; never use a ground-truth target box at inference.
        roi=F.interpolate(obs.signal_roi.reshape(b*t,1,h,w).float(),size=p2.shape[-2:],mode='area')
        pooled=(p2*roi).sum((2,3))/roi.sum((2,3)).clamp_min(1e-6)
        sl=self.signal(pooled).reshape(b,t,5)
        hh,ww=p2.shape[-2:]
        return {'z':z,'heat':logits.reshape(b,t,-1,hh,ww),
                'boxes':boxes.reshape(b,t,4,hh,ww),'signal':sl}

class InputAdapter(nn.Module):
    def __init__(self,information_mode='camera_map'):
        super().__init__()
        if information_mode not in ('camera_map','camera_map_v2x'):raise ValueError(information_mode)
        self.information_mode=information_mode
    def forward(self,obs,z):
        o=obs.legacy.clone()
        # Signal phase MUST come from images through Z, not the old oracle channel.
        o[:,7]=-1.
        if self.information_mode=='camera_map':
            o[:,8]=1.;v2x=torch.zeros_like(obs.v2x_valid)
        else:
            v2x=obs.v2x_valid.float();o[:,8]=torch.where(v2x[:,0]>0,o[:,8],torch.ones_like(o[:,8]))
        meta=torch.cat([obs.valid[:,-1:].float(),obs.age_s[:,-1:].clamp(0,10),
                        obs.association_valid.float(),v2x],1)
        return torch.cat([o,z,meta],1)

class MLP(nn.Module):
    def __init__(self,inputs,tanh=False):
        super().__init__();layers=[nn.Linear(inputs,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1)]
        if tanh:layers.append(nn.Tanh())
        self.f=nn.Sequential(*layers)
    def forward(self,x):return self.f(x)

@torch.no_grad()
def load_legacy_weights(net,old_state,critic=False):
    """Preserve old first 13 columns and critic action column; zero new Z/meta columns.

    Algebraic equality holds when the first 13 inputs match. Online camera masking
    changes those inputs, so this is NOT a claim of unchanged driving behavior.
    """
    current=net.state_dict()
    for key,value in old_state.items():
        if key=='f.0.weight':
            current[key].zero_();current[key][:,:13].copy_(value[:,:13])
            if critic:current[key][:,-1].copy_(value[:,13])
        else:
            if current[key].shape!=value.shape:raise ValueError((key,current[key].shape,value.shape))
            current[key].copy_(value)
    net.load_state_dict(current)

def perception_loss(output,labels,obs):
    """Minimal focal/box/state losses. A real labeled-data loader must encode targets.

    Dense labels are center heatmaps; box target = normalized l,t,r,b at positives.
    Label masks must include visibility, annotation completeness and frame validity.
    """
    zero=output['z'].sum()*0
    if not labels:return zero
    losses=[]
    if 'heat' in labels:
        y=labels['heat'];logit=output['heat'];p=logit.sigmoid()
        mask=labels['heat_valid']*obs.valid[:,:,None,None,None]
        pt=p*y+(1-p)*(1-y);alpha=.25*y+.75*(1-y)
        focal=alpha*(1-pt).square()*F.binary_cross_entropy_with_logits(logit,y,reduction='none')
        losses.append((focal*mask).sum()/mask.expand_as(focal).sum().clamp_min(1))
    if 'boxes' in labels:
        mask=labels['box_valid']*obs.valid[:,:,None,None,None]
        err=F.smooth_l1_loss(output['boxes'],labels['boxes'],reduction='none')
        losses.append((err*mask).sum()/mask.expand_as(err).sum().clamp_min(1))
    if 'signal' in labels:
        y=labels['signal'].clone();y[~obs.valid]=-100
        # Labels must already be -100 when controlling-light association is unavailable.
        losses.append(F.cross_entropy(output['signal'].reshape(-1,5),y.reshape(-1),ignore_index=-100,
            reduction='sum')/(y!=-100).sum().clamp_min(1))
    return sum(losses,zero)
