"""可运行的轻量 FPN 原型（交接框架原件 + v2 修正），不是已评估/预训练的视觉模型。

相对交接框架原件的修改（对应独立审计 v1 第 4、6 节）：
1. ROI 为空（地图关联不存在或投影出画面）的帧，灯色 ROI 头的输出被强制为 unknown（常量 logits，无梯度），
   不再输出仅由偏置决定的颜色。
2. 新增 signal_z 头：由 Z 直接预测最新帧的控制灯色。视觉监督损失因此训练到 Z 的末端投影（temporal），
   使“灯色头正确”与“Z 保留灯色”不再脱钩；输出中同时给出两路预测，评估分别报告。
3. 新增 decode_boxes()：与渲染器 v2 的框参数化（格内偏移 + 归一化尺寸）配套解码。
4. 热图焦点损失改为按正样本格点数归一化，且热图头偏置按先验 π=0.01 初始化。受控实验（同种子 500 步）：原按全部格点归一化命中率 0.017；仅改归一化不加先验初始损失 2086、特征塌缩、命中率 0.008；先验偏置+正样本归一化命中率 0.788、中心误差 2.29 px（visual_dev/heat_loss_probe.py，输出 runs/probes/）。
5. event 注意力权重 att 对热图 logits 取 detach：热图头只由检测损失训练；Z 头 CE 与联合臂的 TD 梯度仍经 p2/p4 特征进入骨干与 temporal（Z 路径），
   但不再改写热图头。标签格分配规则固定为 floor(c/4)（visual_dev/cell_assign_probe.py 受控对比，见 renderer.CELL_ASSIGN 注释）。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import *

UNKNOWN_LOGITS = torch.tensor([-10., -10., -10., -10., 10.])


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
        self.signal=nn.Linear(32,5)  # red/yellow/green/off/unknown, controlling light only（ROI 头）
        self.temporal=nn.Sequential(nn.Linear(stack*(64+2),128),nn.SiLU(),nn.Linear(128,zdim),nn.LayerNorm(zdim))
        self.signal_z=nn.Linear(zdim,5)   # v2：由 Z 预测最新帧灯色，使视觉监督训练到 Z 末端
        nn.init.constant_(self.heat.bias,-math.log((1-.01)/.01))   # v2b：热图头偏置先验 π=0.01（RetinaNet/CenterNet 口径），配合按正样本归一化的焦点损失

    def forward(self,obs):
        obs.validate(self.stack);b,t,c,h,w=obs.frames.shape
        x=obs.frames.float().reshape(b*t,c,h,w)/255.
        c2=self.c2(self.stem(x));c3=self.c3(c2);c4=self.c4(c3)
        p4=self.lat4(c4);p3=self.lat3(c3)+F.interpolate(p4,size=c3.shape[-2:],mode='nearest')
        p2=self.lat2(c2)+F.interpolate(p3,size=c2.shape[-2:],mode='nearest')
        logits=self.heat(p2);boxes=self.box(p2).sigmoid()
        att=logits.detach().sigmoid().sum(1,keepdim=True)+1e-6   # v2d：注意力权重与热图头切断梯度——热图只由检测监督塑形，Z 侧损失/TD 梯度经 p2 特征进入骨干，不再改写热图头（对抗式审查发现该耦合初始占检测梯度约 25%）
        event=(p2*att).sum((2,3))/att.sum((2,3))
        scene=p4.mean((2,3));f=torch.cat([scene,event],1).reshape(b,t,64)
        f=f*obs.valid.unsqueeze(-1)
        temporal=torch.cat([f,obs.valid.float().unsqueeze(-1),obs.age_s.clamp(0,10).unsqueeze(-1)],-1)
        z=self.temporal(temporal.flatten(1))
        roi=F.interpolate(obs.signal_roi.reshape(b*t,1,h,w).float(),size=p2.shape[-2:],mode='area')
        roi_mass=roi.sum((2,3))                                   # [b*t,1]
        pooled=(p2*roi).sum((2,3))/roi_mass.clamp_min(1e-6)
        sl=self.signal(pooled)
        roi_empty=(obs.signal_roi.reshape(b*t,-1).sum(1)<=0)     # v2：空 ROI 强制 unknown
        sl=torch.where(roi_empty.unsqueeze(1),UNKNOWN_LOGITS.to(sl).expand_as(sl),sl).reshape(b,t,5)
        hh,ww=p2.shape[-2:]
        return {'z':z,'heat':logits.reshape(b,t,-1,hh,ww),
                'boxes':boxes.reshape(b,t,4,hh,ww),'signal':sl,'signal_z':self.signal_z(z),
                'roi_empty':roi_empty.reshape(b,t)}

class InputAdapter(nn.Module):
    def __init__(self,information_mode='camera_map'):
        super().__init__()
        if information_mode not in ('camera_map','camera_map_v2x'):raise ValueError(information_mode)
        self.information_mode=information_mode
    def forward(self,obs,z):
        o=obs.legacy.clone()
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
    current=net.state_dict()
    for key,value in old_state.items():
        if key=='f.0.weight':
            current[key].zero_();current[key][:,:13].copy_(value[:,:13])
            if critic:current[key][:,-1].copy_(value[:,13])
        else:
            if current[key].shape!=value.shape:raise ValueError((key,current[key].shape,value.shape))
            current[key].copy_(value)
    net.load_state_dict(current)

def decode_boxes(heat,boxes,cls=0,cell=4):
    """对每帧取类别 cls 热图的最大格，解码为像素框 (l,t,r,b)；返回 [b,t,4] 与该格得分 [b,t]。"""
    b,t,_,hh,ww=heat.shape
    score,idx=heat[:,:,cls].reshape(b,t,-1).max(-1); i=idx//ww; j=idx%ww
    tgt=torch.stack([boxes[n,k,:,i[n,k],j[n,k]] for n in range(b) for k in range(t)]).reshape(b,t,4)
    import os; off=.5 if os.environ.get('RENDER_CELL_ASSIGN','floor')=='nearest' else 0.
    cx=(j.float()+tgt[...,0]-off)*cell; cy=(i.float()+tgt[...,1]-off)*cell; w=tgt[...,2]*ww*cell; h=tgt[...,3]*hh*cell   # 与 renderer.CELL_ASSIGN 一致
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1),score.sigmoid()

def perception_loss(output,labels,obs):
    zero=output['z'].sum()*0
    if not labels:return zero
    losses=[]
    if 'heat' in labels:
        y=labels['heat'];logit=output['heat'];p=logit.sigmoid()
        mask=labels['heat_valid']*obs.valid[:,:,None,None,None]
        pt=p*y+(1-p)*(1-y);alpha=.25*y+.75*(1-y)
        focal=alpha*(1-pt).square()*F.binary_cross_entropy_with_logits(logit,y,reduction='none')
        losses.append((focal*mask).sum()/(y*mask).sum().clamp_min(1))   # v2b：按正样本格点数归一化（CenterNet 口径）；原按全部格点数归一化会淹没 1/960 的正样本梯度
    if 'boxes' in labels:
        mask=labels['box_valid']*obs.valid[:,:,None,None,None]
        err=F.smooth_l1_loss(output['boxes'],labels['boxes'],reduction='none')
        losses.append((err*mask).sum()/mask.expand_as(err).sum().clamp_min(1))
    if 'signal' in labels:
        y=labels['signal'].clone();y[~obs.valid]=-100;y[output['roi_empty']]=-100   # 空 ROI 帧由规则输出 unknown，不参与 ROI 头训练
        losses.append(F.cross_entropy(output['signal'].reshape(-1,5),y.reshape(-1),ignore_index=-100,
            reduction='sum')/(y!=-100).sum().clamp_min(1))
        yz=labels['signal'][:,-1].clone();yz[~obs.valid[:,-1]]=-100                # v2：Z 头监督最新帧灯色（含 unknown）
        losses.append(F.cross_entropy(output['signal_z'],yz,ignore_index=-100,reduction='sum')/(yz!=-100).sum().clamp_min(1))
    return sum(losses,zero)
