import copy,hashlib,json,os,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from visual_z.contracts import *
from visual_z.model import *
from visual_z.learner import *
from visual_z.replay import *
from visual_z.checkpoint import *
torch.set_num_threads(1)

def observation():
    b,t,h,w=2,4,32,48
    return VisualObservation(torch.randn(b,13),torch.randint(0,256,(b,t,3,h,w),dtype=torch.uint8),
        torch.ones(b,t,dtype=torch.bool),torch.tensor([[.3,.2,.1,0.]]).repeat(b,1),
        torch.ones(b,t,1,h,w),torch.ones(b,1),torch.zeros(b,1))
def batch():
    o=observation();n=observation();b,t,_,h,w=o.frames.shape
    lab={'heat':torch.zeros(b,t,8,h//4,w//4),'heat_valid':torch.ones(b,t,1,h//4,w//4),
         'signal':torch.zeros(b,t,dtype=torch.long)}
    lab['heat'][:,:,0,1,1]=1
    return TransitionBatch(o,n,torch.tensor([[.2],[-.2]]),torch.tensor([[-1.],[-.5]]),torch.tensor([[1.],[0.]]),lab)
def difference(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)

class Contracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(3);np.random.seed(3);random.seed(3)
    def test_original_weight_migration(self):
        x=torch.randn(9,13);extra=torch.randn(9,68);a=torch.randn(9,1)
        for critic in (False,True):
            old=MLP(14 if critic else 13,not critic);new=MLP(82 if critic else 81,not critic)
            load_legacy_weights(new,old.state_dict(),critic)
            before=old(torch.cat([x,a],1) if critic else x)
            after=new(torch.cat([x,extra,a],1) if critic else torch.cat([x,extra],1))
            self.assertTrue(torch.allclose(before,after,atol=1e-7,rtol=0))
    def test_signal_oracle_is_masked_and_acceleration_preserved(self):
        o=observation();p=copy.deepcopy(o);p.legacy[:,7:9]+=200
        z=torch.randn(2,64);ad=InputAdapter()
        self.assertTrue(torch.equal(ad(o,z),ad(p,z)))
        self.assertTrue(torch.equal(ad(o,z)[:,1],o.legacy[:,1]))
        self.assertEqual(ad(o,z).shape,(2,81))
    def test_critic_gradient_routing_and_target_stop(self):
        b=batch()
        for mode in ['frozen','supervised','joint']:
            m=VisualTD3(Config(mode=mode));q,v=m.losses(b);q.backward();g=grad_norm(m.encoder)
            self.assertGreater(g,0) if mode=='joint' else self.assertEqual(g,0)
            self.assertTrue(all(p.grad is None for t in m.targets() for p in t.parameters()))
    def test_actor_updates_without_moving_encoder(self):
        m=VisualTD3();o=observation();before=copy.deepcopy(m.encoder.state_dict());m.actor_step(o)
        self.assertEqual(difference(before,m.encoder.state_dict()),0)
        self.assertTrue(all(p.grad is None for p in m.encoder.parameters()))
    def test_auxiliary_batch_is_independent_of_online_rgb(self):
        b=batch();b.visual_obs=observation();other=copy.deepcopy(b)
        other.obs.frames=255-other.obs.frames
        m=VisualTD3(Config(mode='supervised'))
        _,first=m.losses(b);_,second=m.losses(other)
        self.assertEqual(float(first.detach()),float(second.detach()))
        self.assertEqual(b.to('cpu').visual_obs.frames.device.type,'cpu')
    def test_visual_label_has_live_gradient_and_frozen_arm_stays_frozen(self):
        b=batch();m=VisualTD3(Config(mode='supervised'));q,v=m.losses(b);v.backward()
        self.assertGreater(grad_norm(m.encoder),0)
        f=VisualTD3(Config(mode='frozen'));before=copy.deepcopy(f.encoder.state_dict());f.update(b)
        self.assertEqual(difference(before,f.encoder.state_dict()),0)
    def test_full_checkpoint_exact_next_update_and_rng(self):
        m=VisualTD3();b=batch();m.update(b)
        ext={k:{'test_state':5} for k in REQUIRED_EXTERNAL};identity={'source':'unit_fixture','data':'fixture','protocol':'fixture'}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'resume.pt';save_checkpoint(p,m,ext,identity)
            expected=m.update(b);expected_net=copy.deepcopy(m.state_dict()['nets']);rand=(random.random(),np.random.rand(),torch.rand(2))
            restored=VisualTD3();external=load_checkpoint(p,restored,identity);actual=restored.update(b)
            self.assertEqual(expected,actual);self.assertEqual(external,ext)
            for name in expected_net:self.assertEqual(difference(expected_net[name],restored.state_dict()['nets'][name]),0)
            self.assertEqual(rand[0],random.random());self.assertEqual(rand[1],np.random.rand());self.assertTrue(torch.equal(rand[2],torch.rand(2)))
            with self.assertRaises(ValueError):load_checkpoint(p,restored,{'source':'different'})
    def test_frame_storage_prevents_future_cross_episode_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            s=FrameStore(d);rgb=np.arange(8*12*3,dtype=np.uint8).reshape(8,12,3)
            s.put('frame_1',rgb,1.0,'episode_a');self.assertTrue(np.array_equal(s.get('frame_1','episode_a',1.1),rgb))
            with self.assertRaises(ValueError):s.get('frame_1','episode_a',.9)
            with self.assertRaises(ValueError):s.get('frame_1','episode_b',1.1)
            with self.assertRaises(ValueError):s.put('frame_1',255-rgb,1.,'episode_a')
            np.savez_compressed(Path(d)/'frame_1.npz',rgb=255-rgb)
            with self.assertRaises(ValueError):s.get('frame_1','episode_a',1.1)
    def test_future_time_rejected_by_model(self):
        o=observation();o.age_s[0,-1]=-.1
        with self.assertRaises(AssertionError):VisualEncoder()(o)
    def test_replay_command_is_separate_from_execution_and_wraps(self):
        r=IndexedReplay(2);ob={'frame_ids':['f1'],'episode_id':'ep','decision_time':1.,'legacy':[0.]*13}
        rec={'obs':ob,'next_obs':ob,'u_command':.4,'return_n':-2.,'bootstrap_discount':0.,'actual_n':3,'executed_trace_ref':'trace_1'}
        for i in range(3):r.add({**rec,'u_command':i/10})
        self.assertEqual(len(r.records),2);self.assertEqual(r.records[0]['u_command'],.2)
        with self.assertRaises(ValueError):r.add({**rec,'z':[0.]*64})
    @unittest.skipUnless(os.environ.get('TIV_BASE'),'set TIV_BASE to audit supplied v19')
    def test_actual_v19_hashes_and_actor(self):
        base=Path(os.environ['TIV_BASE']);root=Path(__file__).resolve().parents[1]
        for name,h in json.loads((root/'verification/v19_sources.json').read_text()).items():
            self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(),h)
        state=torch.load(base/'TIV_SPaT_Execution_v1/weights/actor_A.pt',weights_only=True,map_location='cpu')
        state=state['actor']
        old=MLP(13,True);old.load_state_dict(state);new=MLP(81,True);load_legacy_weights(new,state)
        x=torch.randn(10,13);self.assertTrue(torch.allclose(old(x),new(torch.cat([x,torch.randn(10,68)],1)),atol=1e-7,rtol=0))
if __name__=='__main__':unittest.main()
