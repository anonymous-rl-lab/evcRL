"""Reference storage contract. Prototype uses individual lossless NPZ frame files.

Production collector must use measured-capacity shards/ref-count GC. Never cache
only Z when the encoder learns. This module does not implement a rollout collector.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np

class FrameStore:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.index={}
    def put(self,frame_id,rgb,capture_time,episode_id):
        if not frame_id.replace('_','').replace('-','').isalnum():raise ValueError('unsafe frame id')
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('expect HWC RGB uint8')
        rawhash=hashlib.sha256(rgb.tobytes()).hexdigest()
        record={'file':frame_id+'.npz','pixel_sha256':rawhash,'capture_time':float(capture_time),
                'episode_id':str(episode_id),'shape':list(rgb.shape)}
        if frame_id in self.index:
            if self.index[frame_id]!=record:raise ValueError('frame id reused for different content')
            return frame_id
        p=self.root/record['file'];tmp=p.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,rgb=rgb);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p);self.index[frame_id]=record;return frame_id
    def get(self,frame_id,episode_id,decision_time):
        r=self.index[frame_id]
        if r['episode_id']!=str(episode_id):raise ValueError('cross-episode frame')
        if r['capture_time']>decision_time:raise ValueError('future-frame leakage')
        with np.load(self.root/r['file']) as z:rgb=z['rgb'].copy()
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=r['pixel_sha256']:raise ValueError('frame corrupted')
        return rgb
    def state_dict(self):return {'schema':1,'index':self.index}
    def load_state_dict(self,state):
        if state['schema']!=1:raise ValueError('schema')
        self.index=state['index']
        for r in self.index.values():
            if not (self.root/r['file']).is_file():raise FileNotFoundError(r['file'])

class IndexedReplay:
    def __init__(self,capacity):self.capacity=capacity;self.records=[];self.ptr=0
    def add(self,record):
        required={'obs','next_obs','u_command','return_n','bootstrap_discount','actual_n','executed_trace_ref'}
        if not required<=record.keys():raise ValueError('missing transition fields')
        if 'z' in record or 'z' in record['obs'] or 'z' in record['next_obs']:
            raise ValueError('store frame references, not train-time cached Z')
        for obs in (record['obs'],record['next_obs']):
            if not {'frame_ids','episode_id','decision_time','legacy'}<=obs.keys():raise ValueError('missing image observation fields')
        if len(self.records)<self.capacity:self.records.append(record)
        else:self.records[self.ptr]=record
        self.ptr=(self.ptr+1)%self.capacity
    def sample(self,rng,count):return [self.records[i] for i in rng.integers(0,len(self.records),count)]
    def state_dict(self):return {'capacity':self.capacity,'records':self.records,'ptr':self.ptr}
    def load_state_dict(self,state):self.capacity=state['capacity'];self.records=state['records'];self.ptr=state['ptr']
