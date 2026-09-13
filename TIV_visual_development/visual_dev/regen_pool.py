"""按种子确定性再生成预训练池（pool.pt / eval_sets.pt / audit_set.npz），供其他容器复用已推送的编码器而不重训。
用法：python visual_dev/regen_pool.py --out pretrain_v4 --coverage v4 --n-train 3200 --n-dev 320 [--seed 4242]"""
import argparse, sys, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pretrain as PT, pipeline as P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='pretrain_v4'); ap.add_argument('--coverage', default='v4'); ap.add_argument('--n-train', type=int, default=3200); ap.add_argument('--n-dev', type=int, default=320); ap.add_argument('--n-audit', type=int, default=32); ap.add_argument('--seed', type=int, default=4242)
    a = ap.parse_args(); out = ROOT / 'runs' / a.out; out.mkdir(parents=True, exist_ok=True)
    train = PT.make_sequences(a.n_train, a.seed, 'tr', a.coverage); dev = PT.make_sequences(a.n_dev, a.seed + 100, 'dv', a.coverage); audit = PT.make_sequences(a.n_audit, a.seed + 200, 'au', a.coverage)
    for seqs in (train, dev, audit): PT.compact(seqs)
    torch.save(dict(sequences=train, seed=a.seed, hw=(P.H, P.W)), out / 'pool.pt'); torch.save(dict(dev=dev, audit=audit, seed=a.seed), out / 'eval_sets.pt')
    np.savez_compressed(out / 'audit_set.npz', rgb=np.stack([[f['rgb'] for f in s] for s in audit]), roi=np.stack([[f['roi'] for f in s] for s in audit]),
                        signal=np.array([[f['labels']['signal'] for f in s] for s in audit]), map_association=np.array([[f['meta']['map_association'] for f in s] for s in audit]))
    h = hashlib.sha256(); [h.update(f['rgb'].tobytes()) for s in train[:50] for f in s]
    print(f'池已生成：train {len(train)} dev {len(dev)} audit {len(audit)}；前 50 序列 RGB 内容哈希 {h.hexdigest()[:16]}')


if __name__ == '__main__':
    main()
