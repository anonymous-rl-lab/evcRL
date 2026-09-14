"""Explicit inference-only current/legacy evaluation entry point.

check: validates supplied source identity, exact protocol difference and syntax.
run: evaluates one selected seed/arm on all nine original development conditions.
Outputs must be in a new directory outside the extracted source archive.
No archive, checkpoint, learner or training configuration is modified.
"""
import argparse, ast, hashlib, inspect, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRANCH = '                if env.x >= 3999 and env.v <= .3: cmd = 0.\n'
REPLACEMENT = '                # Current protocol: retain the held actor command.\n'
RUNS = ['v4r', 'v4r_s1', 'v4r_s2']
ARMS = ['frozen', 'supervised', 'joint', 'joint_head']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checked_sources(root):
    meta = json.loads((HERE / 'protocol_identity.json').read_text())
    observed = {p: sha(root / p) for p in meta['source_sha256']}
    if observed != meta['source_sha256']:
        raise ValueError('Source identity differs; use the specified archive, not a silently changed implementation.')
    source = (root / 'visual_dev/pipeline.py').read_text()
    assert source.count(BRANCH) == 1, 'Terminal branch must occur exactly once.'
    patched = source.replace(BRANCH, REPLACEMENT)
    before, after = ast.parse(source), ast.parse(patched)
    fn = next(x for x in before.body if isinstance(x, ast.FunctionDef) and x.name == 'evaluate')
    targets = [x for x in ast.walk(fn) if isinstance(x, ast.If) and ast.get_source_segment(source, x).strip() == BRANCH.strip()]
    assert len(targets) == 1, 'Expected branch must belong to evaluate().'
    class RemoveOne(ast.NodeTransformer):
        def visit_If(self, node):
            if node is targets[0]: return None
            return self.generic_visit(node)
    expected = RemoveOne().visit(before)
    assert ast.dump(expected, include_attributes=False) == ast.dump(after, include_attributes=False)
    compile(patched, 'pipeline_current_protocol.py', 'exec')
    return dict(status='PASS', source_sha256=observed, transformed_pipeline_sha256=hashlib.sha256(patched.encode()).hexdigest(), changed_executable_nodes=1, vehicle_rollouts=0, training_steps=0)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['check', 'run'])
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--seed', type=int, choices=range(3))
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--protocol', choices=['current', 'legacy'])
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(); root = a.root.resolve()
    checks = checked_sources(root)
    if a.action == 'check':
        print(json.dumps(checks, indent=2)); return
    if a.seed is None or a.arm is None or a.protocol is None or a.out is None:
        ap.error('run requires --seed, --arm, --protocol and --out')
    expected = 'legacy' if a.arm == 'joint_head' else 'current'
    if a.protocol != expected:
        ap.error('For published records, F/S/J require current; JH requires legacy.')
    out = a.out.resolve()
    if out == root or root in out.parents:
        ap.error('--out must be outside the source archive')
    if out.exists():
        ap.error('--out must be a new directory')
    # Import inference components only after source checks succeed.
    sys.path.insert(0, str(root / 'visual_dev'))
    import torch
    import pipeline as P
    from reevaluate import learner_from_final_nets
    torch.set_num_threads(1)
    if a.protocol == 'current':
        source = inspect.getsource(P.evaluate)
        assert source.count(BRANCH) == 1
        exec(compile(source.replace(BRANCH, REPLACEMENT), str(root / 'visual_dev/pipeline.py'), 'exec'), P.__dict__)
    checkpoint = root / 'runs' / RUNS[a.seed] / 'v4_pilot' / a.arm / 'final_nets.pt'
    learner, ck = learner_from_final_nets(checkpoint); cfg = ck['cfg']
    assert cfg.get('world') == 'v4'
    thresholds = cfg.get('det_thr', .5)
    detector = root / cfg['pretrained_encoder']
    per = P.Perception(detector, 'roi_head', det_thr=thresholds.get('traffic_light', .5) if isinstance(thresholds, dict) else thresholds)
    mk = dict(det_thr=thresholds, hold_s=cfg.get('hold_s', 3.))
    # Keep all nine conditions and their indices; subsetting would alter camera streams.
    conditions = P.S.conditions('development')
    assert len(conditions) == 9
    rows, visual, traces = P.evaluate(learner, conditions, camera_seed=1000, world='v4', perception=per, memory_kw=mk)
    out.mkdir(parents=True, exist_ok=False)
    for row, (trace, log) in zip(rows, traces):
        name = f"dev_{row['condition_id']:02d}"
        P.S.save_trace(out / 'evaluation_traces' / (name + '.npz'), trace)
        (out / 'evaluation_traces' / (name + '_layer.json')).write_text(json.dumps(log, indent=1))
    result = dict(protocol=a.protocol, seed_index=a.seed, arm=a.arm, rows=rows,
                  summary=P.S.summarize(rows), visual=P.visual_summary(visual),
                  source_checks=checks, checkpoint_sha256=sha(checkpoint), detector_sha256=sha(detector),
                  training_steps=0, vehicle_rollouts=len(rows),
                  note='New inference outputs; archived publication records are unchanged. Cross-runtime numerical differences may occur.')
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(output=str(out), protocol=a.protocol, seed=a.seed, arm=a.arm, episodes=len(rows))))

if __name__ == '__main__': main()
