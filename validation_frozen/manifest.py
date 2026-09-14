"""SHA-256 manifest of every delivered file under validation_frozen (raw episode records included)."""
import hashlib, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
out = {}
for root, _, files in os.walk(HERE):
    for f in sorted(files):
        p = Path(root) / f; rel = str(p.relative_to(HERE))
        if '__pycache__' in rel or rel == 'MANIFEST_SHA256.json' or rel.endswith('.tmp') or rel.endswith('.resume.pkl'): continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
(HERE / 'MANIFEST_SHA256.json').write_text(json.dumps(dict(sorted(out.items())), indent=1) + '\n'); print(len(out), 'files')
