"""Verify the delivered files against their SHA256 manifest."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    manifest=json.loads((P/'MANIFEST_SHA256.json').read_text())
    for name,digest in manifest.items():
        assert (P/name).is_file(),('missing',name)
        assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,('changed',name)
    print(f'PASS: {len(manifest)} delivered files match the manifest.')
