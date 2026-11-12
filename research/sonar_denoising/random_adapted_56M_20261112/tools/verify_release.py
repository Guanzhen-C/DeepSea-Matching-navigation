"""Verify public package file hashes; no original data or model required."""
from pathlib import Path
import hashlib,json


def main():
    root=Path(__file__).resolve().parents[1]
    records=json.loads((root/'RELEASE_MANIFEST.json').read_text())['files']
    for row in records:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Path outside release')
        assert path.is_file() and path.stat().st_size==row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['SHA256'],row['path']
    print(json.dumps({'status':'all_public_files_verified','files':len(records),
                      'training_started':False,'raw_data_required':False}))


if __name__=='__main__':main()
