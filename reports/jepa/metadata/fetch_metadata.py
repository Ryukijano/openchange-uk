"""One-off bounded approved metadata fetch; no imagery endpoints."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openchange.provenance import start_run
from openchange.sources import require_approved

ROOT = Path('/home/ubuntu/repos/openchange-uk')
CONFIG = ROOT / 'configs/sources.yaml'
IDS = ('ssl4eo-v11-train-metadata', 'ssl4eo-v11-val-metadata')
CAP = 40_000_000
REVISION = '97a91733db962222e1e6b40a836845ddbbb16b86'


def main():
    sources = [require_approved(sid, 'pilot', CONFIG) for sid in IDS]
    if sum(s.expected_size_bytes for s in sources) > CAP:
        raise ValueError('approved sizes exceed combined cap')
    run_dir, _ = start_run('jepa-metadata', CONFIG, 0, ROOT / 'outputs')
    records = []
    total = 0
    for source in sources:
        require_approved(source.id, 'pilot', CONFIG)
        url = urlparse(source.url)
        if url.scheme != 'https' or url.netloc != 'huggingface.co' or REVISION not in url.path:
            raise ValueError('source must be the pinned approved Hub endpoint')
        name = Path(url.path).name
        if name not in {'train_metadata.parquet', 'val_metadata.parquet'} or source.use != 'auxiliary':
            raise ValueError('metadata endpoints only')
        path = run_dir / name
        temp = path.with_suffix('.parquet.partial')
        digest = hashlib.sha256()
        size = 0
        request = Request(source.url, headers={'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=120) as response, temp.open('xb') as out:
            if response.status != 200 or urlparse(response.url).scheme != 'https':
                raise ValueError('unexpected response status or insecure redirect')
            declared = response.headers.get('Content-Length')
            if declared is not None and int(declared) != source.expected_size_bytes:
                raise ValueError('unexpected Content-Length')
            while True:
                remaining = min(CAP - total, source.expected_size_bytes - size + 1)
                if remaining <= 0:
                    raise ValueError('payload ceiling reached')
                chunk = response.read(min(65536, remaining))
                if not chunk:
                    break
                size += len(chunk)
                total += len(chunk)
                if size > source.expected_size_bytes:
                    raise ValueError('file larger than approved size')
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if size != source.expected_size_bytes or digest.hexdigest() != source.checksum_sha256:
            raise ValueError('size/checksum mismatch; partial file not promoted')
        os.replace(temp, path)
        records.append({'source_id': source.id, 'url': source.url, 'path': str(path),
                        'bytes': size, 'sha256': digest.hexdigest(),
                        'verified': True, 'revision': REVISION})
        print(json.dumps(records[-1]), flush=True)
    report = {'status': 'verified_metadata_only', 'fetched_at': datetime.now(timezone.utc).isoformat(),
              'payload_cap_bytes': CAP, 'actual_payload_bytes': total, 'files': records,
              'approval_quote': 'Approve metadata only', 'imagery_approved': False,
              'split_accepted': False, 'sources_config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest()}
    (run_dir / 'fetch-result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT', str(run_dir / 'fetch-result.json'), flush=True)


if __name__ == '__main__':
    main()
