from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests

sys.path.insert(0, '/home/ubuntu/repos/openchange-uk/src')
from openchange.sources import require_approved

HERE = Path(__file__).resolve().parent
CAP = 300_000_000
LOCK = threading.Lock()
LEDGER = HERE / 'transfer_ledger.json'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def catalog():
    license_url = 'https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96c/'
    license_data = requests.get(license_url, timeout=60).json()
    assert 'CC0' in json.dumps(license_data) or 'No Rights Reserved' in json.dumps(license_data)
    save_json(HERE / 'osf_license.json', license_data)
    files = []
    url = 'https://api.osf.io/v2/nodes/jftw8/files/osfstorage/?page[size]=100'
    while url:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        page = response.json()
        files.extend(page['data'])
        url = page['links'].get('next')
    selected = [f for f in files if f['attributes']['name'] == 'Grids.nc' or re.fullmatch(r'Monitoring_TrucVert_\d{4}-\d{2}-\d{2}\.nc', f['attributes']['name'])]
    assert len(selected) == 295
    rows = [{'source': 'trucvert-monitoring', 'name': f['attributes']['name'], 'size': f['attributes']['size'],
             'url': f['links']['download'], 'sha256': f['attributes']['extra']['hashes'].get('sha256'),
             'destination': str(HERE / 'raw/TRUCVERT' / f['attributes']['name'])} for f in selected]
    dataset = requests.get('https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.n5qb383', timeout=60).json()
    assert dataset['license'].endswith('CC0-1.0.html')
    save_json(HERE / 'dryad_license.json', dataset)
    files = requests.get('https://datadryad.org/api/v2/versions/28137/files?page=2', timeout=60).json()['_embedded']['stash:files']
    match = [f for f in files if f['path'] == 'torrey_mapped_sand_elevations.zip']
    assert len(match) == 1
    file = match[0]
    rows.append({'source': 'torrey-mapped', 'name': file['path'], 'size': file['size'],
                 'url': 'https://datadryad.org' + file['_links']['stash:download']['href'],
                 'sha256': None, 'destination': str(HERE / 'raw' / file['path'])})
    assert sum(row['size'] for row in rows) == 277_835_982
    save_json(HERE / 'download_manifest.json', rows)
    print('Manifest:', len(rows), 'files;', sum(row['size'] for row in rows), 'bytes', flush=True)
    return rows


def fetch(row, ledger):
    require_approved(row['source'], 'pilot', HERE / 'sources.yaml')
    path = Path(row['destination'])
    if path.exists():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size == row['size'] and (not row['sha256'] or row['sha256'] == sha):
            return {**row, 'download_sha256': sha, 'cached': True}
        raise RuntimeError(f'Existing file fails verification: {path.name}')
    part = path.with_suffix(path.suffix + '.partial')
    if part.exists():
        raise RuntimeError(f'Inspect incomplete file before retrying: {part.name}')
    digest, count = hashlib.sha256(), 0
    with requests.get(row['url'], stream=True, timeout=(30, 120), headers={'Accept-Encoding': 'identity'}) as response:
        response.raise_for_status()
        with part.open('xb') as output:
            for chunk in response.iter_content(1024 * 1024):
                with LOCK:
                    if ledger['payload_bytes'] + len(chunk) > CAP:
                        raise RuntimeError('Approved transfer cap reached')
                    ledger['payload_bytes'] += len(chunk)
                    save_json(LEDGER, ledger)
                count += len(chunk)
                if count > row['size']:
                    raise RuntimeError(f'File larger than declared: {path.name}')
                digest.update(chunk)
                output.write(chunk)
    sha = digest.hexdigest()
    assert count == row['size'], (row['name'], count, row['size'])
    assert not row['sha256'] or row['sha256'] == sha, row['name']
    part.rename(path)
    record = {**row, 'download_sha256': sha, 'cached': False, 'acquired_utc': datetime.now(UTC).isoformat()}
    with LOCK, (HERE / 'verified_downloads.jsonl').open('a') as output:
        output.write(json.dumps(record) + '\n')
    print('Verified', path.name, count, flush=True)
    return record


def extract_torrey():
    archive = HERE / 'raw/torrey_mapped_sand_elevations.zip'
    root = HERE / 'raw/TORREYPINES'
    with zipfile.ZipFile(archive) as zipped:
        members = [i for i in zipped.infolist() if i.filename.endswith('.nc')]
        assert members
        total = sum(i.file_size for i in members)
        assert total < 5_000_000_000
        assert total < os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2
        save_json(HERE / 'zip_inventory.json', {'nc_count': len(members), 'expanded_bytes': total, 'members': [i.filename for i in members]})
        for member in members:
            relative = PurePosixPath(member.filename)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert ((member.external_attr >> 16) & 0o170000) != 0o120000
            name = relative.name
            assert re.fullmatch(r'map\d{8}_[A-Za-z0-9_]+\.nc', name), name
            destination = root / name
            if destination.exists():
                assert destination.stat().st_size == member.file_size
                continue
            with zipped.open(member) as source, destination.open('xb') as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
        print('Extracted', len(members), 'NC files;', total, 'bytes', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    for source in ['trucvert-monitoring', 'torrey-mapped']:
        require_approved(source, 'pilot', HERE / 'sources.yaml')
    rows = catalog()
    if args.manifest_only:
        return
    for folder in ['raw', 'raw/TRUCVERT', 'raw/TORREYPINES']:
        (HERE / folder).mkdir(exist_ok=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'cap_bytes': CAP, 'payload_bytes': 0}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda row: fetch(row, ledger), rows))
    save_json(HERE / 'acquisition_results.json', records)
    extract_torrey()
    print('Total survey transfer bytes:', ledger['payload_bytes'], flush=True)


if __name__ == '__main__':
    main()
