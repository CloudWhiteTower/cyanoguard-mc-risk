"""Download the exact source NetCDF files listed in the source manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def fetch(output: Path, manifest: Path) -> int:
    items = json.loads(manifest.read_text('utf-8'))['files']
    root = output.resolve()
    completed = 0
    for item in items:
        target = (root / item['path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Manifest path escapes the output directory')
        expected_url = 'https://www.ncei.noaa.gov/data/oceans/ioos/glos/OSUGI/' + item['path']
        if item['url'] != expected_url:
            raise ValueError('Manifest URL is outside the expected archive')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Existing file checksum differs: {target}')
            continue
        request = urllib.request.Request(item['url'], headers={'User-Agent':'SCAU-MC-Risk/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Download checksum differs: {item["path"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(data)
        completed += 1
        print(f'{completed}: {item["path"]}', flush=True)
    return completed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs/osugi-sources.json')
    args = parser.parse_args()
    fetch(args.output, args.manifest)
