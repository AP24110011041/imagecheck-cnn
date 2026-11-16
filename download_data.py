"""Download the two source shards used in this reproducible student experiment (~1.03 GB)."""
import argparse
import hashlib
from pathlib import Path
import requests

REVISION = 'bce7ac5b95c36c5013389341b94c75aa44882165'
SHARDS = {
    'train-00000-of-00005.parquet': '70e3a44fe57b6f75564fb2f55e770440941e8bf1bd6a55b7c4953d9ccc061fb2',
    'train-00002-of-00005.parquet': '11f709ec5d79b420ddfcb4c3192c88301d951901e5a5aa121031979cbe0b60e5',
}
AUDIT_SHARDS = {
    'train-00001-of-00005.parquet': '1e4c90e36df08862c1226beb3e01c3eb611c90382172c519eb26c2609bd43381',
    'train-00003-of-00005.parquet': 'f1bc98e9147e7736e3197366e698f3a31a41b48617de0a0316ef27f8d6afb22a',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parent/'source_data')
    parser.add_argument('--include-audit',action='store_true',help='Also download the two additional audit shards (~1.11 GB extra)')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    shards={**SHARDS,**(AUDIT_SHARDS if args.include_audit else {})}
    for name, expected in shards.items():
        target = args.destination/name
        if target.exists():
            with target.open('rb') as f:
                valid = hashlib.file_digest(f, 'sha256').hexdigest() == expected
            if valid:
                print(f'Already verified: {name}', flush=True)
                continue
        url = f'https://huggingface.co/datasets/Parveshiiii/AI-vs-Real/resolve/{REVISION}/data/{name}'
        temp = target.with_suffix('.part')
        digest = hashlib.sha256()
        with requests.get(url, stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            with temp.open('wb') as output:
                size = 0
                for chunk in response.iter_content(1024*1024):
                    output.write(chunk); digest.update(chunk); size += len(chunk)
                    if size % (50*1024*1024) < 1024*1024:
                        print(f'{name}: {size/1024/1024:.0f} MB', flush=True)
        if digest.hexdigest() != expected:
            raise RuntimeError(f'Checksum mismatch: {name}. Retry the download.')
        temp.replace(target)
        print(f'Verified: {name}', flush=True)

if __name__ == '__main__':
    main()
