"""Create a reproducible, dimension-checked subset from source parquet files."""
import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--per-class', type=int, default=1000)
    args = p.parse_args()
    rng = random.Random(42)
    candidates = [[], []]
    seen = set()
    rejected = {'undersized': 0, 'duplicate': 0, 'invalid': 0}
    # Keep the experiment unchanged when extra audit shards share the download folder.
    sources=[args.source/f'train-{i:05d}-of-00005.parquet' for i in [0,2]]
    for source in sources:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=32):
            for idx, row in enumerate(batch.to_pylist()):
                try:
                    label = 1 - int(row['binary_label'])  # Source: 0=AI, 1=real.
                    if label not in (0, 1):
                        raise ValueError('Unknown label')
                    image = ImageOps.exif_transpose(Image.open(io.BytesIO(row['image']['bytes']))).convert('RGB')
                    w, h = image.size
                    if min(w, h) < 256:
                        rejected['undersized'] += 1
                        continue
                    normalized = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
                    digest = hashlib.sha256(normalized.tobytes()).hexdigest()
                    if digest in seen:
                        rejected['duplicate'] += 1
                        continue
                    seen.add(digest)
                    # Store all eligible images temporarily; sample deterministically below.
                    folder = ROOT / 'data' / ('ai' if label else 'real')
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f'{digest[:24]}.png'
                    normalized.save(path)
                    candidates[label].append({'path': path.relative_to(ROOT).as_posix(),
                        'label': label, 'original_width': w, 'original_height': h,
                        'sha256': digest, 'source_shard': source.name})
                except (OSError, ValueError, KeyError):
                    rejected['invalid'] += 1
        print(source.name, [len(x) for x in candidates], flush=True)
    count = min(args.per_class, *(len(x) for x in candidates))
    if count < 100:
        raise RuntimeError('Need at least 100 valid images per class; add source shards.')
    selected = []
    for group in candidates:
        rng.shuffle(group)
        selected += group[:count]
    train, held = train_test_split(selected, test_size=0.3, stratify=[r['label'] for r in selected], random_state=42)
    val, test = train_test_split(held, test_size=0.5, stratify=[r['label'] for r in held], random_state=42)
    for split, rows in [('train', train), ('validation', val), ('test', test)]:
        for row in rows:
            row['split'] = split
    with (ROOT/'data/manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(train[0]))
        writer.writeheader()
        writer.writerows(train + val + test)
    summary = {'dataset': 'Parveshiiii/AI-vs-Real',
        'source_revision': 'bce7ac5b95c36c5013389341b94c75aa44882165',
        'prepared_image_size': [256, 256],
        'url': 'https://huggingface.co/datasets/Parveshiiii/AI-vs-Real',
        'source_label_mapping': {'0': 'AI-generated', '1': 'Real'},
        'project_label_mapping': {'0': 'Real', '1': 'AI-generated'},
        'per_class': count, 'splits': {'train': len(train), 'validation': len(val), 'test': len(test)},
        'rejected': rejected, 'seed': 42,
        'limitations': 'Subset of selected shards. Exact normalized-pixel duplicates removed; near-duplicates and source/generator overlap may remain. Random image split does not measure unseen-generator performance.'}
    (ROOT/'data/summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
