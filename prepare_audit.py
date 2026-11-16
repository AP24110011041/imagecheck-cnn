"""Build a fresh same-source audit from shard 3, without using model predictions."""
import argparse,csv,hashlib,io,json,random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parent
def dhash(image):
    a=np.array(image.convert('L').resize((9,8)))
    return int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--per-class',type=int,default=200);p.add_argument('--extra-source',type=Path)
    args=p.parse_args();rng=random.Random(2026)
    with (ROOT/'data/manifest.csv').open() as f: old=list(csv.DictReader(f))
    exact={r['sha256'] for r in old};near=[]
    for row in old:
        with Image.open(ROOT/row['path']) as image: near.append(dhash(image))
    candidates=[[],[]];counts=[0,0];rejected={'undersized':0,'duplicate_or_similar':0,'invalid':0}
    sources=[args.source]+([args.extra_source] if args.extra_source else [])
    for source_file in sources:
        for batch in pq.ParquetFile(source_file).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                try:
                    label=1-int(row['binary_label'])
                    with Image.open(io.BytesIO(row['image']['bytes'])) as source:
                        image=ImageOps.exif_transpose(source).convert('RGB')
                    w,h=image.size
                    if min(w,h)<256: rejected['undersized']+=1;continue
                    image=ImageOps.fit(image,(256,256),method=Image.Resampling.LANCZOS)
                    digest=hashlib.sha256(image.tobytes()).hexdigest();dh=dhash(image)
                    if digest in exact or any((dh^x).bit_count()<=3 for x in near):
                        rejected['duplicate_or_similar']+=1;continue
                    exact.add(digest);near.append(dh);counts[label]+=1
                    item=(image,dict(label=label,original_width=w,original_height=h,sha256=digest,source_shard=source_file.name,split='audit'))
                    if len(candidates[label])<args.per_class:candidates[label].append(item)
                    else:
                        index=rng.randrange(counts[label])
                        if index<args.per_class:candidates[label][index]=item
                except (OSError,ValueError,KeyError): rejected['invalid']+=1
    count=min(map(len,candidates));rows=[]
    if count<100:raise RuntimeError(f'Insufficient unique audit images: {counts}; rejected {rejected}')
    for group in candidates:
        for image,row in group[:count]:
            relative=f"data/audit/{row['sha256'][:24]}.png"
            path=ROOT/relative;path.parent.mkdir(exist_ok=True);image.save(path)
            rows.append(dict(path=relative,**row))
    with (ROOT/'data/audit_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(per_class=count,total=2*count,eligible=counts,rejected=rejected,seed=2026,
        source_shards=[x.name for x in sources],near_duplicate_check='64-bit difference hash, Hamming distance <= 3 excluded; not a comprehensive near-duplicate audit',
        limitation='Fresh images from another shard of the same dataset, not an independent generator/source benchmark.')
    (ROOT/'data/audit_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
