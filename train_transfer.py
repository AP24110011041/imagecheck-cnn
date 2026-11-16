"""Train only a simple classifier on frozen CNN features; select by validation loss."""
import argparse
import csv
import io
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from PIL import Image
from model import TransferCNN,prepare,CLASSES

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'candidate'

class Images(Dataset):
    def __init__(self,rows,augment=False):
        self.rows,self.augment=rows,augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(ROOT/row['path']) as image:
            image=image.convert('RGB')
            if self.augment:
                # Expose the head to recompressed and mirrored examples, only in train.
                buffer=io.BytesIO()
                image.save(buffer,format='JPEG',quality=random.randint(65,95))
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    x=prepare(compressed,augment=True)
            else: x=prepare(image)
        return x,torch.tensor(int(row['label']),dtype=torch.float32)

def extract(model,rows,augment=False):
    features,labels=[],[]
    with torch.inference_mode():
        for x,y in DataLoader(Images(rows,augment),batch_size=32):
            features.append(model.extract(x));labels.append(y)
    return torch.cat(features).clone(),torch.cat(labels).clone()

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=80)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    OUT.mkdir(exist_ok=True)
    with (ROOT/'data/manifest.csv').open() as f: rows=list(csv.DictReader(f))
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='validation']
    start=time.time();model=TransferCNN(pretrained=True).eval()
    print('Extracting frozen CNN features: train original view',flush=True)
    tx,ty=extract(model,train)
    print('Extracting frozen CNN features: train augmented view',flush=True)
    ax,ay=extract(model,train,True)
    print('Extracting validation features (no augmentation)',flush=True)
    vx,vy=extract(model,val)
    loader=DataLoader(TensorDataset(torch.cat([tx,ax]),torch.cat([ty,ay])),batch_size=64,shuffle=True)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=0.001,weight_decay=0.01)
    loss_fn=nn.BCEWithLogitsLoss();best=float('inf');stale=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.head.train();total=correct=count=0
        for x,y in loader:
            optimizer.zero_grad();logits=model.head(x).squeeze(1);loss=loss_fn(logits,y)
            loss.backward();optimizer.step()
            total+=loss.item()*len(y);correct+=((logits>=0)==y.bool()).sum().item();count+=len(y)
        model.head.eval()
        with torch.no_grad():
            logits=model.head(vx).squeeze(1);vl=loss_fn(logits,vy).item()
            va=((logits>=0)==vy.bool()).float().mean().item()
        history.append(dict(epoch=epoch,train_loss=total/count,train_accuracy=correct/count,validation_loss=vl,validation_accuracy=va))
        print(f'Epoch {epoch}: train {correct/count:.3f}; validation {va:.3f}; val loss {vl:.4f}',flush=True)
        if vl<best:
            best=vl;stale=0
            checkpoint=dict(state_dict=model.state_dict(),architecture='mobilenet_v3_small_transfer',epoch=epoch,
                            classes=CLASSES,image_size=256,threshold=0.5,uncertainty_margin=0.15)
            torch.save(checkpoint,OUT/'best_model.pt')
        else: stale+=1
        if stale>=10: break
    (OUT/'history.json').write_text(json.dumps(history,indent=2))
    summary=dict(best_validation_loss=best,epochs_run=len(history),training_seconds=round(time.time()-start,1),
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        architecture='MobileNetV3-Small frozen CNN + Dense(64) + Dense(1)',seed=42)
    (OUT/'training.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    print('Candidate trained. Test set has not been used by this script.',flush=True)

if __name__=='__main__':main()
