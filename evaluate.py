"""Report forced binary metrics and separately report the inconclusive policy."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import load_checkpoint,prepare,CLASSES

ROOT=Path(__file__).resolve().parent
class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(ROOT/self.rows[i]['path']) as image:return prepare(image)

def evaluate(checkpoint_path,rows,out,history=None,training=None,scope='Original test split, reused for version comparison'):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    model,checkpoint=load_checkpoint(checkpoint_path);scores=[]
    with torch.inference_mode():
        for x in DataLoader(Images(rows),batch_size=32):scores.extend(torch.sigmoid(model(x)).tolist())
    y=np.array([int(r['label']) for r in rows]);scores=np.array(scores)
    threshold=checkpoint.get('threshold',0.5);margin=checkpoint.get('uncertainty_margin',0.15)
    pred=(scores>=threshold).astype(int);confident=np.abs(scores-threshold)>margin
    cm=confusion_matrix(y,pred,labels=[0,1]);accuracy=accuracy_score(y,pred)
    metrics=dict(accuracy=accuracy,error_rate=1-accuracy,precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,scores),
        confusion_matrix=cm.tolist(),classification_report=classification_report(y,pred,target_names=CLASSES,output_dict=True,zero_division=0),
        test_size=len(y),threshold=threshold,positive_class='AI-generated',best_epoch=checkpoint['epoch'],
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        architecture=checkpoint.get('architecture','small_cnn'),evaluation_scope=scope,
        decision_policy=dict(uncertainty_margin=margin,inconclusive_count=int((~confident).sum()),
            classified_count=int(confident.sum()),coverage=float(confident.mean()),
            accuracy_on_classified=float(accuracy_score(y[confident],pred[confident])) if confident.any() else None,
            note='Heuristic score band, not calibrated confidence. Binary metrics above include ALL images at threshold 0.50.'),seed=42)
    if training:metrics.update(epochs_run=training['epochs_run'],training_seconds=training['training_seconds'])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    with (out/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['path','actual','predicted','ai_score','display_decision'])
        writer.writerows((r['path'],int(t),int(p),float(s),CLASSES[p] if c else 'Inconclusive') for r,t,p,s,c in zip(rows,y,pred,scores,confident))
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted label',ylabel='Actual label',title='Binary confusion matrix (all images)')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=150);plt.close(fig)
    fpr,tpr,_=roc_curve(y,scores);fig,ax=plt.subplots(figsize=(5,4));ax.plot(fpr,tpr,label=f"AUC = {metrics['roc_auc']:.3f}");ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve');ax.legend();fig.tight_layout();fig.savefig(out/'roc_curve.png',dpi=150);plt.close(fig)
    if history:
        (out/'history.json').write_text(json.dumps(history,indent=2))
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,key in zip(axes,['loss','accuracy']):
            for group in ['train','validation']:ax.plot([h['epoch'] for h in history],[h[f'{group}_{key}'] for h in history],label=group.title())
            ax.set(xlabel='Epoch',ylabel=key.title(),title=key.title());ax.legend()
        fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=ROOT/'candidate/best_model.pt')
    p.add_argument('--audit',action='store_true');args=p.parse_args()
    manifest=ROOT/('data/audit_manifest.csv' if args.audit else 'data/manifest.csv')
    with manifest.open() as f:rows=list(csv.DictReader(f))
    if not args.audit:rows=[r for r in rows if r['split']=='test']
    folder=args.checkpoint.parent
    history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else None
    training=json.loads((folder/'training.json').read_text()) if (folder/'training.json').exists() else None
    evaluate(args.checkpoint,rows,folder/('audit' if args.audit else 'evaluation'),history,training,
        scope='Fresh same-dataset shard audit; no training or checkpoint selection on these images' if args.audit else 'Original test split, reused for version comparison')

if __name__=='__main__':main()
