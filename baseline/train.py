"""Train on train only, select by validation loss, evaluate test exactly once."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_score, recall_score, f1_score)
from model import SmallCNN, prepare, CLASSES

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts'

class Images(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT/row['path']) as image:
            x = prepare(image, self.augment)
        return x, torch.tensor(int(row['label']), dtype=torch.float32)

def run_epoch(model, loader, loss_fn, optimizer=None):
    model.train(optimizer is not None)
    total, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            correct += ((logits >= 0) == y.bool()).sum().item()
            count += len(y)
    return total/count, correct/count

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    OUT.mkdir(exist_ok=True)
    with (ROOT/'data/manifest.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    groups = {s: [r for r in rows if r['split'] == s] for s in ['train', 'validation', 'test']}
    loaders = {s: DataLoader(Images(v, s=='train'), batch_size=args.batch_size, shuffle=s=='train') for s,v in groups.items()}
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best, stale = [], float('inf'), 0
    start = time.time()
    for epoch in range(1, args.epochs+1):
        tl, ta = run_epoch(model, loaders['train'], loss_fn, optimizer)
        vl, va = run_epoch(model, loaders['validation'], loss_fn)
        history.append({'epoch': epoch, 'train_loss': tl, 'validation_loss': vl, 'train_accuracy': ta, 'validation_accuracy': va})
        print(f'Epoch {epoch:02d}: train loss {tl:.4f}, accuracy {ta:.3f}; validation loss {vl:.4f}, accuracy {va:.3f}', flush=True)
        (OUT/'history.json').write_text(json.dumps(history, indent=2))
        if vl < best:
            best, stale = vl, 0
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'classes': CLASSES, 'image_size': 256}, OUT/'best_model.pt')
        else:
            stale += 1
        if stale >= 4:
            print('Early stopping: validation loss did not improve for four epochs.', flush=True)
            break
    checkpoint = torch.load(OUT/'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    probabilities, truth = [], []
    with torch.inference_mode():
        for x, y in loaders['test']:
            probabilities.extend(torch.sigmoid(model(x)).tolist())
            truth.extend(y.int().tolist())
    predicted = (np.array(probabilities) >= 0.5).astype(int)
    cm = confusion_matrix(truth, predicted, labels=[0, 1])
    accuracy = accuracy_score(truth, predicted)
    metrics = {'accuracy': accuracy, 'error_rate': 1-accuracy,
        'precision': precision_score(truth, predicted, zero_division=0),
        'recall': recall_score(truth, predicted, zero_division=0),
        'f1': f1_score(truth, predicted, zero_division=0),
        'roc_auc': roc_auc_score(truth, probabilities), 'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(truth, predicted, labels=[0,1], target_names=CLASSES, output_dict=True, zero_division=0),
        'best_epoch': checkpoint['epoch'], 'epochs_run': len(history),
        'test_size': len(truth), 'threshold': 0.5, 'positive_class': 'AI-generated',
        'parameters': sum(p.numel() for p in model.parameters()),
        'training_seconds': round(time.time()-start, 1), 'seed': 42}
    (OUT/'metrics.json').write_text(json.dumps(metrics, indent=2))
    with (OUT/'test_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','actual','predicted','ai_score'])
        writer.writerows((r['path'],t,int(p),s) for r,t,p,s in zip(groups['test'],truth,predicted,probabilities))
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for ax, key in zip(axes, ['loss','accuracy']):
        ax.plot([h['epoch'] for h in history], [h['train_'+key] for h in history], label='Train')
        ax.plot([h['epoch'] for h in history], [h['validation_'+key] for h in history], label='Validation')
        ax.set(xlabel='Epoch', ylabel=key.title(), title=key.title())
        ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'learning_curves.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=CLASSES, yticklabels=CLASSES, xlabel='Predicted label', ylabel='Actual label', title='Held-out test confusion matrix')
    fig.tight_layout(); fig.savefig(OUT/'confusion_matrix.png', dpi=150); plt.close(fig)
    fpr, tpr, _ = roc_curve(truth, probabilities)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr,tpr,label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Held-out test ROC curve'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'roc_curve.png', dpi=150); plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
