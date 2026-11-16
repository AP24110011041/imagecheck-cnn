"""Train the small transfer head, evaluate, then activate the model."""
import csv,json,shutil
from pathlib import Path
from train_transfer import main as train_head
from evaluate import evaluate

ROOT=Path(__file__).resolve().parent

def activate():
    candidate=ROOT/'candidate';artifacts=ROOT/'artifacts'
    history=json.loads((candidate/'history.json').read_text())
    training=json.loads((candidate/'training.json').read_text())
    with (ROOT/'data/manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
    evaluate(candidate/'best_model.pt',rows,candidate/'evaluation',history,training)
    if (ROOT/'data/audit_manifest.csv').exists():
        with (ROOT/'data/audit_manifest.csv').open() as f:audit=list(csv.DictReader(f))
        evaluate(candidate/'best_model.pt',audit,candidate/'audit',history,training,
                 scope='Additional same-source audit; rerunning this command reuses this benchmark')
    artifacts.mkdir(exist_ok=True)
    for path in (candidate/'evaluation').iterdir():
        if path.is_file():shutil.copy2(path,artifacts/path.name)
    if (candidate/'audit').exists():shutil.copytree(candidate/'audit',artifacts/'audit',dirs_exist_ok=True)
    shutil.copy2(candidate/'training.json',artifacts/'training.json')
    shutil.copy2(candidate/'best_model.pt',artifacts/'best_model.pending')
    (artifacts/'best_model.pending').replace(artifacts/'best_model.pt')
    print('Activated the transfer CNN. Restart app.py if its Python code changed.')

if __name__=='__main__':
    train_head()
    activate()
