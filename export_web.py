"""Export the trained CNN for browser inference and verify numerical parity."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from PIL import Image
from model import load_checkpoint,prepare

ROOT=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(4)
    out=ROOT/'docs';out.mkdir(exist_ok=True)
    model,checkpoint=load_checkpoint(ROOT/'artifacts/best_model.pt')
    x=torch.zeros(1,3,256,256)
    torch.onnx.export(model,x,out/'model.onnx',opset_version=18,dynamo=False,
                      input_names=['image'],output_names=['logit'])
    onnx.checker.check_model(onnx.load(out/'model.onnx'))
    session=ort.InferenceSession(str(out/'model.onnx'),providers=['CPUExecutionProvider'])
    results=[]
    for path in sorted((ROOT/'demo_images').glob('*.png')):
        with Image.open(path) as image:x=prepare(image).unsqueeze(0)
        with torch.inference_mode():expected=torch.sigmoid(model(x)).item()
        logit=float(session.run(None,{'image':x.numpy()})[0][0]);actual=float(1/(1+np.exp(-logit)))
        assert abs(actual-expected)<1e-5,(path,expected,actual)
        results.append(dict(file=path.name,pytorch_score=expected,onnx_score=actual,absolute_error=abs(actual-expected)))
    with (out/'model.onnx').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(model_sha256=digest,input_shape=[1,3,256,256],tolerance=1e-5,examples=results,
                note='Numerical parity uses identical preprocessed tensors. Browser decoding/resampling can differ from Pillow; browser examples are checked separately.')
    (ROOT/'artifacts/export_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    for folder in ['static','artifacts','demo_images']:
        if folder=='artifacts':
            (out/folder).mkdir(exist_ok=True)
            for p in (ROOT/folder).iterdir():
                if p.is_file() and p.suffix in ['.json','.csv','.png']:shutil.copy2(p,out/folder/p.name)
            shutil.copytree(ROOT/'artifacts/audit',out/'artifacts/audit',dirs_exist_ok=True)
        else:shutil.copytree(ROOT/folder,out/folder,dirs_exist_ok=True)
    bundle={k:json.loads((ROOT/path).read_text()) for k,path in {
        'metrics':'artifacts/metrics.json','audit':'artifacts/audit/metrics.json','dataset':'data/summary.json',
        'baseline':'baseline/artifacts/metrics.json'}.items()}
    (out/'metrics.json').write_text(json.dumps(bundle,indent=2))
    (out/'.nojekyll').touch()

if __name__=='__main__':main()
