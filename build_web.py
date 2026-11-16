"""Package a static site; run export_web.py first when model weights change."""
from pathlib import Path
import shutil,json
ROOT=Path(__file__).resolve().parent;out=ROOT/'docs'
out.mkdir(exist_ok=True)
html=(ROOT/'templates/index.html').read_text(encoding='utf-8')
html=html.replace('href="/"','href="./"').replace('="/static/','="./static/').replace('="/artifacts/','="./artifacts/')
html=html.replace('Processed locally on your computer. Uploads are not saved.','Runs in your browser. Images never leave your device.')
html=html.replace('Academic prototype · PyTorch + Flask','Academic prototype · PyTorch + ONNX Runtime')
html=html.replace('<script src="./static/app.js"></script>', '<script src="./vendor/ort.wasm.min.js"></script><script src="./static/inference.js"></script><script src="./static/app.js"></script>')
(out/'index.html').write_text(html,encoding='utf-8')
shutil.copytree(ROOT/'static',out/'static',dirs_exist_ok=True)
shutil.copy2(ROOT/'web_inference.js',out/'static/inference.js')
script=(ROOT/'static/app.js').read_text(encoding='utf-8')
script=script.replace("fetch('/demo/'+button.dataset.example)","fetch('./demo_images/'+button.dataset.example)")
script=script.replace("fetch('/api/metrics')","fetch('./metrics.json')")
start=script.index("    const response = await fetch('/api/predict'")
end=script.index("    $('empty').hidden",start)
script=script[:start]+"    const result = await predictInBrowser(file);\n"+script[end:]
script=script.replace("  const data = new FormData(); data.append('image', file);\n",'')
(out/'static/app.js').write_text(script,encoding='utf-8')
vendor=out/'vendor';vendor.mkdir(exist_ok=True)
dist=ROOT/'node_modules/onnxruntime-web/dist'
for name in ['ort.wasm.min.js','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm']:
    shutil.copy2(dist/name,vendor/name)
package=ROOT/'node_modules/onnxruntime-web'
for name in ['LICENSE','ThirdPartyNotices.txt']:
    if (package/name).exists():shutil.copy2(package/name,vendor/name)
(out/'.nojekyll').touch()
print('Static site ready in docs/: no Python server, paid service, or inference API required.')
