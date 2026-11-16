"""Local Flask website. Uploaded images are processed in memory, never saved."""
import io
import json
import warnings
from pathlib import Path
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from model import load_checkpoint, prepare, CLASSES

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
torch.set_num_threads(4)
cached_model = None
cached_stamp = None
cached_checkpoint = None

def get_model():
    global cached_model, cached_stamp, cached_checkpoint
    path = ROOT/'artifacts/best_model.pt'
    if not path.exists():
        raise FileNotFoundError('Model not trained yet. Run python train.py first.')
    stamp = path.stat().st_mtime_ns
    if cached_model is None or stamp != cached_stamp:
        cached_model, cached_checkpoint = load_checkpoint(path)
        cached_stamp = stamp
    return cached_model, cached_checkpoint

@app.get('/')
def home():
    return render_template('index.html')

@app.get('/api/metrics')
def metrics():
    result = {}
    for name, relative in [('metrics', 'artifacts/metrics.json'), ('history', 'artifacts/history.json'), ('dataset', 'data/summary.json'), ('audit','artifacts/audit/metrics.json'), ('baseline','baseline/artifacts/metrics.json')]:
        path = ROOT/relative
        result[name] = json.loads(path.read_text()) if path.exists() else None
    result['ready'] = (ROOT/'artifacts/best_model.pt').exists()
    return jsonify(result)

@app.post('/api/predict')
def predict():
    upload = request.files.get('image')
    if upload is None or not upload.filename:
        return jsonify(error='Choose a JPEG, PNG or WebP image.'), 400
    content_kind = request.form.get('content_kind', 'photo')
    if content_kind not in {'photo','diagram','screenshot'}:
        return jsonify(error='Choose the image content type.'), 400
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(upload.read())) as image:
                if image.format not in ('JPEG', 'PNG', 'WEBP'):
                    return jsonify(error='Supported formats: JPEG, PNG and WebP.'), 400
                if image.width * image.height > 25_000_000:
                    return jsonify(error='Please use an image with at most 25 million pixels.'), 400
                size = list(image.size)
                x = prepare(image).unsqueeze(0)
        if content_kind != 'photo':
            return jsonify(label='Origin cannot be determined', decision='unsupported_content',
                binary_label=None, ai_score=None, real_score=None, original_size=size,input_size=[256,256],
                content_kind=content_kind,
                note='This photo model has not been trained or evaluated to infer the origin of diagrams or mixed screenshots. Human-made and AI-assisted graphics can have identical pixels. For a screenshot of a photo, crop out the surrounding interface and analyze the photo itself.')
        model, checkpoint = get_model()
        with torch.inference_mode():
            score = torch.sigmoid(model(x)).item()
        threshold = checkpoint.get('threshold',0.5)
        margin = checkpoint.get('uncertainty_margin',0.15)
        uncertain = abs(score-threshold) <= margin
        label = 'Inconclusive' if uncertain else ('Likely AI-generated' if score>=threshold else 'Likely real photograph')
        return jsonify(label=label, binary_label=CLASSES[int(score>=threshold)],
                       decision='inconclusive' if uncertain else 'classified', ai_score=score,
                       real_score=1-score, original_size=size, input_size=[256,256],
                       content_kind=content_kind,threshold=threshold,uncertainty_margin=margin,
                       note='Scores from 35% to 65% are marked inconclusive. This band is a display rule, not a calibrated confidence interval. Even a strong score can be wrong on unfamiliar images.')
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        return jsonify(error=str(e) if isinstance(e, ValueError) else 'This image could not be safely decoded. Try another file.'), 400

@app.errorhandler(413)
def too_large(error):
    return jsonify(error='File is too large. Maximum upload size is 12 MB.'), 413

@app.get('/artifacts/<name>')
def artifact(name):
    if name not in {'learning_curves.png','confusion_matrix.png','roc_curve.png','metrics.json','test_predictions.csv'}:
        return jsonify(error='Not found'), 404
    return send_from_directory(ROOT/'artifacts', name)

@app.get('/demo/<name>')
def demo(name):
    if name not in {'1_actual_real.png','2_actual_ai.png'}:
        return jsonify(error='Not found'),404
    return send_from_directory(ROOT/'demo_images',name)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
