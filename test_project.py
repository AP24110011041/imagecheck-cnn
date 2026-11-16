"""Integration checks: real trained inference, upload rules, split integrity and metrics."""
import csv
import io
import json
import unittest
from unittest.mock import patch
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from app import app, ROOT

def png(size):
    stream = io.BytesIO()
    Image.new('RGB', size, 'gray').save(stream, format='PNG')
    stream.seek(0)
    return stream

class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
    def test_home_and_metrics(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertTrue(self.client.get('/api/metrics').json['ready'])
    def test_minimum_dimensions(self):
        for size in [(255,256),(256,255),(32,32)]:
            response = self.client.post('/api/predict', data={'image':(png(size),'small.png')})
            self.assertEqual(response.status_code, 400)
            self.assertIn('256', response.json['error'])
    def test_valid_boundary(self):
        response = self.client.post('/api/predict', data={'image':(png((256,256)),'valid.png')})
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.json['ai_score']+response.json['real_score'],1)
        self.assertIn(response.json['binary_label'], ['Real','AI-generated'])
        self.assertIn(response.json['label'], ['Likely real photograph','Likely AI-generated','Inconclusive'])
    def test_graphics_do_not_get_origin_claims(self):
        with patch('app.get_model',side_effect=AssertionError('Photo model must not classify graphics')):
            for kind in ['diagram','screenshot']:
                response=self.client.post('/api/predict',data={'image':(png((256,256)),'graphic.png'),'content_kind':kind})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json['decision'],'unsupported_content')
                self.assertIsNone(response.json['ai_score'])
        response=self.client.post('/api/predict',data={'image':(png((256,256)),'graphic.png'),'content_kind':'unknown'})
        self.assertEqual(response.status_code,400)
    def test_uncertainty_policy(self):
        # Confirm borderline scores are not silently converted to confident real/AI labels.
        for score,decision in [(0.2,'classified'),(0.38,'inconclusive'),(0.5,'inconclusive'),(0.6,'inconclusive'),(0.8,'classified')]:
            fake=lambda x:torch.tensor([np.log(score/(1-score))])
            with patch('app.get_model',return_value=(fake,{'threshold':0.5,'uncertainty_margin':0.15})):
                response=self.client.post('/api/predict',data={'image':(png((256,256)),'photo.png'),'content_kind':'photo'})
                self.assertEqual(response.json['decision'],decision)
    def test_invalid_uploads(self):
        self.assertEqual(self.client.post('/api/predict').status_code, 400)
        self.assertEqual(self.client.post('/api/predict', data={'image':(io.BytesIO(b'not an image'),'fake.png')}).status_code,400)
        body = b'--boundary\r\nContent-Disposition: form-data; name="image"; filename="big.png"\r\nContent-Type: image/png\r\n\r\n' + b'x'*(13*1024*1024) + b'\r\n--boundary--\r\n'
        self.assertEqual(self.client.post('/api/predict', data=body, content_type='multipart/form-data; boundary=boundary').status_code,413)
    def test_manifest_and_metrics(self):
        with (ROOT/'data/manifest.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows),len({r['sha256'] for r in rows}))
        self.assertTrue(all(int(r['original_width'])>=256 and int(r['original_height'])>=256 for r in rows))
        self.assertEqual({r['split'] for r in rows},{'train','validation','test'})
        with (ROOT/'artifacts/test_predictions.csv').open() as f:
            test = list(csv.DictReader(f))
        self.assertEqual({r['path'] for r in test},{r['path'] for r in rows if r['split']=='test'})
        y, pred = [int(r['actual']) for r in test], [int(r['predicted']) for r in test]
        metrics = json.loads((ROOT/'artifacts/metrics.json').read_text())
        self.assertEqual(confusion_matrix(y,pred).tolist(),metrics['confusion_matrix'])
        self.assertAlmostEqual(accuracy_score(y,pred),metrics['accuracy'])
        self.assertAlmostEqual(metrics['accuracy']+metrics['error_rate'],1)
        # Same saved model, same preprocessing, same held-out scores in website.
        for row in test[:3]:
            with (ROOT/row['path']).open('rb') as f:
                response = self.client.post('/api/predict', data={'image':(io.BytesIO(f.read()),'test.png')})
            self.assertEqual(response.status_code,200)
            self.assertAlmostEqual(response.json['ai_score'],float(row['ai_score']),places=5)
        self.assertEqual({r['predicted'] for r in test},{'0','1'},'Model must not predict just one class')
    def test_additional_audit(self):
        with (ROOT/'data/manifest.csv').open() as f:original=list(csv.DictReader(f))
        with (ROOT/'data/audit_manifest.csv').open() as f:audit=list(csv.DictReader(f))
        self.assertFalse({r['sha256'] for r in original}&{r['sha256'] for r in audit})
        self.assertEqual(len(audit),len({r['sha256'] for r in audit}))
        self.assertTrue(all(min(int(r['original_width']),int(r['original_height']))>=256 for r in audit))
        self.assertEqual(sum(r['label']=='0' for r in audit),sum(r['label']=='1' for r in audit))
        with (ROOT/'artifacts/audit/test_predictions.csv').open() as f:predictions=list(csv.DictReader(f))
        metrics=json.loads((ROOT/'artifacts/audit/metrics.json').read_text())
        self.assertEqual({r['path'] for r in predictions},{r['path'] for r in audit})
        self.assertEqual(confusion_matrix([int(r['actual']) for r in predictions],[int(r['predicted']) for r in predictions]).tolist(),metrics['confusion_matrix'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
