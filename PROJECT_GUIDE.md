# ImageCheck — lightweight CNN transfer-learning project

## Start the website

Double-click **start.bat**, or open this folder in a terminal and run `python app.py`. Open http://127.0.0.1:5000. Keep the terminal open. No model API or paid service is used; uploaded images stay in memory.

Dependencies are already installed on the development computer. On another computer:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

The trained checkpoint and prepared data are included. No retraining or download is necessary for the demo. Package versions from the completed run are in `environment.txt`.

## What changed after the first version

The original four-block CNN sometimes missed AI-generated photos. The website also gave a forced binary answer for unsupported graphics. The replacement uses **MobileNetV3-Small as a frozen CNN feature extractor**, with a small classifier trained on our labeled data. The original experiment is preserved in `baseline/`.

The new model was chosen using validation loss, before its test and additional audit scores were calculated. The original 250-image test set is now a **reused comparison benchmark**, not a fresh independent validation. See `RESULTS.md` for the measured comparison and the additional audit.

## A simple explanation for your evaluation

“My project uses CNN transfer learning. A pretrained MobileNet extracts visual features from each 256 × 256 RGB image. I keep that CNN fixed and train a small dense network to distinguish real from AI-generated photos. I select the best epoch using validation loss, then report accuracy, error rate, precision, recall, F1, ROC-AUC and a confusion matrix. The website also handles uncertainty instead of claiming every input is definitely real or AI.”

This is **not training the whole CNN from scratch**. Explain that distinction honestly. The previous from-scratch experiment remains available for comparison.

## What the website can and cannot classify

The website now uses a direct upload-and-analyze flow, without a content-type dropdown or a How it works section. Every UI upload is sent to the photo classifier. Removing the selector does not expand the trained model's domain: diagrams, charts, illustrations and mixed screenshots are not validated inputs, and their origin cannot reliably be established by this photo model. The API retains its optional content-kind handling for compatibility, but the website does not expose it.

For photos, scores from **0.35 through 0.65** display **Inconclusive**. Below 0.35 displays “Likely real photograph”; above 0.65 displays “Likely AI-generated”. This fixed band is a heuristic display rule, not a calibrated confidence interval. A high score can still be wrong on unfamiliar images.

The quantitative binary evaluation still uses **0.50 on every test image**, including borderline ones. We separately report the number of inconclusive results, coverage, and accuracy among classified images. We do not remove difficult cases to inflate headline accuracy.

## Faculty's 256 × 256 requirement

Both source width and height must be at least 256 pixels. Smaller images are rejected, never enlarged to pass this rule. Eligible source images are converted to RGB, center-cropped and resized to 256 × 256. The source dimensions are recorded in the manifests. Earlier resizing by the dataset publisher cannot be independently ruled out.

The transfer CNN receives pixel values scaled to [0,1], then ImageNet normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Normalization is inside the model, so training and website inference match.

## Dataset and splits

Source: [Parvesh Rawal — AI-vs-Real (2025)](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real), revision `bce7ac5b95c36c5013389341b94c75aa44882165`. The dataset card declares MIT licensing. We use source labels as provided; this project does not independently authenticate each image or claim complete generator provenance.

Source labels are **0 = AI, 1 = real**. Project labels are remapped to **0 = real, 1 = AI**, making AI the positive class.

- Original selected subset: **1,666 images**, 833 per class.
- Train: **1,166**, 583 per class.
- Validation: **250**, 125 per class.
- Original test: **250**, 125 per class.
- Stratified 70/15/15 split, seed 42. Exact normalized-pixel duplicates removed before splitting.
- The additional audit uses images outside this original manifest, with exact-hash exclusion and a basic difference-hash similarity filter. It is still from the same dataset, not proof of unseen-generator generalization. Its final counts are in `data/audit_summary.json`.

The dataset card's overall balance is not assumed: the selected source shards have very different class counts. Our manifests explicitly balance the selected classes.

## Architecture

```text
256 × 256 RGB image
    ↓ ImageNet normalization
Frozen MobileNetV3-Small convolutional features
    ↓ Global average pooling
576 feature values
    ↓ Dense(64) → ReLU → Dropout(0.3)
Dense(1) → logit
    ↓ Sigmoid
AI score
```

**964,001 total parameters; 36,993 trained parameters.** Pretrained convolution weights and batch-normalization statistics remain frozen. The original ImageNet classification head is discarded. Only our two dense layers learn the AI-versus-real task.

Weights come from [official TorchVision MobileNetV3](https://docs.pytorch.org/vision/main/models/mobilenetv3.html). The original 9.83 MB weights are included in `pretrained/` to enable offline retraining. The classifier checkpoint is `artifacts/best_model.pt`.

## Machine-learning steps

1. **Define the problem:** supervised binary photo classification; explicitly distinguish unsupported inputs.
2. **Validate data:** decode, check dimensions, normalize image format, remove exact duplicates.
3. **Split:** stratified train/validation/test split, keeping labels balanced.
4. **Preprocess:** crop/resize to 256 × 256 RGB and normalize consistently.
5. **Augment training:** add a second recompressed/mirrored view for each training image. Validation and test images are unchanged.
6. **Extract CNN features:** run both training views and validation images through the frozen CNN once. Caching these features makes CPU training fast; it does not change which weights learn.
7. **Train the head:** AdamW, learning rate 0.001, weight decay 0.01, batch size 64, binary cross-entropy with logits, at most 80 epochs.
8. **Select by validation loss:** early stop after 10 epochs without improvement. Save the best checkpoint, not automatically the final epoch.
9. **Evaluate:** fixed 0.50 binary threshold, all test images included. Report confusion matrix, class metrics, ROC curve and learning curves. Separately report the inconclusive display policy.
10. **Deploy locally:** Flask validates uploads, runs the saved CNN and displays its scores with the inconclusive policy.

## Commands

```powershell
python train.py
python test_project.py
python app.py
```

`train.py` retrains the transfer head on the prepared data, evaluates the saved benchmark and additional audit if available, then activates the checkpoint. Repeated evaluations reuse those datasets; do not describe them as newly unseen each time.

For the original data preparation: `python download_data.py` downloads source shards 0 and 2 (~1.03 GB) with SHA-256 verification; `python prepare_data.py --source source_data --per-class 1000` reconstructs the original subset. The preparation script selects only shards 0 and 2 even if audit shards are present. The exact audit source files are recorded in its summary; prepared audit data is included.

`python train_transfer.py` trains a candidate without using test images. `python evaluate.py` evaluates that candidate on the original test split; `python evaluate.py --audit` evaluates the additional audit. The original scratch-CNN sources in `baseline/` are archival; the supported default training command is `python train.py`.

## Metrics and confusion matrix

AI is the positive class:

| Actual / predicted | Real | AI |
|---|---|---|
| Real | TN: correct real | FP: real incorrectly flagged AI |
| AI | FN: AI missed | TP: correct AI |

- Accuracy = `(TP + TN) / N`.
- Error rate = `(FP + FN) / N = 1 − accuracy`.
- Precision = `TP / (TP + FP)`.
- Recall = `TP / (TP + FN)`.
- F1 = `2 × precision × recall / (precision + recall)`.
- ROC-AUC measures score ranking across thresholds; it is not accuracy.
- Loss is the quantity optimized during training, not the classification error rate.

Learning curves compare training and validation behavior. If training improves while validation loss worsens, that suggests overfitting. The saved model uses the lowest validation loss.

## Demo and viva

Use the website's **Real photo** and **AI photo** test-example buttons to check the complete upload-to-model path. They load real image files and run the CNN; labels and filenames are never used by the prediction model. A documented failed example is also included when available. These examples are part of the measured benchmark, not a separate accuracy test.

**Why CNN?** It learns spatial image features with shared convolution weights.

**Why transfer learning?** Reuse features learned on many images; train a small task-specific head with limited data and CPU resources.

**What is frozen?** The CNN convolution weights and batch-normalization statistics. Only the dense classifier is trained here.

**Why dropout?** It reduces dependence on particular dense features during training; it is disabled at inference.

**Why three splits?** Training learns weights, validation selects the checkpoint, testing measures performance outside those steps.

**Does a 90% score prove authenticity?** No. Scores are uncalibrated and can be wrong outside the training distribution.

**Why not classify every diagram?** The visual output can be identical regardless of whether AI helped create it. We must not invent an origin claim unsupported by training or evidence.

## Limitations

The original test set was reused to compare versions. The additional audit is fresh relative to this training run but still uses the same dataset, whose per-image source/generator information is limited. Near-duplicate filtering is basic, not a comprehensive provenance audit. Real-world screenshots, new generators, edits, styles and compression can still cause errors. Diagram/screenshot authorship detection remains outside this project's evaluated scope. No universal “best detector” or 100% accuracy is claimed.

## Reproduce the additional audit

The prepared audit is included. To reconstruct it from the pinned sources:

```powershell
python download_data.py --include-audit
python prepare_audit.py --source source_data/train-00001-of-00005.parquet --extra-source source_data/train-00003-of-00005.parquet --per-class 200
python evaluate.py --audit
```

Pretraining-data overlap with this dataset has not been independently audited.
