# Simple project-evaluation explanation

## Your 90-second introduction

“My project is a local website for distinguishing real photographs from AI-generated photo-like images. Each input must be at least 256 × 256 pixels. I use CNN transfer learning: a pretrained MobileNet extracts visual features, and I train a small dense classifier for my two labels.

“I use separate training, validation and test sets. Training updates the dense layers; validation selects the best checkpoint. I report a confusion matrix, accuracy, precision, recall, F1, ROC-AUC and error rate.

“The improved model achieved 94.5% accuracy on 400 additional images from the same dataset. Its error rate was 5.5%. It correctly detected 182 of 200 AI images and correctly recognized 196 of 200 real images.

“The website shows inconclusive for borderline scores. It does not claim to determine authorship of diagrams or mixed screenshots, because identical graphics can be produced through human or AI-assisted workflows.”

## Explain the architecture in one line

**256 × 256 RGB → frozen CNN → 576 features → Dense 64 → Dense 1 → sigmoid score.**

Only **36,993 parameters** are trained here. The pretrained CNN stays frozen. This is transfer learning, not training the complete CNN from scratch.

## Explain the confusion matrix

| Actual / predicted | Real | AI |
|---|---:|---:|
| Real | 196 (TN) | 4 (FP) |
| AI | 18 (FN) | 182 (TP) |

- Correct = 196 + 182 = **378**.
- Wrong = 4 + 18 = **22**.
- Accuracy = 378 / 400 = **94.5%**.
- Error rate = 22 / 400 = **5.5%**.
- Precision = 182 / (182 + 4) = **97.85%**.
- Recall = 182 / (182 + 18) = **91%**.
- F1 = **94.30%**.

## Demo sequence

1. Open the website and click **Real photo**, then **Analyze image**.
2. Click **AI photo**, then **Analyze image**. These buttons load actual test-image files; the backend runs the model on their pixels.
3. Upload another photograph and explain that the score is an estimate, not proof of origin.
4. Open **Model evaluation**. Distinguish the reused 250-image comparison benchmark from the fresh 400-image same-dataset audit.
5. Use the offline README.md to explain the training and evaluation steps; that explanation is not displayed on the website.

## If the faculty asks why the original version failed

“The small CNN trained from scratch had limited features and missed 26 of 125 AI images in the original test set. Transfer learning reduced that to 4 of 125 on the same benchmark. Separately, the new 400-image audit reached 94.5% accuracy. The model still has limitations outside that dataset.”

## Claims to avoid

- “It detects every AI image.”
- “A 99% model score proves the image is AI.”
- “It can establish who authored a diagram or screenshot.”
- “The full CNN was trained from scratch.”
- “The reused test comparison is a new independent benchmark.”

The strongest presentation is clear about what was trained, what was measured, and what remains uncertain.
