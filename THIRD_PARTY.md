# Third-party acknowledgements

- **Dataset:** Parvesh Rawal, AI-vs-Real (2025), https://huggingface.co/datasets/Parveshiiii/AI-vs-Real . The publisher declares MIT licensing. Dataset image provenance and earlier preprocessing are not independently verified here. Only a few labeled demo images are redistributed in this repository; training images can be downloaded with the supplied scripts.
- **Pretrained model:** TorchVision MobileNetV3-Small ImageNet weights from https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth . PyTorch/TorchVision and pretrained-weight usage are subject to their applicable terms; this project replaces the classification head and exports its own trained checkpoint.
- **ONNX Runtime Web:** Microsoft, MIT license. Version 1.31.1 browser assets are vendored under docs/vendor/ with the package's license and notices where supplied.
- **Other dependencies:** PyTorch, TorchVision, Flask, Pillow, NumPy, scikit-learn, Matplotlib, requests, pyarrow and ONNX retain their upstream licenses.

Publishing this repository does not change the licenses of third-party data, model weights or software. No license for unrelated third-party images is implied.
