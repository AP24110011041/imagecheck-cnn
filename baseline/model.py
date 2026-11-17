"""Shared CNN and preprocessing for training and the website."""
import numpy as np
import torch
from torch import nn
from PIL import ImageOps, Image

CLASSES = ['Real', 'AI-generated']

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 12, 24, 48, 64]
        for a, b in zip(channels, channels[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                  nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)

def prepare(image, augment=False):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if min(image.size) < 256:
        raise ValueError('Image width and height must both be at least 256 pixels.')
    image = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
    if augment and torch.rand(1).item() < 0.5:
        image = ImageOps.mirror(image)
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
