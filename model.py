"""Shared CNN and preprocessing for training and the website."""
import numpy as np
import torch
from torch import nn
from PIL import ImageOps, Image
from pathlib import Path

CLASSES = ['Real', 'AI-generated']

class TransferCNN(nn.Module):
    """Frozen MobileNet CNN features + a small binary classifier trained here."""
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        backbone = mobilenet_v3_small(weights=None)
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            state = torch.hub.load_state_dict_from_url(
                weights.url, model_dir=str(Path(__file__).resolve().parent/'pretrained'), check_hash=True)
            backbone.load_state_dict(state)
        self.encoder = backbone.features
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = nn.Sequential(nn.Linear(576,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,1))
        self.register_buffer('mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Keep pretrained batch-normalization statistics frozen.
        return self

    def extract(self,x):
        return self.encoder((x-self.mean)/self.std).mean(dim=(2,3))

    def forward(self,x):
        return self.head(self.extract(x)).squeeze(1)

def load_checkpoint(path):
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    model = TransferCNN() if checkpoint.get('architecture')=='mobilenet_v3_small_transfer' else SmallCNN()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint

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
