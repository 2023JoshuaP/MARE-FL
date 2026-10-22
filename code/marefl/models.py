from __future__ import annotations
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int, image_flatten: bool = False):
        super().__init__()
        self.image_flatten = image_flatten
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(1)
        return self.net(x)


class SmallCifarCNN(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(kind: str, input_dim: int, hidden: int, classes: int):
    if kind == "cnn_cifar":
        return SmallCifarCNN(classes)
    return MLP(input_dim, hidden, classes, image_flatten=(kind == "mlp_image"))
