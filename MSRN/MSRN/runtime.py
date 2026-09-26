"""Shared model creation, device selection and evaluation metrics."""

import random

import numpy as np
import torch
from torchvision.models import Swin_V2_B_Weights, swin_v2_b


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA device is available")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_classifier(num_classes, pretrained=False):
    weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
    model = swin_v2_b(weights=weights)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model


def imagenet_norm(image):
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def classification_metrics(targets, predictions, classes):
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("Targets and predictions must have equal, nonzero length")
    count = len(classes)
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    accuracy = float(np.trace(matrix) / matrix.sum())
    expected = float(matrix.sum(0) @ matrix.sum(1) / float(matrix.sum()) ** 2)
    kappa = float((accuracy - expected) / (1 - expected)) if expected < 1 else None
    recalls = np.divide(matrix.diagonal(), matrix.sum(1), out=np.zeros(count), where=matrix.sum(1) != 0)
    return {"samples": len(targets), "accuracy": accuracy, "kappa": kappa,
            "classes": classes, "per_class_recall": recalls.tolist(), "confusion_matrix": matrix.tolist()}
