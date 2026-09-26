"""The original alternating meta-learning and reconstruction/classification phases."""

from copy import deepcopy

import torch
from torch.nn import functional as F

from .losses import clear_meta, inner_update, outer_update, reconstruction_loss
from .runtime import imagenet_norm


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(v) for v in value)
    return deepcopy(value)


def meta_step(reconstruction, classifier, lpn, classifier_optimizer,
              support, query, rgb_bands, coefficients, lr):
    """Update only LPN; roll back the virtual classifier and its Adam moments."""
    classifier_state = _cpu_copy(classifier.state_dict())
    optimizer_state = _cpu_copy(classifier_optimizer.state_dict())
    try:
        images, labels = support
        rgb, spectral = reconstruction(images)
        loss = reconstruction_loss(rgb, spectral, images, lpn(images), rgb_bands, coefficients)
        inner_update(reconstruction, loss, lr)
        # Preserve the original classifier adaptation on the ordinary R output.
        classifier_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            rgb, _ = reconstruction(images)
        F.cross_entropy(classifier(imagenet_norm(rgb)), labels).backward()
        classifier_optimizer.step()

        images, labels = query
        rgb, _ = reconstruction(images, meta=True)
        task_loss = F.cross_entropy(classifier(imagenet_norm(rgb)), labels)
        outer_update(lpn, task_loss, lr)
        return float(loss.detach()), float(task_loss.detach())
    finally:
        clear_meta(reconstruction)
        classifier.load_state_dict(classifier_state)
        classifier_optimizer.load_state_dict(optimizer_state)
        classifier_optimizer.zero_grad(set_to_none=True)


def train_step(reconstruction, classifier, lpn, reconstruction_optimizer,
               classifier_optimizer, batch, rgb_bands, coefficients):
    images, labels = batch
    reconstruction_optimizer.zero_grad(set_to_none=True)
    rgb, spectral = reconstruction(images)
    with torch.no_grad():
        weight = lpn(images)
    loss = reconstruction_loss(rgb, spectral, images, weight, rgb_bands, coefficients)
    loss.backward()
    reconstruction_optimizer.step()

    classifier_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        rgb, _ = reconstruction(images)
    task_loss = F.cross_entropy(classifier(imagenet_norm(rgb)), labels)
    task_loss.backward()
    classifier_optimizer.step()
    return float(loss.detach()), float(task_loss.detach())
