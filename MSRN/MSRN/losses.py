"""Original reconstruction losses and differentiable meta updates."""

import torch
from torch.nn import functional as F


def loss_int(rgb, reference, weight):
    return ((rgb - reference).square() * weight).mean()


def loss_spec(reconstructed, reference, weight):
    return ((reconstructed - reference).square() * weight).mean()


def loss_grad(rgb, reference):
    # Retain the original cross-channel Sobel response and signed-square target.
    def sobel(image):
        kx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = image.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernels = torch.stack((kx, ky))[:, None].repeat(1, image.shape[1], 1, 1)
        return F.conv2d(image, kernels, padding=1)

    prediction, target = sobel(rgb), sobel(reference)
    return 2 * F.l1_loss(prediction, target.abs() * target)


def reconstruction_loss(rgb, reconstructed, images, weight, rgb_bands, coefficients):
    intensity = loss_int(rgb, images[:, rgb_bands], weight)
    spectral = loss_spec(reconstructed, images, weight)
    gradient = loss_grad(rgb, images[:, rgb_bands])
    return coefficients[0] * intensity + coefficients[1] * spectral + coefficients[2] * gradient


def inner_update(model, loss, lr):
    leaves = [(module, name, value) for module in model.modules()
              if hasattr(module, "named_leaves") for name, value in module.named_leaves()
              if value is not None]
    gradients = torch.autograd.grad(loss, [p for _, _, p in leaves], create_graph=True)
    for (module, name, parameter), gradient in zip(leaves, gradients):
        setattr(module, name + "_meta", parameter - lr * gradient)


def clear_meta(model):
    for module in model.modules():
        if hasattr(module, "named_leaves"):
            for name, _ in module.named_leaves():
                setattr(module, name + "_meta", None)


def outer_update(model, loss, lr):
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(loss, parameters)
    maximum = torch.stack([g.detach().abs().max() for g in gradients]).max()
    if not torch.isfinite(maximum):
        raise FloatingPointError("Non-finite meta gradient")
    if maximum > 0:
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient / maximum, alpha=-lr)
