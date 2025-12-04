import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, **kwargs):
    """Fast Gradient Sign Method"""
    if epsilon == 0:
        return images

    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbed_image = images + epsilon * data_grad.sign()
    return torch.clamp(perturbed_image, 0, 1)


def pgd_attack(
    model,
    images,
    labels,
    epsilon,
    alpha=2 / 255,
    num_steps=10,
    **kwargs,
):
    """Projected Gradient Descent"""
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)
    original_images = images.clone().detach()

    # Random start
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return images


def mim_attack(
    model, images, labels, epsilon, alpha=2 / 255, num_steps=10, decay=1.0, **kwargs
):
    """Momentum Iterative Method"""
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)
    original_images = images.clone().detach()

    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    momentum = torch.zeros_like(images).detach().to(images.device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = images.grad.data
        # Normalisation du gradient L1 pour le momentum
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = momentum * decay + grad

        images = images.detach() + alpha * momentum.sign()
        delta = torch.clamp(images - original_images, -epsilon, epsilon)
        images = torch.clamp(original_images + delta, 0, 1)

    return images
