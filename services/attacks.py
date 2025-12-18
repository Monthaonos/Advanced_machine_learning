import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, **kwargs):
    """
    Fast Gradient Sign Method (FGSM).
    One-step attack that moves the image in the direction of the gradient.
    """
    # If no perturbation is allowed, return original images
    if epsilon == 0:
        return images

    # Create a copy of the images to avoid modifying the original tensor in-place
    # .detach() ensures we start with a leaf tensor for the new graph
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    images.requires_grad = True

    # Forward pass
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # Backward pass to calculate gradients w.r.t the input image
    model.zero_grad()
    loss.backward()

    # Get the direction of the gradient (Sign of the gradient)
    data_grad = images.grad.data

    # Create the perturbed image by adjusting each pixel by epsilon
    perturbed_image = images + epsilon * data_grad.sign()

    # Clip the image to ensure pixel values remain in the valid range [0, 1]
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
    """
    Projected Gradient Descent (PGD).
    Iterative version of FGSM with random start and projection.
    """
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)

    # Keep a copy of original images to compute the perturbation constraint (eta) later
    original_images = images.clone().detach()

    # Random Start: Initialize with uniform random noise within [-epsilon, epsilon]
    # This helps explore the loss landscape better than starting at 0
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Update images: Move a small step (alpha) in the direction of the gradient
        adv_images = images + alpha * images.grad.sign()

        # Projection Step:
        # 1. Calculate the total perturbation (adv - original)
        # 2. Clip this perturbation to be within [-epsilon, epsilon] (L-infinity constraint)
        eta = torch.clamp(
            adv_images - original_images, min=-epsilon, max=epsilon
        )

        # 3. Apply perturbation to original image and clip to valid pixel range [0, 1]
        # .detach() is crucial here to prevent memory leaks (stops building a huge graph)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return images


def mim_attack(
    model,
    images,
    labels,
    epsilon,
    alpha=2 / 255,
    num_steps=10,
    decay=1.0,
    **kwargs,
):
    """
    Momentum Iterative Method (MIM).
    Iterative attack that uses momentum to escape local maxima and stabilize the update direction.
    """
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)
    original_images = images.clone().detach()

    # Apply random start (often used in MIM as well to improve robustness)
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    # Initialize momentum buffer
    momentum = torch.zeros_like(images).detach().to(images.device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = images.grad.data

        # Normalize the gradient by its L1 norm (Mean Absolute Value)
        # This ensures the scale of the gradient doesn't affect the update magnitude too much
        # Added 1e-12 to avoid division by zero
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad / (grad_norm + 1e-12)

        # Update momentum: accumulate gradients with decay factor
        momentum = momentum * decay + grad

        # Update images using the sign of the accumulated momentum
        images = images.detach() + alpha * momentum.sign()

        # Projection Step (same as PGD)
        delta = torch.clamp(images - original_images, -epsilon, epsilon)
        images = torch.clamp(original_images + delta, 0, 1)

    return images
