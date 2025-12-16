"""
Utility functions for saving and loading PyTorch models.

This module provides small helper functions to persist trained models to disk
and restore them later.  Using these helpers keeps the training code in
``main.py`` (or other scripts) concise and clear.  You can extend this
module with additional functionality (for example, checkpointing optimizer
states or tracking the current epoch) as your experiments grow in
complexity.
"""

import os
from pathlib import Path
from typing import Optional
import torch
from torch import nn


def save_model(model: nn.Module, path: str) -> None:
    """Save a model's state_dict to a file.

    Args:
        model: The PyTorch module to save.
        path: Destination filepath.  Any missing parent directories will be
            created automatically.

    Note:
        Only the model's parameters are saved.  If you need to resume
        training with the exact optimizer state, consider saving the
        optimizer's state_dict as well (extend this helper as needed).
    """
    path_obj = Path(path)
    if path_obj.suffix == "":
        # default to .pth if no suffix is provided
        path_obj = path_obj.with_suffix(".pth")
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path_obj)


def load_model(
    model: nn.Module, path: str, device: Optional[str] = None
) -> nn.Module:
    """Load a model's parameters from disk into the provided instance.

    Args:
        model: The model instance whose parameters will be updated.
        path: Path to a file produced by ``save_model`` (i.e. a state_dict).
        device: Optional device specifier for ``torch.load``; if provided,
            the state will be loaded onto that device.  Defaults to the
            current device of the model's parameters.

    Returns:
        The model passed in, now containing the loaded parameters.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    map_location = None
    if device is not None:
        map_location = device
    state_dict = torch.load(path_obj, map_location=map_location)
    model.load_state_dict(state_dict)
    return model
