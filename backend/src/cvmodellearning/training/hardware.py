"""Runtime training-backend detection."""

import torch


def detect_training_backend() -> str:
    """Return the best PyTorch accelerator available to this process."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
