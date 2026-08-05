from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Sampler

from cvmodellearning.models.registry import CLASSIFIER_HEAD_PATHS


class RepeatedAugmentationSampler(Sampler[int]):
    """Sample repeated augmented views without multiplying the epoch length.

    This follows the reference repeated-augmentation idea: build shuffled,
    repeated candidates and consume one dataset-length subset per epoch. The
    same source image can therefore be transformed independently more than
    once, while schedulers and early stopping retain their normal epoch units.
    """

    def __init__(self, dataset, repetitions: int = 1, *, seed: int = 0):
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        self.dataset = dataset
        self.repetitions = repetitions
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic but different candidate subset each epoch."""
        self.epoch = epoch

    def __iter__(self):
        if len(self.dataset) == 0:
            return iter([])
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.arange(len(self.dataset)).repeat_interleave(self.repetitions)
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        return iter(shuffled[:len(self.dataset)].tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def classifier_head(model, model_name: str):
    """Return the registered classifier module for a supported model."""
    try:
        return model.get_submodule(CLASSIFIER_HEAD_PATHS[model_name])
    except KeyError as exc:
        raise ValueError(f"No executable classifier-head mapping for {model_name}.") from exc


def classifier_training_module(model, model_name: str):
    """Return the head region that should remain in train mode while frozen.

    MobileNet and EfficientNet dropout belongs to the classifier container even
    though only its final Linear layer is trainable. Other registered models
    can use the final classifier module directly.
    """
    if model_name.startswith("efficientnet_b") or model_name in {
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
    }:
        return model.classifier
    return classifier_head(model, model_name)


def set_backbone_trainable(model, model_name: str, trainable: bool) -> None:
    """Freeze/unfreeze the backbone while always leaving the classifier trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in classifier_head(model, model_name).parameters():
        parameter.requires_grad = True


def classification_parameter_groups(model, model_name: str, config: dict):
    """Apply a head learning-rate multiplier without model-specific optimizers."""
    base_lr = float(config["learning_rate"])
    head_lr = base_lr * float(config.get("head_learning_rate_multiplier", 1.0))
    head_parameters = list(classifier_head(model, model_name).parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    return [
        {"params": backbone_parameters, "lr": base_lr},
        {"params": head_parameters, "lr": head_lr},
    ]


def swin_parameter_groups(model, config: dict) -> list[dict]:
    """Build Swin AdamW groups with head LR scaling and decay exclusions."""
    base_lr = float(config["learning_rate"])
    head_multiplier = float(config.get("head_learning_rate_multiplier", 1.0))
    weight_decay = float(config.get("weight_decay", 0.0))
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}

    for name, parameter in model.named_parameters():
        is_head = name.startswith("head.")
        # Match the reference recipe's zero decay for biases, normalization
        # vectors and transformer embedding-like one-dimensional parameters.
        is_transformer_embedding = any(
            token in name
            for token in ("class_token", "position_embedding", "relative_position_bias_table")
        )
        use_decay = parameter.ndim > 1 and not name.endswith(".bias") and not is_transformer_embedding
        groups.setdefault((is_head, use_decay), []).append(parameter)

    return [
        {
            "params": parameters,
            "lr": base_lr * (head_multiplier if is_head else 1.0),
            "weight_decay": weight_decay if use_decay else 0.0,
        }
        for (is_head, use_decay), parameters in groups.items()
    ]


def make_epoch_scheduler(optimizer, config: dict):
    """Create TorchVision-compatible linear warmup and epoch-wise scheduling."""
    if config.get("scheduler_name", "none") == "none" and int(config.get("warmup_epochs", 0)) == 0:
        return None

    total_epochs = int(config["num_epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 0))
    start_factor = float(config.get("warmup_start_factor", 0.01))
    scheduler_name = config.get("scheduler_name", "none")

    if scheduler_name == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=float(config.get("min_learning_rate", 0.0)),
        )
    elif scheduler_name == "step":
        main_scheduler = StepLR(
            optimizer,
            step_size=int(config.get("scheduler_step_size", 7)),
            gamma=float(config.get("scheduler_gamma", 0.1)),
        )
    else:
        main_scheduler = None

    if warmup_epochs == 0:
        return main_scheduler

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=warmup_epochs,
    )
    if main_scheduler is None:
        return warmup_scheduler
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )


def make_model_ema(model, config: dict, *, effective_batch_size: int, device):
    """Create an EMA whose decay has TorchVision's update-frequency adjustment."""
    configured_decay = float(config.get("model_ema_decay", 0.99998))
    update_steps = int(config.get("model_ema_steps", 32))
    epochs = int(config["num_epochs"])
    world_size = 1
    adjustment = world_size * effective_batch_size * update_steps / epochs
    alpha = min(1.0, (1.0 - configured_decay) * adjustment)
    effective_decay = 1.0 - alpha
    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(effective_decay),
        use_buffers=True,
        device=device,
    )
    return ema_model, effective_decay, update_steps


def apply_swin_activation_checkpointing(model) -> None:
    """Wrap individual Swin blocks with non-reentrant activation checkpointing."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    from torchvision.models.swin_transformer import SwinTransformerBlockV2

    def wrap_children(module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, SwinTransformerBlockV2):
                setattr(
                    module,
                    name,
                    checkpoint_wrapper(child, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                )
            else:
                wrap_children(child)

    wrap_children(model.features)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    *,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float = 0.0,
    scaler=None,
    batch_augmentation=None,
    on_optimizer_step: Callable[[], None] | None = None,
    frozen_backbone=None,
    trainable_head=None,
    cancel_check: Callable[[], None] | None = None,
):
    model.train()
    if frozen_backbone is not None:
        # Frozen feature extraction should be deterministic: keep stochastic
        # depth/dropout disabled while the newly initialized head trains.
        frozen_backbone.eval()
        if trainable_head is not None:
            trainable_head.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    amp_enabled = scaler is not None and scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (images, targets) in enumerate(loader):
        if cancel_check is not None:
            cancel_check()
        images = images.to(device)
        targets = targets.to(device)
        hard_targets = targets
        if batch_augmentation is not None:
            images, targets = batch_augmentation(images, targets)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Average over the actual accumulation group. Without this adjustment,
        # a final partial group is underweighted whenever the loader length is
        # not divisible by gradient_accumulation_steps.
        group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
        accumulation_group_size = min(
            gradient_accumulation_steps,
            len(loader) - group_start,
        )
        scaled_loss = loss / accumulation_group_size
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_update = (batch_index + 1) % gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
        if is_update:
            if gradient_clip_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if on_optimizer_step is not None:
                on_optimizer_step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == hard_targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc
