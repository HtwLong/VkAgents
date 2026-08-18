"""LoRA layers and trainability policy for the registered Ultralytics RT-DETR-L model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


LORA_TARGET_PROFILES = {
    "decoder_attention": (
        ".decoder.layers.",
        (".cross_attn.value_proj", ".cross_attn.output_proj"),
    ),
    "decoder_attention_and_ffn": (
        ".decoder.layers.",
        (
            ".cross_attn.value_proj",
            ".cross_attn.output_proj",
            ".linear1",
            ".linear2",
        ),
    ),
}

HEAD_MODULE_MARKERS = (
    ".enc_score_head",
    ".enc_bbox_head",
    ".dec_score_head",
    ".dec_bbox_head",
)


class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank residual branch."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling
        return self.base(inputs) + residual

    def merged_linear(self) -> nn.Linear:
        """Return an ordinary Linear layer with the adapter folded into its weight."""
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = self.lora_B.weight @ self.lora_A.weight
        merged.weight.data.copy_(self.base.weight.data + delta.to(self.base.weight.dtype) * self.scaling)
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


@dataclass(frozen=True)
class RTDETRLoRASummary:
    target_modules: tuple[str, ...]
    total_parameters: int
    trainable_parameters: int

    @property
    def trainable_percent(self) -> float:
        if not self.total_parameters:
            return 0.0
        return 100.0 * self.trainable_parameters / self.total_parameters


def _target_suffixes(profile: str) -> tuple[str, tuple[str, ...]]:
    try:
        return LORA_TARGET_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported RT-DETR LoRA target profile: {profile}") from exc


def _parent_and_child(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def apply_rtdetr_lora(model: nn.Module, config: Mapping[str, Any]) -> RTDETRLoRASummary:
    """Inject adapters into validated decoder projections and freeze the detector base."""
    profile = str(config.get("lora_target_profile", "decoder_attention"))
    required_fragment, suffixes = _target_suffixes(profile)
    rank = int(config.get("lora_rank", 8))
    alpha = int(config.get("lora_alpha", 16))
    dropout = float(config.get("lora_dropout", 0.05))

    matches = [
        name
        for name, module in model.named_modules()
        if required_fragment in name
        and name.endswith(suffixes)
        and isinstance(module, nn.Linear)
        and not isinstance(module, LoRALinear)
    ]
    expected = 12 if profile == "decoder_attention" else 24
    if len(matches) != expected:
        raise ValueError(
            f"RT-DETR LoRA profile '{profile}' expected {expected} Linear targets, "
            f"but matched {len(matches)}: {matches}"
        )

    model.requires_grad_(False)
    for name in matches:
        parent, child_name = _parent_and_child(model, name)
        base = getattr(parent, child_name)
        setattr(
            parent,
            child_name,
            LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout),
        )
    set_rtdetr_lora_trainability(model)
    return rtdetr_lora_summary(model)


def set_rtdetr_lora_trainability(model: nn.Module) -> None:
    """Reassert adapter-only tuning plus full class/box heads before optimizer creation."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name or any(
            marker in name for marker in HEAD_MODULE_MARKERS
        ):
            parameter.requires_grad = True


def rtdetr_lora_summary(model: nn.Module) -> RTDETRLoRASummary:
    targets = tuple(name for name, module in model.named_modules() if isinstance(module, LoRALinear))
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not targets:
        raise ValueError("RT-DETR LoRA model contains no adapter targets.")
    if not any(any(marker in name for marker in HEAD_MODULE_MARKERS) for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise ValueError("RT-DETR LoRA requires trainable classification and box heads.")
    return RTDETRLoRASummary(targets, total, trainable)


def rtdetr_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only adapter tensors and the custom-class detection heads."""
    state = model.state_dict()
    selected = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if ".lora_A." in name
        or ".lora_B." in name
        or any(marker in name for marker in HEAD_MODULE_MARKERS)
    }
    if not any(".lora_A." in name for name in selected):
        raise ValueError("RT-DETR checkpoint has no LoRA adapter tensors.")
    if not any(any(marker in name for marker in HEAD_MODULE_MARKERS) for name in selected):
        raise ValueError("RT-DETR checkpoint has no custom detection-head tensors.")
    return selected


def load_rtdetr_lora_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Strictly load a compact adapter/head state into an identically injected base."""
    expected = set(rtdetr_lora_state_dict(model))
    received = set(state)
    if expected != received:
        raise ValueError(
            "Invalid RT-DETR LoRA state: "
            f"missing={sorted(expected - received)}, unexpected={sorted(received - expected)}"
        )
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected RT-DETR LoRA keys: {incompatible.unexpected_keys}")


def merge_rtdetr_lora_(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base Linear layer in place."""
    targets = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name in targets:
        parent, child_name = _parent_and_child(model, name)
        setattr(parent, child_name, getattr(parent, child_name).merged_linear())
    return model
