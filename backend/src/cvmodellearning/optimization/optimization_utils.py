from __future__ import annotations
from typing import Optional, List, Union, Any
from typing_extensions import TypedDict, Literal
from agents import function_tool, RunContextWrapper
import torch.nn as nn
import torch.optim as optim
import torch
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config

def make_optimizer(params, config: dict):
    config = training_compatible_hpo_config(config)
    name = config.get("optimizer_name")
    if name == "adamw":
        return optim.AdamW(
            params,
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
            betas=tuple([config.get("beta1", 0.9), config.get("beta2", 0.999)]),
            eps=config.get("eps", 1e-8),
        )
    if name == "sgd":
        return optim.SGD(
            params,
            lr=config.get("learning_rate", 1e-3),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
            nesterov=config.get("nesterov", False),
        )
    if name == "rmsprop":
        return optim.RMSprop(
            params,
            lr=config.get("learning_rate", 1e-3),
            alpha=config.get("alpha", 0.99),
            eps=config.get("eps", 1e-8),
            momentum=config.get("momentum", 0.0),
            weight_decay=config.get("weight_decay", 0.0),
            centered=config.get("centered", False),
        )
    raise ValueError(f"Unsupported optimizer: {name}")

def make_criterion(config: dict):
    config = training_compatible_hpo_config(config)
    name = config["criterion_name"]
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=config.get("label_smoothing", 0.0),
        )
    if name == "bce_with_logit":
        pos_weight = config.get("pos_weight")
        pos_tensor = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_tensor)
    raise ValueError(f"Unsupported criterion: {name}")
