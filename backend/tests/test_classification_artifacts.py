from __future__ import annotations

import json
from zipfile import ZipFile

import torch
from safetensors.torch import load as load_safetensors
from torch import nn

from cvmodellearning import paths
from cvmodellearning.models.classification_artifacts import (
    ensure_lora_adapter_bundle,
    ensure_merged_lora_model,
)
from cvmodellearning.models.classification_lora import (
    CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
    LORA_CHECKPOINT_FORMAT,
    apply_classification_lora,
    classification_lora_metadata,
    classification_lora_state_dict,
)
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from routers.artifacts import artifact_manifest, artifact_model


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(self.qkv(inputs))


class _Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.head(self.attn(inputs))


def _config() -> dict:
    return {
        "model_name": "clip_vit_b16",
        "model_weights": "default",
        "training_mode": "lora",
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "image_size": 224,
        "classes": ["cat", "dog"],
    }


def _write_lora_checkpoint(job_id: str):
    config = _config()
    torch.manual_seed(5)
    model = apply_classification_lora(_Classifier(), config["model_name"], config)
    checkpoint = {
        "epoch": 1,
        "model_state_dict": classification_lora_state_dict(model),
        "checkpoint_format": LORA_CHECKPOINT_FORMAT,
        "checkpoint_format_version": CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
        "adapter_metadata": classification_lora_metadata(config["model_name"], config),
        "classes": config["classes"],
        "class_to_idx": {"cat": 0, "dog": 1},
        "config": config,
        "resolved_preprocessing": {"evaluation": "Resize(224)"},
        "best_metric_name": "val_loss",
        "best_metric_value": 0.2,
    }
    destination = paths.best_model_path(job_id)
    torch.save(checkpoint, destination)
    return model, checkpoint


def test_lora_bundle_is_self_describing_and_uses_safetensors(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    _, checkpoint = _write_lora_checkpoint("adapter-job")

    bundle = ensure_lora_adapter_bundle("adapter-job")

    with ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "README.txt",
            "adapter_config.json",
            "adapter_model.safetensors",
            "classes.json",
            "manifest.json",
            "metrics.json",
            "preprocessing.json",
            "training_config.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        adapter_state = load_safetensors(archive.read("adapter_model.safetensors"))

    assert manifest["standalone"] is False
    assert manifest["required_base_model"].startswith("timm:")
    assert set(adapter_state) == set(checkpoint["model_state_dict"])


def test_lora_manifest_exposes_adapter_merged_model_and_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    _write_lora_checkpoint("manifest-job")
    hpo_path = paths.hpo_config_path("manifest-job")
    hpo_path.write_text("{}", encoding="utf-8")

    manifest = artifact_manifest("manifest-job")
    artifacts = {artifact["kind"]: artifact for artifact in manifest["artifacts"]}

    assert manifest["training_mode"] == "lora"
    assert artifacts["lora_adapter_bundle"]["standalone"] is False
    assert artifacts["merged_model"]["standalone"] is True
    assert artifacts["merged_model"]["generated_on_download"] is True
    assert "configuration" in artifacts

    response = artifact_model("manifest-job")
    assert response.headers["content-disposition"].endswith('filename="best_lora_adapter.zip"')


def test_full_model_manifest_keeps_single_standalone_model(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    torch.save(
        {
            "checkpoint_format": "full_model",
            "checkpoint_format_version": CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
            "config": {"training_mode": "fine_tune_pretrained"},
        },
        paths.best_model_path("full-job"),
    )

    manifest = artifact_manifest("full-job")
    model_artifacts = [
        artifact for artifact in manifest["artifacts"] if artifact["kind"] == "full_model"
    ]

    assert len(model_artifacts) == 1
    assert model_artifacts[0]["standalone"] is True
    assert model_artifacts[0]["filename"] == "best_model.pth"


def test_merged_export_is_a_loadable_plain_model(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    adapter_model, _ = _write_lora_checkpoint("merged-job")
    monkeypatch.setattr(
        ClassificationPipeline,
        "_restore_checkpoint_model",
        staticmethod(lambda *_args, **_kwargs: adapter_model),
    )
    inputs = torch.randn(2, 4)
    adapter_model.eval()
    expected = adapter_model(inputs).detach()

    merged_path = ensure_merged_lora_model("merged-job")
    merged_checkpoint = torch.load(merged_path, map_location="cpu")
    restored = _Classifier()
    restored.load_state_dict(merged_checkpoint["model_state_dict"])
    restored.eval()

    assert merged_checkpoint["checkpoint_format"] == "full_model"
    assert merged_checkpoint["config"]["training_mode"] == "fine_tune_pretrained"
    assert torch.allclose(expected, restored(inputs))
