from __future__ import annotations

import json
from zipfile import ZipFile

import torch
from safetensors.torch import load as load_safetensors
from ultralytics import RTDETR

from cvmodellearning import paths
from cvmodellearning.models.detection_models.rtdetr_artifacts import (
    ensure_merged_rtdetr_lora_model,
    ensure_rtdetr_lora_adapter_bundle,
)
from cvmodellearning.models.detection_models.rtdetr_lora import (
    HEAD_MODULE_MARKERS,
    LoRALinear,
    apply_rtdetr_lora,
)
from routers.artifacts import artifact_manifest, artifact_model


def _config() -> dict:
    return {
        "task_type": "detection",
        "model_name": "rtdetr_hgnetv2_l",
        "model_weights": "default",
        "training_mode": "lora",
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "lora_target_profile": "decoder_attention",
        "train_detection_head": True,
        "classes": ["car"],
    }


def _write_checkpoint(job_id: str) -> None:
    wrapper = RTDETR("rtdetr-l.yaml")
    apply_rtdetr_lora(wrapper.model, _config())
    for name, parameter in wrapper.model.named_parameters():
        if ".lora_B." in name:
            parameter.data.normal_(std=0.01)
        elif any(marker in name for marker in HEAD_MODULE_MARKERS) and parameter.is_floating_point():
            parameter.data.add_(0.001)
    wrapper.save(paths.best_yolo_model_path(job_id))
    paths.hpo_config_path(job_id).write_text(json.dumps(_config()), encoding="utf-8")


def test_rtdetr_adapter_bundle_contains_only_adapter_and_heads(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    _write_checkpoint("adapter-job")

    bundle = ensure_rtdetr_lora_adapter_bundle("adapter-job")
    with ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "README.txt",
            "adapter_config.json",
            "adapter_model.safetensors",
            "classes.json",
            "manifest.json",
            "training_config.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        adapter_config = json.loads(archive.read("adapter_config.json"))
        state = load_safetensors(archive.read("adapter_model.safetensors"))

    assert manifest["standalone"] is False
    assert manifest["required_base_model"] == "rtdetr-l.pt"
    assert manifest["classes"] == ["car"]
    assert len(adapter_config["resolved_target_modules"]) == 12
    assert all(
        ".lora_A." in name
        or ".lora_B." in name
        or any(marker in name for marker in HEAD_MODULE_MARKERS)
        for name in state
    )
    assert any(".lora_A." in name for name in state)
    assert any("dec_score_head" in name for name in state)


def test_merged_rtdetr_checkpoint_is_plain_and_numerically_equivalent(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    _write_checkpoint("merged-job")
    source = RTDETR(str(paths.best_yolo_model_path("merged-job")))
    target = "model.28.decoder.layers.0.cross_attn.output_proj"
    inputs = torch.randn(2, 5, 256)
    expected = source.model.get_submodule(target)(inputs).detach()

    merged_path = ensure_merged_rtdetr_lora_model("merged-job")
    restored = RTDETR(str(merged_path))

    assert not any(isinstance(module, LoRALinear) for module in restored.model.modules())
    actual = restored.model.get_submodule(target)(inputs).detach()
    assert torch.allclose(expected, actual, atol=2e-3, rtol=2e-3)


def test_manifest_exposes_rtdetr_adapter_and_merged_download(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    _write_checkpoint("manifest-job")

    manifest = artifact_manifest("manifest-job")
    artifacts = {artifact["kind"]: artifact for artifact in manifest["artifacts"]}

    assert manifest["training_mode"] == "lora"
    assert artifacts["lora_adapter_bundle"]["standalone"] is False
    assert artifacts["merged_model"]["standalone"] is True
    assert artifacts["merged_model"]["filename"] == "best_merged_model.pt"
    response = artifact_model("manifest-job")
    assert response.headers["content-disposition"].endswith('filename="best_merged_model.pt"')
