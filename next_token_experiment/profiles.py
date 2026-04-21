from __future__ import annotations

from dataclasses import replace

from .config import HardwareConfig, StorageConfig, TransformerConfig
from .protocol import build_default_experiment_config

PROFILE_DESCRIPTIONS = {
    "cpu_baseline": "Baseline reproducible y comparable para CPU.",
    "gpu_extended": "Perfil extendido para una GPU decente en la misma tarea.",
}


def list_profiles() -> tuple[str, ...]:
    return tuple(PROFILE_DESCRIPTIONS.keys())


def profile_requires_scope_validation(profile_name: str) -> bool:
    return profile_name == "cpu_baseline"


def build_profile_config(
    profile_name: str,
    *,
    corpus_root: str | None = None,
    results_root: str | None = None,
):
    config = build_default_experiment_config()

    if corpus_root is not None:
        config = replace(config, corpus=replace(config.corpus, root_dir=corpus_root))

    resolved_results_root = results_root or config.storage.results_root
    config = replace(config, storage=StorageConfig(results_root=resolved_results_root))

    if profile_name == "cpu_baseline":
        return replace(
            config,
            hardware=HardwareConfig(
                target_device="cpu",
                cpu_threads=4,
                memory_gib=config.hardware.memory_gib,
                gpu_required=False,
                dataloader_workers=0,
                pin_memory=False,
                precision="fp32",
                deterministic=True,
            ),
            transformer=TransformerConfig(
                architecture="decoder_only",
                n_layers=3,
                d_model=128,
                n_heads=4,
                ff_dim=256,
                dropout=0.1,
                learning_rate=3e-4,
                weight_decay=0.01,
                batch_size=16,
                max_epochs=12,
                early_stopping_patience=4,
                gradient_accumulation_steps=1,
                grad_clip_norm=1.0,
                label_smoothing=0.0,
                lr_scheduler_factor=0.5,
                lr_scheduler_patience=2,
                min_learning_rate=1e-5,
                tie_input_output_embeddings=True,
            ),
            notes=config.notes + ("profile:cpu_baseline", PROFILE_DESCRIPTIONS["cpu_baseline"]),
        )

    if profile_name == "gpu_extended":
        return replace(
            config,
            hardware=HardwareConfig(
                target_device="cuda",
                cpu_threads=8,
                memory_gib=config.hardware.memory_gib,
                gpu_required=True,
                dataloader_workers=2,
                pin_memory=True,
                precision="bf16",
                deterministic=False,
            ),
            transformer=TransformerConfig(
                architecture="decoder_only",
                n_layers=4,
                d_model=256,
                n_heads=8,
                ff_dim=512,
                dropout=0.1,
                learning_rate=2e-4,
                weight_decay=0.02,
                batch_size=64,
                max_epochs=40,
                early_stopping_patience=8,
                gradient_accumulation_steps=1,
                grad_clip_norm=1.0,
                label_smoothing=0.05,
                lr_scheduler_factor=0.5,
                lr_scheduler_patience=3,
                min_learning_rate=1e-5,
                tie_input_output_embeddings=True,
            ),
            notes=config.notes + ("profile:gpu_extended", PROFILE_DESCRIPTIONS["gpu_extended"]),
        )

    raise ValueError(f"Unknown profile: {profile_name}")
