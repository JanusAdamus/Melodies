from __future__ import annotations

from .config import (
    CorpusConfig,
    ExperimentConfig,
    FiniteHMMConfig,
    HDPHMMConfig,
    HardwareConfig,
    MetricsConfig,
    PreprocessingConfig,
    RepresentationConfig,
    SplitConfig,
    StorageConfig,
    TaskConfig,
    TransformerConfig,
    WindowConfig,
)

SUPPORTED_REPRESENTATIONS = {"pitch_class", "pitch_class_duration", "event_pitch_duration_metrical"}
PRIMARY_VOCAB_SIZE = {
    "pitch_class": 12,
    "pitch_class_duration": 60,
    "event_pitch_duration_metrical": 325,
}


def estimate_transformer_parameter_count(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    ff_dim: int,
    max_positions: int,
    tie_embeddings: bool = True,
) -> int:
    """Approximate parameter count for a small decoder-only Transformer."""

    token_embeddings = vocab_size * d_model
    positional_embeddings = max_positions * d_model
    attention_weights = 4 * d_model * d_model
    feedforward_weights = 2 * d_model * ff_dim
    layer_norm_and_bias = (8 * d_model) + (ff_dim + vocab_size)
    per_layer = attention_weights + feedforward_weights + layer_norm_and_bias
    output_projection = 0 if tie_embeddings else (d_model * vocab_size + vocab_size)
    return int(token_embeddings + positional_embeddings + n_layers * per_layer + output_projection)


def build_default_experiment_config() -> ExperimentConfig:
    """Freeze the minimal protocol proposed for the thesis comparison."""

    return ExperimentConfig(
        task=TaskConfig(),
        corpus=CorpusConfig(
            name="library_scores",
            root_dir="external/library/scores",
        ),
        preprocessing=PreprocessingConfig(),
        representation=RepresentationConfig(),
        windows=WindowConfig(),
        split=SplitConfig(),
        metrics=MetricsConfig(),
        hardware=HardwareConfig(),
        finite_hmm=FiniteHMMConfig(),
        hdp_hmm=HDPHMMConfig(),
        transformer=TransformerConfig(),
        storage=StorageConfig(),
        notes=(
            "Single-corpus experiment for thesis comparability.",
            "Primary representation is pitch_class.",
            "Transformer remains intentionally small and CPU-feasible.",
            "Any richer representation or larger model belongs to the research track, not the thesis baseline.",
            "Generated results are stored under artifacts/.",
        ),
    )


def validate_experiment_scope(config: ExperimentConfig) -> list[str]:
    """Return protocol violations instead of silently allowing scope creep."""

    issues: list[str] = []

    if config.task.task_name != "next_token_prediction":
        issues.append("The experiment must stay focused on next-token prediction.")

    if config.representation.primary not in SUPPORTED_REPRESENTATIONS:
        issues.append("Primary representation is not supported.")

    if config.representation.primary != "pitch_class":
        issues.append("The first comparison must keep pitch_class as the main representation.")

    if config.representation.alternative not in {None, "pitch_class_duration", "pitch_class"}:
        issues.append("Only pitch_class_duration or pitch_class are allowed as secondary representations in the baseline track.")

    if config.preprocessing.include_rests:
        issues.append("Version 1 should exclude rests to keep the comparison simple.")

    if config.preprocessing.transpose_to_canonical_key:
        issues.append("Version 1 should avoid tonal transposition to keep preprocessing conservative.")

    if config.preprocessing.include_metadata_features:
        issues.append("Metadata features are out of scope for the first experiment.")

    if config.windows.max_context_length > 128:
        issues.append("Context length must remain at or below 128 tokens.")

    if config.windows.min_window_length < 16:
        issues.append("Minimum window length is too small for a meaningful comparison.")

    total_ratio = config.split.train_ratio + config.split.validation_ratio + config.split.test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        issues.append("Data split ratios must sum to 1.0.")

    if config.hardware.gpu_required:
        issues.append("The default experiment must remain CPU-feasible.")

    if config.hardware.target_device not in {"cpu", "auto"}:
        issues.append("The baseline experiment must target cpu or auto, not a GPU-specific device.")

    if config.hardware.precision != "fp32":
        issues.append("The baseline experiment should stay in full precision for CPU reproducibility.")

    if max(config.finite_hmm.candidate_num_states) > 16:
        issues.append("Finite HMM state grid exceeds the agreed complexity cap.")

    if config.hdp_hmm.truncation_level > 24:
        issues.append("HDP-HMM truncation exceeds the agreed complexity cap.")

    if config.transformer.architecture != "decoder_only":
        issues.append("The first Transformer should be decoder-only for simplicity.")

    if config.transformer.attention_implementation != "eager":
        issues.append("The baseline experiment must keep eager attention for strict reproducibility.")

    if config.transformer.use_relative_position_bias:
        issues.append("Relative position bias belongs to the research track, not the thesis baseline.")

    if config.transformer.n_layers < 2 or config.transformer.n_layers > 4:
        issues.append("Transformer depth must stay between 2 and 4 layers.")

    if config.transformer.n_heads != 4:
        issues.append("Transformer must keep 4 attention heads in version 1.")

    if config.transformer.d_model not in {128, 256}:
        issues.append("Transformer embedding size must stay at 128 or 256.")

    vocab_size = PRIMARY_VOCAB_SIZE[config.representation.primary]
    estimated_params = estimate_transformer_parameter_count(
        vocab_size=vocab_size,
        d_model=config.transformer.d_model,
        n_layers=config.transformer.n_layers,
        ff_dim=config.transformer.ff_dim,
        max_positions=config.windows.max_context_length,
        tie_embeddings=config.transformer.tie_input_output_embeddings,
    )
    if estimated_params > 500_000:
        issues.append("Transformer parameter count exceeds the thesis-friendly cap of 500k.")

    if config.transformer.gradient_accumulation_steps < 1:
        issues.append("Gradient accumulation must be at least 1.")

    if config.transformer.label_smoothing < 0.0 or config.transformer.label_smoothing >= 1.0:
        issues.append("Label smoothing must stay in the [0.0, 1.0) interval.")

    return issues
