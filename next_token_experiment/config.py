from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TaskConfig:
    task_name: str = "next_token_prediction"
    description: str = "Predict the probability distribution of the next discrete musical token."


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    root_dir: str
    extensions: tuple[str, ...] = (".mxl", ".musicxml", ".xml")
    min_events_per_piece: int = 32
    group_by_canonical_work: bool = True


@dataclass(frozen=True)
class PreprocessingConfig:
    include_rests: bool = False
    prefer_treble: bool = True
    transpose_to_canonical_key: bool = False
    include_metadata_features: bool = False


@dataclass(frozen=True)
class RepresentationConfig:
    primary: str = "pitch_class"
    alternative: str | None = "pitch_class_duration"
    duration_bins: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class WindowConfig:
    max_context_length: int = 128
    min_window_length: int = 32
    train_stride: int = 64
    eval_stride: int = 128


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 7


@dataclass(frozen=True)
class MetricsConfig:
    primary: str = "nll_per_token"
    secondary: tuple[str, ...] = (
        "perplexity",
        "accuracy",
        "train_wall_clock_s",
        "eval_wall_clock_s",
    )


@dataclass(frozen=True)
class HardwareConfig:
    target_device: str = "cpu"
    cpu_threads: int = 12
    memory_gib: int = 15
    gpu_required: bool = False
    dataloader_workers: int = 0
    pin_memory: bool = False
    precision: str = "fp32"
    deterministic: bool = True


@dataclass(frozen=True)
class FiniteHMMConfig:
    candidate_num_states: tuple[int, ...] = (8, 12, 16)
    max_em_iterations: int = 50
    tolerance: float = 1e-4


@dataclass(frozen=True)
class HDPHMMConfig:
    truncation_level: int = 20
    alpha: float = 8.0
    alpha0: float = 4.0
    gamma: float = 2.0
    eta: float = 1.0
    kappa: float = 0.0
    n_iters: int = 100
    burn_in: int = 50
    seed: int = 7


@dataclass(frozen=True)
class TransformerConfig:
    architecture: str = "decoder_only"
    n_layers: int = 3
    d_model: int = 128
    n_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 25
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    min_learning_rate: float = 1e-5
    tie_input_output_embeddings: bool = True


@dataclass(frozen=True)
class StorageConfig:
    results_root: str = "artifacts/next_token_experiment/results"
    save_piece_metrics: bool = True
    save_validation_curves: bool = True
    save_exclusion_manifest: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    task: TaskConfig
    corpus: CorpusConfig
    preprocessing: PreprocessingConfig
    representation: RepresentationConfig
    windows: WindowConfig
    split: SplitConfig
    metrics: MetricsConfig
    hardware: HardwareConfig
    finite_hmm: FiniteHMMConfig
    hdp_hmm: HDPHMMConfig
    transformer: TransformerConfig
    storage: StorageConfig
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)
