from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import HardwareConfig
from ..experiment.metrics import perplexity_from_nll, summarize_average
from .base import NextTokenModel


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _clone_state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _length_bucket(n_tokens: int) -> str:
    if n_tokens < 64:
        return "short"
    if n_tokens < 192:
        return "medium"
    if n_tokens < 512:
        return "long"
    return "very_long"


def _safe_average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / count


def _top_k_hits(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    if targets.numel() == 0:
        return 0
    capped_k = max(1, min(int(k), logits.size(-1)))
    topk = torch.topk(logits, k=capped_k, dim=-1).indices
    hits = topk.eq(targets.unsqueeze(-1)).any(dim=-1)
    return int(hits.sum().item())


def _append_slice_stat(target: dict[str, dict[str, float]], key: str, nll_sum: float, n_tokens: int, correct: int, top3_hits: int, top5_hits: int) -> None:
    stats = target.setdefault(
        key,
        {
            "nll_sum": 0.0,
            "n_tokens": 0,
            "correct": 0,
            "top3_hits": 0,
            "top5_hits": 0,
        },
    )
    stats["nll_sum"] += float(nll_sum)
    stats["n_tokens"] += int(n_tokens)
    stats["correct"] += int(correct)
    stats["top3_hits"] += int(top3_hits)
    stats["top5_hits"] += int(top5_hits)


def _finalize_slice_stats(slice_name: str, stats_by_key: dict[str, dict[str, float]]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for key in sorted(stats_by_key):
        stats = stats_by_key[key]
        n_tokens = int(stats["n_tokens"])
        nll_per_token = _safe_average(float(stats["nll_sum"]), n_tokens)
        rows.append(
            {
                "slice_name": slice_name,
                "slice_value": key,
                "n_tokens": n_tokens,
                "nll_per_token": nll_per_token,
                "perplexity": perplexity_from_nll(nll_per_token) if n_tokens > 0 else 0.0,
                "accuracy": _safe_average(float(stats["correct"]), n_tokens),
                "top_3_accuracy": _safe_average(float(stats["top3_hits"]), n_tokens),
                "top_5_accuracy": _safe_average(float(stats["top5_hits"]), n_tokens),
            }
        )
    return rows


@dataclass(frozen=True)
class SmallTransformerStudySpec:
    architecture: str
    n_layers: int
    d_model: int
    n_heads: int
    ff_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    gradient_accumulation_steps: int
    grad_clip_norm: float
    label_smoothing: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    min_learning_rate: float
    tie_input_output_embeddings: bool
    use_relative_position_bias: bool
    relative_attention_num_buckets: int
    relative_attention_max_distance: int
    generation_num_prompts: int
    generation_prompt_length: int
    generation_max_new_tokens: int
    generation_temperature: float
    generation_top_k: int


class RelativePositionBias(nn.Module):
    """T5-style bucketed relative position bias for causal self-attention."""

    def __init__(self, n_heads: int, num_buckets: int = 32, max_distance: int = 256) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.embedding = nn.Embedding(self.num_buckets, self.n_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        distance = (-relative_position).clamp(min=0)
        max_exact = max(1, self.num_buckets // 2)
        is_small = distance < max_exact
        large_distance = distance.float().clamp(min=1.0)
        large_bucket = max_exact + (
            torch.log(large_distance / max_exact)
            / math.log(max(self.max_distance / max_exact, 1.0001))
            * (self.num_buckets - max_exact)
        ).to(torch.long)
        large_bucket = large_bucket.clamp(max=self.num_buckets - 1)
        return torch.where(is_small, distance.to(torch.long), large_bucket)

    def forward(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, device=device).unsqueeze(1)
        memory_position = torch.arange(key_length, device=device).unsqueeze(0)
        relative_position = memory_position - context_position
        buckets = self._relative_position_bucket(relative_position)
        bias = self.embedding(buckets)
        return bias.permute(2, 0, 1).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    """Decoder-only multi-head self-attention with optional relative position bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        use_relative_position_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 256,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.relative_position_bias = (
            RelativePositionBias(
                n_heads=n_heads,
                num_buckets=relative_attention_num_buckets,
                max_distance=relative_attention_max_distance,
            )
            if use_relative_position_bias
            else None
        )
        self._causal_mask: torch.Tensor | None = None

    def _get_causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        mask = self._causal_mask
        if mask is None or mask.device != device or mask.size(0) < sequence_length:
            mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
                diagonal=1,
            )
            self._causal_mask = mask
        return mask[:sequence_length, :sequence_length]

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        causal_mask = self._get_causal_mask(sequence_length, hidden_states.device)
        attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attention_scores = attention_scores.masked_fill(~attention_mask[:, None, None, :], float("-inf"))

        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias(sequence_length, sequence_length, hidden_states.device)

        attention_probs = torch.softmax(attention_scores.float(), dim=-1).to(query.dtype)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        attention_output = self.out_proj(attention_output)
        return attention_output * attention_mask.unsqueeze(-1).to(attention_output.dtype)


class TransformerBlock(nn.Module):
    """Pre-norm causal Transformer block with optional relative position bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        use_relative_position_bias: bool,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_relative_position_bias=use_relative_position_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        attn_input = self.norm1(hidden_states)
        hidden_states = residual + self.dropout1(self.attn(attn_input, attention_mask))

        residual = hidden_states
        ff_input = self.norm2(hidden_states)
        ff_output = self.ff(ff_input)
        hidden_states = residual + self.dropout2(ff_output)
        return hidden_states


class SmallTransformerLM(nn.Module):
    """Small decoder-only Transformer for symbolic next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        max_context_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        tie_input_output_embeddings: bool,
        use_relative_position_bias: bool,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.d_model = d_model
        self.tie_input_output_embeddings = tie_input_output_embeddings
        self.use_relative_position_bias = use_relative_position_bias

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = None if use_relative_position_bias else nn.Embedding(max_context_length, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    use_relative_position_bias=use_relative_position_bias,
                    relative_attention_num_buckets=relative_attention_num_buckets,
                    relative_attention_max_distance=relative_attention_max_distance,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = None if tie_input_output_embeddings else nn.Linear(d_model, vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds model context {self.max_context_length}."
            )

        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = self.final_norm(hidden_states)
        if self.output_projection is None:
            return F.linear(hidden_states, self.token_embedding.weight)
        return self.output_projection(hidden_states)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SmallTransformerNextTokenModel(NextTokenModel):
    """Autoregressive Transformer with stronger evaluation and reproducible generation."""

    def __init__(
        self,
        spec: SmallTransformerStudySpec,
        vocab_size: int,
        bos_token_id: int,
        pad_token_id: int,
        max_context_length: int,
        cpu_threads: int | None = None,
        hardware: HardwareConfig | None = None,
        seed: int = 7,
        piece_metadata_by_id: dict[str, dict[str, object]] | None = None,
        token_rarity_by_id: dict[int, str] | None = None,
    ) -> None:
        base_hardware = hardware or HardwareConfig()
        if cpu_threads is not None:
            base_hardware = replace(base_hardware, cpu_threads=int(cpu_threads))

        self.spec = spec
        self.hardware = base_hardware
        self.vocab_size = vocab_size
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.max_context_length = max_context_length
        self.seed = int(seed)
        self.piece_metadata_by_id = piece_metadata_by_id or {}
        self.token_rarity_by_id = token_rarity_by_id or {}
        self.special_token_ids = {self.bos_token_id, self.pad_token_id}

        torch.set_num_threads(max(1, int(self.hardware.cpu_threads)))
        _set_seed(self.seed, deterministic=bool(self.hardware.deterministic))

        self.device = self._resolve_device(self.hardware)
        self.autocast_dtype, self.actual_precision = self._resolve_precision(self.hardware)
        self.use_grad_scaler = self.device.type == "cuda" and self.autocast_dtype == torch.float16
        try:
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        except Exception:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)
        self.non_blocking_transfers = self.device.type == "cuda" and bool(self.hardware.pin_memory)

        self.model = SmallTransformerLM(
            vocab_size=vocab_size,
            max_context_length=max_context_length,
            d_model=spec.d_model,
            n_layers=spec.n_layers,
            n_heads=spec.n_heads,
            ff_dim=spec.ff_dim,
            dropout=spec.dropout,
            tie_input_output_embeddings=spec.tie_input_output_embeddings,
            use_relative_position_bias=spec.use_relative_position_bias,
            relative_attention_num_buckets=spec.relative_attention_num_buckets,
            relative_attention_max_distance=spec.relative_attention_max_distance,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=spec.lr_scheduler_factor,
            patience=spec.lr_scheduler_patience,
            min_lr=spec.min_learning_rate,
        )
        self.best_state_dict = _clone_state_dict_to_cpu(self.model.state_dict())
        self.best_validation_nll = math.inf
        self.best_epoch = 0

    @staticmethod
    def _resolve_device(hardware: HardwareConfig) -> torch.device:
        target = hardware.target_device.lower()

        if target == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif target == "cpu":
            device = torch.device("cpu")
        elif target == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available in this environment.")
            device = torch.device("cuda")
        elif target == "mps":
            if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
                raise ValueError("MPS was requested but is not available in this environment.")
            device = torch.device("mps")
        else:
            raise ValueError(f"Unsupported target_device: {hardware.target_device}")

        if hardware.gpu_required and device.type == "cpu":
            raise ValueError("A GPU-capable device was required but only CPU is available.")
        return device

    def _resolve_precision(self, hardware: HardwareConfig) -> tuple[torch.dtype | None, str]:
        precision = hardware.precision.lower()
        if precision == "fp32" or self.device.type != "cuda":
            return None, "fp32"
        if precision == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, "bf16"
            return torch.float16, "fp16_fallback_from_bf16"
        if precision == "fp16":
            return torch.float16, "fp16"
        raise ValueError(f"Unsupported precision: {hardware.precision}")

    def describe_runtime(self) -> dict[str, int | bool | str]:
        return {
            "device": str(self.device),
            "device_type": self.device.type,
            "requested_device": self.hardware.target_device,
            "requested_precision": self.hardware.precision,
            "actual_precision": self.actual_precision,
            "cpu_threads": int(self.hardware.cpu_threads),
            "dataloader_workers": int(self.hardware.dataloader_workers),
            "pin_memory": bool(self.hardware.pin_memory),
            "deterministic": bool(self.hardware.deterministic),
            "gradient_accumulation_steps": int(self.spec.gradient_accumulation_steps),
            "effective_batch_size": int(self.spec.batch_size * self.spec.gradient_accumulation_steps),
            "use_relative_position_bias": bool(self.spec.use_relative_position_bias),
            "relative_attention_num_buckets": int(self.spec.relative_attention_num_buckets),
            "relative_attention_max_distance": int(self.spec.relative_attention_max_distance),
            "seed": int(self.seed),
        }

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _move_batch(self, batch: dict) -> dict:
        return {
            "input_ids": batch["input_ids"].to(self.device, non_blocking=self.non_blocking_transfers),
            "target_ids": batch["target_ids"].to(self.device, non_blocking=self.non_blocking_transfers),
            "attention_mask": batch["attention_mask"].to(self.device, non_blocking=self.non_blocking_transfers),
            "piece_ids": batch["piece_ids"],
            "start_indices": batch["start_indices"],
            "stop_indices": batch["stop_indices"],
            "sequence_lengths": batch["sequence_lengths"],
        }

    def _compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.spec.label_smoothing,
        )

    def _optimizer_step(self) -> None:
        if self.use_grad_scaler:
            self.grad_scaler.unscale_(self.optimizer)

        if self.spec.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.spec.grad_clip_norm)

        if self.use_grad_scaler:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    def _run_epoch(self, dataloader, train: bool) -> dict[str, float]:
        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_top3_hits = 0
        total_top5_hits = 0
        n_batches = 0
        optimizer_steps = 0
        accumulation_steps = max(1, int(self.spec.gradient_accumulation_steps))
        start_time = time.perf_counter()

        for batch_index, batch in enumerate(dataloader, start=1):
            batch = self._move_batch(batch)
            with torch.set_grad_enabled(train):
                with self._autocast_context():
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    raw_loss = self._compute_loss(logits, batch["target_ids"])

                if train:
                    loss = raw_loss / accumulation_steps
                    if self.use_grad_scaler:
                        self.grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    should_step = batch_index % accumulation_steps == 0 or batch_index == len(dataloader)
                    if should_step:
                        self._optimizer_step()
                        self.optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1

            target_mask = batch["target_ids"] != -100
            valid_targets = batch["target_ids"][target_mask]
            valid_logits = logits[target_mask]
            total_loss += float(raw_loss.detach().float().cpu()) * int(valid_targets.numel())
            total_tokens += int(valid_targets.numel())
            if valid_targets.numel() > 0:
                predictions = torch.argmax(valid_logits, dim=-1)
                total_correct += int((predictions == valid_targets).sum().item())
                total_top3_hits += _top_k_hits(valid_logits, valid_targets, k=3)
                total_top5_hits += _top_k_hits(valid_logits, valid_targets, k=5)
            n_batches += 1

        elapsed = time.perf_counter() - start_time
        average_nll = summarize_average(total_loss, total_tokens)
        return {
            "loss": average_nll,
            "nll_per_token": average_nll,
            "perplexity": perplexity_from_nll(average_nll),
            "accuracy": summarize_average(float(total_correct), total_tokens),
            "top_3_accuracy": summarize_average(float(total_top3_hits), total_tokens),
            "top_5_accuracy": summarize_average(float(total_top5_hits), total_tokens),
            "n_tokens": float(total_tokens),
            "n_batches": float(n_batches),
            "optimizer_steps": float(optimizer_steps),
            "wall_clock_s": elapsed,
        }

    @torch.no_grad()
    def evaluate(self, test_data) -> dict:
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0
        total_correct = 0
        total_top3_hits = 0
        total_top5_hits = 0
        piece_stats: dict[str, dict[str, float]] = {}
        slice_piece_length: dict[str, dict[str, float]] = {}
        slice_composer: dict[str, dict[str, float]] = {}
        slice_token_rarity: dict[str, dict[str, float]] = {}
        start_time = time.perf_counter()

        for batch in test_data:
            batch = self._move_batch(batch)
            with self._autocast_context():
                logits = self.model(batch["input_ids"], batch["attention_mask"])
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            predictions = torch.argmax(log_probs, dim=-1)
            target_ids = batch["target_ids"]
            valid_mask = target_ids != -100

            batch_indices, time_indices = torch.nonzero(valid_mask, as_tuple=True)
            target_values = target_ids[valid_mask]
            valid_logits = logits[valid_mask]
            token_log_probs = log_probs[batch_indices, time_indices, target_values]
            token_nll = -token_log_probs

            total_nll += float(token_nll.sum().cpu())
            total_tokens += int(target_values.numel())
            total_correct += int((predictions[valid_mask] == target_values).sum().item())
            total_top3_hits += _top_k_hits(valid_logits, target_values, k=3)
            total_top5_hits += _top_k_hits(valid_logits, target_values, k=5)

            token_top3 = torch.topk(valid_logits, k=min(3, valid_logits.size(-1)), dim=-1).indices.eq(target_values.unsqueeze(-1)).any(dim=-1)
            token_top5 = torch.topk(valid_logits, k=min(5, valid_logits.size(-1)), dim=-1).indices.eq(target_values.unsqueeze(-1)).any(dim=-1)
            token_correct = predictions[valid_mask] == target_values

            for token_id, token_loss, correct, top3_hit, top5_hit in zip(
                target_values.tolist(),
                token_nll.tolist(),
                token_correct.tolist(),
                token_top3.tolist(),
                token_top5.tolist(),
            ):
                rarity_bucket = self.token_rarity_by_id.get(int(token_id), "unseen")
                _append_slice_stat(
                    slice_token_rarity,
                    rarity_bucket,
                    nll_sum=float(token_loss),
                    n_tokens=1,
                    correct=int(correct),
                    top3_hits=int(top3_hit),
                    top5_hits=int(top5_hit),
                )

            for local_index, piece_id in enumerate(batch["piece_ids"]):
                local_mask = valid_mask[local_index]
                local_targets = target_ids[local_index][local_mask]
                if local_targets.numel() == 0:
                    continue
                local_log_probs = log_probs[local_index][local_mask]
                local_logits = logits[local_index][local_mask]
                local_predictions = predictions[local_index][local_mask]
                gathered = local_log_probs.gather(1, local_targets.unsqueeze(-1)).squeeze(-1)
                local_top3_hits = _top_k_hits(local_logits, local_targets, k=3)
                local_top5_hits = _top_k_hits(local_logits, local_targets, k=5)
                stats = piece_stats.setdefault(
                    piece_id,
                    {
                        "piece_id": piece_id,
                        "nll_sum": 0.0,
                        "n_tokens": 0,
                        "correct": 0,
                        "top3_hits": 0,
                        "top5_hits": 0,
                    },
                )
                stats["nll_sum"] += float((-gathered).sum().cpu())
                stats["n_tokens"] += int(local_targets.numel())
                stats["correct"] += int((local_predictions == local_targets).sum().item())
                stats["top3_hits"] += int(local_top3_hits)
                stats["top5_hits"] += int(local_top5_hits)

        elapsed = time.perf_counter() - start_time
        summary_nll = summarize_average(total_nll, total_tokens)
        piece_metrics = []
        for piece_id in sorted(piece_stats):
            stats = piece_stats[piece_id]
            metadata = self.piece_metadata_by_id.get(piece_id, {})
            piece_n_tokens = int(stats["n_tokens"])
            piece_nll = summarize_average(stats["nll_sum"], piece_n_tokens)
            composer = str(metadata.get("composer", "unknown"))
            length_bucket = str(metadata.get("length_bucket", _length_bucket(piece_n_tokens)))
            piece_metrics.append(
                {
                    "piece_id": piece_id,
                    "title": metadata.get("title", piece_id),
                    "composer": composer,
                    "length_bucket": length_bucket,
                    "n_tokens": piece_n_tokens,
                    "nll_per_token": piece_nll,
                    "perplexity": perplexity_from_nll(piece_nll),
                    "accuracy": summarize_average(float(stats["correct"]), piece_n_tokens),
                    "top_3_accuracy": summarize_average(float(stats["top3_hits"]), piece_n_tokens),
                    "top_5_accuracy": summarize_average(float(stats["top5_hits"]), piece_n_tokens),
                }
            )
            _append_slice_stat(
                slice_piece_length,
                length_bucket,
                nll_sum=float(stats["nll_sum"]),
                n_tokens=piece_n_tokens,
                correct=int(stats["correct"]),
                top3_hits=int(stats["top3_hits"]),
                top5_hits=int(stats["top5_hits"]),
            )
            _append_slice_stat(
                slice_composer,
                composer,
                nll_sum=float(stats["nll_sum"]),
                n_tokens=piece_n_tokens,
                correct=int(stats["correct"]),
                top3_hits=int(stats["top3_hits"]),
                top5_hits=int(stats["top5_hits"]),
            )

        slice_metrics = {
            "piece_length": _finalize_slice_stats("piece_length", slice_piece_length),
            "composer": _finalize_slice_stats("composer", slice_composer),
            "token_rarity": _finalize_slice_stats("token_rarity", slice_token_rarity),
        }
        return {
            "summary": {
                "nll_per_token": summary_nll,
                "perplexity": perplexity_from_nll(summary_nll),
                "accuracy": summarize_average(float(total_correct), total_tokens),
                "top_3_accuracy": summarize_average(float(total_top3_hits), total_tokens),
                "top_5_accuracy": summarize_average(float(total_top5_hits), total_tokens),
                "n_tokens": total_tokens,
                "eval_wall_clock_s": elapsed,
                "parameter_count": self.model.parameter_count(),
                "best_epoch": self.best_epoch,
                "runtime": self.describe_runtime(),
            },
            "piece_metrics": piece_metrics,
            "slice_metrics": slice_metrics,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        do_sample: bool = True,
    ) -> list[int]:
        if not prompt_tokens:
            raise ValueError("Prompt tokens cannot be empty.")
        if max_new_tokens <= 0:
            return []

        self.model.eval()
        generated = list(prompt_tokens)
        generator_device = "cpu" if self.device.type == "mps" else self.device.type
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.seed if seed is None else int(seed))

        for _ in range(max_new_tokens):
            context_tokens = generated[-(self.max_context_length - 1) :]
            input_ids = torch.tensor(
                [[self.bos_token_id] + context_tokens],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            with self._autocast_context():
                logits = self.model(input_ids, attention_mask)[0, -1].float()

            for token_id in self.special_token_ids:
                logits[token_id] = float("-inf")

            if top_k is not None and 0 < top_k < logits.numel():
                cutoff = torch.topk(logits, k=int(top_k)).values[-1]
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

            if (not do_sample) or temperature <= 0.0:
                next_token = int(torch.argmax(logits).item())
            else:
                scaled_logits = logits / max(float(temperature), 1e-6)
                probabilities = torch.softmax(scaled_logits, dim=-1)
                next_token = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())

            generated.append(next_token)

        return generated[len(prompt_tokens) :]

    def fit(self, train_data, validation_data) -> dict:
        patience = 0
        train_log: list[dict[str, float]] = []

        for epoch in range(1, self.spec.max_epochs + 1):
            train_metrics = self._run_epoch(train_data, train=True)
            validation_metrics = self._run_epoch(validation_data, train=False)
            self.scheduler.step(validation_metrics["nll_per_token"])

            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            log_row = {
                "epoch": epoch,
                "train_nll_per_token": train_metrics["nll_per_token"],
                "train_perplexity": train_metrics["perplexity"],
                "train_accuracy": train_metrics["accuracy"],
                "train_top_3_accuracy": train_metrics["top_3_accuracy"],
                "train_top_5_accuracy": train_metrics["top_5_accuracy"],
                "train_wall_clock_s": train_metrics["wall_clock_s"],
                "train_optimizer_steps": train_metrics["optimizer_steps"],
                "validation_nll_per_token": validation_metrics["nll_per_token"],
                "validation_perplexity": validation_metrics["perplexity"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_top_3_accuracy": validation_metrics["top_3_accuracy"],
                "validation_top_5_accuracy": validation_metrics["top_5_accuracy"],
                "validation_wall_clock_s": validation_metrics["wall_clock_s"],
                "learning_rate": learning_rate,
            }
            train_log.append(log_row)

            if validation_metrics["nll_per_token"] < self.best_validation_nll:
                self.best_validation_nll = validation_metrics["nll_per_token"]
                self.best_state_dict = _clone_state_dict_to_cpu(deepcopy(self.model.state_dict()))
                self.best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= self.spec.early_stopping_patience:
                    break

        self.model.load_state_dict(self.best_state_dict)
        return {
            "train_log": train_log,
            "summary": {
                "best_epoch": self.best_epoch,
                "best_validation_nll": self.best_validation_nll,
                "best_validation_perplexity": perplexity_from_nll(self.best_validation_nll),
                "parameter_count": self.model.parameter_count(),
                "epochs_completed": len(train_log),
                "early_stopped": len(train_log) < self.spec.max_epochs,
                "runtime": self.describe_runtime(),
            },
        }

    def save(self, path) -> None:
        payload = {
            "spec": asdict(self.spec),
            "hardware": asdict(self.hardware),
            "runtime": self.describe_runtime(),
            "state_dict": self.best_state_dict,
            "best_validation_nll": self.best_validation_nll,
            "best_epoch": self.best_epoch,
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "max_context_length": self.max_context_length,
        }
        torch.save(payload, path)
