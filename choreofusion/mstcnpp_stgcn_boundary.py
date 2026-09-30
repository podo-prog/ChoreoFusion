from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .annotations import DEFAULT_LAYER
from .boundary import (
    PALETTE,
    _annotations_from_boundaries,
    _load_raw_motion,
    _template_payload,
    decode_boundaries,
    match_training_pairs,
    motion_features,
    read_boundaries,
    soft_targets,
)

SAMPLE_RATE = 120.0
TARGET_SIGMA_SECONDS = 0.075
MIN_BOUNDARY_GAP_SECONDS = 0.375
BOUNDARY_OFFSET_SECONDS = 0.0
SCORE_THRESHOLD = 0.30
SEED = 17
MS_CHANNELS = 48
MS_LAYERS = 7
MS_STAGES = 4
STGCN_CHANNELS = (32, 48, 64)
STGCN_TEMPORAL_DILATIONS = (1, 2, 4)
STGCN_KERNEL_SIZE = 9


@dataclass
class MotionSequence:
    clip_id: str
    modality: str
    path: Path
    times: np.ndarray
    features: np.ndarray
    duration: float
    boundaries: list[float]
    target: np.ndarray


def _read_joint_parents(path: Path, joint_names: list[str]) -> list[int]:
    index = {name.casefold(): i for i, name in enumerate(joint_names)}
    parents: list[int | None] = [None] * len(joint_names)
    stack: list[int | None] = []
    pending: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*(ROOT|JOINT)\s+([^\s{]+)", line)
        if match:
            name = match.group(2).casefold()
            if name not in index:
                raise ValueError(f"Joint {match.group(2)!r} in {path} is absent from the feature order")
            pending = index[name]
            parents[pending] = next((item for item in reversed(stack) if item is not None), None)
        for char in line:
            if char == "{":
                stack.append(pending)
                pending = None
            elif char == "}" and stack:
                stack.pop()
    if any(parent is None for parent in parents[1:]):
        raise ValueError(f"Could not recover the full BVH joint hierarchy from {path}")
    return [-1 if parent is None else int(parent) for parent in parents]


def _graph_partitions(parent_indices: list[int]) -> torch.Tensor:
    count = len(parent_indices)
    identity = np.eye(count, dtype=np.float32)
    inward = np.zeros((count, count), dtype=np.float32)
    outward = np.zeros((count, count), dtype=np.float32)
    for child, parent in enumerate(parent_indices):
        if parent < 0:
            continue
        inward[child, parent] = 1.0
        outward[parent, child] = 1.0
    for matrix in (inward, outward):
        degree = matrix.sum(axis=0, keepdims=True)
        matrix /= np.maximum(degree, 1.0)
    return torch.from_numpy(np.stack((identity, inward, outward)))


def load_sequences(data_dir: str | Path) -> tuple[list[MotionSequence], list[str], list[int]]:
    pairs = match_training_pairs(data_dir)
    joint_names, _, _, _, _ = _load_raw_motion(pairs[0]["bvh"])
    parents = _read_joint_parents(pairs[0]["bvh"], joint_names)
    sequences: list[MotionSequence] = []
    for pair in pairs:
        payload = json.loads(pair["json"].read_text(encoding="utf-8"))
        json_duration = float(payload.get("videoDuration") or 0.0)
        boundaries: list[float] | None = None
        effective_duration = 0.0
        for modality in ("bvh", "fbx"):
            path = pair[modality]
            times, features, duration = motion_features(
                path,
                joint_names,
                json_duration or None,
                sample_rate=SAMPLE_RATE,
            )
            local_duration = min(duration, json_duration or duration)
            if boundaries is None:
                boundaries, _ = read_boundaries(pair["json"], local_duration, DEFAULT_LAYER)
                effective_duration = local_duration
            local_boundaries = [value for value in boundaries if 0.0 < value < effective_duration]
            target = soft_targets(times, local_boundaries)
            sequences.append(
                MotionSequence(
                    clip_id=pair["key"],
                    modality=modality,
                    path=path,
                    times=times,
                    features=features,
                    duration=effective_duration,
                    boundaries=local_boundaries,
                    target=target,
                )
            )
    return sequences, joint_names, parents


def _fit_scaler(sequences: list[MotionSequence]) -> tuple[np.ndarray, np.ndarray]:
    total = None
    total_sq = None
    count = 0
    for item in sequences:
        values = item.features.astype(np.float64, copy=False)
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            total_sq = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    if not count or total is None or total_sq is None:
        raise ValueError("Cannot fit feature normalization on an empty training split")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0))
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _normalized(item: MotionSequence, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((item.features - mean) / std, -8.0, 8.0).astype(np.float32)


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm(self.conv(x)))
        return F.relu(x + self.dropout(y))


class DualDilatedLayer(nn.Module):
    def __init__(self, channels: int, local_dilation: int, global_dilation: int, dropout: float = 0.15):
        super().__init__()
        self.local = nn.Conv1d(channels, channels, kernel_size=3, padding=local_dilation, dilation=local_dilation)
        self.global_context = nn.Conv1d(channels, channels, kernel_size=3, padding=global_dilation, dilation=global_dilation)
        self.mix = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((self.local(x), self.global_context(x)), dim=1)
        y = F.relu(self.norm(self.mix(joined)))
        return F.relu(x + self.dropout(y))


class PredictionGenerationStage(nn.Module):
    def __init__(self, input_dim: int, channels: int = MS_CHANNELS, layers: int = MS_LAYERS):
        super().__init__()
        self.project = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DualDilatedLayer(channels, 2**i, 2 ** (layers - i - 1))
                for i in range(layers)
            ]
        )
        self.head = nn.Conv1d(channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.project(x))
        for layer in self.layers:
            y = layer(y)
        return self.head(y)


class RefinementStage(nn.Module):
    def __init__(self, channels: int = MS_CHANNELS, layers: int = MS_LAYERS):
        super().__init__()
        self.project = nn.Conv1d(2, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(channels, 2**i) for i in range(layers)]
        )
        self.head = nn.Conv1d(channels, 2, kernel_size=1)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.project(probabilities))
        for layer in self.layers:
            y = layer(y)
        return self.head(y)


class MSTCNPPBoundary(nn.Module):
    """MS-TCN++-style multi-stage boundary confidence network for unlabeled intervals."""

    def __init__(
        self,
        input_dim: int,
        channels: int = MS_CHANNELS,
        layers: int = MS_LAYERS,
        stages: int = MS_STAGES,
    ):
        super().__init__()
        if stages < 2:
            raise ValueError("The multi-stage model needs at least two stages")
        self.generator = PredictionGenerationStage(input_dim, channels, layers)
        self.refiners = nn.ModuleList(
            [RefinementStage(channels, layers) for _ in range(stages - 1)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        logits = self.generator(x)
        outputs = [logits]
        for refiner in self.refiners:
            logits = refiner(torch.softmax(logits, dim=1))
            outputs.append(logits)
        return outputs


class STGCNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        partitions: torch.Tensor,
        temporal_dilation: int,
        kernel_size: int = STGCN_KERNEL_SIZE,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.register_buffer("partitions", partitions.clone())
        self.spatial = nn.ModuleList(
            [nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False) for _ in range(len(partitions))]
        )
        self.spatial_norm = nn.BatchNorm2d(output_channels)
        padding = (kernel_size // 2) * temporal_dilation
        self.temporal = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            dilation=(temporal_dilation, 1),
            bias=False,
        )
        self.temporal_norm = nn.BatchNorm2d(output_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        supports = [
            torch.einsum("bctv,vw->bctw", x, adjacency)
            for adjacency in self.partitions
        ]
        spatial = sum(layer(value) for layer, value in zip(self.spatial, supports, strict=True))
        y = F.relu(self.spatial_norm(spatial))
        y = self.temporal_norm(self.temporal(y))
        return F.relu(self.residual(x) + self.dropout(y))


class STGCNBoundaryVerifier(nn.Module):
    """ST-GCN adapted to dense boundary confidence, not action-name or quality classification."""

    def __init__(
        self,
        partitions: torch.Tensor,
        input_channels: int = 14,
        channels: tuple[int, ...] = STGCN_CHANNELS,
        temporal_dilations: tuple[int, ...] = STGCN_TEMPORAL_DILATIONS,
    ):
        super().__init__()
        if len(channels) != len(temporal_dilations):
            raise ValueError("ST-GCN channel and temporal dilation counts must match")
        self.blocks = nn.ModuleList()
        current = input_channels
        for width, dilation in zip(channels, temporal_dilations, strict=True):
            self.blocks.append(STGCNBlock(current, width, partitions, dilation))
            current = width
        self.head = nn.Conv2d(current, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x).mean(dim=-1)


def _frame_loss(logits: torch.Tensor, target: torch.Tensor, smooth_weight: float) -> torch.Tensor:
    target = target.reshape(1, -1).clamp(0.0, 1.0)
    soft_labels = torch.stack((1.0 - target, target), dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    frame_loss = -(soft_labels * log_probs).sum(dim=1)
    weights = 1.0 + 4.0 * target
    classification = (frame_loss * weights).sum() / weights.sum().clamp_min(1.0)
    if logits.shape[-1] > 1 and smooth_weight > 0:
        changes = (log_probs[:, :, 1:] - log_probs[:, :, :-1]).square()
        smoothing = changes.clamp(max=16.0).mean()
    else:
        smoothing = classification * 0.0
    return classification + smooth_weight * smoothing


def _graph_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray, root_index: int = 0) -> torch.Tensor:
    normalized = np.clip((features - mean) / std, -8.0, 8.0).astype(np.float32)
    if normalized.shape[1] != 51 * 8 + 6:
        raise ValueError(f"Expected 51x8 joint features plus six root features, found {normalized.shape[1]}")
    time = len(normalized)
    joints = normalized[:, :51 * 8].reshape(time, 51, 8)
    graph_input = np.zeros((time, 51, 14), dtype=np.float32)
    graph_input[:, :, :8] = joints
    graph_input[:, root_index, 8:] = normalized[:, 51 * 8:]
    return torch.from_numpy(graph_input.transpose(2, 0, 1)[None])


def _network_input(
    kind: str,
    item: MotionSequence,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if kind == "ms_tcnpp":
        values = _normalized(item, mean, std)
        return torch.from_numpy(values.T[None]).to(device)
    return _graph_features(item.features, mean, std).to(device)


def _model_logits(model: nn.Module, kind: str, item: MotionSequence, mean: np.ndarray, std: np.ndarray, device: torch.device):
    x = _network_input(kind, item, mean, std, device)
    return model(x)


def _sequence_loss(model: nn.Module, kind: str, item: MotionSequence, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    target = torch.from_numpy(item.target).to(device)
    if kind == "ms_tcnpp":
        outputs = _model_logits(model, kind, item, mean, std, device)
        losses = [_frame_loss(logits, target, 0.10) for logits in outputs]
        return torch.stack(losses).sum()
    logits = _model_logits(model, kind, item, mean, std, device)
    return _frame_loss(logits, target, 0.05)


def _train_epoch(
    model: nn.Module,
    kind: str,
    sequences: list[MotionSequence],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> float:
    training = optimizer is not None
    model.train(training)
    ordered = list(sequences)
    if training:
        random.Random(seed).shuffle(ordered)
    losses = []
    for item in ordered:
        with torch.set_grad_enabled(training):
            loss = _sequence_loss(model, kind, item, mean, std, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _new_model(kind: str, input_dim: int, partitions: torch.Tensor) -> nn.Module:
    if kind == "ms_tcnpp":
        return MSTCNPPBoundary(input_dim)
    return STGCNBoundaryVerifier(partitions)


def _fit_with_validation(
    kind: str,
    train_sequences: list[MotionSequence],
    validation_sequences: list[MotionSequence],
    input_dim: int,
    partitions: torch.Tensor,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, int, list[dict[str, float]]]:
    mean, std = _fit_scaler(train_sequences)
    torch.manual_seed(SEED)
    model = _new_model(kind, input_dim, partitions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model, kind, train_sequences, mean, std, device, optimizer, SEED + epoch
        )
        validation_loss = _train_epoch(
            model, kind, validation_sequences, mean, std, device, None, SEED
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(
            f"{kind} epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{kind} did not produce a usable checkpoint")
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, history


def _refit_all(
    kind: str,
    sequences: list[MotionSequence],
    input_dim: int,
    partitions: torch.Tensor,
    epochs: int,
    device: torch.device,
    seed_offset: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray, list[float]]:
    mean, std = _fit_scaler(sequences)
    torch.manual_seed(SEED + seed_offset)
    model = _new_model(kind, input_dim, partitions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history = []
    for epoch in range(1, max(1, epochs) + 1):
        loss = _train_epoch(
            model,
            kind,
            sequences,
            mean,
            std,
            device,
            optimizer,
            SEED + seed_offset + epoch,
        )
        history.append(loss)
        print(f"{kind} full-data epoch {epoch}/{max(1, epochs)} loss={loss:.4f}", flush=True)
    return model, mean, std, history


def train_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 24,
    patience: int = 5,
) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading 120Hz BVH/FBX features on {device}...", flush=True)
    sequences, joint_names, parents = load_sequences(data_dir)
    clip_ids = sorted({item.clip_id for item in sequences})
    if len(clip_ids) < 10:
        raise ValueError(f"Need at least 10 independent songs for clip-level validation, found {len(clip_ids)}")
    order = list(np.random.default_rng(SEED).permutation(clip_ids))
    heldout_ids = set(order[:max(3, round(len(order) * 0.2))])
    rest = [clip_id for clip_id in order if clip_id not in heldout_ids]
    validation_ids = set(rest[:max(3, round(len(rest) * 0.2))])
    train_ids = set(rest) - validation_ids
    train_sequences = [item for item in sequences if item.clip_id in train_ids]
    validation_sequences = [item for item in sequences if item.clip_id in validation_ids]
    print(
        f"Loaded {len(clip_ids)} songs / {len(sequences)} sequences; "
        f"train={len(train_ids)}, validation={len(validation_ids)}, heldout={len(heldout_ids)}.",
        flush=True,
    )
    partitions = _graph_partitions(parents)
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, nn.Module] = {}
    checkpoint_paths = {
        "ms_tcnpp": model_dir / "ms_tcnpp_boundary_120hz_29songs.pt",
        "st_gcn": model_dir / "st_gcn_boundary_verifier_120hz_29songs.pt",
    }
    report: dict[str, Any] = {
        "schema": "choreofusion.ms-tcnpp-stgcn-boundary-report.v1",
        "sample_rate_hz": SAMPLE_RATE,
        "training_clips": len(clip_ids),
        "training_sequences": len(sequences),
        "input_feature_dim": int(sequences[0].features.shape[1]),
        "joint_count": len(joint_names),
        "annotation_text_used": False,
        "annotation_tags_used": False,
        "label_target": "frame-level generic boundary confidence from manually segmented intervals",
        "action_type_or_quality_classification": "not trained; no such labels in the dataset",
        "split": {
            "train_clips": sorted(train_ids),
            "validation_clips": sorted(validation_ids),
            "heldout_clips": sorted(heldout_ids),
        },
        "heldout_test_score": "not computed",
        "device": str(device),
        "models": {},
        "limitations": [
            "MS-TCN++ is adapted from framewise action classes to a binary soft boundary-confidence target because all segment labels are blank.",
            "ST-GCN is adapted to score generic temporal boundaries; it does not classify choreography names or execution quality.",
            "FACT is a music-conditioned 3D dance-generation model rather than an interval detector; SlowFast requires original video frames, which are not part of these inputs.",
            "ChillKill is excluded from training and its test annotation JSON has no ground-truth segments; the output is for manual review.",
        ],
    }
    configurations = [
        ("ms_tcnpp", "MS-TCN++", 100),
        ("st_gcn", "ST-GCN boundary verifier", 200),
    ]
    for kind, display_name, offset in configurations:
        _, _, _, best_epoch, history = _fit_with_validation(
            kind,
            train_sequences,
            validation_sequences,
            sequences[0].features.shape[1],
            partitions,
            epochs,
            patience,
            device,
        )
        final_model, mean, std, full_history = _refit_all(
            kind,
            sequences,
            sequences[0].features.shape[1],
            partitions,
            best_epoch,
            device,
            offset,
        )
        models[kind] = final_model.eval()
        checkpoint = {
            "schema": f"choreofusion.{kind}-motion-boundary.v1",
            "model_type": kind,
            "state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
            "sample_rate_hz": SAMPLE_RATE,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "joint_names": joint_names,
            "parent_indices": parents,
            "graph_partitions": partitions.tolist(),
            "target_sigma_seconds": TARGET_SIGMA_SECONDS,
            "minimum_boundary_gap_seconds": MIN_BOUNDARY_GAP_SECONDS,
            "boundary_offset_seconds": BOUNDARY_OFFSET_SECONDS,
            "score_threshold": SCORE_THRESHOLD,
            "metadata": {
                "training_clips": len(clip_ids),
                "training_sequences": len(sequences),
                "labels_used": "blank interval boundaries only",
                "annotation_text_used": False,
                "device": str(device),
            },
        }
        if kind == "ms_tcnpp":
            checkpoint.update({
                "channels": MS_CHANNELS,
                "layers_per_stage": MS_LAYERS,
                "stages": MS_STAGES,
                "first_stage": "dual dilated local/global temporal residual layers",
                "refinement_stages": "three dilated temporal residual stages",
            })
        else:
            checkpoint.update({
                "input_node_channels": 14,
                "channels": list(STGCN_CHANNELS),
                "temporal_dilations": list(STGCN_TEMPORAL_DILATIONS),
                "temporal_kernel_size": STGCN_KERNEL_SIZE,
            })
        torch.save(checkpoint, checkpoint_paths[kind])
        report["models"][kind] = {
            "display_name": display_name,
            "checkpoint": str(checkpoint_paths[kind]),
            "best_epoch": best_epoch,
            "validation_history": history,
            "full_data_refit_losses": full_history,
        }
        print(f"Saved {display_name} model to {checkpoint_paths[kind]}", flush=True)
    report_path = report_dir / "training_report_ms_tcnpp_st_gcn_120hz_29songs.json"
    report["models"]["ms_tcnpp"]["checkpoint"] = str(checkpoint_paths["ms_tcnpp"])
    report["models"]["st_gcn"]["checkpoint"] = str(checkpoint_paths["st_gcn"])
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _load_model(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    kind = checkpoint["model_type"]
    if kind == "ms_tcnpp":
        model = MSTCNPPBoundary(
            input_dim=len(checkpoint["mean"]),
            channels=int(checkpoint["channels"]),
            layers=int(checkpoint["layers_per_stage"]),
            stages=int(checkpoint["stages"]),
        )
    elif kind == "st_gcn":
        model = STGCNBoundaryVerifier(
            torch.as_tensor(checkpoint["graph_partitions"], dtype=torch.float32),
            input_channels=int(checkpoint["input_node_channels"]),
            channels=tuple(checkpoint["channels"]),
            temporal_dilations=tuple(checkpoint["temporal_dilations"]),
        )
    else:
        raise ValueError(f"Unsupported boundary model type {kind!r}")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_pipeline(
    ms_model_path: str | Path,
    st_model_path: str | Path,
    inputs: list[str | Path],
    template_path: str | Path,
    output_path: str | Path,
    threshold: float = SCORE_THRESHOLD,
    min_gap: float = MIN_BOUNDARY_GAP_SECONDS,
    title: str | None = None,
    boundary_offset_seconds: float = BOUNDARY_OFFSET_SECONDS,
) -> list[dict[str, Any]]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ms_model, ms_ck = _load_model(ms_model_path, device)
    st_model, st_ck = _load_model(st_model_path, device)
    joint_names = ms_ck["joint_names"]
    if [name.casefold() for name in joint_names] != [name.casefold() for name in st_ck["joint_names"]]:
        raise ValueError("MS-TCN++ and ST-GCN checkpoints have different skeleton definitions")
    paths = [Path(item) for item in inputs]
    predicted = []
    durations = []
    for path in paths:
        times, features, duration = motion_features(
            path,
            joint_names,
            sample_rate=float(ms_ck["sample_rate_hz"]),
        )
        durations.append(duration)
        item = MotionSequence(
            clip_id=path.stem,
            modality=path.suffix.casefold().lstrip("."),
            path=path,
            times=times,
            features=features,
            duration=duration,
            boundaries=[],
            target=np.zeros(len(times), dtype=np.float32),
        )
        with torch.no_grad():
            ms_logits = _model_logits(
                ms_model,
                "ms_tcnpp",
                item,
                np.asarray(ms_ck["mean"], dtype=np.float32),
                np.asarray(ms_ck["std"], dtype=np.float32),
                device,
            )[-1]
            st_logits = _model_logits(
                st_model,
                "st_gcn",
                item,
                np.asarray(st_ck["mean"], dtype=np.float32),
                np.asarray(st_ck["std"], dtype=np.float32),
                device,
            )
            ms_scores = torch.softmax(ms_logits, dim=1)[0, 1].detach().cpu().numpy()
            st_scores = torch.softmax(st_logits, dim=1)[0, 1].detach().cpu().numpy()
        predicted.append((times, 0.5 * ms_scores + 0.5 * st_scores))
    base_times, base_scores = predicted[0]
    aligned = [base_scores]
    for times, scores in predicted[1:]:
        if np.array_equal(times, base_times):
            aligned.append(scores)
        else:
            aligned.append(np.interp(base_times, times, scores, left=float(scores[0]), right=float(scores[-1])))
    fused_scores = np.mean(np.stack(aligned), axis=0)
    duration = max(durations)
    boundaries = decode_boundaries(base_times, fused_scores, threshold, min_gap)
    raw_times = [float(time) for time, _ in boundaries]
    offsets = [0.0] * len(raw_times)
    requested_offset = max(0.0, float(boundary_offset_seconds))
    next_time = duration
    next_offset = 0.0
    for index in range(len(raw_times) - 1, -1, -1):
        following_duration = next_time - raw_times[index]
        room = max(0.0, following_duration + next_offset - min_gap)
        offsets[index] = min(requested_offset, room)
        next_time = raw_times[index]
        next_offset = offsets[index]
    shifted = [time + offset for time, offset in zip(raw_times, offsets)]
    annotations, _ = _annotations_from_boundaries(shifted, duration)
    payload = _template_payload(paths[0], duration, template_path, title)
    layer = next(row for row in payload["layers"] if row.get("name") == DEFAULT_LAYER)
    layer["annotations"] = annotations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MS-TCN++ and ST-GCN boundary models for 120Hz BVH/FBX motion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--patience", type=int, default=5)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--ms-model", required=True)
    predict.add_argument("--st-model", required=True)
    predict.add_argument("--input", action="append", required=True)
    predict.add_argument("--template", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    predict.add_argument("--min-gap", type=float, default=MIN_BOUNDARY_GAP_SECONDS)
    predict.add_argument("--boundary-offset", type=float, default=BOUNDARY_OFFSET_SECONDS)
    predict.add_argument("--title", default=None)
    args = parser.parse_args()
    if args.command == "train":
        train_pipeline(args.data_dir, args.output_dir, args.epochs, args.patience)
    else:
        annotations = predict_pipeline(
            args.ms_model,
            args.st_model,
            args.input,
            args.template,
            args.output,
            args.threshold,
            args.min_gap,
            args.title,
            args.boundary_offset,
        )
        print(json.dumps({"output": args.output, "annotation_count": len(annotations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
