from __future__ import annotations

import argparse
import json
import math
import random
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
    _load_raw_motion,
    _template_payload,
    match_training_pairs,
    motion_features,
    read_boundaries,
)

SAMPLE_RATE = 120.0
SEED = 17
FEATURE_DIM = 96
ATTENTION_HEADS = 4
ATTENTION_WINDOW = 19
BRANCH_LEVELS = 5
SCORE_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.40
MIN_BOUNDARY_GAP = 0.30
REGRESSION_RANGES = (
    (0.0, 0.20),
    (0.20, 0.40),
    (0.40, 0.75),
    (0.75, 1.25),
    (1.25, 2.00),
    (2.00, float("inf")),
)


@dataclass
class MotionSequence:
    clip_id: str
    modality: str
    features: np.ndarray
    duration: float
    segments: np.ndarray


def _raw_json_segments(json_path: Path, duration: float) -> np.ndarray:
    boundaries, payload = read_boundaries(json_path, duration, DEFAULT_LAYER)
    clipped_duration = min(
        duration,
        float(payload.get("videoDuration") or duration),
    )
    points = [0.0, *[x for x in boundaries if 0.0 < x < clipped_duration], clipped_duration]
    return np.asarray(
        [(left, right) for left, right in zip(points, points[1:]) if right > left],
        dtype=np.float32,
    )


def load_sequences(data_dir: str | Path) -> tuple[list[MotionSequence], list[str]]:
    pairs = match_training_pairs(data_dir)
    joint_names, _, _, _, _ = _load_raw_motion(pairs[0]["bvh"])
    sequences: list[MotionSequence] = []
    for pair in pairs:
        payload = json.loads(pair["json"].read_text(encoding="utf-8"))
        clip_duration = float(payload.get("videoDuration") or 0.0)
        for modality in ("bvh", "fbx"):
            path = pair[modality]
            times, features, duration = motion_features(
                path,
                joint_names,
                clip_duration or None,
                sample_rate=SAMPLE_RATE,
            )
            effective_duration = min(duration, clip_duration or duration)
            segments = _raw_json_segments(pair["json"], effective_duration)
            if features.shape[1] != len(joint_names) * 8 + 6:
                raise ValueError(f"Unexpected feature width {features.shape[1]} for {path}")
            sequences.append(
                MotionSequence(
                    clip_id=pair["key"],
                    modality=modality,
                    features=features,
                    duration=effective_duration,
                    segments=segments,
                )
            )
    return sequences, joint_names


def _fit_scaler(sequences: list[MotionSequence]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = None
    total_sq = None
    for item in sequences:
        x = item.features.astype(np.float64, copy=False)
        if total is None:
            total = np.zeros(x.shape[1], dtype=np.float64)
            total_sq = np.zeros(x.shape[1], dtype=np.float64)
        total += x.sum(axis=0)
        total_sq += np.square(x).sum(axis=0)
        count += len(x)
    if not count or total is None or total_sq is None:
        raise ValueError("Cannot fit normalization without motion frames")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _feature_tensor(
    item: MotionSequence,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    x = np.clip((item.features - mean) / std, -8.0, 8.0).astype(np.float32)
    return torch.from_numpy(x.T[None]).to(device)


def _positional_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros((length, dim), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype).unsqueeze(0)


class LocalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding width must be divisible by the attention head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window if window % 2 else window + 1
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        pad = self.window // 2
        k_pad = F.pad(k.permute(0, 2, 3, 1), (pad, pad))
        v_pad = F.pad(v.permute(0, 2, 3, 1), (pad, pad))
        k_win = k_pad.unfold(-1, self.window, 1).permute(0, 3, 4, 1, 2)
        v_win = v_pad.unfold(-1, self.window, 1).permute(0, 3, 4, 1, 2)

        valid = torch.ones(length, device=x.device, dtype=torch.bool)
        valid = F.pad(valid, (pad, pad), value=False).unfold(0, self.window, 1)
        logits = torch.einsum("bthd,btwhd->bthw", q, k_win) * self.scale
        logits = logits.masked_fill(~valid[None, :, None, :], -1e4)
        weights = self.attention_dropout(torch.softmax(logits, dim=-1))
        attended = torch.einsum("bthw,btwhd->bthd", weights, v_win)
        attended = attended.reshape(batch, length, self.dim)
        return x + self.projection_dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, dropout: float = 0.1):
        super().__init__()
        self.attention = LocalSelfAttention(dim, heads, window, dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        return x + self.mlp(self.norm(x))


class ConvTransformerBackbone(nn.Module):
    """ActionFormer-style convolutional embedding and multi-scale local-attention backbone."""

    def __init__(
        self,
        input_dim: int,
        dim: int = FEATURE_DIM,
        heads: int = ATTENTION_HEADS,
        window: int = ATTENTION_WINDOW,
        branch_levels: int = BRANCH_LEVELS,
    ):
        super().__init__()
        self.embedding = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim if i == 0 else dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.GELU(),
                )
                for i in range(2)
            ]
        )
        self.stem = nn.ModuleList(
            [TransformerBlock(dim, heads, window) for _ in range(2)]
        )
        self.downsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 3, stride=2, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.GELU(),
                )
                for _ in range(branch_levels)
            ]
        )
        self.branch = nn.ModuleList(
            [TransformerBlock(dim, heads, window) for _ in range(branch_levels)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        for layer in self.embedding:
            x = layer(x)
        x = x.transpose(1, 2)
        x = x + _positional_encoding(x.shape[1], x.shape[2], x.device, x.dtype)
        for block in self.stem:
            x = block(x)
        features = [x.transpose(1, 2)]
        x = x.transpose(1, 2)
        for downsample, block in zip(self.downsample, self.branch):
            x = downsample(x)
            x = block(x.transpose(1, 2)).transpose(1, 2)
            features.append(x)
        return features


class TemporalFeaturePyramid(nn.Module):
    def __init__(self, dim: int, levels: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv1d(dim, dim, 1) for _ in range(levels)])
        self.output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.GELU(),
                )
                for _ in range(levels)
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        top = self.lateral[-1](features[-1])
        outputs: list[torch.Tensor] = [top] * len(features)
        outputs[-1] = self.output[-1](top)
        for level in range(len(features) - 2, -1, -1):
            lateral = self.lateral[level](features[level])
            top = lateral + F.interpolate(
                top, size=lateral.shape[-1], mode="linear", align_corners=False
            )
            outputs[level] = self.output[level](top)
        return outputs


class ConvTower(nn.Module):
    def __init__(self, dim: int, layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.GELU(),
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ActionFormerMotion(nn.Module):
    """Single-class ActionFormer-style temporal localization for motion features."""

    def __init__(self, input_dim: int, sample_rate: float = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.strides = [2**level for level in range(BRANCH_LEVELS + 1)]
        levels = len(self.strides)
        self.backbone = ConvTransformerBackbone(input_dim, branch_levels=BRANCH_LEVELS)
        self.neck = TemporalFeaturePyramid(FEATURE_DIM, levels)
        self.cls_tower = ConvTower(FEATURE_DIM)
        self.reg_tower = ConvTower(FEATURE_DIM)
        self.cls_head = nn.Conv1d(FEATURE_DIM, 1, 3, padding=1)
        self.reg_head = nn.Conv1d(FEATURE_DIM, 2, 3, padding=1)
        self.level_scales = nn.Parameter(torch.zeros(levels))
        nn.init.constant_(self.cls_head.bias, -math.log((1.0 - 0.01) / 0.01))
        nn.init.constant_(self.reg_head.bias, -0.3)

    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        pyramid = self.neck(self.backbone(x))
        outputs = []
        for level, (feature, stride) in enumerate(zip(pyramid, self.strides)):
            class_logits = self.cls_head(self.cls_tower(feature)).squeeze(0).squeeze(0)
            raw_offsets = self.reg_head(self.reg_tower(feature)).squeeze(0).transpose(0, 1)
            offsets = F.softplus(raw_offsets) * torch.exp(self.level_scales[level])
            point_times = (
                torch.arange(feature.shape[-1], device=feature.device, dtype=feature.dtype) + 0.5
            ) * (stride / self.sample_rate)
            outputs.append(
                {
                    "logits": class_logits,
                    "offsets": offsets,
                    "times": point_times,
                    "stride": torch.as_tensor(stride, device=feature.device),
                }
            )
        return outputs


def _targets_for_levels(
    outputs: list[dict[str, torch.Tensor]],
    segments: np.ndarray,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    result = []
    gt = torch.as_tensor(segments, dtype=torch.float32, device=device)
    for level, output in enumerate(outputs):
        times = output["times"]
        labels = torch.zeros_like(output["logits"])
        offsets = torch.zeros((len(times), 2), dtype=torch.float32, device=device)
        assigned_len = torch.full_like(times, float("inf"))
        lo, hi = REGRESSION_RANGES[level]
        for start, end in gt:
            left = times - start
            right = end - times
            max_dist = torch.maximum(left, right)
            inside = (left >= 0) & (right > 0)
            in_range = (max_dist >= lo) & (max_dist < hi)
            duration = end - start
            choose = inside & in_range & (duration < assigned_len)
            labels[choose] = 1.0
            offsets[choose, 0] = left[choose]
            offsets[choose, 1] = right[choose]
            assigned_len[choose] = duration
        result.append((labels, offsets))
    return result


def _diou_1d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(min=1e-5)
    target = target.clamp(min=1e-5)
    intersection = torch.minimum(pred[:, 0], target[:, 0]) + torch.minimum(
        pred[:, 1], target[:, 1]
    )
    union = torch.maximum(pred[:, 0], target[:, 0]) + torch.maximum(
        pred[:, 1], target[:, 1]
    )
    iou = intersection / union.clamp(min=1e-5)
    center_delta = ((pred[:, 1] - pred[:, 0]) - (target[:, 1] - target[:, 0])).abs() * 0.5
    enclosing = union.clamp(min=1e-5)
    return 1.0 - iou + (center_delta / enclosing).square()


def _loss(
    outputs: list[dict[str, torch.Tensor]],
    segments: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = _targets_for_levels(outputs, segments, device)
    positive_count = sum(int(label.sum().item()) for label, _ in targets)
    normalizer = max(positive_count, 1)
    cls_total = outputs[0]["logits"].sum() * 0.0
    reg_total = outputs[0]["logits"].sum() * 0.0
    for output, (labels, target_offsets) in zip(outputs, targets):
        logits = output["logits"]
        probabilities = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = probabilities * labels + (1.0 - probabilities) * (1.0 - labels)
        alpha_t = 0.25 * labels + 0.75 * (1.0 - labels)
        cls_total = cls_total + (bce * (1.0 - p_t).square() * alpha_t).sum()
        positive = labels > 0
        if positive.any():
            reg_total = reg_total + _diou_1d(
                output["offsets"][positive], target_offsets[positive]
            ).sum()
    cls_loss = cls_total / normalizer
    reg_loss = reg_total / normalizer
    return cls_loss + reg_loss, {
        "classification": float(cls_loss.detach()),
        "regression": float(reg_loss.detach()),
        "positives": float(positive_count),
    }


def _epoch(
    model: ActionFormerMotion,
    sequences: list[MotionSequence],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    rng: random.Random | None = None,
) -> float:
    model.train(optimizer is not None)
    order = list(sequences)
    if rng is not None:
        rng.shuffle(order)
    losses = []
    for item in order:
        x = _feature_tensor(item, mean, std, device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(x)
            loss, _ = _loss(outputs, item.segments, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _split_clips(sequences: list[MotionSequence]) -> tuple[set[str], set[str], set[str]]:
    clip_ids = sorted({item.clip_id for item in sequences})
    order = list(np.random.default_rng(SEED).permutation(clip_ids))
    test_ids = set(order[: max(3, round(len(order) * 0.2))])
    remainder = [clip_id for clip_id in order if clip_id not in test_ids]
    val_ids = set(remainder[: max(3, round(len(remainder) * 0.2))])
    train_ids = set(remainder) - val_ids
    return train_ids, val_ids, test_ids


def _fit(
    train_sequences: list[MotionSequence],
    val_sequences: list[MotionSequence],
    input_dim: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> tuple[ActionFormerMotion, np.ndarray, np.ndarray, int, list[dict[str, float]]]:
    mean, std = _fit_scaler(train_sequences)
    torch.manual_seed(seed)
    model = ActionFormerMotion(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss = _epoch(
            model,
            train_sequences,
            mean,
            std,
            device,
            optimizer,
            random.Random(seed + epoch),
        )
        val_loss = _epoch(model, val_sequences, mean, std, device, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(
            f"epoch {epoch}/{max_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("ActionFormer did not produce a usable validation checkpoint")
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, history


def train_actionformer(
    data_dir: str | Path,
    output_dir: str | Path,
    max_epochs: int = 30,
    patience: int = 6,
) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading 120Hz BVH/FBX features on {device}...", flush=True)
    sequences, joint_names = load_sequences(data_dir)
    train_ids, val_ids, test_ids = _split_clips(sequences)
    train_sequences = [x for x in sequences if x.clip_id in train_ids]
    val_sequences = [x for x in sequences if x.clip_id in val_ids]
    print(
        f"Loaded {len({x.clip_id for x in sequences})} songs / {len(sequences)} sequences; "
        f"train={len(train_ids)}, validation={len(val_ids)}, heldout={len(test_ids)}.",
        flush=True,
    )
    _, _, _, best_epoch, history = _fit(
        train_sequences,
        val_sequences,
        sequences[0].features.shape[1],
        max_epochs,
        patience,
        device,
        SEED,
    )
    all_mean, all_std = _fit_scaler(sequences)
    final_model = ActionFormerMotion(sequences[0].features.shape[1]).to(device)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-4)
    print(f"Refitting on all {len(train_ids | val_ids | test_ids)} songs for {best_epoch} epochs.", flush=True)
    for epoch in range(1, max(best_epoch, 1) + 1):
        loss = _epoch(
            final_model,
            sequences,
            all_mean,
            all_std,
            device,
            final_optimizer,
            random.Random(SEED + 1000 + epoch),
        )
        print(f"final epoch {epoch}/{max(best_epoch, 1)} loss={loss:.4f}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "actionformer_120hz_29songs.pt"
    checkpoint = {
        "schema": "choreofusion.actionformer-motion-single-class.v1",
        "state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
        "input_dim": int(sequences[0].features.shape[1]),
        "sample_rate_hz": SAMPLE_RATE,
        "feature_dim": FEATURE_DIM,
        "attention_heads": ATTENTION_HEADS,
        "attention_window": ATTENTION_WINDOW,
        "branch_levels": BRANCH_LEVELS,
        "regression_ranges_seconds": [list(x) for x in REGRESSION_RANGES],
        "mean": all_mean.tolist(),
        "std": all_std.tolist(),
        "joint_names": joint_names,
        "score_threshold": SCORE_THRESHOLD,
        "minimum_boundary_gap_seconds": MIN_BOUNDARY_GAP,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "metadata": {
            "training_clips": len({x.clip_id for x in sequences}),
            "training_sequences": len(sequences),
            "labels_used": "all annotation intervals mapped to one generic action class; text and tags ignored",
            "device": str(device),
        },
    }
    torch.save(checkpoint, model_path)
    report = {
        "schema": "choreofusion.actionformer-training-report.v1",
        "model": str(model_path),
        "architecture": "ActionFormer-style local-attention ConvTransformer + temporal FPN + point-based classification and boundary regression",
        "sample_rate_hz": SAMPLE_RATE,
        "training_clips": len({x.clip_id for x in sequences}),
        "training_sequences": len(sequences),
        "input_feature_dim": int(sequences[0].features.shape[1]),
        "generic_action_classes": 1,
        "annotation_text_used": False,
        "minimum_boundary_gap_seconds": MIN_BOUNDARY_GAP,
        "score_threshold_for_manual_test": SCORE_THRESHOLD,
        "regression_ranges_seconds": [list(x) for x in REGRESSION_RANGES],
        "split": {
            "train_clips": sorted(train_ids),
            "validation_clips": sorted(val_ids),
            "heldout_clips": sorted(test_ids),
        },
        "best_epoch": best_epoch,
        "validation_history": history,
        "heldout_test_score": "not computed",
        "note": "ChillKill is excluded from training; its supplied annotation JSON has no ground-truth segments.",
    }
    report_path = output_dir / "actionformer_training_report_120hz_29songs.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(model_path), "report": str(report_path), "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
    return report


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def _decode(
    outputs: list[dict[str, torch.Tensor]],
    duration: float,
    score_threshold: float = SCORE_THRESHOLD,
    min_gap: float = MIN_BOUNDARY_GAP,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    topk: int = 400,
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for output in outputs:
        scores = torch.sigmoid(output["logits"]).detach().cpu().numpy()
        offsets = output["offsets"].detach().cpu().numpy()
        times = output["times"].detach().cpu().numpy()
        indices = np.flatnonzero(scores >= score_threshold)
        if not len(indices):
            continue
        if len(indices) > topk:
            indices = indices[np.argpartition(scores[indices], -topk)[-topk:]]
        for idx in indices:
            start = max(0.0, float(times[idx] - offsets[idx, 0]))
            end = min(duration, float(times[idx] + offsets[idx, 1]))
            if end <= start:
                continue
            candidates.append({"start": start, "end": end, "score": float(scores[idx])})
    candidates.sort(key=lambda row: row["score"], reverse=True)
    selected: list[dict[str, float]] = []
    for candidate in candidates:
        center = (candidate["start"] + candidate["end"]) * 0.5
        if any(
            _interval_iou((candidate["start"], candidate["end"]), (row["start"], row["end"])) > iou_threshold
            or abs(center - (row["start"] + row["end"]) * 0.5) < min_gap
            for row in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= 120:
            break
    return sorted(selected, key=lambda row: (row["start"], row["end"]))


def _timeline_segments(
    detections: list[dict[str, float]], duration: float
) -> list[dict[str, float]]:
    """Turn NMS-selected ActionFormer proposals into a full, non-overlapping timeline."""
    if duration <= 0:
        return []
    if not detections:
        return [{"start": 0.0, "end": duration}]
    centers = sorted(
        max(0.0, min(duration, (row["start"] + row["end"]) * 0.5))
        for row in detections
    )
    boundaries = [(left + right) * 0.5 for left, right in zip(centers, centers[1:])]
    points = [0.0, *boundaries, duration]
    return [
        {"start": left, "end": right}
        for left, right in zip(points, points[1:])
        if right > left
    ]


def predict_actionformer(
    model_path: str | Path,
    inputs: list[str | Path],
    template_path: str | Path,
    output_path: str | Path,
    score_threshold: float | None = None,
    title: str | None = None,
) -> list[dict[str, Any]]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = ActionFormerMotion(int(checkpoint["input_dim"]), float(checkpoint["sample_rate_hz"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    threshold = float(score_threshold if score_threshold is not None else checkpoint["score_threshold"])
    input_paths = [Path(item) for item in inputs]
    raw = []
    durations = []
    for path in input_paths:
        times, features, duration = motion_features(
            path,
            checkpoint["joint_names"],
            sample_rate=float(checkpoint["sample_rate_hz"]),
        )
        item = MotionSequence(path.stem, path.suffix.casefold().lstrip("."), features, duration, np.empty((0, 2), dtype=np.float32))
        x = _feature_tensor(item, np.asarray(checkpoint["mean"], dtype=np.float32), np.asarray(checkpoint["std"], dtype=np.float32), device)
        with torch.no_grad():
            output = model(x)
        raw.append((times, output))
        durations.append(duration)
    base_times, base_outputs = raw[0]
    for times, outputs in raw[1:]:
        if len(times) != len(base_times) or not np.allclose(times, base_times):
            raise ValueError("All fused motion inputs must have matching timestamps")
    if len(raw) > 1:
        fused = []
        for level in range(len(base_outputs)):
            logits = torch.stack([row[1][level]["logits"] for row in raw]).mean(dim=0)
            offsets = torch.stack([row[1][level]["offsets"] for row in raw]).mean(dim=0)
            fused.append({**base_outputs[level], "logits": logits, "offsets": offsets})
        outputs = fused
    else:
        outputs = base_outputs
    duration = max(durations)
    detections = _decode(
        outputs,
        duration,
        threshold,
        float(checkpoint["minimum_boundary_gap_seconds"]),
        NMS_IOU_THRESHOLD,
    )
    segments = _timeline_segments(detections, duration)
    annotations = [
        {
            "start": round(row["start"], 6),
            "end": round(row["end"], 6),
            "label": "",
            "color": PALETTE[i % len(PALETTE)],
            "tags": [],
        }
        for i, row in enumerate(segments)
    ]
    payload = _template_payload(input_paths[0], duration, template_path, title)
    layer = next(row for row in payload["layers"] if row.get("name") == DEFAULT_LAYER)
    layer["annotations"] = annotations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run ActionFormer-style localization on BVH/FBX motion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--patience", type=int, default=6)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", action="append", required=True, help="repeat for BVH and FBX fusion")
    predict.add_argument("--template", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--threshold", type=float, default=None)
    predict.add_argument("--title", default=None)
    args = parser.parse_args()
    if args.command == "train":
        train_actionformer(args.data_dir, args.output_dir, args.epochs, args.patience)
    else:
        annotations = predict_actionformer(
            args.model,
            args.input,
            args.template,
            args.output,
            args.threshold,
            args.title,
        )
        print(json.dumps({"output": args.output, "annotation_count": len(annotations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
