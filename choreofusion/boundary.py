from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .annotations import DEFAULT_LAYER
from .motion import load_bvh

SAMPLE_RATE = 30.0
TARGET_SIGMA = 0.075
BOUNDARY_TOLERANCE = 0.0
THRESHOLD_TUNING_TOLERANCE = 0.5 / SAMPLE_RATE
MIN_BOUNDARY_GAP = 0.30
SEED = 17
PALETTE = ["#F97316", "#3B82F6", "#8B5CF6", "#F59E0B", "#10B981", "#EC4899", "#14B8A6", "#EF4444"]
BODY_LAYERS = ["구간 성격 (S/H/R/E)", "머리", "왼팔", "오른팔", "몸통", "골반", "왼다리", "오른다리"]


def _normal_key(value: str) -> str:
    value = re.sub(r"\s*\(\d+\)$", "", value.strip())
    value = re.sub(r"^M4\.0\s*[/_-]?\s*", "", value, flags=re.I)
    return "".join(c.casefold() for c in value if c.isalnum())


def match_training_pairs(data_dir: str | Path) -> list[dict[str, Path]]:
    root = Path(data_dir)
    bvhs = sorted(root.rglob("*.bvh"))
    fbxs = sorted([*root.rglob("*.FBX"), *root.rglob("*.fbx")])
    jsons = sorted(root.rglob("*.json"))
    if not bvhs or not fbxs or not jsons:
        raise ValueError(f"Expected BVH, FBX, and annotation JSON files in {root}")
    by_key: dict[str, dict[str, list[Path]]] = {}
    for kind, files in (("bvh", bvhs), ("fbx", fbxs)):
        for path in files:
            by_key.setdefault(_normal_key(path.stem), {}).setdefault(kind, []).append(path)
    json_by_key: dict[str, list[Path]] = {}
    for path in jsons:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = _normal_key(str(payload.get("videoTitle", "")))
        json_by_key.setdefault(key, []).append(path)
    rows = []
    for key, group in sorted(by_key.items()):
        if not group.get("bvh") and not group.get("fbx"):
            continue
        if len(group.get("bvh", [])) != 1 or len(group.get("fbx", [])) != 1 or len(json_by_key.get(key, [])) != 1:
            raise ValueError(
                f"Expected exactly one BVH, one FBX, and one JSON for key {key!r}; "
                f"got BVH={len(group.get('bvh', []))}, FBX={len(group.get('fbx', []))}, "
                f"JSON={len(json_by_key.get(key, []))}"
            )
        rows.append({"key": key, "bvh": group["bvh"][0], "fbx": group["fbx"][0], "json": json_by_key[key][0]})
    if len(rows) != len(bvhs) or len(rows) != len(fbxs) or len(rows) != len(jsons):
        raise ValueError(f"Unmatched files in {root}: BVH={len(bvhs)}, FBX={len(fbxs)}, JSON={len(jsons)}, pairs={len(rows)}")
    return rows


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    v = aw * bv + bw * av + np.cross(av, bv)
    w = aw * bw - np.sum(av * bv, axis=-1, keepdims=True)
    return np.concatenate([v, w], axis=-1)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    norm2 = np.sum(q * q, axis=-1, keepdims=True)
    return out / np.maximum(norm2, 1e-12)


def _quat_axis(axis: str, degrees: np.ndarray) -> np.ndarray:
    half = np.deg2rad(degrees) * 0.5
    result = np.zeros((len(degrees), 4), dtype=np.float64)
    result[:, "XYZ".index(axis.upper())] = np.sin(half)
    result[:, 3] = np.cos(half)
    return result


def _continuous_quats(quats: np.ndarray) -> np.ndarray:
    result = np.asarray(quats, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
    for i in range(1, len(result)):
        flip = np.sum(result[i - 1] * result[i], axis=-1) < 0
        result[i, flip] *= -1.0
    return result


def _resample_quats(times: np.ndarray, quats: np.ndarray, grid: np.ndarray) -> np.ndarray:
    q = _continuous_quats(quats)
    if len(times) == 1:
        return np.repeat(q, len(grid), axis=0)
    out = np.empty((len(grid), q.shape[1], 4), dtype=np.float64)
    for joint in range(q.shape[1]):
        for component in range(4):
            out[:, joint, component] = np.interp(grid, times, q[:, joint, component])
    return _continuous_quats(out)


def _resample_xyz(times: np.ndarray, xyz: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(times) == 1:
        return np.repeat(xyz, len(grid), axis=0)
    out = np.empty((len(grid), 3), dtype=np.float64)
    for component in range(3):
        out[:, component] = np.interp(grid, times, xyz[:, component])
    return out


def _load_raw_motion(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, float]:
    path = Path(path)
    if path.suffix.casefold() == ".bvh":
        motion = load_bvh(path)
        joint_names = list(dict.fromkeys(name.rsplit(".", 1)[0] for name in motion.channel_names))
        channel_lookup = {name: index for index, name in enumerate(motion.channel_names)}
        q_by_joint = []
        for joint in joint_names:
            rotation_channels = []
            for index, channel_name in enumerate(motion.channel_names):
                channel_joint, channel = channel_name.rsplit(".", 1)
                if channel_joint.casefold() == joint.casefold() and channel.casefold().endswith("rotation"):
                    rotation_channels.append((index, channel[0].upper()))
            q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(motion.frames), 1))
            for index, axis in rotation_channels:
                q = _quat_mul(q, _quat_axis(axis, motion.frames[:, index]))
            q_by_joint.append(q)
        quats = np.stack(q_by_joint, axis=1)
        times = np.arange(len(motion.frames), dtype=np.float64) * motion.frame_time
        root_values = np.zeros((len(motion.frames), 3), dtype=np.float64)
        for axis in "XYZ":
            column = channel_lookup.get(f"{motion.root_joint}.{axis}position")
            if column is not None:
                root_values[:, "XYZ".index(axis)] = motion.frames[:, column]
        # The paired FBX files use X-right, Y-up, Z-forward coordinates.
        # Convert BVH's X-right, Y-forward, Z-up translation to that basis.
        root_values = root_values[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
        duration = len(motion.frames) * motion.frame_time
        return joint_names, times, quats, root_values, duration
    if path.suffix.casefold() not in (".fbx",):
        raise ValueError(f"Input must be a .bvh or .fbx file: {path}")
    try:
        from pufbx import anim_to_array
    except ImportError as exc:
        raise RuntimeError("FBX input requires pufbx. Install requirements.txt with Python 3.12.") from exc
    values, times, joint_names = anim_to_array(str(path))
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 7 or len(times) != values.shape[1]:
        raise ValueError(f"Unexpected FBX animation array shape {values.shape} from {path}")
    times = times - times[0]
    order = np.argsort(times)
    times = times[order]
    values = values[:, order, :]
    if len(times) < 2:
        raise ValueError(f"FBX contains fewer than two animation samples: {path}")
    delta = np.diff(times)
    step = float(np.median(delta[delta > 0]))
    duration = float(times[-1] + step)
    joint_names = [str(name) for name in joint_names]
    quats = np.transpose(values[:, :, 3:7], (1, 0, 2))
    root_candidates = [i for i, name in enumerate(joint_names) if name.casefold() in ("hips", "root")]
    root_index = root_candidates[0] if root_candidates else 0
    root_values = values[root_index, :, :3]
    return joint_names, times, quats, root_values, duration


def motion_features(path: str | Path, joint_names: list[str], duration: float | None = None, sample_rate: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    sample_rate = float(SAMPLE_RATE if sample_rate is None else sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"Invalid sample rate {sample_rate}")
    names, times, quats, root, measured_duration = _load_raw_motion(path)
    lookup = {name.casefold(): index for index, name in enumerate(names)}
    missing = [name for name in joint_names if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"Skeleton does not match the training skeleton; missing joints: {missing[:8]}")
    selection = [lookup[name.casefold()] for name in joint_names]
    quats = quats[:, selection, :]
    clip_duration = float(duration or measured_duration)
    clip_duration = min(clip_duration, measured_duration + 1.0 / sample_rate)
    if clip_duration <= 0:
        raise ValueError(f"Invalid motion duration {clip_duration} for {path}")
    grid = np.arange(0.0, clip_duration, 1.0 / sample_rate, dtype=np.float64)
    if len(grid) < 2:
        grid = np.array([0.0, clip_duration], dtype=np.float64)
    q = _resample_quats(times, quats, grid)
    root = _resample_xyz(times, root, grid)

    relative_q = _quat_mul(_quat_inverse(q[:1]), q)
    relative_q *= np.where(relative_q[..., 3:4] < 0, -1.0, 1.0)
    relative_vec = relative_q[..., :3]
    relative_norm = np.linalg.norm(relative_vec, axis=-1, keepdims=True)
    relative_angle = 2.0 * np.arctan2(relative_norm[..., 0], np.maximum(relative_q[..., 3], 0.0))
    relative_axis_angle = relative_vec * (relative_angle[..., None] / np.maximum(relative_norm, 1e-8))
    relative_axis_angle[relative_norm[..., 0] < 1e-8] = 0.0

    if len(q) > 1:
        step_q = _quat_mul(_quat_inverse(q[:-1]), q[1:])
        step_q *= np.where(step_q[..., 3:4] < 0, -1.0, 1.0)
        vec = step_q[..., :3]
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        angle = 2.0 * np.arctan2(norm[..., 0], np.maximum(step_q[..., 3], 0.0))
        angular_velocity = vec * (angle[..., None] / np.maximum(norm, 1e-8)) * sample_rate
        angular_velocity[norm[..., 0] < 1e-8] = 0.0
        angular_velocity = np.concatenate([angular_velocity[:1], angular_velocity], axis=0)
    else:
        angular_velocity = np.zeros_like(relative_axis_angle)
    angular_speed = np.linalg.norm(angular_velocity, axis=-1, keepdims=True)
    relative_angle = np.linalg.norm(relative_axis_angle, axis=-1, keepdims=True)
    relative_root = (root - root[:1]) / 100.0
    root_velocity = np.gradient(root, 1.0 / sample_rate, axis=0) / 100.0
    features = np.concatenate(
        [relative_axis_angle.reshape(len(grid), -1), angular_velocity.reshape(len(grid), -1),
         angular_speed.reshape(len(grid), -1), relative_angle.reshape(len(grid), -1), relative_root, root_velocity],
        axis=1,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return grid, features, clip_duration


def read_boundaries(json_path: str | Path, duration: float, layer_name: str = DEFAULT_LAYER) -> tuple[list[float], dict[str, Any]]:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"No layers array in {json_path}")
    layer = next((item for item in layers if item.get("name") == layer_name), None)
    if layer is None:
        raise ValueError(f"Layer {layer_name!r} not found in {json_path}")
    annotations = sorted(layer.get("annotations", []), key=lambda row: (float(row["start"]), float(row["end"])))
    if len(annotations) < 2:
        raise ValueError(f"Need at least two segments in {json_path}")
    intervals = []
    for row in annotations:
        start, end = float(row["start"]), float(row["end"])
        if not np.isfinite(start + end) or end <= start:
            raise ValueError(f"Invalid range [{start}, {end}] in {json_path}")
        intervals.append((max(0.0, start), min(duration, end)))
    boundaries = []
    for left, right in zip(intervals, intervals[1:]):
        gap_or_overlap = right[0] - left[1]
        if abs(gap_or_overlap) > 0.10:
            raise ValueError(f"Adjacent ranges have a {gap_or_overlap:.3f}s gap/overlap in {json_path}")
        boundaries.append((left[1] + right[0]) * 0.5)
    return boundaries, payload


def soft_targets(times: np.ndarray, boundaries: list[float]) -> np.ndarray:
    target = np.zeros(len(times), dtype=np.float32)
    sigma = TARGET_SIGMA
    for boundary in boundaries:
        lo = max(0, int(np.searchsorted(times, boundary - sigma * 4)))
        hi = min(len(times), int(np.searchsorted(times, boundary + sigma * 4, side="right")))
        local = np.exp(-0.5 * ((times[lo:hi] - boundary) / sigma) ** 2).astype(np.float32)
        target[lo:hi] = np.maximum(target[lo:hi], local)
    return target


@dataclass
class Example:
    clip_id: str
    modality: str
    features: np.ndarray
    times: np.ndarray
    target: np.ndarray
    truth: list[float]


def collect_examples(data_dir: str | Path, layer_name: str = DEFAULT_LAYER) -> tuple[list[Example], list[str], dict[str, Any]]:
    pairs = match_training_pairs(data_dir)
    reference_names, _, _, _, _ = _load_raw_motion(pairs[0]["bvh"])
    examples: list[Example] = []
    clips: dict[str, Any] = {}
    counts = []
    colors: set[str] = set()
    for pair in pairs:
        clip_id = pair["key"]
        json_data = json.loads(pair["json"].read_text(encoding="utf-8"))
        clip_duration = float(json_data.get("videoDuration") or 0.0)
        clip_truth = None
        for modality, path in (("bvh", pair["bvh"]), ("fbx", pair["fbx"])):
            times, features, duration = motion_features(path, reference_names, clip_duration or None)
            if clip_truth is None:
                clip_truth, payload = read_boundaries(pair["json"], min(duration, clip_duration or duration), layer_name)
                clips[clip_id] = {"duration": min(duration, clip_duration or duration), "json": pair["json"], "bvh": pair["bvh"], "fbx": pair["fbx"], "boundaries": clip_truth}
                layer = next(x for x in payload["layers"] if x.get("name") == layer_name)
                counts.append(len(layer.get("annotations", [])))
                for row in layer.get("annotations", []):
                    if isinstance(row.get("color"), str):
                        colors.add(row["color"])
            local_duration = clips[clip_id]["duration"]
            boundaries = [b for b in clips[clip_id]["boundaries"] if 0.0 < b < local_duration]
            examples.append(Example(clip_id, modality, features, times, soft_targets(times, boundaries), boundaries))
    metadata = {
        "clip_count": len(pairs),
        "example_count": len(examples),
        "segment_counts": counts,
        "mean_segments_per_clip": float(np.mean(counts)),
        "annotation_text_used": False,
        "annotation_layer": layer_name,
        "colors": sorted(colors),
    }
    return examples, reference_names, {"clips": clips, **metadata}


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.12):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.norm(self.conv(x)))
        return F.relu(x + self.dropout(residual))


class BoundaryTCN(nn.Module):
    def __init__(self, input_channels: int, width: int = 48, dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.project = nn.Conv1d(input_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(*(ResidualTemporalBlock(width, d) for d in dilations))
        self.head = nn.Conv1d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(F.relu(self.project(x)))).squeeze(1)


def _fit_scaler(examples: list[Example]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([x.features for x in examples], axis=0).astype(np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _normalized(example: Example, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((example.features - mean) / std, -8.0, 8.0).astype(np.float32)


def _batches(examples: list[Example], mean: np.ndarray, std: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    indices = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = [examples[int(i)] for i in indices[start:start + batch_size]]
        max_len = max(len(x.times) for x in selected)
        width = selected[0].features.shape[1]
        xbatch = np.zeros((len(selected), width, max_len), dtype=np.float32)
        ybatch = np.zeros((len(selected), max_len), dtype=np.float32)
        mask = np.zeros((len(selected), max_len), dtype=np.float32)
        for i, example in enumerate(selected):
            n = len(example.times)
            xbatch[i, :, :n] = _normalized(example, mean, std).T
            ybatch[i, :n] = example.target
            mask[i, :n] = 1.0
        yield torch.from_numpy(xbatch), torch.from_numpy(ybatch), torch.from_numpy(mask), selected


def _loss_for(model: BoundaryTCN, examples: list[Example], mean: np.ndarray, std: np.ndarray, batch_size: int, training: bool, optimizer=None, seed: int = SEED) -> float:
    model.train(training)
    total, denom = 0.0, 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, mask, _ in _batches(examples, mean, std, batch_size, training, seed):
            logits = model(x)
            raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            weights = mask * (1.0 + 2.0 * y)
            loss = (raw * weights).sum() / weights.sum().clamp_min(1.0)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            total += float(loss.detach()) * float(weights.sum())
            denom += float(weights.sum())
    return total / max(denom, 1.0)


def fit_model(train: list[Example], validation: list[Example] | None, input_channels: int, epochs: int, patience: int = 8, seed: int = SEED) -> tuple[BoundaryTCN, np.ndarray, np.ndarray, int, dict[str, Any]]:
    torch.manual_seed(seed)
    mean, std = _fit_scaler(train)
    model = BoundaryTCN(input_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    history = {"train_loss": [], "validation_loss": []}
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        train_loss = _loss_for(model, train, mean, std, 2, True, optimizer, seed + epoch)
        val_loss = _loss_for(model, validation, mean, std, 2, False) if validation else train_loss
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        if val_loss < best_loss - 1e-4:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss, best_epoch, stale = val_loss, epoch, 0
        else:
            stale += 1
        if validation and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_epoch, {"best_loss": best_loss, "loss_source": "validation" if validation else "training", "history": history}


def predict_probabilities(model: BoundaryTCN, example: Example, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(_normalized(example, mean, std).T[None, :, :])
    with torch.no_grad():
        return torch.sigmoid(model(x))[0].cpu().numpy()


def decode_boundaries(times: np.ndarray, probabilities: np.ndarray, threshold: float, min_gap: float = MIN_BOUNDARY_GAP) -> list[tuple[float, float]]:
    if len(probabilities) < 3:
        return []
    smoothed = np.convolve(probabilities, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    candidates = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= threshold and smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            candidates.append((float(times[i]), float(smoothed[i])))
    chosen: list[tuple[float, float]] = []
    for time, score in sorted(candidates, key=lambda row: row[1], reverse=True):
        if all(abs(time - other_time) >= min_gap for other_time, _ in chosen):
            chosen.append((time, score))
    return sorted(chosen)


def _annotations_from_boundaries(boundary_times: list[float], duration: float) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    times = sorted(float(t) for t in boundary_times if 0.0 < float(t) < duration)
    endpoints = [0.0, *times, float(duration)]
    annotations: list[dict[str, Any]] = []
    for start, end in zip(endpoints, endpoints[1:]):
        if end <= start:
            continue
        annotations.append({
            "start": round(float(start), 6), "end": round(float(end), 6),
            "label": "", "color": PALETTE[len(annotations) % len(PALETTE)], "tags": [],
        })
    return annotations, []


def _match_counts(truth: list[float], predicted: list[tuple[float, float]], tolerance: float = BOUNDARY_TOLERANCE) -> tuple[int, int, int, list[float]]:
    pred = [x[0] for x in predicted]
    i = j = tp = 0
    errors = []
    while i < len(truth) and j < len(pred):
        delta = pred[j] - truth[i]
        if abs(delta) <= tolerance:
            tp += 1
            errors.append(abs(delta))
            i += 1
            j += 1
        elif delta < -tolerance:
            j += 1
        else:
            i += 1
    return tp, len(pred) - tp, len(truth) - tp, errors


def _metrics(samples: list[Example], outputs: list[np.ndarray], threshold: float, tolerance: float = BOUNDARY_TOLERANCE) -> dict[str, Any]:
    tp = fp = fn = 0
    count_errors: list[int] = []
    boundary_errors: list[float] = []
    for example, probs in zip(samples, outputs, strict=True):
        predicted = decode_boundaries(example.times, probs, threshold)
        a, b, c, errors = _match_counts(example.truth, predicted, tolerance)
        tp += a
        fp += b
        fn += c
        count_errors.append(len(predicted) - len(example.truth))
        boundary_errors.extend(errors)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "boundary_tolerance_seconds": tolerance,
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_absolute_boundary_error_seconds": float(np.mean(boundary_errors)) if boundary_errors else None,
        "mean_count_error_per_clip_modality": float(np.mean(count_errors)) if count_errors else None,
    }


def _best_threshold(samples: list[Example], outputs: list[np.ndarray], tolerance: float = THRESHOLD_TUNING_TOLERANCE) -> tuple[float, dict[str, Any]]:
    best_threshold, best = 0.40, None
    for threshold in np.arange(0.10, 0.801, 0.025):
        metrics = _metrics(samples, outputs, float(threshold), tolerance=tolerance)
        rank = (metrics["f1"], metrics["precision"], -abs(metrics["mean_count_error_per_clip_modality"] or 0.0))
        if best is None or rank > best[0]:
            best = (rank, metrics)
            best_threshold = float(threshold)
    return best_threshold, best[1]


def _save_checkpoint(path: Path, model: BoundaryTCN, mean: np.ndarray, std: np.ndarray, joint_names: list[str], threshold: float, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "choreofusion.boundary-tcn.v1",
        "state_dict": model.state_dict(),
        "input_channels": int(len(mean)),
        "width": 48,
        "dilations": [1, 2, 4, 8, 16, 32, 64],
        "mean": mean.tolist(), "std": std.tolist(),
        "joint_names": joint_names,
        "sample_rate": SAMPLE_RATE,
        "threshold": threshold,
        "min_boundary_gap": MIN_BOUNDARY_GAP,
        "metadata": metadata,
    }, path)


def train_boundary_model(data_dir: str | Path, output_dir: str | Path, layer_name: str = DEFAULT_LAYER, epochs: int = 45) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    examples, joint_names, data_meta = collect_examples(data_dir, layer_name)
    clip_ids = sorted({item.clip_id for item in examples})
    if len(clip_ids) < 10:
        raise ValueError(f"Need at least 10 independent clips for clip-level validation, found {len(clip_ids)}")
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(clip_ids))
    test_ids = set(order[:max(3, round(len(order) * 0.2))])
    remainder = [clip for clip in order if clip not in test_ids]
    val_ids = set(remainder[:max(3, round(len(remainder) * 0.2))])
    train_ids = set(remainder) - val_ids
    train = [x for x in examples if x.clip_id in train_ids]
    validation = [x for x in examples if x.clip_id in val_ids]
    test = [x for x in examples if x.clip_id in test_ids]
    cv_model, cv_mean, cv_std, best_epoch, cv_meta = fit_model(train, validation, examples[0].features.shape[1], epochs, seed=SEED)
    val_probabilities = [predict_probabilities(cv_model, x, cv_mean, cv_std) for x in validation]
    threshold, threshold_tuning_metrics = _best_threshold(validation, val_probabilities)
    validation_metrics = _metrics(validation, val_probabilities, threshold, tolerance=BOUNDARY_TOLERANCE)
    test_probabilities = [predict_probabilities(cv_model, x, cv_mean, cv_std) for x in test]
    test_metrics = _metrics(test, test_probabilities, threshold, tolerance=BOUNDARY_TOLERANCE)
    final_epochs = max(8, best_epoch)
    final_model, final_mean, final_std, _, final_meta = fit_model(examples, None, examples[0].features.shape[1], final_epochs, patience=final_epochs + 1, seed=SEED)
    output_dir = Path(output_dir)
    checkpoint = output_dir / "boundary_tcn.pt"
    report_path = output_dir / "boundary_training_report.json"
    model_meta = {
        "training_clips": len(clip_ids),
        "training_examples": len(examples),
        "annotation_layer": layer_name,
        "annotation_text_used": False,
        "modalities": ["BVH", "FBX"],
        "input_joint_count": len(joint_names),
        "target_rate_hz": SAMPLE_RATE,
        "final_epochs": final_epochs,
        "train_clip_ids": sorted(train_ids),
        "validation_clip_ids": sorted(val_ids),
        "test_clip_ids": sorted(test_ids),
    }
    _save_checkpoint(checkpoint, final_model, final_mean, final_std, joint_names, threshold, model_meta)
    report = {
        "schema": "choreofusion.boundary-tcn-report.v1",
        "model": str(checkpoint),
        "clips": len(clip_ids),
        "motion_examples": len(examples),
        "segments_per_clip": {"min": int(min(data_meta["segment_counts"])), "max": int(max(data_meta["segment_counts"])), "mean": data_meta["mean_segments_per_clip"]},
        "labels_used": {"layer": layer_name, "boundaries_only": True, "label_text_used": False, "tags_used": False},
        "threshold_tuning_tolerance_seconds": THRESHOLD_TUNING_TOLERANCE,
        "threshold_tuning_validation": {"threshold": threshold, **threshold_tuning_metrics},
        "validation": {"threshold": threshold, **validation_metrics},
        "held_out_test": {"threshold": threshold, **test_metrics},
        "split": {"train_clips": sorted(train_ids), "validation_clips": sorted(val_ids), "test_clips": sorted(test_ids)},
        "best_inner_epoch": best_epoch,
        "cross_validation_history": cv_meta["history"],
        "final_training": {"epochs": final_epochs, "loss_source": final_meta["loss_source"], "loss_history": final_meta["history"]},
        "limitations": [
            "Only the shared 51-joint skeleton present in this corpus is supported.",
            "The test metrics are from held-out songs, with both paired file formats kept in the same split.",
            "Predicted segment count is variable and depends on the probability threshold.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report



def fine_tune_boundary_model(
    data_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    new_clip_names: list[str] | None = None,
    epochs: int = 10,
    replay_clips: int = 4,
) -> dict[str, Any]:
    """Continue training an existing model on new clips with a small replay set."""
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data_dir = Path(data_dir).resolve()
    examples, joint_names, data_meta = collect_examples(data_dir)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if [x.casefold() for x in checkpoint["joint_names"]] != [x.casefold() for x in joint_names]:
        raise ValueError("New data skeleton does not match the saved model's joint order")

    all_pairs = match_training_pairs(data_dir)
    if new_clip_names:
        new_ids = {_normal_key(name) for name in new_clip_names}
    else:
        new_ids = {row["key"] for row in all_pairs if row["bvh"].parent.resolve() == data_dir}
    known_ids = {row["key"] for row in all_pairs}
    unknown = new_ids - known_ids
    if unknown:
        raise ValueError(f"Requested new clips are not in {data_dir}: {sorted(unknown)}")
    new_examples = [x for x in examples if x.clip_id in new_ids]
    if not new_examples:
        raise ValueError("No new clips selected; pass --new-clip names or place the added files in the data folder root")

    old_ids = sorted({x.clip_id for x in examples if x.clip_id not in new_ids})
    if not old_ids:
        raise ValueError("Fine-tuning needs the original training clips for replay")
    rng = np.random.default_rng(SEED)
    replay_ids = sorted(rng.choice(old_ids, size=min(replay_clips, len(old_ids)), replace=False).tolist())
    replay_examples = [x for x in examples if x.clip_id in replay_ids]
    training_examples = new_examples + replay_examples

    model = BoundaryTCN(int(checkpoint["input_channels"]), int(checkpoint["width"]), tuple(checkpoint["dilations"]))
    model.load_state_dict(checkpoint["state_dict"])
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    history = []
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        total, denom = 0.0, 0.0
        for x, y, mask, _ in _batches(training_examples, mean, std, 2, True, SEED + epoch):
            logits = model(x)
            raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            weights = mask * (1.0 + 2.0 * y)
            loss = (raw * weights).sum() / weights.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * float(weights.sum())
            denom += float(weights.sum())
        history.append(total / max(denom, 1.0))

    output_dir = Path(output_dir)
    output_model = output_dir / "boundary_tcn.pt"
    report_path = output_dir / "boundary_training_report.json"
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.update({
        "training_clips": len(known_ids),
        "training_examples": len(examples),
        "fine_tuned_added_clips": sorted(new_ids),
        "fine_tune_replay_clips": replay_ids,
        "fine_tune_epochs": max(1, epochs),
        "fine_tuning_mode": "continued training with replay; original scaler and threshold retained",
    })
    _save_checkpoint(
        output_model, model, mean, std, joint_names,
        float(checkpoint["threshold"]), metadata,
    )
    report = {
        "schema": "choreofusion.boundary-tcn-finetune-report.v1",
        "model": str(output_model),
        "base_model": str(model_path),
        "total_clips": len(known_ids),
        "new_clips": sorted(new_ids),
        "replay_clips": replay_ids,
        "motion_examples_used_this_run": len(training_examples),
        "epochs": max(1, epochs),
        "training_loss": history,
        "threshold_retained_from_base_model": float(checkpoint["threshold"]),
        "independent_test_evaluation": "not run; awaiting the user's separate test clip",
        "labels_used": {"layer": DEFAULT_LAYER, "boundaries_only": True, "label_text_used": False, "tags_used": False},
        "segment_counts_in_corpus": {"min": int(min(data_meta["segment_counts"])), "max": int(max(data_meta["segment_counts"])), "mean": data_meta["mean_segments_per_clip"]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _template_payload(input_path: Path, duration: float, template_path: str | Path | None, title: str | None) -> dict[str, Any]:
    if template_path:
        payload = json.loads(Path(template_path).read_text(encoding="utf-8"))
    else:
        payload = {
            "videoTitle": input_path.stem,
            "videoUrl": "",
            "videoDuration": duration,
            "layers": [{"name": DEFAULT_LAYER, "annotationMode": "text", "annotations": []}] + [
                {"name": name, "annotationMode": "text", "annotations": []} for name in BODY_LAYERS
            ],
        }
    payload["videoTitle"] = title or payload.get("videoTitle") or input_path.stem
    payload["videoUrl"] = payload.get("videoUrl") or ""
    payload["videoDuration"] = float(duration)
    layers = payload.get("layers")
    if not isinstance(layers, list):
        layers = []
    if not any(row.get("name") == DEFAULT_LAYER for row in layers):
        layers.insert(0, {"name": DEFAULT_LAYER, "annotationMode": "text", "annotations": []})
    for row in layers:
        row["annotations"] = []
    payload["layers"] = layers
    payload["exportedAt"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return payload


def segment_file(model_path: str | Path, input_path: str | Path, output_path: str | Path, template_path: str | Path | None = None, title: str | None = None, threshold: float | None = None) -> list[dict[str, Any]]:
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = BoundaryTCN(int(checkpoint["input_channels"]), int(checkpoint["width"]), tuple(checkpoint["dilations"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    input_path = Path(input_path)
    times, features, duration = motion_features(input_path, checkpoint["joint_names"])
    example = Example(input_path.stem, input_path.suffix.casefold().lstrip("."), features, times, np.zeros(len(times), dtype=np.float32), [])
    probabilities = predict_probabilities(model, example, np.asarray(checkpoint["mean"], dtype=np.float32), np.asarray(checkpoint["std"], dtype=np.float32))
    chosen_threshold = float(threshold if threshold is not None else checkpoint["threshold"])
    predicted = decode_boundaries(times, probabilities, chosen_threshold, float(checkpoint["min_boundary_gap"]))
    annotations, _ = _annotations_from_boundaries([x[0] for x in predicted], duration)
    payload = _template_payload(input_path, duration, template_path, title)
    layer = next(row for row in payload["layers"] if row.get("name") == DEFAULT_LAYER)
    layer["annotations"] = annotations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return annotations


def add_boundary_cli(subparsers) -> None:
    train = subparsers.add_parser("train-boundary", help="train a TCN that detects segment boundaries from BVH and FBX")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", default="outputs/boundary/01_DilatedTCN_48ch_7layers_30Hz_29songs/training_runs")
    train.add_argument("--layer", default=DEFAULT_LAYER)
    train.add_argument("--epochs", type=int, default=45)
    fine = subparsers.add_parser("fine-tune-boundary", help="continue training the saved TCN on new songs")
    fine.add_argument("--data-dir", required=True)
    fine.add_argument("--model", default="outputs/boundary/01_DilatedTCN_48ch_7layers_30Hz_29songs/models/archive/tcn_dilated_30hz_w48_d7_25songs.pt")
    fine.add_argument("--output-dir", default="outputs/boundary/01_DilatedTCN_48ch_7layers_30Hz_29songs/fine_tuning_runs")
    fine.add_argument("--new-clip", action="append", help="new clip stem; repeat once per new song")
    fine.add_argument("--epochs", type=int, default=10)
    fine.add_argument("--replay-clips", type=int, default=4)
    predict = subparsers.add_parser("segment", help="detect boundaries with the final MS-TCN++ + ST-GCN model")
    predict.add_argument("--model", default="outputs/boundary/05_MS-TCNpp_ST-GCN_120Hz_29songs/models/ms_tcnpp_boundary_120hz_29songs.pt")
    predict.add_argument("--st-model", default="outputs/boundary/05_MS-TCNpp_ST-GCN_120Hz_29songs/models/st_gcn_boundary_verifier_120hz_29songs.pt")
    predict.add_argument("--input", action="append", required=True, help="motion file; repeat to combine paired BVH and FBX inputs")
    predict.add_argument("--output", default="outputs/boundary/05_MS-TCNpp_ST-GCN_120Hz_29songs/predictions/predicted.annotations.json")
    predict.add_argument("--template")
    predict.add_argument("--title")
    predict.add_argument("--threshold", type=float, default=0.30)
    predict.add_argument("--min-gap", type=float, default=0.375)
    predict.add_argument("--boundary-offset", type=float, default=0.0, help="shift predicted cuts later in seconds")

    legacy_predict = subparsers.add_parser("segment-tcn", help="run a legacy single-checkpoint TCN boundary model")
    legacy_predict.add_argument("--model", default="outputs/boundary/01_DilatedTCN_48ch_7layers_30Hz_29songs/models/tcn_dilated_30hz_w48_d7_29songs.pt")
    legacy_predict.add_argument("--input", required=True)
    legacy_predict.add_argument("--output", default="outputs/boundary/01_DilatedTCN_48ch_7layers_30Hz_29songs/predictions/predicted.annotations.json")
    legacy_predict.add_argument("--template")
    legacy_predict.add_argument("--title")
    legacy_predict.add_argument("--threshold", type=float)


def run_boundary_cli(args) -> None:
    if args.command == "train-boundary":
        report = train_boundary_model(args.data_dir, args.output_dir, args.layer, args.epochs)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "fine-tune-boundary":
        report = fine_tune_boundary_model(args.data_dir, args.model, args.output_dir, args.new_clip, args.epochs, args.replay_clips)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "segment":
        from .mstcnpp_stgcn_boundary import predict_pipeline

        rows = predict_pipeline(
            args.model,
            args.st_model,
            args.input,
            args.template,
            args.output,
            args.threshold,
            args.min_gap,
            args.title,
            args.boundary_offset,
        )
        print(f"Wrote {len(rows)} blank-label ranges to {args.output}")
    else:
        rows = segment_file(args.model, args.input, args.output, args.template, args.title, args.threshold)
        print(f"Wrote {len(rows)} blank-label ranges to {args.output}")
