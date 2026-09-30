from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass
class BVHMotion:
    path: Path
    frame_time: float
    frames: np.ndarray
    channel_names: list[str]
    root_joint: str

    @property
    def duration(self) -> float:
        return len(self.frames) * self.frame_time

    def channel_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.channel_names)}


def load_bvh(path: str | Path) -> BVHMotion:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    channel_names: list[str] = []
    current_joint: str | None = None
    root_joint: str | None = None

    for line in text.splitlines():
        joint_match = re.match(r"\s*(ROOT|JOINT)\s+([^\s]+)", line)
        if joint_match:
            current_joint = joint_match.group(2)
            if root_joint is None:
                root_joint = current_joint
            continue
        channel_match = re.match(r"\s*CHANNELS\s+(\d+)\s+(.+?)\s*$", line)
        if channel_match:
            if current_joint is None:
                raise ValueError(f"CHANNELS line has no preceding joint in {path}")
            count = int(channel_match.group(1))
            names = channel_match.group(2).split()
            if len(names) != count:
                raise ValueError(f"Malformed CHANNELS line in {path}: expected {count}, found {len(names)}")
            channel_names.extend(f"{current_joint}.{name}" for name in names)

    motion_match = re.search(
        r"\bMOTION\s+Frames:\s*(\d+)\s+Frame\s+Time:\s*([0-9eE.+-]+)\s*",
        text,
        flags=re.IGNORECASE,
    )
    if not motion_match:
        raise ValueError(f"Could not read BVH MOTION header in {path}")
    frame_count = int(motion_match.group(1))
    frame_time = float(motion_match.group(2))
    if frame_time <= 0 or not channel_names:
        raise ValueError(f"Invalid frame time or empty channel list in {path}")

    number_text = text[motion_match.end():]
    values = np.fromstring(number_text, sep=" ", dtype=np.float64)
    expected = frame_count * len(channel_names)
    if values.size != expected:
        raise ValueError(
            f"{path} declares {frame_count} frames x {len(channel_names)} channels "
            f"({expected} values), but contains {values.size}"
        )
    frames = values.reshape(frame_count, len(channel_names))
    return BVHMotion(path, frame_time, frames, channel_names, root_joint or "Hips")


_FINGER_WORDS = ("thumb", "index", "middle", "ring", "pinky", "little", "finger")
_ROTATION_SUFFIXES = ("xrotation", "yrotation", "zrotation")
_POSITION_SUFFIXES = ("xposition", "yposition", "zposition")


def default_feature_channels(motion: BVHMotion) -> list[str]:
    selected: list[str] = []
    for name in motion.channel_names:
        joint, channel = name.rsplit(".", 1)
        lower_joint = joint.casefold()
        lower_channel = channel.casefold()
        if any(word in lower_joint for word in _FINGER_WORDS):
            continue
        if lower_channel.endswith(_ROTATION_SUFFIXES):
            selected.append(name)
        elif joint.casefold() == motion.root_joint.casefold() and lower_channel.endswith(_POSITION_SUFFIXES):
            selected.append(name)
    rotations = [name for name in selected if name.casefold().endswith(_ROTATION_SUFFIXES)]
    if not rotations:
        raise ValueError(f"No usable rotation channels found in {motion.path}")
    return selected


def resample_rows(values: np.ndarray, sample_count: int) -> np.ndarray:
    if len(values) == 1:
        return np.repeat(values, sample_count, axis=0)
    old_axis = np.linspace(0.0, 1.0, num=len(values))
    new_axis = np.linspace(0.0, 1.0, num=sample_count)
    out = np.empty((sample_count, values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        out[:, column] = np.interp(new_axis, old_axis, values[:, column])
    return out


def segment_features(
    motion: BVHMotion,
    start_seconds: float,
    end_seconds: float,
    feature_channels: list[str],
    sample_count: int = 32,
) -> tuple[np.ndarray, int, int]:
    if not np.isfinite(start_seconds) or not np.isfinite(end_seconds) or end_seconds <= start_seconds:
        raise ValueError(f"Invalid segment range [{start_seconds}, {end_seconds}] in {motion.path}")

    start_frame = max(0, min(len(motion.frames) - 1, int(np.floor(start_seconds / motion.frame_time))))
    end_frame = max(start_frame + 1, min(len(motion.frames), int(np.ceil(end_seconds / motion.frame_time))))
    channel_map = motion.channel_index()
    missing = [name for name in feature_channels if name not in channel_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"BVH skeleton mismatch in {motion.path}; missing channels: {preview}")

    clip = motion.frames[start_frame:end_frame]
    indices = [channel_map[name] for name in feature_channels]
    selected = clip[:, indices]
    rotation_columns = [
        i for i, name in enumerate(feature_channels)
        if name.casefold().endswith(_ROTATION_SUFFIXES)
    ]
    position_columns = [
        i for i, name in enumerate(feature_channels)
        if name.casefold().endswith(_POSITION_SUFFIXES)
    ]

    duration = max(end_seconds - start_seconds, motion.frame_time)
    parts: list[np.ndarray] = []

    # Relative angles capture pose change; circular pose summaries preserve
    # starting posture without angle-wrap discontinuities.
    if rotation_columns:
        angles = np.deg2rad(selected[:, rotation_columns])
        angles = np.unwrap(angles, axis=0)
        relative = angles - angles[0:1, :]
        sampled_relative = resample_rows(relative, sample_count) / np.pi
        sampled_absolute = resample_rows(angles, sample_count)
        pose_summary = np.concatenate(
            [np.mean(np.sin(sampled_absolute), axis=0), np.mean(np.cos(sampled_absolute), axis=0)]
        )
        velocity = np.diff(sampled_relative, axis=0) * ((sample_count - 1) / duration)
        parts.extend([sampled_relative.ravel(), velocity.ravel(), pose_summary])

    if position_columns:
        positions = selected[:, position_columns]
        relative_position = positions - positions[0:1, :]
        sampled_position = resample_rows(relative_position, sample_count) / 100.0
        position_velocity = np.diff(sampled_position, axis=0) * ((sample_count - 1) / duration)
        parts.extend([sampled_position.ravel(), position_velocity.ravel()])

    parts.append(np.asarray([np.log1p(duration)], dtype=np.float64))
    feature = np.concatenate(parts)
    return feature, start_frame, end_frame
