from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import math

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .annotations import DEFAULT_LAYER, match_pairs, read_segments
from .motion import default_feature_channels, load_bvh, segment_features


def _cluster_candidates(sample_count: int, requested: str) -> tuple[list[int], int | None]:
    if requested != "auto":
        count = int(requested)
        if count < 1 or count > sample_count:
            raise ValueError(f"Cluster count must be between 1 and {sample_count}")
        return [count], count
    if sample_count < 3:
        return [1], 1
    upper = min(12, max(2, int(math.sqrt(sample_count))), sample_count - 1)
    return list(range(2, upper + 1)), None


def _canonicalize(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    old_ids = sorted(set(int(item) for item in labels), key=lambda cluster: int(np.flatnonzero(labels == cluster)[0]))
    mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=1)}
    return np.asarray([mapping[int(item)] for item in labels], dtype=int), mapping


def _collect(data_dir: str | Path, layer_name: str):
    pairs = match_pairs(data_dir)
    loaded = []
    common_channels: list[str] | None = None

    for bvh_path, json_path in pairs:
        motion = load_bvh(bvh_path)
        candidates = default_feature_channels(motion)
        if common_channels is None:
            common_channels = candidates
        else:
            candidate_set = set(candidates)
            common_channels = [name for name in common_channels if name in candidate_set]
        segments, _metadata = read_segments(json_path, layer_name)
        loaded.append((bvh_path.stem, json_path, motion, segments))

    if not common_channels or not any(name.casefold().endswith(("xrotation", "yrotation", "zrotation")) for name in common_channels):
        raise ValueError("Paired BVH files have no common body rotation channels")
    rows = []
    records = []
    for clip_id, json_path, motion, segments in loaded:
        for segment in segments:
            feature, start_frame, end_frame = segment_features(
                motion, segment["start"], segment["end"], common_channels
            )
            rows.append(feature)
            records.append({
                "clip_id": clip_id,
                "annotation_file": json_path.name,
                "segment_index": segment["segment_index"],
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["end"] - segment["start"],
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "frame_time": motion.frame_time,
            })
    return np.vstack(rows), records, common_channels, len(loaded)


def _write_assignments(out_path: Path, records: list[dict], labels: list[int]) -> None:
    result = [
        {**record, "cluster_id": int(cluster)}
        for record, cluster in zip(records, labels, strict=True)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    columns = [
        "clip_id", "segment_index", "start", "end", "duration",
        "start_frame", "end_frame_exclusive", "cluster_id",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in result)


def fit(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "outputs",
    layer_name: str = DEFAULT_LAYER,
    clusters: str = "auto",
) -> dict:
    output_dir = Path(output_dir)
    matrix, records, channel_names, clip_count = _collect(data_dir, layer_name)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)

    component_count = min(64, standardized.shape[0] - 1, standardized.shape[1])
    if component_count >= 2:
        reducer = PCA(n_components=component_count, svd_solver="full", random_state=17)
        embedding = reducer.fit_transform(standardized)
    else:
        reducer = None
        embedding = standardized

    candidates, forced_count = _cluster_candidates(len(records), clusters)
    scores = []
    best_score = -float("inf")
    best_labels: np.ndarray | None = None
    best_model: KMeans | None = None
    chosen_count = forced_count or 1

    for count in candidates:
        if count == 1:
            model = KMeans(n_clusters=1, n_init=10, random_state=17)
            labels = model.fit_predict(embedding)
            score = None
        else:
            model = KMeans(n_clusters=count, n_init=20, random_state=17)
            labels = model.fit_predict(embedding)
            distinct = len(set(int(item) for item in labels))
            score = float(silhouette_score(embedding, labels)) if 1 < distinct < len(records) else -1.0
        scores.append({"clusters": count, "silhouette": score})
        comparable_score = score if score is not None else 0.0
        if forced_count is not None:
            if count == forced_count:
                best_labels, best_model, chosen_count = labels, model, count
        elif comparable_score > best_score:
            best_score, best_labels, best_model, chosen_count = comparable_score, labels, model, count

    if best_labels is None or best_model is None:
        raise RuntimeError("Could not fit a cluster model")
    final_labels, label_map = _canonicalize(best_labels)
    counts = Counter(int(item) for item in final_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "schema": "choreofusion.unsupervised-motion.v1",
        "scaler": scaler,
        "pca": reducer,
        "kmeans": best_model,
        "cluster_id_map": label_map,
        "feature_channels": channel_names,
        "sample_count": 32,
        "layer_name": layer_name,
        "training_cluster_count": chosen_count,
        "training_clip_count": clip_count,
    }
    joblib.dump(model_bundle, output_dir / "model.joblib")
    _write_assignments(output_dir / "assignments.json", records, final_labels.tolist())

    report = {
        "schema": "choreofusion.unsupervised-motion-report.v1",
        "training_clips": clip_count,
        "segments": len(records),
        "annotation_layer": layer_name,
        "annotation_text_used": False,
        "feature_channels": len(channel_names),
        "motion_frame_rate_hz_by_clip": {
            record["clip_id"]: round(1.0 / record["frame_time"], 6)
            for record in records
        },
        "cluster_selection": "user-fixed" if forced_count is not None else "silhouette search over PCA motion features",
        "candidate_scores": scores,
        "selected_cluster_count": chosen_count,
        "cluster_sizes": {str(key): value for key, value in sorted(counts.items())},
        "limitations": [
            "Cluster IDs are arbitrary and do not name human-readable actions.",
            "The current corpus has one clip; cross-clip generalization cannot be assessed.",
            "Classification of a new clip requires its BVH motion and segment time ranges.",
        ],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def classify_one(
    model_path: str | Path,
    bvh_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
    layer_name: str | None = None,
) -> list[dict]:
    bundle = joblib.load(model_path)
    active_layer = layer_name or bundle["layer_name"]
    motion = load_bvh(bvh_path)
    segments, _metadata = read_segments(annotation_path, active_layer)
    output = []
    for segment in segments:
        features, start_frame, end_frame = segment_features(
            motion, segment["start"], segment["end"], bundle["feature_channels"], bundle["sample_count"]
        )
        vector = features.reshape(1, -1)
        standardized = bundle["scaler"].transform(vector)
        embedding = bundle["pca"].transform(standardized) if bundle["pca"] is not None else standardized
        raw_id = int(bundle["kmeans"].predict(embedding)[0])
        output.append({
            "clip_id": Path(bvh_path).stem,
            "segment_index": segment["segment_index"],
            "start": segment["start"],
            "end": segment["end"],
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "cluster_id": bundle["cluster_id_map"][raw_id],
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
