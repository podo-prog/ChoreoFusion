from __future__ import annotations

import json
from pathlib import Path


DEFAULT_LAYER = "전체 동작 설명"


def read_segments(path: str | Path, layer_name: str = DEFAULT_LAYER) -> tuple[list[dict], dict]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"No annotation layers found in {path}")
    layer = next((item for item in layers if item.get("name") == layer_name), None)
    if layer is None:
        names = [str(item.get("name", "")) for item in layers]
        raise ValueError(f"Layer {layer_name!r} not found in {path}. Available layers: {names}")

    annotations = layer.get("annotations", [])
    segments = []
    for index, annotation in enumerate(annotations, start=1):
        # Deliberately read only interval boundaries. Text, tags, and category
        # values are not used as model inputs or targets.
        start = float(annotation["start"])
        end = float(annotation["end"])
        if end <= start:
            raise ValueError(f"Invalid annotation interval {index} in {path}")
        segments.append({"segment_index": index, "start": start, "end": end})
    segments.sort(key=lambda item: (item["start"], item["end"]))
    if not segments:
        raise ValueError(f"No segments found in layer {layer_name!r} in {path}")
    return segments, payload


def normalized_key(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def annotation_title(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload.get("videoTitle", ""))


def match_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    data_dir = Path(data_dir)
    bvh_files = sorted(data_dir.glob("*.bvh"))
    json_files = sorted(data_dir.glob("*.json"))
    if not bvh_files or not json_files:
        raise ValueError(f"Expected matching .bvh and annotation .json files in {data_dir}")

    json_keys = [(path, normalized_key(annotation_title(path))) for path in json_files]
    pairs: list[tuple[Path, Path]] = []
    used_json: set[Path] = set()
    for bvh in bvh_files:
        bvh_key = normalized_key(bvh.stem)
        matches = [
            path for path, title_key in json_keys
            if path not in used_json and bvh_key and (bvh_key in title_key or title_key in bvh_key)
        ]
        if not matches and len(bvh_files) == 1 and len(json_files) == 1:
            matches = [json_files[0]]
        if len(matches) != 1:
            raise ValueError(
                f"Could not match {bvh.name} to exactly one annotation JSON by videoTitle. "
                f"Matching title candidates: {[path.name for path in matches]}"
            )
        pairs.append((bvh, matches[0]))
        used_json.add(matches[0])
    if len(used_json) != len(json_files):
        unused = [path.name for path in json_files if path not in used_json]
        raise ValueError(f"Unmatched annotation JSON files: {unused}")
    return pairs
