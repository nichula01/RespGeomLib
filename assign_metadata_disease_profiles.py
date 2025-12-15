from pathlib import Path
from typing import Any, Dict, List

import copy
import yaml


def load_tree(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return []
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a top-level list in {path}, got {type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(f"Entry {idx} in {path} is not a mapping/dict")
    return data


def save_tree(segments: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(segments, f, sort_keys=False)


def _get_generation(seg: Dict[str, Any]) -> int:
    meta = seg.get("meta") or {}
    gen = meta.get("generation", None)
    if isinstance(gen, int) and not isinstance(gen, bool):
        return gen
    raise RuntimeError(f"Segment {seg.get('id')} missing valid generation in meta")


def apply_asthma_profile(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = copy.deepcopy(segments)
    modified = 0
    for seg in result:
        meta = seg.setdefault("meta", {}) or {}
        gen = _get_generation(seg)

        meta.setdefault("disease", "healthy_or_unaffected")

        if meta.get("zone") != "conducting":
            continue
        if 5 <= gen <= 10:
            wt = meta.get("wall_thickness", None)
            E = meta.get("E", None)
            if wt is not None:
                meta["wall_thickness"] = float(wt) * 1.5
            if E is not None:
                meta["E"] = float(E) * 1.5
            meta["disease"] = "asthma"
            meta["severity"] = "mild"
            modified += 1
    return result, modified


def apply_copd_profile(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = copy.deepcopy(segments)
    modified_bands = {"0-2": 0, "3-6": 0, "7-10": 0}
    for seg in result:
        meta = seg.setdefault("meta", {}) or {}
        gen = _get_generation(seg)

        meta.setdefault("disease", "healthy_or_unaffected")

        if 0 <= gen <= 2:
            wt = meta.get("wall_thickness", None)
            E = meta.get("E", None)
            if wt is not None:
                meta["wall_thickness"] = float(wt) * 1.2
            if E is not None:
                meta["E"] = float(E) * 1.2
            meta["disease"] = "COPD"
            meta["severity_central"] = "mild"
            modified_bands["0-2"] += 1
        elif 3 <= gen <= 6:
            wt = meta.get("wall_thickness", None)
            E = meta.get("E", None)
            if wt is not None:
                meta["wall_thickness"] = float(wt) * 1.4
            if E is not None:
                meta["E"] = float(E) * 1.3
            meta["disease"] = "COPD"
            meta["severity_medium"] = "moderate"
            modified_bands["3-6"] += 1
        elif 7 <= gen <= 10:
            wt = meta.get("wall_thickness", None)
            E = meta.get("E", None)
            if wt is not None:
                meta["wall_thickness"] = float(wt) * 1.3
            if E is not None:
                meta["E"] = float(E) * 0.8
            meta["disease"] = "COPD"
            meta["severity_distal"] = "moderate"
            modified_bands["7-10"] += 1
        else:
            meta.setdefault("disease", "healthy_or_unaffected")

    return result, modified_bands


def main() -> None:
    base_path = Path("trees") / "human_central_v2_healthy_with_metadata.yaml"
    asthma_path = Path("trees") / "human_central_v2_asthma_mild.yaml"
    copd_path = Path("trees") / "human_central_v2_copd_moderate.yaml"

    healthy_segments = load_tree(base_path)

    asthma_segments, asthma_modified = apply_asthma_profile(healthy_segments)
    save_tree(asthma_segments, asthma_path)
    print(f"Asthma profile: modified {asthma_modified} segments (gen 5-10) -> {asthma_path}")

    copd_segments, copd_modified_bands = apply_copd_profile(healthy_segments)
    save_tree(copd_segments, copd_path)
    print(f"COPD profile saved to {copd_path}")
    print("  Modified segments per band:")
    for band in ["0-2", "3-6", "7-10"]:
        print(f"    {band}: {copd_modified_bands.get(band, 0)}")


if __name__ == "__main__":
    main()
