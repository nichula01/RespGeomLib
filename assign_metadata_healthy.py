from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"YAML file not found: {path}")
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


def infer_generations(segments: List[Dict[str, Any]]) -> Dict[str, int]:
    by_id = {}
    for seg in segments:
        sid = seg.get("id")
        if sid is None:
            raise RuntimeError("Segment entry missing 'id'")
        by_id[sid] = seg

    gen_cache: Dict[str, int] = {}
    visiting = set()

    def _infer(seg_id: str) -> int:
        if seg_id in gen_cache:
            return gen_cache[seg_id]
        if seg_id in visiting:
            raise RuntimeError(f"Cycle detected while inferring generation for {seg_id!r}")
        visiting.add(seg_id)
        try:
            seg = by_id.get(seg_id)
            if seg is None:
                raise RuntimeError(f"Missing parent segment {seg_id!r}")
            meta = seg.get("meta") or {}
            meta_gen = meta.get("generation", None)
            if isinstance(meta_gen, int) and not isinstance(meta_gen, bool):
                gen = meta_gen
            else:
                parent_id = seg.get("parent_id", None)
                if parent_id is None:
                    gen = 0
                else:
                    gen = _infer(parent_id) + 1
            gen_cache[seg_id] = gen
            return gen
        finally:
            visiting.discard(seg_id)

    for seg in segments:
        _infer(seg["id"])
    return gen_cache


def assign_healthy_meta(segments: List[Dict[str, Any]], gen_map: Dict[str, int]) -> None:
    for seg in segments:
        sid = seg.get("id")
        if sid is None:
            raise RuntimeError("Segment entry missing 'id'")

        meta = seg.get("meta")
        if meta is None or not isinstance(meta, dict):
            meta = {}
            seg["meta"] = meta

        meta.setdefault("zone", "conducting")

        gen = gen_map.get(sid)
        if gen is None:
            raise RuntimeError(f"No generation found for segment {sid!r}")
        meta["generation"] = gen

        if gen <= 2:
            profile = {
                "wall_thickness": 0.15,
                "E": 0.5e6,
                "cartilage_presence": "ringed",
                "cartilage_model": "rings",
                "ring_spacing": 0.4,
                "ring_thickness": 0.05,
                "ring_width": 0.2,
                "mucosa_type": "ciliated",
            }
        elif gen <= 6:
            profile = {
                "wall_thickness": 0.08,
                "E": 0.3e6,
                "cartilage_presence": "partial",
                "cartilage_model": "rings",
                "ring_spacing": 0.3,
                "ring_thickness": 0.03,
                "ring_width": 0.15,
                "mucosa_type": "ciliated",
            }
        elif gen <= 10:
            profile = {
                "wall_thickness": 0.05,
                "E": 0.2e6,
                "cartilage_presence": "partial",
                "cartilage_model": "none",
                "mucosa_type": "ciliated",
            }
        else:
            profile = {
                "wall_thickness": 0.03,
                "E": 0.15e6,
                "cartilage_presence": "none",
                "cartilage_model": "none",
                "mucosa_type": "terminal_bronchiole",
            }

        meta.update(profile)


def main() -> None:
    base_path = Path("trees") / "human_central_v2_weibel_with_right_subtree.yaml"
    out_path = Path("trees") / "human_central_v2_healthy_with_metadata.yaml"

    segments = load_yaml_list(base_path)
    gen_map = infer_generations(segments)
    assign_healthy_meta(segments, gen_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(segments, f, sort_keys=False)

    band_counts = {"0-2": 0, "3-6": 0, "7-10": 0, ">10": 0}
    for gen in gen_map.values():
        if gen <= 2:
            band_counts["0-2"] += 1
        elif gen <= 6:
            band_counts["3-6"] += 1
        elif gen <= 10:
            band_counts["7-10"] += 1
        else:
            band_counts[">10"] += 1

    total = len(segments)
    print(f"Processed {total} segments from {base_path}")
    print("Generation bands:")
    for band in ["0-2", "3-6", "7-10", ">10"]:
        print(f"  {band}: {band_counts[band]}")


if __name__ == "__main__":
    main()
