import os
from collections import defaultdict, deque
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_segments_from_yaml(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of segments in {path}, got {type(data).__name__}")

    for idx, segment in enumerate(data):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment at index {idx} is not a mapping")
    return data


def compute_generations(segments: List[Dict[str, Any]]) -> Dict[str, int]:
    id_map: Dict[str, Dict[str, Any]] = {}
    for idx, seg in enumerate(segments):
        seg_id = seg.get("id")
        if seg_id is None:
            raise ValueError(f"Segment at index {idx} is missing 'id'")
        if seg_id in id_map:
            raise ValueError(f"Duplicate segment id '{seg_id}' found")
        id_map[seg_id] = seg

    for seg_id, seg in id_map.items():
        parent_id = seg.get("parent_id")
        if parent_id is not None and parent_id not in id_map:
            raise ValueError(f"Segment '{seg_id}' references missing parent_id '{parent_id}'")

    visiting = set()
    visited = set()

    def walk(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError(f"Cycle detected involving segment '{node_id}'")
        visiting.add(node_id)
        parent_id = id_map[node_id].get("parent_id")
        if parent_id is not None:
            walk(parent_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in id_map:
        walk(node_id)

    roots = [sid for sid, seg in id_map.items() if seg.get("parent_id") is None]
    if not roots:
        raise ValueError("No root segment found (parent_id is null)")
    if len(roots) > 1:
        raise ValueError(f"Multiple root segments found: {', '.join(sorted(roots))}")
    root_id = roots[0]

    children_map: Dict[str, List[str]] = defaultdict(list)
    for seg_id, seg in id_map.items():
        parent_id = seg.get("parent_id")
        if parent_id is not None:
            children_map[parent_id].append(seg_id)

    generations: Dict[str, int] = {root_id: 0}
    queue: deque[str] = deque([root_id])
    while queue:
        current = queue.popleft()
        for child_id in children_map.get(current, []):
            if child_id in generations:
                raise ValueError(f"Cycle detected while assigning generation for '{child_id}'")
            generations[child_id] = generations[current] + 1
            queue.append(child_id)

    if len(generations) != len(id_map):
        missing = sorted(set(id_map) - set(generations))
        raise ValueError(
            f"Unreachable segments detected (possible cycle or disconnected root): {', '.join(missing)}"
        )

    return generations


def _get_param(params: Dict[str, Any], seg_id: str, key: str) -> float:
    if not isinstance(params, dict):
        raise ValueError(f"Segment '{seg_id}' is missing 'params' mapping")
    try:
        return float(params[key])
    except KeyError:
        raise ValueError(f"Segment '{seg_id}' is missing required parameter '{key}'")
    except Exception as exc:
        raise ValueError(f"Segment '{seg_id}' has invalid parameter '{key}': {exc}")


def build_branch_records(
    segments: List[Dict[str, Any]], generations: Dict[str, int]
) -> List[Dict[str, Any]]:
    branches: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = seg["id"]
        if seg_id not in generations:
            raise ValueError(f"Missing generation assignment for segment '{seg_id}'")
        seg_gen = generations[seg_id]
        kind_value = seg.get("kind")
        if kind_value is None:
            raise ValueError(f"Segment '{seg_id}' is missing 'kind'")
        kind = str(kind_value).lower()
        params = seg.get("params", {})

        if kind == "pipe":
            length = _get_param(params, seg_id, "length")
            d_in = _get_param(params, seg_id, "d_in")
            d_out = _get_param(params, seg_id, "d_out")
            diameter = 0.5 * (d_in + d_out)
            branches.append(
                {
                    "segment_id": seg_id,
                    "branch_role": "pipe",
                    "generation": seg_gen,
                    "length": length,
                    "diameter": diameter,
                    "kind": kind,
                }
            )
        elif kind == "y2":
            trunk_length = _get_param(params, seg_id, "length_trunk")
            child1_length = _get_param(params, seg_id, "length_child1")
            child2_length = _get_param(params, seg_id, "length_child2")
            trunk_d = _get_param(params, seg_id, "d_trunk")
            child1_d = _get_param(params, seg_id, "d_child1")
            child2_d = _get_param(params, seg_id, "d_child2")
            branches.extend(
                [
                    {
                        "segment_id": seg_id,
                        "branch_role": "trunk",
                        "generation": seg_gen,
                        "length": trunk_length,
                        "diameter": trunk_d,
                        "kind": kind,
                    },
                    {
                        "segment_id": seg_id,
                        "branch_role": "child1",
                        "generation": seg_gen + 1,
                        "length": child1_length,
                        "diameter": child1_d,
                        "kind": kind,
                    },
                    {
                        "segment_id": seg_id,
                        "branch_role": "child2",
                        "generation": seg_gen + 1,
                        "length": child2_length,
                        "diameter": child2_d,
                        "kind": kind,
                    },
                ]
            )
        elif kind == "y3":
            trunk_length = _get_param(params, seg_id, "length_trunk")
            child1_length = _get_param(params, seg_id, "length_child1")
            child2_length = _get_param(params, seg_id, "length_child2")
            child3_length = _get_param(params, seg_id, "length_child3")
            trunk_d = _get_param(params, seg_id, "d_trunk")
            child1_d = _get_param(params, seg_id, "d_child1")
            child2_d = _get_param(params, seg_id, "d_child2")
            child3_d = _get_param(params, seg_id, "d_child3")
            branches.extend(
                [
                    {
                        "segment_id": seg_id,
                        "branch_role": "trunk",
                        "generation": seg_gen,
                        "length": trunk_length,
                        "diameter": trunk_d,
                        "kind": kind,
                    },
                    {
                        "segment_id": seg_id,
                        "branch_role": "child1",
                        "generation": seg_gen + 1,
                        "length": child1_length,
                        "diameter": child1_d,
                        "kind": kind,
                    },
                    {
                        "segment_id": seg_id,
                        "branch_role": "child2",
                        "generation": seg_gen + 1,
                        "length": child2_length,
                        "diameter": child2_d,
                        "kind": kind,
                    },
                    {
                        "segment_id": seg_id,
                        "branch_role": "child3",
                        "generation": seg_gen + 1,
                        "length": child3_length,
                        "diameter": child3_d,
                        "kind": kind,
                    },
                ]
            )
        else:
            raise ValueError(f"Unsupported segment kind '{kind_value}' for segment '{seg_id}'")

    return branches


def summarize_generations(branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for branch in branches:
        grouped[branch["generation"]].append(branch)

    summary: List[Dict[str, Any]] = []
    for gen in sorted(grouped):
        group = grouped[gen]
        diameters = np.array([b["diameter"] for b in group], dtype=float)
        lengths = np.array([b["length"] for b in group], dtype=float)
        summary.append(
            {
                "generation": gen,
                "n": len(group),
                "d_mean": float(diameters.mean()),
                "d_min": float(diameters.min()),
                "d_max": float(diameters.max()),
                "l_mean": float(lengths.mean()),
                "l_min": float(lengths.min()),
                "l_max": float(lengths.max()),
            }
        )
    return summary


def print_summary_table(summary: List[Dict[str, Any]]) -> None:
    if not summary:
        print("No branches to summarize.")
        return

    header_fmt = "{:<10} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
    row_fmt = "{:<10d} {:>4d} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}"

    print(header_fmt.format("Generation", "n", "D_mean", "D_min", "D_max", "L_mean", "L_min", "L_max"))
    for row in summary:
        print(
            row_fmt.format(
                row["generation"],
                row["n"],
                row["d_mean"],
                row["d_min"],
                row["d_max"],
                row["l_mean"],
                row["l_min"],
                row["l_max"],
            )
        )


def plot_metrics(summary: List[Dict[str, Any]], output_dir: str) -> None:
    if not summary:
        return

    os.makedirs(output_dir, exist_ok=True)

    generations = np.array([row["generation"] for row in summary], dtype=int)

    diam_mean = np.array([row["d_mean"] for row in summary], dtype=float)
    diam_min = np.array([row["d_min"] for row in summary], dtype=float)
    diam_max = np.array([row["d_max"] for row in summary], dtype=float)

    plt.figure()
    yerr = np.vstack((diam_mean - diam_min, diam_max - diam_mean))
    plt.errorbar(generations, diam_mean, yerr=yerr, fmt="o", capsize=5)
    plt.xlabel("Generation")
    plt.ylabel("Diameter")
    plt.title("Diameter vs Generation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diameter_vs_generation.png"))
    plt.close()

    length_mean = np.array([row["l_mean"] for row in summary], dtype=float)
    length_min = np.array([row["l_min"] for row in summary], dtype=float)
    length_max = np.array([row["l_max"] for row in summary], dtype=float)

    plt.figure()
    yerr = np.vstack((length_mean - length_min, length_max - length_mean))
    plt.errorbar(generations, length_mean, yerr=yerr, fmt="o", capsize=5)
    plt.xlabel("Generation")
    plt.ylabel("Length")
    plt.title("Length vs Generation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "length_vs_generation.png"))
    plt.close()


def main() -> None:
    yaml_path = "trees/human_central_v2_weibel_with_right_subtree.yaml"
    segments = load_segments_from_yaml(yaml_path)
    if not segments:
        raise ValueError(f"No segments loaded from {yaml_path}")

    generations = compute_generations(segments)
    branches = build_branch_records(segments, generations)
    summary = summarize_generations(branches)

    print_summary_table(summary)

    output_dir = os.path.join("results", "morphometry")
    plot_metrics(summary, output_dir)
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
