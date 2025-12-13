import numpy as np
import pyvista as pv

from tree_builder import load_specs_from_yaml, build_tree


def color_for_segment_id(seg_id: str) -> str:
    """Pick a PyVista color name based on anatomical side/role encoded in the id."""
    sid = seg_id.lower()
    if sid == "trachea":
        return "lightgray"
    if sid.startswith("y_"):
        return "mistyrose"
    if "right_" in sid:
        return "coral"
    if "left_" in sid:
        return "lightskyblue"
    return "tan"


def main():
    yaml_path = "trees/human_central_v2_weibel.yaml"

    specs = load_specs_from_yaml(yaml_path)
    root_origin = np.array([0.0, 0.0, 0.0])
    root_z_world = np.array([0.0, 0.0, -1.0])

    built = build_tree(specs, root_origin=root_origin, root_z_world=root_z_world)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_text(
        "Human central airways (Weibel-inspired) – coloured by side/lobe",
        font_size=12,
    )

    for seg_id, seg in sorted(built.items()):
        pts = seg.points_world
        faces = seg.faces
        faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).ravel()
        mesh = pv.PolyData(pts, faces_flat)
        color = color_for_segment_id(seg_id)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, show_edges=False)

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    main()
