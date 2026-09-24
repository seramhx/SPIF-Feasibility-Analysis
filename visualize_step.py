"""
STEP FACE VISUALIZER — PyVista
================================
Loads STL + JSON face map.
Shows mesh in an interactive PyVista window with:
  - Each STEP face colored differently
  - Face ID label at the centroid of each STEP face
  - Window stays open while user reads labels
  - Then prompts for face ID selection in terminal
"""

import json
import numpy as np
import pyvista as pv
import trimesh
from pathlib import Path


def _distinct_colors(n):
    import colorsys
    colors = []
    for i in range(n):
        # Hue range skips greens (0.20–0.55) to avoid conflict with bright green selection color
        hue = i / n
        hue = 0.55 + hue * 0.70   # spans 0.55 to 1.25
        hue = hue % 1.0           # wrap around: covers blues/purples/reds/yellows

        # Extra safety: if hue falls in bright green range, shift it
        if 0.20 <= hue <= 0.55:
            hue = (hue + 0.30) % 1.0

        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
        colors.append([r, g, b])
    return colors


def load_mapping(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def visualize_and_select_faces(stl_file, json_file):
    """
    Opens a PyVista window showing the mesh with colored STEP faces
    and face ID labels at centroids.
    User rotates in the window, then types IDs in terminal.
    Returns list of selected STEP face tag strings.
    """
    mapping       = load_mapping(json_file)
    face_map      = mapping["step_surface_tag_to_stl_triangle_indices"]
    centroids_map = mapping["step_surface_tag_to_centroid"]
    areas_map     = mapping["step_surface_tag_to_area_mm2"]

    # Load mesh with trimesh first to get clean arrays
    tm = trimesh.load(stl_file, force='mesh')
    tm.fix_normals()

    vertices = np.array(tm.vertices, dtype=np.float32)
    faces    = np.array(tm.faces,    dtype=np.int32)

    # Build PyVista mesh
    # PyVista face array format: [3, v0, v1, v2,  3, v0, v1, v2, ...]
    n_faces   = len(faces)
    pv_faces  = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int32), faces
    ]).ravel()
    pv_mesh   = pv.PolyData(vertices, pv_faces)

    # Build per-face color scalar array (one integer per triangle)
    # Each STEP face gets a unique integer ID
    face_tags    = sorted(face_map.keys(), key=lambda x: int(x))
    n_step_faces = len(face_tags)
    colors_rgb   = _distinct_colors(n_step_faces)

    # Precompute reverse lookup: triangle index -> STEP tag
    tri_to_tag = {}
    for tag in face_tags:
        for tri_idx in face_map[tag]:
            if tri_idx < n_faces:
                tri_to_tag[tri_idx] = tag   # int key, no str()

    # Base color index per triangle (for distinct face colors)
    base_color_id = np.zeros(n_faces, dtype=np.int32)
    for color_idx, tag in enumerate(face_tags):
        for tri_idx in face_map[tag]:
            if tri_idx < n_faces:
                base_color_id[tri_idx] = color_idx

    # Display color array — will be mutated on pick
    # n_step_faces + 1 is reserved for "selected" (green)
    SELECTED_COLOR_ID = n_step_faces
    face_color_id = base_color_id.copy()
    pv_mesh.cell_data['face_id'] = face_color_id

    # Build label points and text
    label_points = []
    label_texts  = []
    for tag in face_tags:
        c = centroids_map.get(tag, [0.0, 0.0, 0.0])
        label_points.append(c)
        label_texts.append(str(tag))

    label_points = np.array(label_points, dtype=np.float32)
    label_cloud  = pv.PolyData(label_points)

    # Build a lookup table (LUT) for the distinct colors
    lut = pv.LookupTable()
    lut.n_values = n_step_faces + 1  # +1 for selected state

    # Distinct colors for unselected faces
    # Distinct colors for unselected faces (fully opaque)
    lut_colors = np.array(
        [[int(r*255), int(g*255), int(b*255), 255]  # changed from 220 → 255
        for r, g, b in colors_rgb],
        dtype=np.uint8
    )

    # Selected = bright green, fully opaque
    selected_color = np.array([[0, 255, 0, 255]], dtype=np.uint8)
    lut_colors = np.vstack([lut_colors, selected_color])

    lut.values = lut_colors

    selected_tags = set()
    selection_confirmed = [False]

    def update_colors():
        new_colors = base_color_id.copy()
        for tag in selected_tags:
            for tri_idx in face_map[tag]:
                if tri_idx < n_faces:
                    new_colors[tri_idx] = SELECTED_COLOR_ID
        pv_mesh.cell_data['face_id'] = new_colors
        pl.render()

    pl = pv.Plotter(window_size=[1400, 900], off_screen=False)
    pl.set_background('white')

    actor = pl.add_mesh(
        pv_mesh,
        scalars='face_id',
        cmap='tab20',        # placeholder; will be overridden by custom LUT
        clim=[0, n_step_faces],
        show_scalar_bar=False,
        show_edges=True,
        edge_color='black',
        line_width=0.3,
        pickable=True,
        opacity=1.0,  # fully opaque
    )

    # Attach your custom LUT to the mapper
    actor.GetMapper().SetLookupTable(lut)

    pl.add_point_labels(
        label_cloud,
        label_texts,
        font_size=14,
        font_family='arial',
        bold=True,
        text_color='black',
        point_color='red',
        point_size=8,
        render_points_as_spheres=True,
        always_visible=True,
        shadow=True,
    )

    pl.add_title(
        f"STEP Face Selection  —  {Path(stl_file).stem}\n"
        f"RIGHT CLICK to select/deselect  |  ENTER to confirm  |  R to reset",
        font_size=11,
        color='black',
    )
    pl.add_axes(line_width=3)

    # ── VTK right-click observer ──────────────────────────────────────────────
    import vtk

    def on_right_click(obj, event):
        x, y = pl.iren.interactor.GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.Pick(x, y, 0, pl.renderer)

        tri_id = picker.GetCellId()
        if tri_id < 0:
            return

        tag = tri_to_tag.get(tri_id)   # tri_to_tag keys are ints here
        if tag is None:
            return

        if tag in selected_tags:
            selected_tags.discard(tag)
            print(f"  Deselected face {tag}  (area: {areas_map.get(tag, 0):.1f} mm2)")
        else:
            selected_tags.add(tag)
            print(f"  Selected   face {tag}  (area: {areas_map.get(tag, 0):.1f} mm2)")

        update_colors()

    pl.iren.interactor.AddObserver('RightButtonPressEvent', on_right_click)

    def on_confirm():
        if not selected_tags:
            print("  No faces selected yet.")
            return
        selection_confirmed[0] = True
        pl.close()

    pl.add_key_event('Return', on_confirm)
    pl.add_key_event('r', lambda: [selected_tags.clear(),
                                    update_colors(),
                                    print("  Selection reset.")])

    print(f"\n  RIGHT CLICK : select / deselect a face (orange = selected)")
    print(f"  R           : reset selection")
    print(f"  ENTER       : confirm and continue\n")

    pl.show()

    # ── Terminal face selection ───────────────────────────────────────────
    # If user closed window without confirming, fall back to terminal
    if not selection_confirmed[0]:
        print("\n  Window closed without confirmation — falling back to terminal.")
        print(f"\nAvailable STEP face IDs:")
        for tag in face_tags:
            area = areas_map.get(tag, 0)
            print(f"  Face {tag:>4s}  —  area: {area:.1f} mm2")

        while True:
            raw = input(
                "\nEnter face IDs (comma-separated, e.g. 1,3,5): "
            ).strip()
            if not raw:
                print("  No input. Please enter at least one face ID.")
                continue
            parts   = [p.strip() for p in raw.replace(' ', '').split(',')]
            valid   = [p for p in parts if p in face_map]
            invalid = [p for p in parts if p not in face_map]
            if invalid:
                print(f"  Unknown IDs: {invalid}")
                ans = input("  Continue with valid only? (y/n): ").strip().lower()
                if ans != 'y':
                    continue
            if not valid:
                print("  No valid IDs. Try again.")
                continue
            print(f"\n  Selected: {valid}")
            confirm = input("  Confirm? (y/n): ").strip().lower()
            if confirm == 'y':
                return valid

    valid = sorted(selected_tags)
    print(f"\n  Confirmed selection: {valid}")
    for tag in valid:
        print(f"    Face {tag:>4s}  —  area: {areas_map.get(tag, 0):.1f} mm2")
    return valid


def get_selected_triangle_indices(selected_face_tags, mapping):
    """
    Return flat sorted list of STL triangle indices for selected STEP faces.
    """
    face_map    = mapping["step_surface_tag_to_stl_triangle_indices"]
    all_indices = []
    for tag in selected_face_tags:
        all_indices.extend(face_map.get(tag, []))
    return sorted(set(all_indices))