"""
SPIF COMBINED SHOWCASE — geometry / formability / accessibility / force
=========================================================================
Runs entirely off files already produced by the main pipeline
(step2stl.py + main.py --mode 5axis) -- it does NOT re-run ray casting or
GPU wall-angle computation. The 5-axis .npz already stores the full
(n_faces x n_dirs) direction_angles_all / accessible / depth_proj_all
matrices for every hemisphere direction, so the 3-axis result (which is just
a per-direction score search over that same matrix -- see analyze_3axis in
spif_analysis.py) is re-derived here with plain numpy, using the identical
formulas. No CUDA/GPU is required to run this script.

Wall-angle reference per strategy (see WALL ANGLE REFERENCE in
spif_analysis.py):
  3-axis     : each candidate direction = work-plane rotation (blank
               re-clamped normal to it), so direction_angles_all IS the wall
               angle for that candidate.
  multi-pass : fixed blank normal (npz['blank_normal']).
  5-axis     : fixed blank normal -- tool tilt only changes accessibility, so
               the 5-axis severity is the SAME blank-plane wall angle / force
               as multi-pass; what 5-axis adds is reachability.

Needs, per part (pass --stem, or the four paths individually):
  <stem>.STEP                 original CAD file (only its name is shown)
  <stem>.stl / <stem>.json    tessellated mesh + STEP-face map (step2stl.py)
  <stem>_5axis_db.npz         5-axis database (main.py --mode 5axis / both)

Opens four PyVista windows in sequence, each closing to open the next:

  1. GEOMETRY        — CAD (STL colored per original STEP face) vs the
                        tessellated analysis mesh (triangle edges shown)
  2. FORMABILITY      — single-pass 3-axis (optimized work plane) / multi-pass
                        (discrete pass-count colors) / 5-axis (fixed blank
                        plane, grey = unreachable by any allowed tilt)
  3. ACCESSIBILITY    — ray-cast reachability at the 3-axis tool direction
                        vs at each face's best 5-axis direction
  4. FORCE            — predicted forming force (Aerens et al. 2010) for
                        3-axis vs 5-axis

--material / --sheet_thickness / --tool_radius / --max_step_pass are only
used for quantities NOT already baked into the .npz (multi-pass pass counts,
and force, which needs sheet thickness + tool diameter) -- pass the same
values used to originally build the .npz for a fully consistent comparison.
The forming limit itself is read directly from the .npz, not recomputed.

Usage:
  python spif_showcase_all.py --stem test
  python spif_showcase_all.py --stem test2 --material DC01_steel --tool_radius 6.0
  python spif_showcase_all.py --step test.STEP --stl test.stl --json test.json \
                               --npz test_5axis_db.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from spif_analysis import (FORMING_LIMITS_DEG, estimate_forming_force_fz,
                            analyze_multipass, TOOL_DIAMETER_DEFAULT_MM,
                            load_5axis_database, force_model_caveat)
from spif_hover import enable_hover_tooltip, value_label


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_pyvista_mesh(vertices, faces):
    n = len(faces)
    pv_faces = np.hstack([
        np.full((n, 1), 3, dtype=np.int32), faces.astype(np.int32)
    ]).ravel()
    return pv.PolyData(vertices.astype(np.float32), pv_faces)


def _background_mesh(pl, vertices, faces, sel_idx, opacity=0.2):
    unsel_mask = np.ones(len(faces), dtype=bool)
    unsel_mask[sel_idx] = False
    unsel_faces = faces[unsel_mask]
    if len(unsel_faces) > 0:
        m = _build_pyvista_mesh(vertices, unsel_faces)
        pl.add_mesh(m, color='lightgrey', opacity=opacity, show_edges=False)


# Pass 1-5 fixed to green/yellow/blue/red/white (as requested); anything
# beyond 5 distinct pass counts continues with these extra distinct colors.
_DISTINCT_COUNT_COLORS = np.array([
    [46,  204, 113],   # green
    [241, 196, 15],    # yellow
    [41,  128, 185],   # blue
    [231, 76,  60],    # red
    [255, 255, 255],   # white
    [230, 126, 34],    # orange
    [155, 89,  182],   # purple
    [52,  73,  94],    # dark slate
    [26,  188, 156],   # teal
    [149, 165, 166],   # grey
], dtype=np.uint8)

_DISTINCT_COUNT_NAMES = [
    'green', 'yellow', 'blue', 'red', 'white',
    'orange', 'purple', 'dark slate', 'teal', 'grey',
]


def _discrete_int_lut(int_values):
    """Discrete per-integer-value LUT (e.g. pass counts). See spif_visualize.py."""
    values = np.asarray(int_values, dtype=np.int64)
    vmin = int(values.min())
    vmax = int(values.max())
    n_cat = vmax - vmin + 1
    category = (values - vmin).astype(np.int32)

    colors = np.zeros((n_cat, 4), dtype=np.uint8)
    for i in range(n_cat):
        colors[i, :3] = _DISTINCT_COUNT_COLORS[min(i, len(_DISTINCT_COUNT_COLORS) - 1)]
        colors[i, 3] = 255

    lut = pv.LookupTable()
    lut.n_values = n_cat
    lut.values = colors

    legend_lines = [
        (str(vmin + i), int((values == vmin + i).sum()), tuple(colors[i, :3]))
        for i in range(n_cat)
    ]
    return category, lut, legend_lines, vmin, vmax


def _bool_lut():
    """2-value LUT: 0 = blocked (red), 1 = accessible (green)."""
    lut = pv.LookupTable()
    lut.n_values = 2
    lut.values = np.array([
        [220, 50, 50, 255],   # 0: blocked
        [60, 180, 75, 255],   # 1: accessible
    ], dtype=np.uint8)
    return lut


def _scalar_bar(title):
    return dict(
        title=title, n_labels=5, label_font_size=13, title_font_size=14,
        position_x=0.82, position_y=0.05, height=0.50, width=0.12,
        vertical=True, fmt='%.1f',
    )


def _wall_margin_scalars(wall_angles, forming_limit, ceiling=1.5):
    return np.clip(wall_angles / (forming_limit + 1e-9), 0.0, ceiling)


def _inaccessible_label(local_cell_id):
    return "Inaccessible / infeasible\n(no reliable value here)"


def _make_wall_angle_label(wall_angles_deg, forming_limit_deg):
    def _label(local_cell_id):
        angle = float(wall_angles_deg[local_cell_id])
        return value_label("Wall angle", angle, fmt='{:.1f}', unit='°',
                           pass_fail=(angle <= forming_limit_deg),
                           extra=f"limit {forming_limit_deg:.0f}°")
    return _label


def _make_force_label(force_N):
    def _label(local_cell_id):
        return value_label("Predicted force", float(force_N[local_cell_id]),
                           fmt='{:.0f}', unit=' N')
    return _label


def _make_passes_label(passes_required):
    def _label(local_cell_id):
        n = int(passes_required[local_cell_id])
        return value_label("Passes required", n, fmt='{:d}',
                           pass_fail=(n == 1),
                           pass_word='PASS (single pass)',
                           fail_word=f'FAIL (needs {n} passes)')
    return _label


def _make_accessibility_label(accessible_bool, reach_depth_mm=None):
    """
    reach_depth_mm : optional per-face array, same cell order as
        accessible_bool -- the measured insertion depth (mm) at this
        face+direction (see _compute_reach_depth in spif_analysis.py), only
        available if the .npz was built with --tool_length. Shown as an
        extra tooltip line when present; omitted otherwise.
    """
    def _label(local_cell_id):
        ok = bool(accessible_bool[local_cell_id])
        extra = (f"Tool reach: {reach_depth_mm[local_cell_id]:.1f}mm"
                 if reach_depth_mm is not None else '')
        return value_label("Accessibility", 'reachable' if ok else 'blocked',
                           fmt='{}', pass_fail=ok, extra=extra)
    return _label


def _step_face_boundary_edges(vertices, tm, face_id):
    """
    BREP-style edges: mesh edges where the two adjacent triangles belong to
    different original STEP surfaces (i.e. the true face-to-face boundary
    curves of the CAD model), as a black polyline overlay.
    """
    adjacency = tm.face_adjacency
    boundary_mask = face_id[adjacency[:, 0]] != face_id[adjacency[:, 1]]
    boundary_edges = tm.face_adjacency_edges[boundary_mask]

    if len(boundary_edges) == 0:
        return None

    n = len(boundary_edges)
    lines = np.hstack([
        np.full((n, 1), 2, dtype=np.int32), boundary_edges.astype(np.int32)
    ]).ravel()
    edges_poly = pv.PolyData(vertices.astype(np.float32))
    edges_poly.lines = lines
    return edges_poly


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_part(step_path, stl_path, json_path, npz_path):
    tm = trimesh.load(stl_path, force='mesh')
    tm.fix_normals()
    vertices = np.array(tm.vertices, dtype=np.float32)
    faces = np.array(tm.faces, dtype=np.int32)

    with open(json_path, 'r') as f:
        mapping = json.load(f)

    npz = load_5axis_database(npz_path)

    sel_idx = npz['face_indices_global'].astype(int)
    if sel_idx.max() >= len(faces):
        raise ValueError(
            f"{npz_path} references triangle {sel_idx.max()} but {stl_path} "
            f"only has {len(faces)} triangles -- these files don't match.")

    return {
        'step_path': step_path,
        'tm': tm,
        'vertices': vertices,
        'faces': faces,
        'mapping': mapping,
        'npz': npz,
        'sel_idx': sel_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RE-DERIVE 3-AXIS RESULT FROM THE 5-AXIS DATABASE (pure numpy, no GPU)
# ─────────────────────────────────────────────────────────────────────────────

def recompute_3axis(npz):
    """
    Reproduces analyze_3axis()'s per-direction scoring (spif_analysis.py)
    from the full (n_faces x n_dirs) matrices already saved in the 5-axis
    .npz, vectorized over all directions at once instead of looping.
    Each candidate direction is a work-plane rotation, so the per-direction
    angle matrix is the wall angle for that candidate orientation. The
    baseline (no rotation) is the fixed blank normal: blank_wall_angles /
    nominal_accessible from the .npz.
    """
    wall_angles_all = npz['direction_angles_all']  # (n_faces, n_dirs)
    accessible = npz['accessible']                 # (n_faces, n_dirs)
    depth_proj_all = npz['depth_proj_all']          # (n_faces, n_dirs)
    face_areas = npz['face_areas']                  # (n_faces,)
    directions = npz['directions']                  # (n_dirs, 3)
    forming_limit_deg = float(npz['forming_limit_deg'][0])

    total_area = float(face_areas.sum())
    area_col = face_areas[:, None]

    wall_viol_matrix = wall_angles_all > forming_limit_deg
    blocked_matrix = ~accessible
    combined_matrix = wall_viol_matrix | blocked_matrix

    wall_violation_pct = (area_col * wall_viol_matrix).sum(axis=0) / total_area * 100.0
    blocked_pct = (area_col * blocked_matrix).sum(axis=0) / total_area * 100.0
    combined_violation_pct = (area_col * combined_matrix).sum(axis=0) / total_area * 100.0

    draw_distance_mm = depth_proj_all.max(axis=0) - depth_proj_all.min(axis=0)
    draw_uniformity = depth_proj_all.std(axis=0)

    nv = combined_violation_pct / (combined_violation_pct.max() + 1e-9)
    nd = draw_distance_mm / (draw_distance_mm.max() + 1e-9)
    nu = draw_uniformity / (draw_uniformity.max() + 1e-9)
    score = 0.60 * nv + 0.25 * nd + 0.15 * nu

    best_idx = int(np.argmin(score))
    best_direction = directions[best_idx]
    best_wall_angles = wall_angles_all[:, best_idx]
    best_accessible = accessible[:, best_idx]

    return {
        'forming_limit_deg': forming_limit_deg,
        'directions': directions,
        'best_idx': best_idx,
        'best_direction': best_direction,
        'best_wall_angles': best_wall_angles,
        'best_accessible': best_accessible,
        'combined_violation_pct': combined_violation_pct,
        'baseline_direction': npz['blank_normal'],
        'baseline_wall_angles': npz['blank_wall_angles'],
        'baseline_accessible': npz['nominal_accessible'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 1 — GEOMETRY: CAD (STEP faces) vs TESSELLATED MESH
# ─────────────────────────────────────────────────────────────────────────────



def _make_face_id_array(n_tri, face_map, face_tags):
    """Assign each triangle its STEP face index. Raises loudly instead of
    silently defaulting unmapped triangles to face 0 (that silent default
    is what causes the black speckle artifact)."""
    face_id = np.full(n_tri, -1, dtype=np.int32)
    for color_idx, tag in enumerate(face_tags):
        idxs = np.asarray(face_map[tag], dtype=np.int64)
        bad = idxs[(idxs < 0) | (idxs >= n_tri)]
        if len(bad):
            raise ValueError(
                f"face tag {tag}: {len(bad)} triangle indices out of range "
                f"(n_tri={n_tri}), e.g. {bad[:5]}"
            )
        face_id[idxs] = color_idx

    unmapped = np.count_nonzero(face_id == -1)
    if unmapped:
        raise ValueError(
            f"{unmapped}/{n_tri} triangles were never assigned a STEP face "
            f"id -- face_map does not cover the whole mesh. Fix the mapping "
            f"before extracting boundary edges."
        )
    return face_id


def _extract_brep_boundary_polydata(vertices, faces, face_id):
    """Return a PolyData of the true BREP face-boundary edges (edges whose
    two adjacent triangles belong to different STEP faces, or edges that
    belong to only one triangle) as proper connected line segments."""
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be an (n_tri, 3) triangle index array")

    edge_owner = {}  # (v0, v1) sorted -> list of face_id values touching it
    for tri_idx, tri in enumerate(faces):
        fid = face_id[tri_idx]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            edge_owner.setdefault(key, []).append(fid)

    boundary_edges = [
        key for key, owners in edge_owner.items()
        if len(owners) == 1 or len(set(owners)) > 1
    ]

    if not boundary_edges:
        return None

    n_edges = len(boundary_edges)
    lines = np.empty((n_edges, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1:] = np.asarray(boundary_edges, dtype=np.int64)

    poly = pv.PolyData()
    poly.points = np.asarray(vertices, dtype=np.float64)
    poly.lines = lines.reshape(-1)
    return poly


def _make_pv_trimesh(vertices, faces):
    faces = np.asarray(faces)
    n_tri = faces.shape[0]
    cells = np.hstack([np.full((n_tri, 1), 3, dtype=np.int64), faces]).reshape(-1)
    return pv.PolyData(np.asarray(vertices, dtype=np.float64), cells)


def show_geometry(part, step_path):
    vertices, faces, mapping = part['vertices'], part['faces'], part['mapping']
    face_map = mapping['step_surface_tag_to_stl_triangle_indices']
    face_tags = sorted(face_map.keys(), key=lambda x: int(x))
    n_step_faces = len(face_tags)
    n_tri = len(faces)

    face_id = _make_face_id_array(n_tri, face_map, face_tags)
    edges_poly = _extract_brep_boundary_polydata(vertices, faces, face_id)

    pl = pv.Plotter(shape=(1, 2), window_size=[1800, 900], border=False)
    pl.set_background('white')

    # Panel 1: CAD -- single flat color, BREP face-boundary edges in black
    pl.subplot(0, 0)
    m1 = _make_pv_trimesh(vertices, faces)
    pl.add_mesh(m1, color='lightgrey', show_edges=False, lighting=False)
    if edges_poly is not None:
        pl.add_mesh(edges_poly, color='black', line_width=2.0)
    pl.add_text(
        f"CAD GEOMETRY  —  {Path(step_path).name}\n"
        f"{n_step_faces} STEP surfaces -- black lines = BREP face boundaries\n"
        f"(rendered via its STL tessellation -- no native BREP viewer here)",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)

    # Panel 2: tessellated analysis mesh -- triangulation shown via edges
    pl.subplot(0, 1)
    m2 = _make_pv_trimesh(vertices, faces)
    pl.add_mesh(m2, color='lightsteelblue', show_edges=True,
                edge_color='black', line_width=0.4)
    mesh_params = mapping.get('mesh_params', {})
    pl.add_text(
        f"TESSELLATED MESH  —  {Path(mapping.get('stl_file', '')).name or 'STL'}\n"
        f"Triangles: {n_tri}   Vertices: {len(vertices)}\n"
        + (f"Mesh size: min {mesh_params.get('minh', 0):.2f}  "
           f"max {mesh_params.get('maxh', 0):.2f} mm\n" if mesh_params else ""),
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)

    pl.link_views()
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 2 — FORMABILITY: single-pass 3-axis / multi-pass 3-axis / single-pass 5-axis
# ─────────────────────────────────────────────────────────────────────────────

def show_formability(part, r3, multipass_res, material_key,
                     accessible_multipass=None):
    """
    accessible_multipass : optional bool array aligned with sel_idx, used to
        grey out inaccessible faces in Panel 2 (see caller in main() -- it's
        npz['nominal_accessible'], accessibility at the blank normal, since
        this showcase script deliberately does no GPU ray casting of its own).
        None: Panel 2 colors every selected face, unchanged from before this
        parameter existed.
    """
    vertices, faces, sel_idx = part['vertices'], part['faces'], part['sel_idx']
    npz = part['npz']
    sel_faces = faces[sel_idx]
    forming_limit = r3['forming_limit_deg']

    pl = pv.Plotter(shape=(1, 3), window_size=[2400, 900], border=False)
    pl.set_background('white')
    hover_layers = {0: [], 1: [], 2: []}

    # Panel 1: single-pass 3-axis (margin to limit at optimized direction).
    # Faces blocked at the optimized direction (r3['best_accessible']) are
    # painted flat grey instead of by wall-angle margin -- an unreachable
    # face doesn't have a meaningful "will it fracture" answer here, same
    # treatment as Panel 3's 5-axis map below.
    # This panel carries the ONLY scalar bar for the margin scale -- panel 3
    # shares the exact same clim/cmap, so its bar is hidden to avoid a
    # redundant, overlapping second legend.
    pl.subplot(0, 0)
    _background_mesh(pl, vertices, faces, sel_idx)
    best_accessible = np.asarray(r3['best_accessible'], dtype=bool)
    margin3 = _wall_margin_scalars(r3['best_wall_angles'], forming_limit)
    if best_accessible.any():
        m1 = _build_pyvista_mesh(vertices, sel_faces[best_accessible])
        m1.cell_data['margin'] = margin3[best_accessible]
        actor1 = pl.add_mesh(
            m1, scalars='margin', cmap='RdYlGn_r', clim=[0.0, 1.5],
            show_edges=True, edge_color='black', line_width=0.3,
            annotations={1.0: 'FAIL'},
            scalar_bar_args=_scalar_bar(f'Wall angle / {forming_limit:.0f}\xb0  (shared, both panels)'),
        )
        hover_layers[0].append({'actor': actor1, 'label_fn': _make_wall_angle_label(
            r3['best_wall_angles'][best_accessible], forming_limit)})
    if (~best_accessible).any():
        m1g = _build_pyvista_mesh(vertices, sel_faces[~best_accessible])
        actor1g = pl.add_mesh(m1g, color='grey', show_edges=True,
                              edge_color='black', line_width=0.3)
        hover_layers[0].append({'actor': actor1g, 'label_fn': _inaccessible_label})
    bd = r3['best_direction']
    viol3 = float((margin3[best_accessible] > 1.0).sum()) if best_accessible.any() else 0.0
    n_blocked3 = int((~best_accessible).sum())
    pl.add_text(
        f"FORMABILITY  —  material: {material_key}  |  limit: {forming_limit:.0f}\xb0\n\n"
        f"SINGLE-PASS  —  3-AXIS (optimized work-plane orientation)\n"
        f"Dir: [{bd[0]:.2f},{bd[1]:.2f},{bd[2]:.2f}]\n"
        f"Violations: {int(viol3)} / {len(sel_idx)} faces\n"
        f"Inaccessible (grey): {n_blocked3} faces\n"
        f"(green=safe, red=at/beyond FAIL line = forming limit)",
        position='upper_left', font_size=10, color='black',
    )

    # Panel 2: multi-pass 3-axis (discrete pass count, named colors).
    # Faces blocked at the nominal direction (accessible_multipass, if given)
    # are painted flat grey instead of by pass count.
    pl.subplot(0, 1)
    _background_mesh(pl, vertices, faces, sel_idx)
    passes_required = np.asarray(multipass_res['passes_required'])

    if accessible_multipass is not None:
        acc_mp = np.asarray(accessible_multipass, dtype=bool)
        legend_text = "(no accessible faces)"
        if acc_mp.any():
            category, lut, legend_lines, vmin, vmax = _discrete_int_lut(
                passes_required[acc_mp])
            m2 = _build_pyvista_mesh(vertices, sel_faces[acc_mp])
            m2.cell_data['pass_cat'] = category
            clim = [0, vmax - vmin] if vmax > vmin else [-0.5, 0.5]
            actor2 = pl.add_mesh(
                m2, scalars='pass_cat', clim=clim, show_scalar_bar=False,
                show_edges=True, edge_color='black', line_width=0.3,
            )
            actor2.GetMapper().SetLookupTable(lut)
            legend_text = "\n".join(
                f"  {label} pass{'es' if label != '1' else ''} "
                f"({_DISTINCT_COUNT_NAMES[i % len(_DISTINCT_COUNT_NAMES)]}): {count} faces"
                for i, (label, count, _) in enumerate(legend_lines))
            hover_layers[1].append({'actor': actor2, 'label_fn': _make_passes_label(
                passes_required[acc_mp])})
        if (~acc_mp).any():
            m2g = _build_pyvista_mesh(vertices, sel_faces[~acc_mp])
            actor2g = pl.add_mesh(m2g, color='grey', show_edges=True,
                                  edge_color='black', line_width=0.3)
            hover_layers[1].append({'actor': actor2g, 'label_fn': _inaccessible_label})
        n_blocked_mp = int((~acc_mp).sum())
        blocked_mp_line = f"Inaccessible at nominal axis (grey): {n_blocked_mp} faces\n"
    else:
        category, lut, legend_lines, vmin, vmax = _discrete_int_lut(passes_required)
        m2 = _build_pyvista_mesh(vertices, sel_faces)
        m2.cell_data['pass_cat'] = category
        clim = [0, vmax - vmin] if vmax > vmin else [-0.5, 0.5]
        actor2 = pl.add_mesh(
            m2, scalars='pass_cat', clim=clim, show_scalar_bar=False,
            show_edges=True, edge_color='black', line_width=0.3,
        )
        actor2.GetMapper().SetLookupTable(lut)
        legend_text = "\n".join(
            f"  {label} pass{'es' if label != '1' else ''} "
            f"({_DISTINCT_COUNT_NAMES[i % len(_DISTINCT_COUNT_NAMES)]}): {count} faces"
            for i, (label, count, _) in enumerate(legend_lines))
        blocked_mp_line = ""
        hover_layers[1].append({'actor': actor2, 'label_fn': _make_passes_label(passes_required)})

    pl.add_text(
        f"MULTI-PASS  —  3-AXIS (nominal direction, staged forming)\n"
        f"Pass count: Tier 2 practice-based rule\n"
        f"Max passes: {int(passes_required.max())}\n"
        f"{blocked_mp_line}\n{legend_text}",
        position='upper_left', font_size=10, color='black',
    )

    # Panel 3: single-pass 5-axis. The sheet stays clamped normal to the
    # blank normal, so the wall angle is the FIXED blank-plane angle for
    # every face whatever the tool tilt (see WALL ANGLE REFERENCE in
    # spif_analysis.py). Tilt only decides reachability: faces with no
    # collision-free, within-reach, within-tilt-bound direction (access_ok
    # False) are grey.
    pl.subplot(0, 2)
    _background_mesh(pl, vertices, faces, sel_idx)
    blank_wa = npz['blank_wall_angles']
    access_ok = np.asarray(npz['access_ok'], dtype=bool)
    grey_mask = ~access_ok
    colored_mask = access_ok

    if colored_mask.any():
        margin5 = _wall_margin_scalars(blank_wa[colored_mask], forming_limit)
        m3 = _build_pyvista_mesh(vertices, sel_faces[colored_mask])
        m3.cell_data['margin'] = margin5
        actor3 = pl.add_mesh(
            m3, scalars='margin', cmap='RdYlGn_r', clim=[0.0, 1.5],
            show_edges=True, edge_color='black', line_width=0.3,
            show_scalar_bar=False,
        )
        hover_layers[2].append({'actor': actor3, 'label_fn': _make_wall_angle_label(
            blank_wa[colored_mask], forming_limit)})
    if grey_mask.any():
        m3g = _build_pyvista_mesh(vertices, sel_faces[grey_mask])
        actor3g = pl.add_mesh(m3g, color='grey', show_edges=True,
                              edge_color='black', line_width=0.3)
        hover_layers[2].append({'actor': actor3g, 'label_fn': _inaccessible_label})

    n_grey = int(grey_mask.sum())
    n_wall_fail = int((colored_mask & ~np.asarray(npz['wall_ok'], dtype=bool)).sum())
    pl.add_text(
        f"SINGLE-PASS  —  5-AXIS (tilting tool, sheet clamped at nominal)\n"
        f"Wall angle vs. FIXED blank plane -- tilt does not change it\n"
        f"Wall angle fail, reachable (red): {n_wall_fail} faces\n"
        f"Unreachable by any allowed tilt (grey): {n_grey} faces\n"
        f"(green=safe, red=at/beyond FAIL line, grey=unreachable)",
        position='upper_left', font_size=10, color='black',
    )

    pl.link_views()
    enable_hover_tooltip(pl, hover_layers)
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 3 — ACCESSIBILITY: 3-axis vs 5-axis
# ─────────────────────────────────────────────────────────────────────────────

def show_accessibility(part, r3, tool_diameter_mm):
    vertices, faces, sel_idx = part['vertices'], part['faces'], part['sel_idx']
    npz = part['npz']
    sel_faces = faces[sel_idx]

    best_dir_indices = npz['best_dir_indices']
    n_faces = len(sel_idx)
    # Reachable with SOME collision-free, within-reach direction inside the
    # tilt bound (best_dir_indices points at the minimum-tilt such direction).
    accessible_5axis = np.asarray(npz['access_ok'], dtype=bool)

    # Per-face reach depth (mm) at the direction each panel actually shows --
    # only present in the .npz if main.py was run with --tool_length (see
    # _compute_accessibility / build_5axis_database). None if not available,
    # in which case the hover tooltip just omits that line.
    reach_depth_matrix = npz['reach_depth_mm'] if 'reach_depth_mm' in npz.files else None
    reach_3axis = (reach_depth_matrix[:, r3['best_idx']]
                   if reach_depth_matrix is not None else None)
    reach_5axis = (reach_depth_matrix[np.arange(n_faces), best_dir_indices]
                   if reach_depth_matrix is not None else None)

    # The ray-casting clearance ring radius used to build the .npz's
    # `accessible` matrix isn't itself stored in the .npz -- only the
    # resulting booleans are. tool_diameter_mm below is THIS script's
    # --tool_radius setting (shown for both panels, since both read off the
    # same saved matrix), not necessarily what main.py used originally.
    panels = [
        (r3['best_accessible'], reach_3axis,
         f"3-AXIS  (optimized direction "
         f"[{r3['best_direction'][0]:.2f},{r3['best_direction'][1]:.2f},"
         f"{r3['best_direction'][2]:.2f}])"),
        (accessible_5axis, reach_5axis,
         "5-AXIS  (each face's minimum-tilt collision-free direction)"),
    ]

    pl = pv.Plotter(shape=(1, 2), window_size=[1800, 900], border=False)
    pl.set_background('white')
    hover_layers = {}

    for col, (acc, reach_depth, label) in enumerate(panels):
        acc = np.asarray(acc, dtype=bool)
        pl.subplot(0, col)
        _background_mesh(pl, vertices, faces, sel_idx)
        m = _build_pyvista_mesh(vertices, sel_faces)
        m.cell_data['acc'] = acc.astype(np.int32)
        actor = pl.add_mesh(
            m, scalars='acc', clim=[0, 1], show_scalar_bar=False,
            show_edges=True, edge_color='black', line_width=0.3,
        )
        actor.GetMapper().SetLookupTable(_bool_lut())
        hover_layers[col] = [{'actor': actor, 'label_fn': _make_accessibility_label(
            acc, reach_depth_mm=reach_depth)}]
        n_acc = int(acc.sum())
        n_blocked = len(acc) - n_acc
        pl.add_text(
            f"{label}\n"
            f"Tool diameter checked: {tool_diameter_mm:.1f} mm\n"
            f"  Accessible : {n_acc} faces\n"
            f"  Blocked    : {n_blocked} faces\n"
            f"(green=reachable, red=blocked by ray casting)",
            position='upper_left', font_size=11, color='black',
        )
        pl.add_axes(line_width=3)

    pl.link_views()
    enable_hover_tooltip(pl, hover_layers)
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 4 — FORCE: 3-axis vs 5-axis
# ─────────────────────────────────────────────────────────────────────────────

def show_force(part, r3, material_key, sheet_thickness_mm, tool_diameter_mm):
    vertices, faces, sel_idx = part['vertices'], part['faces'], part['sel_idx']
    npz = part['npz']
    sel_faces = faces[sel_idx]

    force_3axis = estimate_forming_force_fz(
        r3['best_wall_angles'], sheet_thickness_mm, material_key,
        tool_diameter_mm=tool_diameter_mm)

    # 5-axis force: fixed blank-plane wall angle and t0 (tool tilt does not
    # change it). Recomputed with THIS script's material/t0/tool so both
    # panels share identical inputs.
    force_5axis = estimate_forming_force_fz(
        npz['blank_wall_angles'], sheet_thickness_mm, material_key,
        tool_diameter_mm=tool_diameter_mm)

    # Panel 1 (3-axis): grey where the optimized direction is blocked.
    # Panel 2 (5-axis): grey where the face is infeasible (wall angle beyond
    # the formable range the force model was fitted on, or unreachable).
    best_accessible = np.asarray(r3['best_accessible'], dtype=bool)
    is_feasible_5ax = np.asarray(npz['is_feasible'], dtype=bool)

    shown = np.concatenate([force_3axis[best_accessible], force_5axis[is_feasible_5ax]])
    fmax = (float(shown.max()) if len(shown) else 1.0) + 1e-6
    caveat = force_model_caveat(material_key)

    panels = [
        (force_3axis, "3-AXIS  (optimized work-plane orientation)", best_accessible, "Inaccessible"),
        (force_5axis, "5-AXIS  (fixed blank plane -- tilt-independent)", is_feasible_5ax, "Infeasible"),
    ]

    pl = pv.Plotter(shape=(1, 2), window_size=[1800, 900], border=False)
    pl.set_background('white')
    hover_layers = {}

    for col, (force, label, ok_mask, grey_label) in enumerate(panels):
        pl.subplot(0, col)
        _background_mesh(pl, vertices, faces, sel_idx)
        hover_layers[col] = []

        if ok_mask.any():
            m = _build_pyvista_mesh(vertices, sel_faces[ok_mask])
            m.cell_data['force_N'] = force[ok_mask]
            actor = pl.add_mesh(
                m, scalars='force_N', cmap='inferno', clim=[0.0, fmax],
                show_edges=True, edge_color='black', line_width=0.3,
                scalar_bar_args=_scalar_bar('Predicted axial force Fz_s (N)'),
            )
            hover_layers[col].append({'actor': actor, 'label_fn': _make_force_label(force[ok_mask])})
        if (~ok_mask).any():
            mg = _build_pyvista_mesh(vertices, sel_faces[~ok_mask])
            actor_g = pl.add_mesh(mg, color='grey', show_edges=True,
                                  edge_color='black', line_width=0.3)
            hover_layers[col].append({'actor': actor_g, 'label_fn': _inaccessible_label})

        n_grey = int((~ok_mask).sum())
        fmean = float(force[ok_mask].mean()) if ok_mask.any() else float('nan')
        fmax_shown = float(force[ok_mask].max()) if ok_mask.any() else float('nan')
        grey_line = f"{grey_label} (grey): {n_grey} faces\n" if n_grey else ""
        pl.add_text(
            f"{label}\n"
            f"Material: {material_key}  |  t0={sheet_thickness_mm:.2f}mm  |  "
            f"tool dia={tool_diameter_mm:.1f}mm\n"
            f"Max: {fmax_shown:.0f} N   Mean: {fmean:.0f} N\n"
            f"{grey_line}"
            + (f"CAVEAT: {caveat}\n" if caveat else "")
            + f"(shared color scale across both panels; relative effort proxy)",
            position='upper_left', font_size=10, color='black',
        )
        pl.add_axes(line_width=3)

    pl.link_views()
    enable_hover_tooltip(pl, hover_layers)
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_step_path(stem):
    for ext in ('.STEP', '.step', '.stp', '.STP'):
        p = Path(stem + ext)
        if p.exists():
            return str(p)
    return stem + '.STEP'   # fall back to the name even if not found


def main():
    parser = argparse.ArgumentParser(
        description='SPIF combined showcase: geometry, formability, '
                     'accessibility, and force -- 3-axis vs 5-axis, side by side.')
    parser.add_argument('--stem', default=None,
                         help='Path stem, e.g. "test" -> test.STEP, test.stl, '
                              'test.json, test_5axis_db.npz')
    parser.add_argument('--step', default=None)
    parser.add_argument('--stl', default=None)
    parser.add_argument('--json', default=None)
    parser.add_argument('--npz', default=None)
    parser.add_argument('--material', default='AA1050',
                         choices=list(FORMING_LIMITS_DEG.keys()),
                         help='Only affects force (t^1.57 term via Rm) and '
                              'multi-pass force -- forming limit itself is '
                              'read from the .npz, not recomputed.')
    parser.add_argument('--sheet_thickness', type=float, default=1.0,
                         help='Sheet thickness t0 (mm) -- for force + multi-pass.')
    parser.add_argument('--tool_radius', type=float, default=0.0,
                         help='Tool radius (mm) -- drives force tool diameter '
                              '(2x radius); 10mm fallback if 0, matching main.py.')
    parser.add_argument('--max_step_pass', type=float, default=10.0,
                         help='Max wall-angle step per multi-pass stage (deg).')
    parser.add_argument('--windows', default='geometry,formability,accessibility,force',
                         help='Comma-separated subset/order of windows to show.')
    args = parser.parse_args()

    if args.stem:
        step_path = args.step or _resolve_step_path(args.stem)
        stl_path = args.stl or (args.stem + '.stl')
        json_path = args.json or (args.stem + '.json')
        npz_path = args.npz or (args.stem + '_5axis_db.npz')
    else:
        missing = [n for n, v in
                   [('--step', args.step), ('--stl', args.stl),
                    ('--json', args.json), ('--npz', args.npz)] if v is None]
        if missing:
            parser.error(f"Either pass --stem, or all of: {missing}")
        step_path, stl_path, json_path, npz_path = (
            args.step, args.stl, args.json, args.npz)

    if args.tool_radius > 0:
        tool_diameter_mm = 2.0 * args.tool_radius
    else:
        tool_diameter_mm = TOOL_DIAMETER_DEFAULT_MM

    print(f"\n{'#'*60}")
    print(f"SPIF COMBINED SHOWCASE")
    print(f"{'#'*60}")
    print(f"  STEP : {step_path}")
    print(f"  STL  : {stl_path}")
    print(f"  JSON : {json_path}")
    print(f"  NPZ  : {npz_path}")
    print(f"  Material: {args.material}  |  t0={args.sheet_thickness}mm  |  "
          f"tool dia={tool_diameter_mm}mm  |  max_step_pass={args.max_step_pass}deg")

    part = load_part(step_path, stl_path, json_path, npz_path)
    forming_limit = float(part['npz']['forming_limit_deg'][0])
    print(f"  Forming limit (from .npz): {forming_limit:.1f} deg")
    print(f"  Selected faces: {len(part['sel_idx'])}")

    windows = [w.strip() for w in args.windows.split(',') if w.strip()]

    if 'geometry' in windows:
        print(f"\n[1] Geometry: CAD (STEP faces) vs tessellated mesh...")
        show_geometry(part, step_path)

    r3 = None
    if any(w in windows for w in ('formability', 'accessibility', 'force')):
        r3 = recompute_3axis(part['npz'])
        print(f"\n  3-axis optimized direction: "
              f"[{r3['best_direction'][0]:.3f},{r3['best_direction'][1]:.3f},"
              f"{r3['best_direction'][2]:.3f}]  "
              f"(combined violation {r3['combined_violation_pct'][r3['best_idx']]:.1f}%)")

    if 'formability' in windows:
        print(f"\n[2] Formability: single-pass 3-axis / multi-pass 3-axis / "
              f"single-pass 5-axis...")
        sel_idx = part['sel_idx']
        face_normals = np.array(part['tm'].face_normals, dtype=np.float32)[sel_idx]
        face_areas = part['npz']['face_areas']
        nominal_dir = part['npz']['blank_normal']   # the confirmed hemisphere
                                                     # axis (fixed clamping)
        multipass_res = analyze_multipass(
            face_normals=face_normals,
            face_areas=face_areas,
            forming_limit_deg=forming_limit,
            max_step_per_pass_deg=args.max_step_pass,
            t0_mm=args.sheet_thickness,
            material_key=args.material,
            tool_diameter_mm=tool_diameter_mm,
            nominal_dir=nominal_dir,
        )
        # Accessibility at the blank normal, as saved by the 5-axis build
        # (exact for new runs, where the confirmed axis is directions[0];
        # nearest-sample approximation for databases regenerated from runs
        # made before that change -- see regenerate_5axis_db.py).
        accessible_multipass = part['npz']['nominal_accessible']
        show_formability(part, r3, multipass_res, args.material,
                         accessible_multipass=accessible_multipass)

    if 'accessibility' in windows:
        print(f"\n[3] Accessibility: 3-axis vs 5-axis reachability...")
        show_accessibility(part, r3, tool_diameter_mm)

    if 'force' in windows:
        print(f"\n[4] Force: 3-axis vs 5-axis predicted forming force...")
        show_force(part, r3, args.material, args.sheet_thickness, tool_diameter_mm)

    print(f"\n{'#'*60}")
    print(f"SHOWCASE COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
