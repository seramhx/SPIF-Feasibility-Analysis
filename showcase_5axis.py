"""
5-AXIS DATABASE SHOWCASE VISUALIZER
===================================
The sheet is clamped ONCE (blank normal = confirmed hemisphere axis); only
the tool tilts. Wall angle, sine-law thinning and force are therefore fixed
per face by the blank plane -- tool tilt changes ACCESSIBILITY only. See
WALL ANGLE REFERENCE in spif_analysis.py.

Window 1 — FEASIBILITY MAP (root cause per face)
Window 2 — TILT MAP (tilt of the minimum-tilt reachable direction)
Window 3 — FORMING SEVERITY: fixed-plane wall angle vs limit | sine-law thinning
Window 4 — DIRECTION FIELD (arrows colored by tool-approach angle, kinematic only)
Window 5 — CRITICAL FACES (feasible faces closest to the forming limit)
Window 6 — CRITICAL TILT FACES (highest tool tilt)
Window 7 — MULTI-PASS: 3-axis vs 5-axis REACHABILITY (pass counts are the
           same for both -- tilt only changes which faces can be reached)
Window 8 — PREDICTED FORMING FORCE (Aerens et al. 2010, fixed-plane wall angle, t0)

Requires a schema-v2 database (regenerate older ones with
regenerate_5axis_db.py).
"""

import argparse
import numpy as np
import pyvista as pv
import trimesh
from pathlib import Path

from spif_hover import enable_hover_tooltip, value_label
from spif_analysis import load_5axis_database, force_model_caveat


def _inaccessible_label(local_cell_id):
    return "Infeasible\n(no reliable value here)"


def _unreachable_label(local_cell_id):
    return "Unreachable\n(no collision-free direction within bounds)"


def _make_approach_label(approach_deg):
    def _label(local_cell_id):
        return value_label("Tool-approach angle", float(approach_deg[local_cell_id]),
                           fmt='{:.1f}', unit='°',
                           extra="kinematic only -- not a wall angle")
    return _label


def _make_thinning_label(thinning_pct, wall_ok):
    def _label(local_cell_id):
        return value_label("Sine-law thinning", float(thinning_pct[local_cell_id]),
                           fmt='{:.1f}', unit=' %',
                           pass_fail=bool(wall_ok[local_cell_id]),
                           fail_word='FAIL (below critical thickness)')
    return _label


def _make_wall_angle_label(wall_angles_deg, forming_limit_deg):
    def _label(local_cell_id):
        angle = float(wall_angles_deg[local_cell_id])
        return value_label("Wall angle", angle, fmt='{:.1f}', unit='°',
                           pass_fail=(angle <= forming_limit_deg),
                           extra=f"limit {forming_limit_deg:.0f}°")
    return _label


def _make_tilt_label(tilt_deg):
    def _label(local_cell_id):
        return value_label("Tilt from nominal", float(tilt_deg[local_cell_id]),
                           fmt='{:.1f}', unit='°')
    return _label


def _make_force_label(force_N):
    def _label(local_cell_id):
        return value_label("Predicted force", float(force_N[local_cell_id]),
                           fmt='{:.0f}', unit=' N')
    return _label


def _make_passes_label(passes, saved=False):
    def _label(local_cell_id):
        n = int(passes[local_cell_id])
        if saved:
            return value_label("Passes saved (5-axis vs 3-axis)", n, fmt='{:d}')
        return value_label("Passes required", n, fmt='{:d}',
                           pass_fail=(n == 1),
                           pass_word='PASS (single pass)',
                           fail_word=f'FAIL (needs {n} passes)')
    return _label


_FEASIBILITY_REASON_TEXT = {
    0: ('Feasible', True),
    1: ('Wall angle > limit (fixed blank plane)', False),
    2: ('Blocked by collision', False),
    3: ('Excluded by tilt bound', False),
    4: ('Too deep for tool', False),
}


def _make_feasibility_label(infeasible_reason):
    def _label(local_cell_id):
        reason = int(infeasible_reason[local_cell_id])
        text, ok = _FEASIBILITY_REASON_TEXT.get(reason, ('Unknown', False))
        return value_label("Feasibility", text, fmt='{}', pass_fail=ok)
    return _label


def _build_pyvista_mesh(vertices, faces):
    n    = len(faces)
    pv_f = np.hstack([
        np.full((n, 1), 3, dtype=np.int32),
        faces.astype(np.int32)
    ]).ravel()
    return pv.PolyData(vertices.astype(np.float32), pv_f)


def _background_mesh(pl, vertices, faces, sel_idx):
    unsel_mask = np.ones(len(faces), dtype=bool)
    unsel_mask[sel_idx] = False
    unsel_faces = faces[unsel_mask]
    if len(unsel_faces) > 0:
        m = _build_pyvista_mesh(vertices, unsel_faces)
        pl.add_mesh(m, color='lightgrey', opacity=0.15, show_edges=False)


# Distinct (non-gradient) palette for small integer counts, e.g. pass counts.
# Repeats the last color for values beyond the palette length.
_DISTINCT_COUNT_COLORS = np.array([
    [46,  204, 113],   # green
    [241, 196, 15],    # yellow
    [230, 126, 34],    # orange
    [231, 76,  60],    # red
    [155, 89,  182],   # purple
    [52,  73,  94],    # dark slate
    [26,  188, 156],   # teal
    [149, 165, 166],   # grey
], dtype=np.uint8)


def _discrete_int_lut(int_values):
    """
    Discrete (non-gradient) PyVista LookupTable for a small range of integer
    values (e.g. pass counts). Returns (category_array, lut, legend_lines,
    vmin, vmax): category_array is int_values re-based to start at 0 (for use
    as cell_data with clim=[0, n_categories-1]); legend_lines is a list of
    (label, count, rgb_color) per distinct integer value, in ascending order.
    """
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


def _scalar_bar(title):
    return dict(
        title=title,
        n_labels=5,
        label_font_size=13,
        title_font_size=14,
        position_x=0.82,
        position_y=0.05,
        height=0.50,
        width=0.12,
        vertical=True,
        fmt='%.1f',
    )


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 1 — FEASIBILITY MAP
# ─────────────────────────────────────────────────────────────────────────────

def show_feasibility(vertices, faces, sel_idx,
                     is_feasible, infeasible_reason, forming_limit,
                     max_tilt_deg=None, access_ok=None):
    """
    Categories come directly from `infeasible_reason` (see
    evaluate_5axis_from_matrices in spif_analysis.py):

    0 feasible | 1 wall angle vs. the FIXED blank plane exceeds the limit (no
    tool tilt can fix this) | 2 blocked by collision from every direction |
    3 reachable only beyond the tilt bound | 4 collision-clear but too deep
    for --tool_length everywhere. Code 1 takes priority; access_ok (if
    given) reports how many wall-failing faces are also unreachable.
    """
    n_sel    = len(sel_idx)
    category = np.asarray(infeasible_reason, dtype=np.int32)

    n_feas  = int(is_feasible.sum())
    n_wall  = int((category == 1).sum())
    n_wall_unreach = (int(((category == 1) & ~np.asarray(access_ok, dtype=bool)).sum())
                      if access_ok is not None else None)
    n_block = int((category == 2).sum())
    n_tilt  = int((category == 3).sum())
    n_reach = int((category == 4).sum())

    sel_faces = faces[sel_idx]
    mesh      = _build_pyvista_mesh(vertices, sel_faces)
    mesh.cell_data['category'] = category

    lut = pv.LookupTable()
    lut.n_values = 5
    colors = np.array([
        [60,  180, 75,  255],   # Green (0: Feasible)
        [220, 50,  50,  255],   # Red (1: Wall fail everywhere)
        [255, 140, 0,   255],   # Orange (2: Blocked despite an ok wall angle)
        [80,  80,  220, 255],   # Blue (3: Tilt bound excludes the only ok+accessible dirs)
        [155, 89,  182, 255],   # Purple (4: Ray-clear but too deep for the tool)
    ], dtype=np.uint8)
    pad = np.tile(colors[-1], (256 - 5, 1))
    lut.values = np.vstack([colors, pad])[:256]

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')
    _background_mesh(pl, vertices, faces, sel_idx)

    actor = pl.add_mesh(
        mesh,
        scalars='category',
        clim=[0, 4],
        show_scalar_bar=False,
        show_edges=True,
        edge_color='black',
        line_width=0.3,
    )
    actor.GetMapper().SetLookupTable(lut)

    tilt_line = (f"  ■ Tilt bound excl.  : {n_tilt}  faces "
                 f"(bound={max_tilt_deg:.1f} deg)\n" if max_tilt_deg is not None
                 else f"  ■ Tilt bound excl.  : {n_tilt}  faces\n")
    reach_line = f"  ■ Too deep for tool  : {n_reach} faces\n" if n_reach else ""
    wall_extra = (f"  (also unreachable: {n_wall_unreach})"
                  if n_wall_unreach is not None else "")

    pl.add_text(
        f"FEASIBILITY MAP  (5-axis, sheet clamped once)\n"
        f"Forming limit: {forming_limit:.0f} deg vs. FIXED blank plane\n\n"
        f"  ■ Feasible          : {n_feas}  faces\n"
        f"  ■ Wall angle fail   : {n_wall}  faces{wall_extra}\n"
        f"      -> tilt cannot fix: redesign / multi-pass / re-clamp\n"
        f"  ■ Ray blocked       : {n_block} faces\n"
        f"{tilt_line}"
        f"{reach_line}\n"
        f"Total analyzed: {n_sel} faces",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)
    enable_hover_tooltip(pl, {0: [{'actor': actor,
                                    'label_fn': _make_feasibility_label(infeasible_reason)}]})
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 2 — TILT MAP
# ─────────────────────────────────────────────────────────────────────────────

def show_tilt_map(vertices, faces, sel_idx,
                  tilt_needed, is_feasible):
    """is_feasible here = the mask of faces to color; main() passes
    access_ok (tilt is meaningful for every REACHABLE face, including
    wall-failing ones)."""

    feas_faces   = faces[sel_idx[is_feasible]]
    infeas_faces = faces[sel_idx[~is_feasible]]
    tilt_feas    = tilt_needed[is_feasible]

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')
    _background_mesh(pl, vertices, faces, sel_idx)
    hover_layers = []

    if infeas_faces.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, infeas_faces)
        actor = pl.add_mesh(m, color='lightgrey', opacity=0.5,
                            show_edges=True, edge_color='darkgrey', line_width=0.3)
        hover_layers.append({'actor': actor, 'label_fn': _unreachable_label})

    if feas_faces.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, feas_faces)
        m.cell_data['tilt_deg'] = tilt_feas
        actor = pl.add_mesh(
            m,
            scalars='tilt_deg',
            cmap='cool',
            clim=[0, float(tilt_feas.max()) + 1e-3],
            show_edges=True,
            edge_color='black',
            line_width=0.3,
            scalar_bar_args=_scalar_bar('Tool tilt from nominal (deg)'),
        )
        hover_layers.append({'actor': actor, 'label_fn': _make_tilt_label(tilt_feas)})

    mean_tilt = float(tilt_feas.mean()) if is_feasible.any() else 0.0
    max_tilt  = float(tilt_feas.max())  if is_feasible.any() else 0.0
    zero_tilt = int((tilt_feas < 1.0).sum())

    pl.add_text(
        f"TILT MAP  (reachable faces, minimum-tilt collision-free direction)\n"
        f"Cyan = 0 deg (no tilt needed)  →  Magenta = max tilt\n\n"
        f"  Mean tilt : {mean_tilt:.1f} deg\n"
        f"  Max tilt  : {max_tilt:.1f} deg\n"
        f"  No tilt needed (<1 deg): {zero_tilt} faces\n"
        f"  Grey = unreachable ({(~is_feasible).sum()} faces)",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)
    enable_hover_tooltip(pl, {0: hover_layers})
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 3 — FORMING SEVERITY (fixed blank plane)
# ─────────────────────────────────────────────────────────────────────────────

def show_forming_severity(vertices, faces, sel_idx,
                          blank_wa, thinning_pct, wall_ok, forming_limit,
                          t0_mm, access_ok=None):
    """
    Forming severity against the FIXED blank plane -- identical for every
    tool direction, so there is one value per face (no nominal-vs-5-axis
    comparison: tool tilt does not change wall angle or thinning).

    Left : wall angle / forming limit (1.0 = limit).
    Right: pure-shear sine-law thinning 1 - cos(a), %. Faces beyond the
           forming limit (below the critical thickness) are outlined red.

    Every face is colored -- these are real geometric values regardless of
    reachability. access_ok (optional) is only used for the stats text.
    """
    sel_faces = faces[sel_idx]
    wall_ok   = np.asarray(wall_ok, dtype=bool)
    norm_wa   = np.clip(blank_wa / (forming_limit + 1e-9), 0, 1.5)

    pl = pv.Plotter(shape=(1, 2), window_size=[1800, 900], border=False)
    pl.set_background('white')
    hover_layers = {0: [], 1: []}

    # ── Left: wall angle / limit ──────────────────────────────────────────
    pl.subplot(0, 0)
    _background_mesh(pl, vertices, faces, sel_idx)
    m = _build_pyvista_mesh(vertices, sel_faces)
    m.cell_data['wall_norm'] = norm_wa
    actor = pl.add_mesh(
        m, scalars='wall_norm', cmap='RdYlGn_r', clim=[0.0, 1.5],
        show_edges=True, edge_color='black', line_width=0.3,
        scalar_bar_args=_scalar_bar(
            f'Wall angle / limit\n(1.0 = {forming_limit:.0f} deg)'),
    )
    hover_layers[0].append({'actor': actor, 'label_fn': _make_wall_angle_label(
        blank_wa, forming_limit)})
    n_viol = int((~wall_ok).sum())
    unreach_note = ""
    if access_ok is not None:
        n_unreach = int((~np.asarray(access_ok, dtype=bool)).sum())
        unreach_note = f"Unreachable by any tilt (see Window 1): {n_unreach} faces\n"
    pl.add_text(
        f"WALL ANGLE vs. FIXED BLANK PLANE  [Tier 1]\n"
        f"Mean : {float(blank_wa.mean()):.1f} deg   Max : {float(blank_wa.max()):.1f} deg\n"
        f"Beyond limit : {n_viol} / {len(sel_idx)} faces\n"
        f"{unreach_note}"
        f"Same for every tool direction -- tilt cannot change it",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)

    # ── Right: sine-law thinning ──────────────────────────────────────────
    pl.subplot(0, 1)
    _background_mesh(pl, vertices, faces, sel_idx)
    crit_thinning = float((1.0 - np.cos(np.radians(forming_limit))) * 100.0)
    m2 = _build_pyvista_mesh(vertices, sel_faces)
    m2.cell_data['thinning_pct'] = thinning_pct
    actor2 = pl.add_mesh(
        m2, scalars='thinning_pct', cmap='viridis_r', clim=[0.0, 100.0],
        show_edges=False,
        scalar_bar_args=_scalar_bar('Sine-law thinning (%)'),
    )
    hover_layers[1].append({'actor': actor2, 'label_fn': _make_thinning_label(
        thinning_pct, wall_ok)})
    if (~wall_ok).any():
        risk = m2.extract_cells(np.nonzero(~wall_ok)[0])
        pl.add_mesh(risk, color='red', style='wireframe', line_width=2,
                    label='Below critical thickness')
        pl.add_legend()
    pl.add_text(
        f"SINE-LAW THINNING 1 - cos(a)  [Tier 1, first-order]\n"
        f"t0 = {t0_mm:.2f} mm  |  critical thinning = {crit_thinning:.0f}% "
        f"(at {forming_limit:.0f} deg)\n"
        f"Mean : {float(thinning_pct.mean()):.1f}%   Max : {float(thinning_pct.max()):.1f}%\n"
        f"Red outline = below critical thickness",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)

    pl.link_views()
    enable_hover_tooltip(pl, hover_layers)
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 4 — DIRECTION FIELD
# ─────────────────────────────────────────────────────────────────────────────

def show_direction_field(vertices, faces, face_centroids,
                          sel_idx, best_dir_indices, directions,
                          approach_deg, is_feasible,
                          max_arrows=400):
    """
    Arrows = each reachable face's minimum-tilt collision-free tool direction
    (is_feasible here = access_ok). Color = TOOL-APPROACH angle between the
    face normal and that tool axis: a kinematic quantity (how side-on the
    tool meets the surface), NOT a wall angle -- it does not enter the
    forming limit, thinning or force.
    """
    feas_sel_idx = sel_idx[is_feasible]
    feas_wall    = approach_deg[is_feasible]
    feas_best_d  = best_dir_indices[is_feasible]

    n_feas_orig = len(feas_sel_idx)
    if len(feas_sel_idx) > max_arrows:
        step         = len(feas_sel_idx) // max_arrows
        feas_sel_idx = feas_sel_idx[::step]
        feas_wall    = feas_wall[::step]
        feas_best_d  = feas_best_d[::step]

    centroids  = face_centroids[feas_sel_idx]
    best_dirs  = directions[feas_best_d]

    bbox_diag   = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    arrow_scale = bbox_diag * 0.015

    cloud = pv.PolyData(centroids.astype(np.float32))
    cloud['vectors']    = (best_dirs * arrow_scale).astype(np.float32)
    cloud['wall_angle'] = feas_wall.astype(np.float32)
    cloud.set_active_vectors('vectors')

    arrows = cloud.glyph(
        orient='vectors',
        scale=False,
        factor=10.0,
        geom=pv.Arrow(),
    )
    n_cells_per_arrow = arrows.n_cells // len(feas_wall) if len(feas_wall) > 0 else 1
    arrows['wall_angle'] = np.repeat(feas_wall, n_cells_per_arrow)

    clim = [0.0, 90.0]

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')
    _background_mesh(pl, vertices, faces, sel_idx)
    hover_layers = []

    infeas_faces = faces[sel_idx[~is_feasible]]
    if infeas_faces.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, infeas_faces)
        actor = pl.add_mesh(m, color='lightgrey', opacity=0.4, show_edges=False)
        hover_layers.append({'actor': actor, 'label_fn': _unreachable_label})

    feas_faces_all = faces[sel_idx[is_feasible]]
    if feas_faces_all.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, feas_faces_all)
        actor = pl.add_mesh(m, color='lightsteelblue', opacity=0.25, show_edges=False)
        hover_layers.append({'actor': actor, 'label_fn': _make_approach_label(
            approach_deg[is_feasible])})

    pl.add_mesh(
        arrows,
        scalars='wall_angle',
        cmap='plasma',
        clim=clim,
        show_scalar_bar=True,
        scalar_bar_args=_scalar_bar('Tool-approach angle (deg)\nkinematic only'),
    )

    mean_wa = float(feas_wall.mean()) if len(feas_wall) > 0 else 0.0
    max_wa  = float(feas_wall.max())  if len(feas_wall) > 0 else 0.0

    pl.add_text(
        f"DIRECTION FIELD  (minimum-tilt collision-free tool direction per face)\n"
        f"Arrow = tool direction  |  Color = tool-approach angle (face normal vs tool axis)\n"
        f"Kinematic diagnostic only -- NOT a wall angle; does not affect\n"
        f"forming limit, thinning or force (sheet is clamped once)\n\n"
        f"  Mean approach angle : {mean_wa:.1f} deg\n"
        f"  Max approach angle  : {max_wa:.1f} deg\n"
        f"  Showing {len(feas_sel_idx)} arrows "
        f"({'subsampled' if n_feas_orig > max_arrows else 'all reachable'})\n"
        f"  Grey = unreachable faces",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)
    enable_hover_tooltip(pl, {0: hover_layers})
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 5 — CRITICAL FACES
# ─────────────────────────────────────────────────────────────────────────────

def show_critical_faces(vertices, faces, face_centroids,
                         sel_idx, best_dir_indices, directions,
                         blank_wa, is_feasible, forming_limit,
                         top_pct=0.15):
    """Feasible faces whose FIXED blank-plane wall angle is closest to the
    forming limit; rays show the tool direction chosen to reach them."""
    bbox_diag  = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    ray_length = bbox_diag * 0.5

    feas_mask    = is_feasible
    feas_sel_idx = sel_idx[feas_mask]
    feas_wall    = blank_wa[feas_mask]
    feas_best_d  = best_dir_indices[feas_mask]

    sort_order   = np.argsort(feas_wall)[::-1]
    n_critical   = max(1, int(np.ceil(len(feas_sel_idx) * top_pct)))
    crit_order   = sort_order[:n_critical]

    crit_sel_idx = feas_sel_idx[crit_order]
    crit_wall    = feas_wall[crit_order]
    crit_best_d  = feas_best_d[crit_order]

    crit_centroids = face_centroids[crit_sel_idx]
    crit_dirs      = directions[crit_best_d]

    ray_starts = crit_centroids.astype(np.float32)
    ray_ends   = (crit_centroids + crit_dirs * ray_length).astype(np.float32)

    n_crit     = len(crit_sel_idx)
    ray_pts    = np.empty((n_crit * 2, 3), dtype=np.float32)
    ray_pts[0::2] = ray_starts
    ray_pts[1::2] = ray_ends

    conn = []
    for i in range(n_crit):
        conn.extend([2, 2 * i, 2 * i + 1])
    conn = np.array(conn, dtype=np.int32)

    rays_poly        = pv.PolyData(ray_pts)
    rays_poly.lines  = conn
    rays_poly['wall_angle'] = np.repeat(crit_wall, 2).astype(np.float32)

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')
    _background_mesh(pl, vertices, faces, sel_idx)
    hover_layers = []

    all_feas_faces = faces[feas_sel_idx]
    if all_feas_faces.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, all_feas_faces)
        pl.add_mesh(m, color='lightsteelblue', opacity=0.2, show_edges=False)

    infeas_faces = faces[sel_idx[~is_feasible]]
    if infeas_faces.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, infeas_faces)
        actor = pl.add_mesh(m, color='lightgrey', opacity=0.35, show_edges=False)
        hover_layers.append({'actor': actor, 'label_fn': _inaccessible_label})

    crit_faces_mesh = _build_pyvista_mesh(vertices, faces[crit_sel_idx])
    crit_faces_mesh.cell_data['wall_angle'] = crit_wall.astype(np.float32)
    crit_actor = pl.add_mesh(
        crit_faces_mesh,
        scalars='wall_angle',
        cmap='RdYlGn_r',
        clim=[0.0, forming_limit],
        show_edges=True,
        edge_color='black',
        line_width=0.4,
        scalar_bar_args=_scalar_bar(
            f'Wall angle (deg)\n(red = {forming_limit:.0f} deg limit)'),
    )
    hover_layers.append({'actor': crit_actor, 'label_fn': _make_wall_angle_label(
        crit_wall, forming_limit)})

    pl.add_mesh(
        rays_poly,
        scalars='wall_angle',
        cmap='RdYlGn_r',
        clim=[0.0, forming_limit],
        line_width=3.0,
        show_scalar_bar=False,
    )

    threshold_wa = float(crit_wall.min())
    mean_wa      = float(crit_wall.mean())

    pl.add_text(
        f"CRITICAL FACES  (top {top_pct*100:.0f}% closest to forming limit)\n"
        f"Forming limit : {forming_limit:.0f} deg  |  wall angle vs. fixed blank plane\n"
        f"Rays show the chosen (minimum-tilt) tool direction\n\n"
        f"  Critical faces    : {n_crit} / {len(feas_sel_idx)} feasible\n"
        f"  Wall angle range  : {threshold_wa:.1f} — {crit_wall.max():.1f} deg\n"
        f"  Mean wall angle   : {mean_wa:.1f} deg\n",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)
    enable_hover_tooltip(pl, {0: hover_layers})
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 6 — CRITICAL TILT FACES
# ─────────────────────────────────────────────────────────────────────────────

def show_critical_tilt_faces(vertices, faces, face_centroids,
                              sel_idx, best_dir_indices, directions,
                              tilt_needed, is_feasible, forming_limit,
                              top_pct=0.05):
    bbox_diag  = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    ray_length = bbox_diag * 0.5

    feas_mask    = is_feasible
    feas_sel_idx = sel_idx[feas_mask]
    feas_tilt    = tilt_needed[feas_mask]
    feas_best_d  = best_dir_indices[feas_mask]

    sort_order = np.argsort(feas_tilt)[::-1]
    n_critical = max(1, int(np.ceil(len(feas_sel_idx) * top_pct)))
    crit_order = sort_order[:n_critical]

    crit_sel_idx = feas_sel_idx[crit_order]
    crit_tilt    = feas_tilt[crit_order]
    crit_best_d  = feas_best_d[crit_order]

    crit_centroids = face_centroids[crit_sel_idx]
    crit_dirs      = directions[crit_best_d]

    ray_starts = crit_centroids.astype(np.float32)
    ray_ends   = (crit_centroids + crit_dirs * ray_length).astype(np.float32)

    n_crit    = len(crit_sel_idx)
    ray_pts   = np.empty((n_crit * 2, 3), dtype=np.float32)
    ray_pts[0::2] = ray_starts
    ray_pts[1::2] = ray_ends

    conn = []
    for i in range(n_crit):
        conn.extend([2, 2 * i, 2 * i + 1])
    conn = np.array(conn, dtype=np.int32)

    rays_poly       = pv.PolyData(ray_pts)
    rays_poly.lines = conn
    rays_poly['tilt_angle'] = np.repeat(crit_tilt, 2).astype(np.float32)

    tilt_clim = [0.0, float(feas_tilt.max()) + 1e-3]

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')
    _background_mesh(pl, vertices, faces, sel_idx)
    hover_layers = []

    all_feas_faces = faces[feas_sel_idx]
    if all_feas_faces.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, all_feas_faces)
        pl.add_mesh(m, color='lightsteelblue', opacity=0.2, show_edges=False)

    infeas_faces = faces[sel_idx[~is_feasible]]
    if infeas_faces.shape[0] > 0:
        m = _build_pyvista_mesh(vertices, infeas_faces)
        actor = pl.add_mesh(m, color='lightgrey', opacity=0.35, show_edges=False)
        hover_layers.append({'actor': actor, 'label_fn': _inaccessible_label})

    crit_faces_mesh = _build_pyvista_mesh(vertices, faces[crit_sel_idx])
    crit_faces_mesh.cell_data['tilt_angle'] = crit_tilt.astype(np.float32)
    crit_actor = pl.add_mesh(
        crit_faces_mesh,
        scalars='tilt_angle',
        cmap='cool',
        clim=tilt_clim,
        show_edges=True,
        edge_color='black',
        line_width=0.4,
        scalar_bar_args=_scalar_bar('Tool tilt from nominal (deg)'),
    )
    hover_layers.append({'actor': crit_actor, 'label_fn': _make_tilt_label(crit_tilt)})

    pl.add_mesh(
        rays_poly,
        scalars='tilt_angle',
        cmap='cool',
        clim=tilt_clim,
        line_width=3.0,
        show_scalar_bar=False,
    )

    min_tilt  = float(crit_tilt.min())
    max_tilt  = float(crit_tilt.max())
    mean_tilt = float(crit_tilt.mean())

    pl.add_text(
        f"CRITICAL TILT FACES  (top {top_pct*100:.0f}% highest tool tilt)\n"
        f"Cyan = low tilt  →  Magenta = high tilt\n\n"
        f"  Critical faces    : {n_crit} / {len(feas_sel_idx)} feasible\n"
        f"  Tilt range        : {min_tilt:.1f} — {max_tilt:.1f} deg\n"
        f"  Mean tilt         : {mean_tilt:.1f} deg\n",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)
    enable_hover_tooltip(pl, {0: hover_layers})
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 7 — MULTI-PASS: 3-AXIS vs 5-AXIS REACHABILITY
# ─────────────────────────────────────────────────────────────────────────────

def show_multipass_reachability(vertices, faces, sel_idx,
                                blank_wa, forming_limit,
                                max_step_pass=10.0, nominal_accessible=None,
                                access_ok=None):
    """
    Multi-pass requirement vs. reachability, 3-axis vs 5-axis.

    The pass count depends only on the fixed blank-plane wall angle, so it is
    IDENTICAL for 3-axis and 5-axis forming of the same clamped sheet -- tool
    tilt does not reduce it. What tilt does change is which faces can be
    reached at all:
      Left : pass count, grey where blocked at the nominal (untilted) axis.
      Right: same pass count, grey where no collision-free direction exists
             within the tilt bound (access_ok).
    Pass count itself is a TIER 2 practice-based rule.
    """
    excess = np.maximum(0.0, blank_wa - forming_limit)
    passes = np.ones_like(blank_wa, dtype=np.int32)
    over = blank_wa > forming_limit
    passes[over] += np.ceil(excess[over] / max_step_pass).astype(np.int32)

    n = len(sel_idx)
    reach_3 = (np.asarray(nominal_accessible, dtype=bool)
               if nominal_accessible is not None else np.ones(n, dtype=bool))
    reach_5 = (np.asarray(access_ok, dtype=bool)
               if access_ok is not None else np.ones(n, dtype=bool))

    pl = pv.Plotter(shape=(1, 2), window_size=[1800, 900], border=False)
    pl.set_background('white')
    hover_layers = {0: [], 1: []}
    sel_faces = faces[sel_idx]

    panels = [
        ("3-AXIS (nominal axis, no tilt)", reach_3),
        ("5-AXIS (tilting tool, same clamping)", reach_5),
    ]
    for col, (label, reach) in enumerate(panels):
        pl.subplot(0, col)
        _background_mesh(pl, vertices, faces, sel_idx)
        legend_text = "(no reachable faces)"
        if reach.any():
            cat, lut, legend, vmin, vmax = _discrete_int_lut(passes[reach])
            m = _build_pyvista_mesh(vertices, sel_faces[reach])
            m.cell_data['pass_cat'] = cat
            clim = [0, vmax - vmin] if vmax > vmin else [-0.5, 0.5]
            actor = pl.add_mesh(
                m, scalars='pass_cat', clim=clim,
                show_scalar_bar=False, show_edges=True,
                edge_color='black', line_width=0.3,
            )
            actor.GetMapper().SetLookupTable(lut)
            legend_text = "\n".join(
                f"  {lbl} pass{'es' if lbl != '1' else ''}: {count} faces"
                for lbl, count, _ in legend)
            hover_layers[col].append({'actor': actor, 'label_fn': _make_passes_label(
                passes[reach])})
        if (~reach).any():
            mg = _build_pyvista_mesh(vertices, sel_faces[~reach])
            actor_g = pl.add_mesh(mg, color='grey', show_edges=True,
                                  edge_color='black', line_width=0.3)
            hover_layers[col].append({'actor': actor_g, 'label_fn': _unreachable_label})

        n_mp_reach = int(((passes > 1) & reach).sum())
        pl.add_text(
            f"MULTI-PASS  —  {label}\n"
            f"Pass count [Tier 2] from the fixed blank-plane wall angle\n"
            f"Multi-pass faces reachable : {n_mp_reach} / {int((passes > 1).sum())}\n"
            f"Unreachable (grey)         : {int((~reach).sum())} faces\n"
            f"\n{legend_text}",
            position='upper_left', font_size=11, color='black'
        )

    pl.link_views()
    enable_hover_tooltip(pl, hover_layers)
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW 8 — PREDICTED FORMING FORCE
# ─────────────────────────────────────────────────────────────────────────────

def show_force_map(vertices, faces, sel_idx, predicted_force_N, is_feasible=None,
                   force_caveat=None, material_key=None, t0_mm=None):
    """
    Per-face predicted steady-state axial force (Aerens et al. 2010) at the
    FIXED blank-plane wall angle and t0 -- independent of tool tilt. See
    spif_analysis.py FORCE MODEL PROVENANCE for sourcing/caveats -- notably
    that Fz_s ~ angle*cos(angle) is NOT monotonic in wall angle.

    is_feasible : optional bool array, greys out infeasible faces (wall angle
        beyond the formable range the model was fitted on, or unreachable).
    """
    sel_faces = faces[sel_idx]
    shown = (np.asarray(is_feasible, dtype=bool) if is_feasible is not None
             else np.ones(len(predicted_force_N), dtype=bool))
    fmax  = float(predicted_force_N[shown].max()) if shown.any() else 1.0
    fmean = float(predicted_force_N[shown].mean()) if shown.any() else 0.0

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')
    _background_mesh(pl, vertices, faces, sel_idx)

    hover_layers = []
    n_grey = 0
    if is_feasible is not None:
        is_feasible = np.asarray(is_feasible, dtype=bool)
        n_grey = int((~is_feasible).sum())

        if is_feasible.any():
            mesh = _build_pyvista_mesh(vertices, sel_faces[is_feasible])
            feas_force = predicted_force_N[is_feasible]
            mesh.cell_data['force_N'] = feas_force
            actor = pl.add_mesh(
                mesh, scalars='force_N', cmap='inferno',
                clim=[0.0, fmax + 1e-6], show_edges=True, edge_color='black',
                line_width=0.3,
                scalar_bar_args=_scalar_bar('Predicted axial force Fz_s (N)'),
            )
            hover_layers.append({'actor': actor, 'label_fn': _make_force_label(feas_force)})
        if (~is_feasible).any():
            mg = _build_pyvista_mesh(vertices, sel_faces[~is_feasible])
            actor_g = pl.add_mesh(mg, color='grey', show_edges=True,
                                  edge_color='black', line_width=0.3)
            hover_layers.append({'actor': actor_g, 'label_fn': _inaccessible_label})
    else:
        mesh = _build_pyvista_mesh(vertices, sel_faces)
        mesh.cell_data['force_N'] = predicted_force_N
        actor = pl.add_mesh(
            mesh,
            scalars='force_N',
            cmap='inferno',
            clim=[0.0, fmax + 1e-6],
            show_edges=True,
            edge_color='black',
            line_width=0.3,
            scalar_bar_args=_scalar_bar('Predicted axial force Fz_s (N)'),
        )
        hover_layers.append({'actor': actor, 'label_fn': _make_force_label(predicted_force_N)})

    grey_note = f"Infeasible (grey): {n_grey} faces\n\n" if n_grey else "\n"
    setup = ""
    if material_key:
        setup = f"Material: {material_key}" + (f"  |  t0 = {t0_mm:.2f} mm" if t0_mm else "") + "\n"
    caveat_line = f"CAVEAT: {force_caveat}\n" if force_caveat else ""
    pl.add_text(
        f"PREDICTED FORMING FORCE  (Aerens et al. 2010, relative effort proxy)\n"
        f"Fixed blank-plane wall angle, t0 -- independent of tool tilt\n"
        f"{setup}\n"
        f"  Max force  : {fmax:.0f} N  (feasible faces)\n"
        f"  Mean force : {fmean:.0f} N\n"
        f"{grey_note}"
        f"{caveat_line}"
        f"NOTE: Fz_s ~ angle*cos(angle) -- NOT monotonic in wall angle,\n"
        f"can be LOWER at very steep walls than at ~50-60 deg. See README.",
        position='upper_left', font_size=11, color='black',
    )
    pl.add_axes(line_width=3)
    enable_hover_tooltip(pl, {0: hover_layers})
    pl.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='5-axis database showcase visualizer')
    parser.add_argument('--npz',            required=True)
    parser.add_argument('--stl',            required=True)
    parser.add_argument('--max_arrows',     type=int, default=400)
    parser.add_argument('--max_step_pass', type=float, default=10.0,
                        help='Must match the value main.py was run with, or '
                             'the Window 7 pass counts will differ from '
                             'main.py\'s multi-pass analysis.')
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"5-AXIS SHOWCASE")
    print(f"{'#'*60}")
    print(f"  NPZ : {args.npz}")
    print(f"  STL : {args.stl}")

    data = load_5axis_database(args.npz)

    face_indices_global = data['face_indices_global']
    directions          = data['directions']
    blank_wa            = data['blank_wall_angles']
    wall_ok             = data['wall_ok']
    thinning_pct        = data['thinning_pct']
    nominal_accessible  = data['nominal_accessible']
    access_ok           = data['access_ok']
    approach_deg        = data['approach_angle_at_best_deg']
    best_dir_indices    = data['best_dir_indices']
    tilt_needed         = data['tilt_needed_deg']
    is_feasible         = data['is_feasible']
    infeasible_reason   = data['infeasible_reason']
    forming_limit       = float(data['forming_limit_deg'][0])
    max_tilt_deg        = float(data['max_tilt_deg'][0])
    t0_mm               = float(data['sheet_thickness_mm'][0])
    material_key        = str(data['material_key'][0]) or None

    n_faces = len(face_indices_global)

    predicted_force_N = data['predicted_force_N'] if 'predicted_force_N' in data.files else None

    print(f"  Faces         : {n_faces}")
    print(f"  Directions    : {len(directions)}")
    print(f"  Blank normal  : {np.round(data['blank_normal'], 3).tolist()}  "
          f"(fixed clamping -- wall angle / thinning / force measured against it)")
    print(f"  Forming limit : {forming_limit:.0f} deg")
    if max_tilt_deg is not None:
        print(f"  Tilt bound    : {max_tilt_deg:.1f} deg  "
              f"(placeholder unless overridden -- see spif_analysis.py header)")
    print(f"  Feasible      : {is_feasible.sum()}  "
          f"({is_feasible.sum()/n_faces*100:.1f}%)")
    print(f"  Infeasible    : {(~is_feasible).sum()}")

    print(f"\n  Loading STL...")
    tm = trimesh.load(args.stl, force='mesh')
    tm.fix_normals()
    vertices       = np.array(tm.vertices,         dtype=np.float32)
    faces          = np.array(tm.faces,            dtype=np.int32)
    face_centroids = np.array(tm.triangles_center, dtype=np.float32)

    sel_idx = face_indices_global.astype(int)

    print(f"\n[1] Feasibility map...")
    show_feasibility(vertices, faces, sel_idx,
                     is_feasible, infeasible_reason, forming_limit,
                     max_tilt_deg=max_tilt_deg, access_ok=access_ok)

    print(f"\n[2] Tilt map...")
    show_tilt_map(vertices, faces, sel_idx,
                  tilt_needed, access_ok)

    print(f"\n[3] Forming severity (fixed blank plane)...")
    show_forming_severity(vertices, faces, sel_idx,
                          blank_wa, thinning_pct, wall_ok, forming_limit,
                          t0_mm, access_ok=access_ok)

    print(f"\n[4] Direction field...")
    show_direction_field(vertices, faces, face_centroids,
                          sel_idx, best_dir_indices, directions,
                          approach_deg, access_ok,
                          max_arrows=args.max_arrows)

    if is_feasible.any():
        print(f"\n[5] Critical faces (top 15% closest to forming limit)...")
        show_critical_faces(vertices, faces, face_centroids,
                             sel_idx, best_dir_indices, directions,
                             blank_wa, is_feasible, forming_limit,
                             top_pct=0.15)

        print(f"\n[6] Critical tilt faces (top 5% highest tool tilt)...")
        show_critical_tilt_faces(vertices, faces, face_centroids,
                                  sel_idx, best_dir_indices, directions,
                                  tilt_needed, is_feasible, forming_limit,
                                  top_pct=0.05)
    else:
        print(f"\n[5]/[6] Critical faces: SKIPPED (no feasible faces)")

    print(f"\n[7] Multi-pass requirement: 3-axis vs 5-axis reachability...")
    show_multipass_reachability(vertices, faces, sel_idx,
                                blank_wa, forming_limit,
                                max_step_pass=args.max_step_pass,
                                nominal_accessible=nominal_accessible,
                                access_ok=access_ok)

    if predicted_force_N is not None:
        print(f"\n[8] Predicted forming force (Aerens et al. 2010)...")
        show_force_map(vertices, faces, sel_idx, predicted_force_N,
                       is_feasible=is_feasible,
                       force_caveat=force_model_caveat(material_key),
                       material_key=material_key, t0_mm=t0_mm)
    else:
        print(f"\n[8] Predicted forming force: SKIPPED "
              f"(.npz was built without material_key -- regenerate with "
              f"the current pipeline to include it)")

    print(f"\n{'#'*60}")
    print(f"SHOWCASE COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()