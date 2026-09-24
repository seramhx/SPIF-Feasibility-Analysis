"""
SPIF 3-AXIS RESULTS VISUALIZER — PyVista
==========================================
Two separate PyVista windows:
  Window 1: Wall angle violation at BASELINE direction (the confirmed
            hemisphere axis's pole -- [0,1,0] only if that axis was left
            at its default; see hemisphere.py)
  Window 2: Wall angle violation at OPTIMIZED direction

Color scale per face:
  Green  = wall angle well within forming limit
  Yellow = approaching limit
  Red    = at or beyond limit (violation)

Non-selected faces shown as transparent grey.
"""

import numpy as np
import pyvista as pv
import trimesh

from spif_hover import enable_hover_tooltip, value_label


def _build_pyvista_mesh(vertices, faces):
    """Build PyVista PolyData from vertex/face arrays."""
    n = len(faces)
    pv_faces = np.hstack([
        np.full((n, 1), 3, dtype=np.int32), faces.astype(np.int32)
    ]).ravel()
    return pv.PolyData(vertices.astype(np.float32), pv_faces)


def _wall_violation_scalars(wall_angles, forming_limit):
    # 0 = 0 deg, 1.0 = at forming limit or beyond
    normed = wall_angles / (forming_limit + 1e-9)
    return np.clip(normed, 0.0, 1.0)


def _inaccessible_label(local_cell_id):
    return "Inaccessible\n(no value at this direction)"


def _make_wall_angle_label(wall_angles_deg, forming_limit_deg):
    """Hover label closure: wall angle (deg) + PASS/FAIL vs the forming limit."""
    def _label(local_cell_id):
        angle = float(wall_angles_deg[local_cell_id])
        return value_label(
            "Wall angle", angle, fmt='{:.1f}', unit='°',
            pass_fail=(angle <= forming_limit_deg),
            extra=f"limit {forming_limit_deg:.0f}°")
    return _label


def _make_force_label(force_N):
    """Hover label closure: predicted force (N). No pass/fail concept."""
    def _label(local_cell_id):
        return value_label("Predicted force", float(force_N[local_cell_id]),
                           fmt='{:.0f}', unit=' N')
    return _label


def _make_passes_label(passes_required):
    """Hover label closure: pass count. PASS = single pass, FAIL = multi-pass needed."""
    def _label(local_cell_id):
        n = int(passes_required[local_cell_id])
        return value_label(f"Passes required", n, fmt='{:d}',
                           pass_fail=(n == 1),
                           pass_word='PASS (single pass)',
                           fail_word=f'FAIL (needs {n} passes)')
    return _label


def _make_thickness_label(thickness_mm, below_critical):
    """Hover label closure: sine-law thickness (mm). FAIL = below critical."""
    def _label(local_cell_id):
        t = float(thickness_mm[local_cell_id])
        risky = bool(below_critical[local_cell_id])
        return value_label("Sine-law thickness", t, fmt='{:.2f}', unit=' mm',
                           pass_fail=(not risky),
                           pass_word='PASS',
                           fail_word='FAIL (below critical -- not certifiable)')
    return _label


def _background_mesh(pl, vertices, faces, sel_idx):
    unsel_mask = np.ones(len(faces), dtype=bool)
    unsel_mask[sel_idx] = False
    unsel_faces = faces[unsel_mask]
    if len(unsel_faces) > 0:
        m = _build_pyvista_mesh(vertices, unsel_faces)
        pl.add_mesh(m, color='lightgrey', opacity=0.25, show_edges=False)


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


def _show_violation_window(stl_vertices, stl_faces,
                             selected_tri_indices,
                             face_wall_angles,
                             forming_limit,
                             direction_label,
                             violation_pct,
                             material_key,
                             window_title,
                             accessible_mask=None):
    """
    Generic function to open one PyVista violation map window.

    accessible_mask : optional bool array aligned with selected_tri_indices.
        Inaccessible faces (False) render flat grey instead of by wall-angle
        color, matching the 5-axis feasibility map's treatment of blocked
        faces -- a face that can't be reached doesn't have a meaningful
        "will it fracture" answer at this direction. None (default):
        original behavior, every selected face colored by wall angle.
    """
    sel_idx    = np.array(selected_tri_indices, dtype=int)
    all_idx    = np.arange(len(stl_faces))
    unsel_mask = np.ones(len(stl_faces), dtype=bool)
    unsel_mask[sel_idx] = False

    sel_faces   = stl_faces[sel_idx]
    unsel_faces = stl_faces[unsel_mask]

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')

    # Unselected faces — grey background
    if len(unsel_faces) > 0:
        unsel_mesh = _build_pyvista_mesh(stl_vertices, unsel_faces)
        pl.add_mesh(unsel_mesh, color='lightgrey', opacity=0.25,
                    show_edges=False)

    hover_layers = []

    n_blocked = 0
    if accessible_mask is not None:
        accessible_mask = np.asarray(accessible_mask, dtype=bool)
        n_blocked = int((~accessible_mask).sum())

        acc_faces = sel_faces[accessible_mask]
        blk_faces = sel_faces[~accessible_mask]

        if len(acc_faces) > 0:
            acc_mesh = _build_pyvista_mesh(stl_vertices, acc_faces)
            acc_wa = face_wall_angles[accessible_mask]
            acc_mesh.cell_data['wall_violation'] = _wall_violation_scalars(
                acc_wa, forming_limit)
            acc_actor = pl.add_mesh(
                acc_mesh, scalars='wall_violation', cmap='RdYlGn_r',
                clim=[0.0, 1.0], show_edges=True, edge_color='black',
                line_width=0.3,
                scalar_bar_args=dict(
                    title=f'Wall angle / {forming_limit:.0f}°  (1.0 = violation)',
                    n_labels=5, label_font_size=13, title_font_size=14,
                    position_x=0.82, position_y=0.05, height=0.50,
                    width=0.12, vertical=True, fmt='%.1f',
                ),
            )
            hover_layers.append({'actor': acc_actor, 'label_fn': _make_wall_angle_label(
                acc_wa, forming_limit)})
        if len(blk_faces) > 0:
            blk_mesh = _build_pyvista_mesh(stl_vertices, blk_faces)
            blk_actor = pl.add_mesh(blk_mesh, color='grey', show_edges=True,
                                    edge_color='black', line_width=0.3)
            hover_layers.append({'actor': blk_actor, 'label_fn': _inaccessible_label})
    else:
        # Original behavior: every selected face colored by wall angle.
        scalars = _wall_violation_scalars(face_wall_angles, forming_limit)
        sel_mesh = _build_pyvista_mesh(stl_vertices, sel_faces)
        sel_mesh.cell_data['wall_violation'] = scalars
        sel_actor = pl.add_mesh(
            sel_mesh,
            scalars='wall_violation',
            cmap='RdYlGn_r',      # green=fine, yellow=caution, red=violation
            clim=[0.0, 1.0],
            show_edges=True,
            edge_color='black',
            line_width=0.3,
            scalar_bar_args=dict(
                title=f'Wall angle / {forming_limit:.0f}°  (1.0 = violation)',
                n_labels=5,
                label_font_size=13,
                title_font_size=14,
                position_x=0.82,
                position_y=0.05,
                height=0.50,
                width=0.12,
                vertical=True,
                fmt='%.1f',
            ),
        )
        hover_layers.append({'actor': sel_actor, 'label_fn': _make_wall_angle_label(
            face_wall_angles, forming_limit)})

    blocked_line = (f"Inaccessible (grey): {n_blocked} faces\n"
                     if accessible_mask is not None else "")
    pl.add_text(
        f"{window_title}\n"
        f"Material: {material_key}  |  Forming limit: {forming_limit:.0f} deg\n"
        f"Direction: {direction_label}\n"
        f"Wall violation: {violation_pct:.1f}% of selected area\n"
        f"{blocked_line}",
        position='upper_left',
        font_size=11,
        color='black',
    )

    pl.add_axes(line_width=3)
    enable_hover_tooltip(pl, {0: hover_layers})
    pl.show()


def _show_force_window(stl_vertices, stl_faces,
                        selected_tri_indices,
                        predicted_force_N,
                        material_key,
                        direction_label,
                        window_title,
                        accessible_mask=None,
                        force_caveat=None):
    """
    Per-face predicted steady-state axial force (Aerens et al. 2010).
    See spif_analysis.py FORCE MODEL PROVENANCE for sourcing/caveats.

    accessible_mask : optional bool array aligned with selected_tri_indices.
        Inaccessible faces render flat grey instead of by force color --
        an unreachable face has no meaningful force estimate at this
        direction. None (default): every selected face colored by force.
    """
    sel_idx    = np.array(selected_tri_indices, dtype=int)
    unsel_mask = np.ones(len(stl_faces), dtype=bool)
    unsel_mask[sel_idx] = False

    sel_faces   = stl_faces[sel_idx]
    unsel_faces = stl_faces[unsel_mask]

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')

    if len(unsel_faces) > 0:
        unsel_mesh = _build_pyvista_mesh(stl_vertices, unsel_faces)
        pl.add_mesh(unsel_mesh, color='lightgrey', opacity=0.25,
                    show_edges=False)

    fmax = float(predicted_force_N.max()) if len(predicted_force_N) else 1.0
    fmean = float(predicted_force_N.mean()) if len(predicted_force_N) else 0.0

    hover_layers = []
    n_blocked = 0
    if accessible_mask is not None:
        accessible_mask = np.asarray(accessible_mask, dtype=bool)
        n_blocked = int((~accessible_mask).sum())

        acc_faces = sel_faces[accessible_mask]
        blk_faces = sel_faces[~accessible_mask]

        if len(acc_faces) > 0:
            acc_mesh = _build_pyvista_mesh(stl_vertices, acc_faces)
            acc_force = predicted_force_N[accessible_mask]
            acc_mesh.cell_data['force_N'] = acc_force
            acc_actor = pl.add_mesh(
                acc_mesh, scalars='force_N', cmap='inferno',
                clim=[0.0, fmax + 1e-6], show_edges=True, edge_color='black',
                line_width=0.3,
                scalar_bar_args=dict(
                    title='Predicted axial force Fz_s (N)',
                    n_labels=5, label_font_size=13, title_font_size=14,
                    position_x=0.82, position_y=0.05, height=0.50,
                    width=0.12, vertical=True, fmt='%.0f',
                ),
            )
            hover_layers.append({'actor': acc_actor, 'label_fn': _make_force_label(acc_force)})
        if len(blk_faces) > 0:
            blk_mesh = _build_pyvista_mesh(stl_vertices, blk_faces)
            blk_actor = pl.add_mesh(blk_mesh, color='grey', show_edges=True,
                                    edge_color='black', line_width=0.3)
            hover_layers.append({'actor': blk_actor, 'label_fn': _inaccessible_label})
    else:
        sel_mesh = _build_pyvista_mesh(stl_vertices, sel_faces)
        sel_mesh.cell_data['force_N'] = predicted_force_N
        sel_actor = pl.add_mesh(
            sel_mesh,
            scalars='force_N',
            cmap='inferno',
            clim=[0.0, fmax + 1e-6],
            show_edges=True,
            edge_color='black',
            line_width=0.3,
            scalar_bar_args=dict(
                title='Predicted axial force Fz_s (N)',
                n_labels=5,
                label_font_size=13,
                title_font_size=14,
                position_x=0.82,
                position_y=0.05,
                height=0.50,
                width=0.12,
                vertical=True,
                fmt='%.0f',
            ),
        )
        hover_layers.append({'actor': sel_actor, 'label_fn': _make_force_label(predicted_force_N)})

    blocked_line = (f"Inaccessible (grey): {n_blocked} faces\n"
                     if accessible_mask is not None else "")
    caveat_line = f"CAVEAT: {force_caveat}\n" if force_caveat else ""
    pl.add_text(
        f"{window_title}\n"
        f"Material: {material_key}  |  Direction: {direction_label}\n"
        f"Force model: Aerens et al. 2010 (steady-state axial Fz_s, relative effort proxy)\n"
        f"Max: {fmax:.0f} N   |   Mean: {fmean:.0f} N\n"
        f"{blocked_line}"
        f"{caveat_line}"
        f"NOTE: Fz_s ~ angle*cos(angle) -- NOT monotonic in wall angle;\n"
        f"can be LOWER at very steep walls than at ~50-60 deg. See README.",
        position='upper_left',
        font_size=11,
        color='black',
    )
    enable_hover_tooltip(pl, {0: hover_layers})

    pl.add_axes(line_width=3)
    pl.show()


def visualize_3axis_results(stl_file, results,
                              selected_tri_indices,
                              forming_limit_deg,
                              material_key):
    """
    Opens two sequential PyVista windows showing violation maps.
    """
    tm = trimesh.load(stl_file, force='mesh')
    tm.fix_normals()
    vertices = np.array(tm.vertices, dtype=np.float32)
    faces    = np.array(tm.faces,    dtype=np.int32)

    baseline_wa   = results['baseline_wall_angles']
    best_wa       = results['best_wall_angles']
    best_dir      = results['best_direction']
    best_viol = float(results['combined_violation_pct'][results['best_idx']])
    directions = results['directions']
    baseline_dir = results['baseline_direction']
    baseline_pole_idx = results.get('baseline_idx')
    if baseline_pole_idx is None:
        baseline_pole_idx = int(np.argmax(directions @ (
            baseline_dir / np.linalg.norm(baseline_dir))))
    baseline_viol = float(results['combined_violation_pct'][baseline_pole_idx])

    # ── Window 1: Baseline ────────────────────────────────────────────────
    print(f"\nOpening Window 1: Baseline direction {results['baseline_label']}...")
    _show_violation_window(
        vertices, faces,
        selected_tri_indices,
        baseline_wa,
        forming_limit_deg,
        direction_label=results['baseline_label'],
        violation_pct=baseline_viol,
        material_key=material_key,
        window_title='BASELINE DIRECTION',
        accessible_mask=results.get('baseline_accessible'),
    )

    # ── Window 2: Optimized ───────────────────────────────────────────────
    print(f"Opening Window 2: Optimized direction...")
    _show_violation_window(
        vertices, faces,
        selected_tri_indices,
        best_wa,
        forming_limit_deg,
        direction_label=(f'[{best_dir[0]:.3f}, {best_dir[1]:.3f}, '
                         f'{best_dir[2]:.3f}]  (optimized)'),
        violation_pct=best_viol,
        material_key=material_key,
        window_title='OPTIMIZED WORK-PLANE ORIENTATION (part re-clamped normal to this direction)',
        accessible_mask=results.get('best_accessible'),
    )

    # ── Window 3: Predicted forming force (optional) ────────────────────────
    if results.get('predicted_force_N') is not None:
        print(f"Opening Window 3: Predicted forming force (optimized direction)...")
        _show_force_window(
            vertices, faces,
            selected_tri_indices,
            results['predicted_force_N'],
            material_key,
            direction_label=(f'[{best_dir[0]:.3f}, {best_dir[1]:.3f}, '
                             f'{best_dir[2]:.3f}]  (optimized)'),
            window_title='PREDICTED FORMING FORCE (optimized work-plane orientation, t0)',
            accessible_mask=results.get('best_accessible'),
            force_caveat=results.get('force_caveat'),
        )

    # ── Terminal summary ──────────────────────────────────────────────────
    face_diff = results['face_difficulty']
    always    = face_diff >= 0.80
    often     = (face_diff >= 0.40) & ~always

    print(f"\n{'='*60}")
    print(f"3-AXIS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Material:          {material_key} (limit: {forming_limit_deg} deg)")
    print(f"Selected faces:    {len(selected_tri_indices)} triangles")
    if results.get('predicted_force_N') is not None:
        fN = results['predicted_force_N']
        print(f"Predicted force:   max {fN.max():.0f} N, mean {fN.mean():.0f} N "
              f"(Aerens et al. 2010, at optimized work-plane orientation)")
        if results.get('force_caveat'):
            print(f"  CAVEAT: {results['force_caveat']}")
    grad = results.get('wall_angle_gradient')
    if grad is not None:
        print(f"Wall-angle gradient [TIER 2, unvalidated heuristic]: "
              f"{int(grad['gradient_risk'].sum())} faces above "
              f"{grad['gradient_threshold_deg_per_mm']:.1f} deg/mm")
    print(f"")

    # Replace the two direction print blocks with:
    same_dir = np.allclose(best_dir, results['baseline_direction'], atol=1e-3)

    print(f"Baseline {results['baseline_label']}:")
    print(f"  Combined violation:  {baseline_viol:.1f}%")
    print(f"    Wall angle only  : {results['wall_violation_pct'][baseline_pole_idx]:.1f}%")
    print(f"    Ray blocked      : {results['blocked_pct'][baseline_pole_idx]:.1f}%")
    print(f"")
    if same_dir:
        print(f"Optimized direction: same as baseline (no improvement found in hemisphere)")
    else:
        print(f"Optimized [{best_dir[0]:.3f},{best_dir[1]:.3f},{best_dir[2]:.3f}]:")
    print(f"  Combined violation:  {best_viol:.1f}%")
    print(f"    Wall angle only  : {results['wall_violation_pct'][results['best_idx']]:.1f}%")
    print(f"    Ray blocked      : {results['blocked_pct'][results['best_idx']]:.1f}%")
    print(f"  Draw distance:       {results['draw_distance_mm'][results['best_idx']]:.1f} mm")
    if not same_dir:
        print(f"  Improvement:         {baseline_viol - best_viol:.1f}% reduction in violation")
    
    print(f"")

    # Replace the final if/elif/else block:
    print(f"Face difficulty across all hemisphere directions:")
    print(f"  Always difficult  (>80% dirs): {always.sum()} faces")
    print(f"  Often difficult (40-80% dirs): {often.sum()} faces")

    if baseline_viol <= 0.5 and best_viol <= 0.5:
        print(f"  -> Part is formable from optimized direction")
    elif best_viol < baseline_viol - 1.0:
        print(f"  -> Optimized direction reduces violation by "
            f"{baseline_viol - best_viol:.1f}% — use this direction")
    elif abs(best_viol - baseline_viol) < 1.0:
        print(f"  -> No better direction found in hemisphere "
            f"({best_viol:.1f}% violation remains)")
        print(f"     Consider multi-stage forming or redesign (5-axis tool tilt "
              f"improves reach, not the wall-angle limit)")
        if best_viol > 20.0:
            print(f"     WARNING: {best_viol:.1f}% area still violates forming limit")
            print(f"     Ref: Duflou et al. 2018 page 749")
    else:
        print(f"  -> {best_viol:.1f}% violation remains after optimization")
        if always.sum() / max(len(face_diff), 1) > 0.20:
            print(f"     Consider heat-assisted SPIF (LASPIF)")
        else:
            print(f"     Multi-stage toolpath recommended")
            print(f"     Ref: Duflou et al. 2018 page 753")

def visualize_multipass_results(stl_path, multipass_results, selected_triangle_indices,
                                 t0_mm, accessible_mask=None):
    """
    Renders 3D scalar visual maps for Multi-Pass SPIF analysis:
    1. Required Passes -- discrete, maximally distinct colors per integer
       pass count (not a gradient/heatmap), with an exact-count legend.
       "Passes required" > 1 means the nominal wall angle at that face
       VIOLATES the material's single-pass forming limit (see
       needs_multipass in analyze_multipass); more passes = a larger
       violation of that single-pass limit, not a different metric.
    2. Pure-shear sine-law thickness t0*cos(a) (path-independent), with
       faces below the critical thickness outlined -- by construction these
       are exactly the faces beyond the single-pass limit, i.e. walls whose
       multi-pass success the geometry-only sine law cannot certify.
    3. Predicted forming force (single-pass-equivalent at the fixed-plane
       wall angle and t0), if available

    accessible_mask : optional bool array aligned with selected_triangle_indices
        (typically multipass_results['accessible'], only present if
        analyze_multipass was given trimesh_mesh/face_centroids). Inaccessible
        faces render flat grey in all three panels instead of by their
        pass-count/thickness/force value -- a face the tool can't reach
        doesn't have a meaningful "how many passes" or "how much force"
        answer, matching the 5-axis feasibility map's treatment of blocked
        faces. None (default): every selected face colored normally,
        unchanged from before this parameter existed.
    """
    tm = trimesh.load(stl_path, force='mesh')
    tm.fix_normals()
    vertices = np.array(tm.vertices, dtype=np.float32)
    faces    = np.array(tm.faces,    dtype=np.int32)

    sel_idx   = np.array(selected_triangle_indices, dtype=int)
    sel_faces = faces[sel_idx]

    passes_required     = np.asarray(multipass_results['passes_required'])
    final_thickness_sel = np.asarray(multipass_results['final_thickness_mm'])
    below_crit_sel   = np.asarray(multipass_results['below_critical_thickness'])

    if accessible_mask is not None:
        accessible_mask = np.asarray(accessible_mask, dtype=bool)
        n_blocked = int((~accessible_mask).sum())
        blocked_line = f"\nInaccessible (grey): {n_blocked} faces"
    else:
        n_blocked = 0
        blocked_line = ""

    has_force = multipass_results.get('predicted_force_N') is not None
    n_cols = 3 if has_force else 2
    plotter = pv.Plotter(shape=(1, n_cols), title="Multi-Pass SPIF Analysis Results")
    hover_layers = {0: [], 1: []}
    if has_force:
        hover_layers[2] = []

    # --- Subplot 1: Pass Count Map (discrete, distinct colors, exact legend) ---
    plotter.subplot(0, 0)
    _background_mesh(plotter, vertices, faces, sel_idx)

    if accessible_mask is not None:
        acc_faces, blk_faces = sel_faces[accessible_mask], sel_faces[~accessible_mask]
        legend_text = "(no accessible faces)"
        if len(acc_faces) > 0:
            category, lut, legend_lines, vmin, vmax = _discrete_int_lut(
                passes_required[accessible_mask])
            m1 = _build_pyvista_mesh(vertices, acc_faces)
            m1.cell_data['pass_cat'] = category
            clim = [0, vmax - vmin] if vmax > vmin else [-0.5, 0.5]
            actor1 = plotter.add_mesh(
                m1, scalars='pass_cat', clim=clim,
                show_scalar_bar=False, show_edges=True,
                edge_color='black', line_width=0.3,
            )
            actor1.GetMapper().SetLookupTable(lut)
            legend_text = "\n".join(
                f"  {label} pass{'es' if label != '1' else ''}: {count} faces"
                for label, count, _ in legend_lines)
            hover_layers[0].append({'actor': actor1, 'label_fn': _make_passes_label(
                passes_required[accessible_mask])})
        if len(blk_faces) > 0:
            m1g = _build_pyvista_mesh(vertices, blk_faces)
            actor1g = plotter.add_mesh(m1g, color='grey', show_edges=True,
                                       edge_color='black', line_width=0.3)
            hover_layers[0].append({'actor': actor1g, 'label_fn': _inaccessible_label})
    else:
        category, lut, legend_lines, vmin, vmax = _discrete_int_lut(passes_required)
        m1 = _build_pyvista_mesh(vertices, sel_faces)
        m1.cell_data['pass_cat'] = category
        clim = [0, vmax - vmin] if vmax > vmin else [-0.5, 0.5]
        actor1 = plotter.add_mesh(
            m1, scalars='pass_cat', clim=clim,
            show_scalar_bar=False, show_edges=True,
            edge_color='black', line_width=0.3,
        )
        actor1.GetMapper().SetLookupTable(lut)
        legend_text = "\n".join(
            f"  {label} pass{'es' if label != '1' else ''}: {count} faces"
            for label, count, _ in legend_lines)
        hover_layers[0].append({'actor': actor1, 'label_fn': _make_passes_label(passes_required)})

    plotter.add_text(
        f"REQUIRED FORMING PASSES  [TIER 2: practice-based rule]\n"
        f"1 pass = within the single-pass forming limit; N passes = nominal\n"
        f"wall angle exceeds that limit, needing N-1 extra stages\n\n"
        f"{legend_text}{blocked_line}",
        font_size=10,
    )

    # --- Subplot 2: Sine-law thickness map ---
    plotter.subplot(0, 1)
    _background_mesh(plotter, vertices, faces, sel_idx)
    plotter.add_text(
        f"Sine-law thickness t0*cos(a)  (t0 = {t0_mm} mm)\n"
        f"pure shear, path-independent -- first-order estimate\n"
        f"red outline = below critical thickness = beyond single-pass\n"
        f"limit: multi-pass success NOT certifiable from geometry"
        f"{blocked_line}", font_size=10)

    if accessible_mask is not None:
        acc_faces, blk_faces = sel_faces[accessible_mask], sel_faces[~accessible_mask]
        acc_risk = below_crit_sel[accessible_mask]
        if len(acc_faces) > 0:
            m2 = _build_pyvista_mesh(vertices, acc_faces)
            acc_thickness = final_thickness_sel[accessible_mask]
            m2.cell_data["Thickness_mm"] = acc_thickness
            actor2 = plotter.add_mesh(m2, scalars="Thickness_mm", cmap="viridis",
                                      show_edges=False, scalar_bar_args={"title": "Thickness (mm)"})
            if acc_risk.any():
                risk_submesh = m2.extract_cells(np.nonzero(acc_risk)[0])
                plotter.add_mesh(risk_submesh, color="red", style="wireframe",
                                 line_width=3, label="Below critical thickness")
                plotter.add_legend()
            hover_layers[1].append({'actor': actor2, 'label_fn': _make_thickness_label(
                acc_thickness, acc_risk)})
        if len(blk_faces) > 0:
            m2g = _build_pyvista_mesh(vertices, blk_faces)
            actor2g = plotter.add_mesh(m2g, color='grey', show_edges=False)
            hover_layers[1].append({'actor': actor2g, 'label_fn': _inaccessible_label})
    else:
        m2 = _build_pyvista_mesh(vertices, sel_faces)
        m2.cell_data["Thickness_mm"] = final_thickness_sel
        actor2 = plotter.add_mesh(m2, scalars="Thickness_mm", cmap="viridis",
                                  show_edges=False, scalar_bar_args={"title": "Thickness (mm)"})
        if below_crit_sel.any():
            risk_submesh = m2.extract_cells(np.nonzero(below_crit_sel)[0])
            plotter.add_mesh(risk_submesh, color="red", style="wireframe",
                             line_width=3, label="Below critical thickness")
            plotter.add_legend()
        hover_layers[1].append({'actor': actor2, 'label_fn': _make_thickness_label(
            final_thickness_sel, below_crit_sel)})

    # --- Subplot 3: Predicted forming force (final pass), if available ---
    if has_force:
        force_sel = np.asarray(multipass_results['predicted_force_N'])

        plotter.subplot(0, 2)
        _background_mesh(plotter, vertices, faces, sel_idx)
        caveat = multipass_results.get('force_caveat')
        plotter.add_text(
            f"Predicted Force Fz_s (Aerens 2010)\n"
            f"single-pass-equivalent: fixed-plane wall angle, t0\n"
            f"(multi-pass per-pass force not modeled)"
            + (f"\nCAVEAT: {caveat}" if caveat else "")
            + blocked_line, font_size=10)

        if accessible_mask is not None:
            acc_faces, blk_faces = sel_faces[accessible_mask], sel_faces[~accessible_mask]
            if len(acc_faces) > 0:
                m3 = _build_pyvista_mesh(vertices, acc_faces)
                acc_force = force_sel[accessible_mask]
                m3.cell_data["Force_N"] = acc_force
                actor3 = plotter.add_mesh(m3, scalars="Force_N", cmap="inferno",
                                          show_edges=False, scalar_bar_args={"title": "Fz_s (N)"})
                hover_layers[2].append({'actor': actor3, 'label_fn': _make_force_label(acc_force)})
            if len(blk_faces) > 0:
                m3g = _build_pyvista_mesh(vertices, blk_faces)
                actor3g = plotter.add_mesh(m3g, color='grey', show_edges=False)
                hover_layers[2].append({'actor': actor3g, 'label_fn': _inaccessible_label})
        else:
            m3 = _build_pyvista_mesh(vertices, sel_faces)
            m3.cell_data["Force_N"] = force_sel
            actor3 = plotter.add_mesh(m3, scalars="Force_N", cmap="inferno",
                                      show_edges=False, scalar_bar_args={"title": "Fz_s (N)"})
            hover_layers[2].append({'actor': actor3, 'label_fn': _make_force_label(force_sel)})

    plotter.link_views()
    enable_hover_tooltip(plotter, hover_layers)
    plotter.show()