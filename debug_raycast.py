"""
RAY CASTING DEBUG VISUALIZER
=============================
Loads the mesh, lets the user pick a single tool direction,
fires rays from all selected faces, and visualizes:
  - The full mesh (grey)
  - Selected faces colored green (accessible) or red (blocked)
  - Ray lines drawn from each face centroid

Usage:
  python debug_raycast.py --stl mypart.stl --json mypart.json
"""

import argparse
import numpy as np
import pyvista as pv
import trimesh
import warp as wp
import json

from spif_analysis import _build_warp_mesh, _ray_cast_kernel
from visualize_step import visualize_and_select_faces, get_selected_triangle_indices

def visualize_hemisphere_raycast_debug(trimesh_mesh, selected_tri_indices,
                                        directions, hemi_axis):
    """
    For each hemisphere direction, compute accessibility for selected faces
    and visualize aggregate results:
      - Each face colored by fraction of directions it is accessible from
        (0 = always blocked, 1 = always accessible)
      - Hemisphere direction points on sphere, colored by % accessible faces
        for that direction
    """
    face_centroids = np.array(trimesh_mesh.triangles_center, dtype=np.float32)
    face_normals   = np.array(trimesh_mesh.face_normals,     dtype=np.float32)
    vertices       = np.array(trimesh_mesh.vertices,         dtype=np.float32)
    faces          = np.array(trimesh_mesh.faces,            dtype=np.int32)

    sel_idx        = np.array(selected_tri_indices, dtype=int)
    sel_centroids  = face_centroids[sel_idx]
    sel_normals    = face_normals[sel_idx]
    n_sel          = len(sel_idx)
    n_dirs         = len(directions)

    bbox_diag      = float(np.linalg.norm(
        trimesh_mesh.bounds[1] - trimesh_mesh.bounds[0]))
    ray_length     = bbox_diag * 2.0
    bounds         = trimesh_mesh.bounds
    center         = ((bounds[0] + bounds[1]) / 2).astype(np.float32)
    sphere_radius  = bbox_diag * 0.7

    # ── Run full accessibility for all hemisphere directions ──────────────
    print(f"\n  Running hemisphere ray cast: {n_dirs} directions "
          f"x {n_sel} faces...")

    from spif_analysis import _build_warp_mesh, _ray_cast_kernel
    import warp as wp

    from spif_analysis import _compute_accessibility

    accessible_all = _compute_accessibility(
        trimesh_mesh, sel_centroids, sel_normals, directions)

    # ── Per-face accessibility fraction (0-1) ─────────────────────────────
    face_access_frac = accessible_all.mean(axis=1)   # (n_sel,)

    # ── Per-direction accessibility fraction (0-1) ────────────────────────
    dir_access_frac  = accessible_all.mean(axis=0)   # (n_dirs,)

    n_always_blocked = int((face_access_frac == 0.0).sum())
    n_always_acc     = int((face_access_frac == 1.0).sum())
    n_partial        = n_sel - n_always_blocked - n_always_acc

    print(f"\n  Face accessibility summary:")
    print(f"    Always accessible  : {n_always_acc}  "
          f"({n_always_acc/n_sel*100:.1f}%)")
    print(f"    Partially accessible: {n_partial}  "
          f"({n_partial/n_sel*100:.1f}%)")
    print(f"    Always blocked     : {n_always_blocked}  "
          f"({n_always_blocked/n_sel*100:.1f}%)")

    # ── Build PyVista scene ───────────────────────────────────────────────
    unsel_mask       = np.ones(len(faces), dtype=bool)
    unsel_mask[sel_idx] = False
    unsel_faces      = faces[unsel_mask]
    sel_faces        = faces[sel_idx]

    pl = pv.Plotter(
        shape=(1, 2),
        window_size=[1800, 900],
        border=False,
    )
    pl.set_background('white')

    # ════════════════════════════════════════════════════════════════════
    # LEFT: Face accessibility map
    # ════════════════════════════════════════════════════════════════════
    pl.subplot(0, 0)

    if len(unsel_faces) > 0:
        unsel_mesh = build_pyvista_mesh(vertices, unsel_faces)
        pl.add_mesh(unsel_mesh, color='lightgrey', opacity=0.15,
                    show_edges=False)

    sel_mesh = build_pyvista_mesh(vertices, sel_faces)
    sel_mesh.cell_data['access_frac'] = face_access_frac

    pl.add_mesh(
        sel_mesh,
        scalars='access_frac',
        cmap='RdYlGn',
        clim=[0.0, 1.0],
        show_edges=True,
        edge_color='black',
        line_width=0.3,
        scalar_bar_args=dict(
            title='Accessible fraction\n(across all dirs)',
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

    # Hemisphere axis arrow
    arrow = pv.Arrow(
        start      = (center - hemi_axis * bbox_diag * 0.6).tolist(),
        direction  = hemi_axis.tolist(),
        scale      = bbox_diag * 0.4,
        tip_length = 0.15,
        tip_radius = 0.04,
        shaft_radius=0.015,
    )
    pl.add_mesh(arrow, color='royalblue')

    pl.add_text(
        f"FACE ACCESSIBILITY MAP\n"
        f"Hemisphere axis: [{hemi_axis[0]:.2f},{hemi_axis[1]:.2f},"
        f"{hemi_axis[2]:.2f}]\n"
        f"{n_dirs} directions tested\n"
        f"Always accessible : {n_always_acc} faces\n"
        f"Always blocked    : {n_always_blocked} faces\n"
        f"Partial           : {n_partial} faces",
        position='upper_left',
        font_size=10,
        color='black',
    )
    pl.add_axes(line_width=3)

    # ════════════════════════════════════════════════════════════════════
    # RIGHT: Hemisphere direction quality map
    # ════════════════════════════════════════════════════════════════════
    pl.subplot(0, 1)

    if len(unsel_faces) > 0:
        unsel_mesh2 = build_pyvista_mesh(vertices, unsel_faces)
        pl.add_mesh(unsel_mesh2, color='lightgrey', opacity=0.15,
                    show_edges=False)

    # Part mesh faded
    sel_mesh2 = build_pyvista_mesh(vertices, sel_faces)
    pl.add_mesh(sel_mesh2, color='lightsteelblue', opacity=0.3,
                show_edges=False)

    # Direction points on sphere surface colored by accessibility fraction
    dir_pts        = directions * sphere_radius + center
    dir_cloud      = pv.PolyData(dir_pts)
    dir_cloud.point_data['dir_access'] = dir_access_frac

    pl.add_mesh(
        dir_cloud,
        scalars='dir_access',
        cmap='RdYlGn',
        clim=[0.0, 1.0],
        point_size=10,
        render_points_as_spheres=True,
        scalar_bar_args=dict(
            title='% faces accessible\nfrom this direction',
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

    # Best direction = highest accessibility fraction
    best_dir_idx = int(np.argmax(dir_access_frac))
    best_dir     = directions[best_dir_idx]
    best_frac    = dir_access_frac[best_dir_idx]

    best_arrow = pv.Arrow(
        start      = center.tolist(),
        direction  = best_dir.tolist(),
        scale      = sphere_radius * 1.1,
        tip_length = 0.15,
        tip_radius = 0.05,
        shaft_radius=0.02,
    )
    pl.add_mesh(best_arrow, color='gold', label='Best direction')

    # Hemisphere axis arrow
    arrow2 = pv.Arrow(
        start      = center.tolist(),
        direction  = hemi_axis.tolist(),
        scale      = sphere_radius * 1.1,
        tip_length = 0.15,
        tip_radius = 0.04,
        shaft_radius=0.015,
    )
    pl.add_mesh(arrow2, color='royalblue', label='Hemisphere axis')

    # Label best direction point
    best_pt_label = (center + best_dir * sphere_radius * 1.15).reshape(1, 3)
    pl.add_point_labels(
        best_pt_label,
        [f"Best\n[{best_dir[0]:.2f},{best_dir[1]:.2f},{best_dir[2]:.2f}]\n"
         f"{best_frac*100:.1f}% accessible"],
        font_size=11,
        bold=True,
        text_color='darkgreen',
        always_visible=True,
        show_points=False,
    )

    pl.add_text(
        f"HEMISPHERE DIRECTION MAP\n"
        f"Each point = one tool direction\n"
        f"Color = fraction of selected faces accessible\n"
        f"Gold arrow = best direction  "
        f"[{best_dir[0]:.2f},{best_dir[1]:.2f},{best_dir[2]:.2f}]\n"
        f"Best accessibility: {best_frac*100:.1f}%",
        position='upper_left',
        font_size=10,
        color='black',
    )
    pl.add_axes(line_width=3)
    pl.add_legend(face='circle', size=(0.15, 0.12))

    pl.show()


def load_mesh_and_mapping(stl_file, json_file):
    mesh = trimesh.load(stl_file, force='mesh')
    mesh.fix_normals()
    with open(json_file, 'r') as f:
        mapping = json.load(f)
    return mesh, mapping


def fire_debug_rays(trimesh_mesh, face_centroids, face_normals, direction):
    """
    Fire one ray per face along direction.
    Returns accessible (n_faces,) bool array.
    """
    n_faces    = len(face_centroids)
    bbox_diag  = np.linalg.norm(
        trimesh_mesh.bounds[1] - trimesh_mesh.bounds[0])
    ray_length = bbox_diag * 2.0
    epsilon    = 0.1

    direction = np.array(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)

    origins_np  = face_centroids.astype(np.float32)
    dirs_np     = np.tile(direction, (n_faces, 1)).astype(np.float32)
    normals_np  = face_normals.astype(np.float32)

    print(f"  Building Warp BVH...")
    mesh_wp = _build_warp_mesh(trimesh_mesh)

    origins_wp    = wp.array(origins_np,  dtype=wp.vec3, device='cuda')
    dirs_wp       = wp.array(dirs_np,     dtype=wp.vec3, device='cuda')
    normals_wp    = wp.array(normals_np,  dtype=wp.vec3, device='cuda')
    accessible_wp = wp.zeros(n_faces,     dtype=int,     device='cuda')

    wp.launch(
        kernel=_ray_cast_kernel,
        dim=n_faces,
        inputs=[mesh_wp.id, origins_wp, dirs_wp, normals_wp,
                accessible_wp, float(ray_length), float(epsilon)],
        device='cuda',
    )

    accessible = accessible_wp.numpy().astype(bool)

    del origins_wp, dirs_wp, normals_wp, accessible_wp

    n_acc     = accessible.sum()
    n_blocked = (~accessible).sum()
    print(f"  Accessible : {n_acc}  ({n_acc/n_faces*100:.1f}%)")
    print(f"  Blocked    : {n_blocked}  ({n_blocked/n_faces*100:.1f}%)")

    return accessible, ray_length


def build_ray_lines(face_centroids, face_normals, direction,
                    accessible, ray_length):
    """
    Build PyVista line segments from each centroid along direction.
    Ray length scaled by accessibility for visual clarity:
      accessible -> full ray_length (green)
      blocked    -> short stub      (red)
    """
    direction = np.array(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)

    n_faces = len(face_centroids)
    epsilon = 0.1

    # Offset origins slightly along face normal (same as kernel)
    origins = face_centroids + face_normals * epsilon + direction * epsilon

    acc_starts  = []
    acc_ends    = []
    blk_starts  = []
    blk_ends    = []

    for i in range(n_faces):
        start = origins[i]
        if accessible[i]:
            end = start + direction * ray_length
            acc_starts.append(start)
            acc_ends.append(end)
        else:
            # Short stub so we can see where it was fired from
            end = start + direction * (ray_length * 0.05)
            blk_starts.append(start)
            blk_ends.append(end)

    def make_lines(starts, ends):
        if not starts:
            return None
        starts = np.array(starts, dtype=np.float32)
        ends   = np.array(ends,   dtype=np.float32)
        n      = len(starts)
        points = np.empty((n * 2, 3), dtype=np.float32)
        points[0::2] = starts
        points[1::2] = ends
        cells  = np.empty(n * 3, dtype=np.int32)
        cells[0::3] = 2
        cells[1::3] = np.arange(0, n * 2, 2)
        cells[2::3] = np.arange(1, n * 2, 2)
        poly        = pv.PolyData(points)
        poly.lines  = cells
        return poly

    acc_lines = make_lines(acc_starts, acc_ends)
    blk_lines = make_lines(blk_starts, blk_ends)

    return acc_lines, blk_lines


def build_pyvista_mesh(vertices, faces):
    n      = len(faces)
    pv_f   = np.hstack([
        np.full((n, 1), 3, dtype=np.int32),
        faces.astype(np.int32)
    ]).ravel()
    return pv.PolyData(vertices.astype(np.float32), pv_f)


def visualize_debug(trimesh_mesh, selected_tri_indices,
                    direction, accessible, ray_length):
    """
    PyVista window showing:
      - Full mesh (grey, transparent)
      - Selected faces: green = accessible, red = blocked
      - Ray lines: green long = escaped, red short = blocked stub
      - Arrow showing tool direction
    """
    vertices = np.array(trimesh_mesh.vertices, dtype=np.float32)
    faces    = np.array(trimesh_mesh.faces,    dtype=np.int32)

    sel_idx    = np.array(selected_tri_indices, dtype=int)
    unsel_mask = np.ones(len(faces), dtype=bool)
    unsel_mask[sel_idx] = False

    sel_faces   = faces[sel_idx]
    unsel_faces = faces[unsel_mask]

    acc_idx  = sel_idx[accessible]
    blk_idx  = sel_idx[~accessible]

    face_centroids = np.array(trimesh_mesh.triangles_center, dtype=np.float32)
    face_normals   = np.array(trimesh_mesh.face_normals,     dtype=np.float32)

    sel_centroids = face_centroids[sel_idx]
    sel_normals   = face_normals[sel_idx]

    acc_lines, blk_lines = build_ray_lines(
        sel_centroids, sel_normals, direction, accessible, ray_length)

    d      = np.array(direction, dtype=np.float32)
    d     /= np.linalg.norm(d)
    bounds = trimesh_mesh.bounds
    center = ((bounds[0] + bounds[1]) / 2).astype(np.float32)
    diag   = float(np.linalg.norm(bounds[1] - bounds[0]))

    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background('white')

    # Full mesh background
    if len(unsel_faces) > 0:
        unsel_mesh = build_pyvista_mesh(vertices, unsel_faces)
        pl.add_mesh(unsel_mesh, color='lightgrey', opacity=0.2,
                    show_edges=False)

    # Accessible faces — green
    if len(acc_idx) > 0:
        acc_mesh = build_pyvista_mesh(vertices, faces[acc_idx])
        pl.add_mesh(acc_mesh, color='limegreen', opacity=0.9,
                    show_edges=True, edge_color='darkgreen',
                    line_width=0.5, label=f'Accessible ({len(acc_idx)})')

    # Blocked faces — red
    if len(blk_idx) > 0:
        blk_mesh = build_pyvista_mesh(vertices, faces[blk_idx])
        pl.add_mesh(blk_mesh, color='tomato', opacity=0.9,
                    show_edges=True, edge_color='darkred',
                    line_width=0.5, label=f'Blocked ({len(blk_idx)})')

    # Ray lines
    if acc_lines is not None:
        pl.add_mesh(acc_lines, color='limegreen', line_width=1.0,
                    opacity=0.4, label='Escaped rays')
    if blk_lines is not None:
        pl.add_mesh(blk_lines, color='tomato', line_width=2.0,
                    opacity=0.9, label='Blocked ray stubs')

    # Tool direction arrow (from above the part, pointing along d)
    arrow_start = center - d * diag * 0.8
    arrow = pv.Arrow(
        start      = arrow_start.tolist(),
        direction  = d.tolist(),
        scale      = diag * 0.5,
        tip_length = 0.15,
        tip_radius = 0.04,
        shaft_radius=0.015,
    )
    pl.add_mesh(arrow, color='royalblue', label='Tool direction')

    # Centroid dots
    centroid_cloud = pv.PolyData(sel_centroids)
    pl.add_mesh(centroid_cloud, color='black', point_size=6,
                render_points_as_spheres=True, label='Face centroids')

    n_sel     = len(sel_idx)
    n_acc     = int(accessible.sum())
    n_blk     = n_sel - n_acc

    pl.add_text(
        f"RAY CASTING DEBUG\n"
        f"Direction : [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}]\n"
        f"Selected faces : {n_sel}\n"
        f"Accessible (green) : {n_acc}  ({n_acc/max(n_sel,1)*100:.1f}%)\n"
        f"Blocked    (red)   : {n_blk}  ({n_blk/max(n_sel,1)*100:.1f}%)",
        position='upper_left',
        font_size=11,
        color='black',
    )

    pl.add_axes(line_width=3)
    pl.add_legend(face='circle', size=(0.18, 0.18))
    pl.show()


def main():
    parser = argparse.ArgumentParser(
        description='Ray casting debug visualizer for SPIF analysis')
    parser.add_argument('--stl',  required=True, help='Path to STL file')
    parser.add_argument('--json', required=True, help='Path to face map JSON')
    parser.add_argument('--dir',  default=None,
        help='Tool direction as x,y,z  e.g.  0,0,-1  '
             '(if omitted, you will be prompted)')
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"RAY CASTING DEBUG")
    print(f"{'#'*60}")
    print(f"  STL  : {args.stl}")
    print(f"  JSON : {args.json}")

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\n[1] Loading mesh...")
    mesh, mapping = load_mesh_and_mapping(args.stl, args.json)
    print(f"    Triangles : {len(mesh.faces)}")
    print(f"    Vertices  : {len(mesh.vertices)}")

    # ── Face selection ────────────────────────────────────────────────────
    print(f"\n[2] Select faces to debug (PyVista window)...")
    selected_tags = visualize_and_select_faces(args.stl, args.json)
    selected_tri  = get_selected_triangle_indices(selected_tags, mapping)

    print(f"    Selected triangles : {len(selected_tri)}")
    if not selected_tri:
        print("  ERROR: No triangles selected. Exiting.")
        return

    face_centroids = np.array(mesh.triangles_center, dtype=np.float32)
    face_normals   = np.array(mesh.face_normals,     dtype=np.float32)
    sel_centroids  = face_centroids[selected_tri]
    sel_normals    = face_normals[selected_tri]

    # ── Direction input ───────────────────────────────────────────────────
    while True:
        if args.dir:
            raw = args.dir
            args.dir = None  # only use CLI arg on first iteration
        else:
            print(f"\n[3] Enter tool approach direction.")
            print(f"    Examples:")
            print(f"      Straight down  : 0,0,-1")
            print(f"      Straight up    : 0,0,1")
            print(f"      From +X side   : -1,0,0")
            print(f"      Diagonal       : -1,0,-1")
            raw = input("    Direction (x,y,z) or 'q' to quit: ").strip()

        if raw.lower() == 'q':
            print("  Exiting.")
            break

        try:
            parts = [float(v.strip()) for v in raw.split(',')]
            if len(parts) != 3:
                raise ValueError("Need exactly 3 values")
            direction = np.array(parts, dtype=np.float32)
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                print("  Zero vector — try again.")
                continue
            direction /= norm
        except ValueError as e:
            print(f"  Could not parse: {e}  — try again.")
            continue

        print(f"\n  Firing rays along [{direction[0]:.3f},"
              f"{direction[1]:.3f},{direction[2]:.3f}]...")

        # ── Ray cast ──────────────────────────────────────────────────────
        accessible, ray_length = fire_debug_rays(
            mesh, sel_centroids, sel_normals, direction)

        # ── Visualize ─────────────────────────────────────────────────────
        print(f"\n[4] Opening visualization window...")
        visualize_debug(
            mesh, selected_tri, direction, accessible, ray_length)

        again = input("\nTest another direction? (y/n): ").strip().lower()
        if again != 'y':
            break

    # ── Hemisphere debug ──────────────────────────────────────────────────
    do_hemi = input(
        "\nRun full hemisphere accessibility debug? (y/n): "
    ).strip().lower()

    if do_hemi == 'y':
        from hemisphere import confirm_hemisphere_with_user
        import pyvista as pv

        n_f     = len(mesh.faces)
        pv_f    = np.hstack([
            np.full((n_f, 1), 3, dtype=np.int32),
            mesh.faces.astype(np.int32)
        ]).ravel()
        part_pv = pv.PolyData(mesh.vertices.astype(np.float32), pv_f)

        directions, hemi_axis = confirm_hemisphere_with_user(
            mesh=mesh,
            part_mesh_pv=part_pv,
            n_directions=1500,
        )

        visualize_hemisphere_raycast_debug(
            mesh, selected_tri, directions, hemi_axis)

    print(f"\n{'#'*60}")
    print(f"DEBUG COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()