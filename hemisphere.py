"""
HEMISPHERE DIRECTION GENERATOR + USER CONFIRMATION — PyVista
=============================================================
Shows the PART inside a full unit sphere.
Hemisphere sample points shown on one half.
User confirms or rotates the hemisphere axis.
On correction, sample points are fully recomputed around new axis.
"""

import numpy as np
import pyvista as pv


def fibonacci_hemisphere(n, axis):
    """
    Generate n uniformly distributed directions over a hemisphere
    aligned to the given axis.

    Strategy:
      1. Generate uniform fibonacci hemisphere always pointing toward +Z
      2. Compute rotation from +Z to the desired axis
      3. Apply that rotation to all sample points
    This guarantees uniform distribution regardless of axis orientation.
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # ── Step 1: Generate uniform hemisphere in +Z space ──────────────────
    # Fibonacci hemisphere: phi in [0, pi/2], full theta
    golden  = (1 + 5**0.5) / 2
    indices = np.arange(n, dtype=float)

    # Map to upper hemisphere only (z >= 0)
    # phi goes from 0 (north pole) to pi/2 (equator)
    phi   = np.arccos(1 - (indices + 0.5) / n)   # [0, ~pi/2]
    theta = 2 * np.pi * indices / golden

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    dirs_z = np.stack([x, y, z], axis=1)   # all z >= 0, uniform hemisphere
    dirs_z /= np.linalg.norm(dirs_z, axis=1, keepdims=True)

    # The first Fibonacci sample sits ~arccos(1 - 0.5/n) off the pole (3.3 deg
    # for n=300). Snap it onto the pole so the confirmed axis itself -- the
    # blank normal every fixed-plane wall angle is measured against -- is an
    # exactly sampled direction (directions[0]), with its own accessibility.
    dirs_z[0] = [0.0, 0.0, 1.0]

    # ── Step 2: Build rotation matrix from +Z to desired axis ────────────
    z_axis  = np.array([0.0, 0.0, 1.0])
    dot     = np.clip(np.dot(z_axis, axis), -1.0, 1.0)

    if abs(dot - 1.0) < 1e-9:
        # Already aligned with +Z, no rotation needed
        rot = np.eye(3)
    elif abs(dot + 1.0) < 1e-9:
        # Exactly opposite to +Z, rotate 180 deg around X
        rot = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=float)
    else:
        # Rodrigues rotation formula
        v     = np.cross(z_axis, axis)          # rotation axis
        s     = np.linalg.norm(v)               # sin(angle)
        c     = dot                              # cos(angle)
        vx    = np.array([                       # skew-symmetric cross-product matrix
            [ 0,   -v[2],  v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1],  v[0],  0  ]
        ], dtype=float)
        rot   = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # ── Step 3: Rotate all hemisphere directions ──────────────────────────
    dirs_rotated = (rot @ dirs_z.T).T
    dirs_rotated /= np.linalg.norm(dirs_rotated, axis=1, keepdims=True)

    return dirs_rotated.astype(np.float32), axis.astype(np.float32)


def _make_sphere_wireframe(radius, n_lines=24):
    """
    Build a wireframe sphere (3 great circles: XY, XZ, YZ planes)
    as a PyVista PolyData of lines.
    """
    angles  = np.linspace(0, 2 * np.pi, 120)
    lines_pts  = []
    lines_conn = []
    offset = 0

    for plane in ['xy', 'xz', 'yz']:
        if plane == 'xy':
            pts = np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros(len(angles))
            ])
        elif plane == 'xz':
            pts = np.column_stack([
                radius * np.cos(angles),
                np.zeros(len(angles)),
                radius * np.sin(angles)
            ])
        else:
            pts = np.column_stack([
                np.zeros(len(angles)),
                radius * np.cos(angles),
                radius * np.sin(angles)
            ])
        lines_pts.append(pts)
        n = len(pts)
        for i in range(n - 1):
            lines_conn.extend([2, offset + i, offset + i + 1])
        offset += n

    all_pts  = np.vstack(lines_pts).astype(np.float32)
    conn     = np.array(lines_conn, dtype=np.int32)
    poly     = pv.PolyData(all_pts)
    poly.lines = conn
    return poly


def _make_hemisphere_boundary(axis, radius, n=120):
    """
    Build the equatorial circle of the hemisphere (boundary between
    the two halves) as a PyVista PolyData of lines.
    """
    ax = np.array(axis, dtype=float)
    ax /= np.linalg.norm(ax)

    if abs(ax[0]) < 0.9:
        p1 = np.cross(ax, [1, 0, 0])
    else:
        p1 = np.cross(ax, [0, 1, 0])
    p1 /= np.linalg.norm(p1)
    p2  = np.cross(ax, p1)
    p2 /= np.linalg.norm(p2)

    angles   = np.linspace(0, 2 * np.pi, n + 1)
    boundary = np.array([
        radius * (p1 * np.cos(a) + p2 * np.sin(a))
        for a in angles
    ], dtype=np.float32)

    n_pts = len(boundary)
    conn  = []
    for i in range(n_pts - 1):
        conn.extend([2, i, i + 1])
    conn = np.array(conn, dtype=np.int32)

    poly       = pv.PolyData(boundary)
    poly.lines = conn
    return poly


def visualize_hemisphere_pyvista(directions, axis,
                                  part_mesh_pv=None,
                                  part_bounds=None):
    """
    Show:
      - The part mesh (grey, inside the sphere)
      - A full wireframe sphere scaled around the part
      - Hemisphere sample directions (blue dots on sphere surface)
      - Hemisphere axis as red arrow
      - Hemisphere boundary circle (red dashed equator)

    part_mesh_pv : pyvista PolyData of the part (optional but recommended)
    part_bounds  : numpy array shape (2,3)  [[xmin,ymin,zmin],[xmax,ymax,zmax]]
                   used to scale the sphere so it always encloses the part
    """
    dirs = np.array(directions, dtype=np.float32)
    ax   = np.array(axis,       dtype=np.float32)
    ax  /= np.linalg.norm(ax)

    # ── Sphere radius: must enclose the part ─────────────────────────────
    if part_bounds is not None:
        bounds     = np.array(part_bounds)
        center     = (bounds[0] + bounds[1]) / 2.0
        half_diag  = np.linalg.norm(bounds[1] - bounds[0]) / 2.0
        radius     = half_diag * 1.4    # 40% margin so sphere is clearly outside part
    else:
        center = np.zeros(3)
        radius = 1.0

    # Scale direction points to sphere surface
    dir_pts = (dirs * radius) + center

    pl = pv.Plotter(window_size=[1100, 900])
    pl.set_background('white')

    # ── Part mesh ─────────────────────────────────────────────────────────
    if part_mesh_pv is not None:
        pl.add_mesh(
            part_mesh_pv,
            color='lightsteelblue',
            opacity=0.85,
            show_edges=True,
            edge_color='black',
            line_width=0.3,
            label='Part',
        )

    # ── Full wireframe sphere ─────────────────────────────────────────────
    sphere_wire = _make_sphere_wireframe(radius)
    # Translate to center
    sphere_wire.points += center.astype(np.float32)
    pl.add_mesh(sphere_wire, color='lightgrey', line_width=1,
                opacity=0.5, label='Reference sphere')

    # ── Hemisphere sample direction points ────────────────────────────────
    dir_cloud = pv.PolyData(dir_pts)
    pl.add_mesh(
        dir_cloud,
        color='royalblue',
        point_size=8,
        render_points_as_spheres=True,
        label=f'Hemisphere directions ({len(dirs)})',
    )

    # ── Hemisphere axis arrow ─────────────────────────────────────────────
    arrow_start = center
    arrow       = pv.Arrow(
        start     = arrow_start.tolist(),
        direction = ax.tolist(),
        scale     = radius * 1.35,
        tip_length= 0.15,
        tip_radius= 0.04,
        shaft_radius=0.015,
    )
    pl.add_mesh(arrow, color='red', label='Hemisphere axis')

    # ── Hemisphere boundary circle ────────────────────────────────────────
    boundary = _make_hemisphere_boundary(ax, radius)
    boundary.points += center.astype(np.float32)
    pl.add_mesh(boundary, color='red', line_width=3,
                label='Hemisphere boundary')

    # ── Axis label ────────────────────────────────────────────────────────
    label_pt = (center + ax * radius * 1.45).reshape(1, 3)
    pl.add_point_labels(
        label_pt,
        [f"Axis [{ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f}]"],
        font_size=13,
        bold=True,
        text_color='red',
        always_visible=True,
        show_points=False,
    )

    # ── Origin / center dot ───────────────────────────────────────────────
    pl.add_mesh(
        pv.PolyData(center.reshape(1, 3).astype(np.float32)),
        color='black', point_size=8,
        render_points_as_spheres=True,
    )

    pl.add_title(
        f"SPIF Hemisphere of Forming Directions\n"
        f"Axis (red): [{ax[0]:.3f},{ax[1]:.3f},{ax[2]:.3f}]  "
        f"|  {len(dirs)} directions  "
        f"|  Close window to continue",
        font_size=10,
        color='black',
    )
    pl.add_axes(line_width=3)
    pl.add_legend(face='circle', size=(0.15, 0.15))
    pl.show()


def confirm_hemisphere_with_user(mesh=None,
                                  part_mesh_pv=None,
                                  n_directions=300):
    """
    Interactive loop:
      1. Show hemisphere with current axis around the part
      2. Ask user to confirm or correct axis
      3. On correction, FULLY RECOMPUTE sample directions around new axis
      4. Re-show and repeat until confirmed

    mesh        : trimesh object (used to get bounds)
    part_mesh_pv: pyvista PolyData for display
    n_directions: number of hemisphere sample directions

    Returns: (directions np.float32 (N,3),  axis np.float32 (3,))
    """
    # Default: tool approaches from +Y (above the sheet)
    axis = np.array([0.0, 1.0, 0.0], dtype=float)
    part_bounds = mesh.bounds if mesh is not None else None

    while True:
        # Fully recompute directions for current axis
        directions, axis_norm = fibonacci_hemisphere(n_directions, axis)

        # Show visualization
        visualize_hemisphere_pyvista(
            directions,
            axis_norm,
            part_mesh_pv=part_mesh_pv,
            part_bounds=part_bounds,
        )

        print(f"\nCurrent hemisphere axis: "
              f"[{axis_norm[0]:.3f}, {axis_norm[1]:.3f}, {axis_norm[2]:.3f}]")
        print(f"The red arrow shows the direction the SPIF tool")
        print(f"approaches the sheet from.")
        print(f"Sample directions (blue dots): {len(directions)}\n")

        ans = input(
            "Is this hemisphere correct for your setup? (y/n): "
        ).strip().lower()

        if ans == 'y':
            print(f"\n  Confirmed.")
            print(f"  Hemisphere axis      : [{axis_norm[0]:.3f},"
                  f"{axis_norm[1]:.3f},{axis_norm[2]:.3f}]")
            print(f"  Directions computed  : {len(directions)}")
            return directions, axis_norm

        print("\nEnter the corrected hemisphere axis.")
        print("Examples:")
        print(f"  Tool approaches from above (default) : 0,0,-1")
        print(f"  Tool approaches from below           : 0,0,1")
        print(f"  Tool approaches from +X side         : -1,0,0")
        print(f"  Angled setup (example)               : -1,0,-1")

        raw = input("New axis (x,y,z): ").strip()
        try:
            parts    = [float(v.strip()) for v in raw.split(',')]
            if len(parts) != 3:
                raise ValueError("Need exactly 3 values")
            new_axis = np.array(parts, dtype=float)
            norm     = np.linalg.norm(new_axis)
            if norm < 1e-6:
                print("  Zero vector is not valid. Try again.")
                continue
            axis = new_axis / norm
            print(f"  Axis updated to: [{axis[0]:.3f},{axis[1]:.3f},"
                  f"{axis[2]:.3f}]  (will recompute directions)")
        except ValueError as e:
            print(f"  Could not parse input: {e}")
            print("  Please enter three comma-separated numbers, e.g.  0,0,1")