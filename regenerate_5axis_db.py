"""
REGENERATE A 5-AXIS DATABASE UNDER THE FIXED-BLANK-PLANE MODEL
================================================================
Databases written before the 2026-09 correction (schema v1) measured wall
angle, thinning and force against each candidate TOOL direction (see WALL
ANGLE REFERENCE in spif_analysis.py). The expensive, still-valid parts of
such a database -- the per-direction angle matrix, the ray-cast accessibility
matrix, the depth matrices -- are reused here; everything derived from the
conflated angle is recomputed by evaluate_5axis_from_matrices().

The face selection and hemisphere axis are recovered from the file, so the
interactive steps of main.py are not repeated:
  - face normals: from the STL (same fix_normals() as main.py) indexed by
    face_indices_global
  - blank normal: the confirmed hemisphere axis, recovered from the stored
    Fibonacci directions and verified by regenerating them

Accessibility at the exact blank normal was never ray-cast in v1 files (the
old "nominal" was the nearest Fibonacci sample, ~3.3 deg off the axis). With
CUDA available it is ray-cast here; the tool radius used originally is not
stored, so candidates are tried against the stored matrix and the one that
reproduces it exactly is used. Without CUDA, the nearest sample is used and
flagged.

Usage:
  python regenerate_5axis_db.py --stem test --material DC01_steel \
      --sheet_thickness 1.5 --tool_diameter 4
"""

import argparse
import sys

import numpy as np
import trimesh

from hemisphere import fibonacci_hemisphere
from spif_analysis import (FORMING_LIMITS_DEG, get_forming_limit,
                           evaluate_5axis_from_matrices, save_5axis_database,
                           _compute_accessibility, FIVE_AXIS_DB_SCHEMA_VERSION)


def recover_hemisphere_axis(directions):
    """Confirmed axis from stored Fibonacci directions (sample 0 may be the
    pre-fix off-pole point, so it is excluded from the check)."""
    guess = directions.mean(axis=0)
    guess /= np.linalg.norm(guess)
    for decimals in (1, 2, 3):
        cand = np.round(guess, decimals)
        if np.linalg.norm(cand) < 1e-6:
            continue
        regen, axis = fibonacci_hemisphere(len(directions), cand)
        if np.abs(regen[1:] - directions[1:]).max() < 1e-4:
            return axis.astype(np.float64)
    raise ValueError("could not recover the hemisphere axis -- pass --axis x,y,z")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--stem', required=True)
    p.add_argument('--material', required=True, choices=list(FORMING_LIMITS_DEG))
    p.add_argument('--sheet_thickness', type=float, required=True)
    p.add_argument('--tool_diameter', type=float, required=True,
                   help='Force-model tool diameter (main.py: 2 x --tool_radius, '
                        'or 10 mm if --tool_radius was 0).')
    p.add_argument('--max_tilt_deg', type=float, default=None,
                   help='Default: the bound stored in the database.')
    p.add_argument('--axis', default=None, help='Override recovered axis, "x,y,z".')
    p.add_argument('--no_recast', action='store_true',
                   help='Skip GPU ray casting even if CUDA is available.')
    args = p.parse_args()

    npz_path = f"{args.stem}_5axis_db.npz"
    old = dict(np.load(npz_path))
    version = int(old['schema_version'][0]) if 'schema_version' in old else 1
    print(f"\n{npz_path}: schema v{version}")

    directions = old['directions'].astype(np.float32)
    angles_all = old.get('direction_angles_all', old.get('wall_angles_all'))
    accessible = old['accessible'].astype(bool)
    face_idx   = old['face_indices_global'].astype(int)
    face_areas = old['face_areas']
    reach      = old.get('reach_depth_mm')
    tool_length = (None if np.isnan(old['tool_length_mm'][0])
                   else float(old['tool_length_mm'][0]))
    max_tilt = (args.max_tilt_deg if args.max_tilt_deg is not None
                else float(old['max_tilt_deg'][0]))

    if args.axis:
        axis = np.array([float(v) for v in args.axis.split(',')])
        axis /= np.linalg.norm(axis)
    elif 'blank_normal' in old:
        axis = old['blank_normal'].astype(np.float64)
    else:
        axis = recover_hemisphere_axis(directions)
    print(f"  Blank normal (confirmed axis): {np.round(axis, 4).tolist()}")

    forming_limit = get_forming_limit(args.material, args.sheet_thickness)
    if not np.isclose(forming_limit, float(old['forming_limit_deg'][0])):
        sys.exit(f"  ERROR: forming limit {forming_limit} for {args.material} @ "
                 f"{args.sheet_thickness} mm != stored {old['forming_limit_deg'][0]} "
                 f"-- wrong --material / --sheet_thickness?")

    mesh = trimesh.load(f"{args.stem}.stl", force='mesh')
    mesh.fix_normals()
    normals   = np.asarray(mesh.face_normals, dtype=np.float32)[face_idx]
    centroids = np.asarray(mesh.triangles_center, dtype=np.float32)[face_idx]

    # Sanity: stored angle matrix must match these normals.
    chk = np.degrees(np.arccos(np.clip(np.abs(normals @ directions[:5].T), 0, 1)))
    err = float(np.abs(chk - angles_all[:, :5]).max())
    print(f"  Angle-matrix consistency vs. STL normals: max |diff| = {err:.4f} deg")
    if err > 0.05:
        sys.exit("  ERROR: STL does not match the database.")

    collision_ok = old.get('collision_ok')
    nominal_accessible = None
    notes = []

    cuda = False
    if not args.no_recast:
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            pass

    if cuda:
        # Identify the original clearance-ring radius from the stored matrix.
        probe = np.arange(0, len(directions), max(1, len(directions) // 12))
        cands = sorted({0.0, round(args.tool_diameter / 2.0, 3)})
        tool_radius = None
        for r in cands:
            col = _compute_accessibility(mesh, centroids, normals, directions[probe],
                                         tool_radius=r, tool_length=None)
            if reach is not None:
                acc = col & ~(reach[:, probe] > tool_length)
            else:
                acc = col
            mism = int((acc != accessible[:, probe]).sum())
            print(f"  tool_radius {r:g} mm: {mism} mismatches on {len(probe)} probe directions")
            if mism == 0:
                tool_radius = r
                break
        if tool_radius is None:
            print("  WARNING: no candidate tool radius reproduces the stored "
                  "accessibility -- keeping nearest-sample approximations.")
        else:
            nominal_accessible = _compute_accessibility(
                mesh, centroids, normals, axis[np.newaxis, :].astype(np.float32),
                tool_radius=tool_radius, tool_length=tool_length)[:, 0]
            notes.append(f"nominal accessibility ray-cast at the exact axis "
                         f"(tool_radius {tool_radius:g} mm)")
            if reach is not None and collision_ok is None:
                collision_ok = _compute_accessibility(
                    mesh, centroids, normals, directions,
                    tool_radius=tool_radius, tool_length=None)
                assert np.array_equal(collision_ok & ~(reach > tool_length), accessible)
                notes.append("collision-only mask recomputed exactly")

    if nominal_accessible is None:
        idx = int(np.argmax(directions @ axis))
        off = np.degrees(np.arccos(np.clip(directions[idx] @ axis, -1, 1)))
        nominal_accessible = accessible[:, idx]
        notes.append(f"nominal accessibility from nearest sample ({off:.1f} deg off axis)")
    if reach is not None and collision_ok is None:
        # Unknown collision state where reach-blocked: assume collision-clear
        # there (may label some code-2 faces as code 4).
        collision_ok = accessible | old['blocked_by_reach']
        notes.append("collision-only mask approximated (reach-blocked assumed collision-clear)")

    result = evaluate_5axis_from_matrices(
        normals, face_areas, directions, angles_all, accessible,
        blank_normal=axis, forming_limit_deg=forming_limit,
        max_tilt_deg=max_tilt, collision_ok=collision_ok,
        reach_depth_mm=reach, tool_length=tool_length,
        material_key=args.material, sheet_thickness_mm=args.sheet_thickness,
        tool_diameter_mm=args.tool_diameter,
        nominal_accessible=nominal_accessible)
    result['depth_proj_all'] = old.get('depth_proj_all')
    save_5axis_database(result, face_idx, f"{args.stem}_5axis_db")
    print(f"  Written as schema v{FIVE_AXIS_DB_SCHEMA_VERSION}. Notes:")
    for n in notes:
        print(f"    - {n}")


if __name__ == '__main__':
    main()
