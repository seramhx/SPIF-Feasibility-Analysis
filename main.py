"""
SPIF ANALYSIS PIPELINE — MAIN ENTRY POINT
==========================================
Usage:
  python main.py --step mypart.step --material AA1050 --mode 3axis
  python main.py --step mypart.step --material Ti_grade5_RT --mode 5axis
  python main.py --step mypart.step --material DC01_steel --mode multipass

Skip reconversion if STL already exists:
  python main.py --step mypart.step --stl mypart.stl --json mypart.json
                 --material AA1050 --mode both
"""

import argparse
import json
import numpy as np
import trimesh
from pathlib import Path

from step2stl       import generate_stl_with_face_map
from visualize_step import (visualize_and_select_faces,
                             get_selected_triangle_indices)
from hemisphere     import confirm_hemisphere_with_user
from spif_analysis  import (get_forming_limit, analyze_3axis,
                             build_5axis_database, analyze_multipass,
                             FORMING_LIMITS_DEG)
from spif_visualize import visualize_3axis_results, visualize_multipass_results


def load_mesh_and_mapping(stl_file, json_file):
    mesh = trimesh.load(stl_file, force='mesh')
    mesh.fix_normals()
    with open(json_file, 'r') as f:
        mapping = json.load(f)
    return mesh, mapping


def get_face_data(mesh, selected_tri_indices):
    idx = np.array(selected_tri_indices, dtype=int)
    return (
        np.array(mesh.face_normals,     dtype=np.float32)[idx],
        np.array(mesh.triangles_center, dtype=np.float32)[idx],
        np.array(mesh.area_faces,       dtype=np.float32)[idx],
    )


def main():
    parser = argparse.ArgumentParser(
        description='SPIF Formability Analysis Pipeline')
    parser.add_argument('--step',      required=True,
                        help='Path to STEP file')
    parser.add_argument('--material',  default='AA1050',
                        choices=list(FORMING_LIMITS_DEG.keys()))
    parser.add_argument('--mode',      default='both',
                        choices=['3axis', '5axis', 'multipass', 'both'])
    parser.add_argument('--dirs',      type=int, default=300,
                        help='Number of hemisphere directions')
    parser.add_argument('--triangles', type=int, default=50000,
                        help='Target triangle count for meshing')
    parser.add_argument('--stl',       default=None,
                        help='Existing STL (skip conversion)')
    parser.add_argument('--json',      default=None,
                        help='Existing face map JSON (skip conversion)')
    parser.add_argument('--tool_radius', type=float, default=0.0,
                        help='Tool radius in mm. Drives both the ray-casting '
                             'clearance ring (0 = single ray, no clearance) '
                             'AND the forming-force tool diameter, which is '
                             'always 2x this value -- there is no separate '
                             '--tool_diameter flag.')
    parser.add_argument('--sheet_thickness', type=float, default=1.0,
                        help='Initial sheet thickness t0 in mm')
    parser.add_argument('--max_step_pass', type=float, default=10.0,
                        help='Max step angle per pass for multi-pass (deg). '
                             'Default 10 matches Duflou et al. 2008 CIRP '
                             'Annals 57(1):253-256 (5-step 50->90deg cones).')
    parser.add_argument('--max_tilt_deg', type=float, default=None,
                        help='5-axis tool tilt bound (deg) from nominal. '
                             'PLACEHOLDER if omitted -- no literature value '
                             'exists for this; set it to your machine/robot\'s '
                             'real kinematic tilt limit.')
    parser.add_argument('--tool_length', type=float, default=None,
                        help='Max tool reach in mm (tip to where the holder/'
                             'shank would start colliding with the part). A '
                             'face can be ray-clear (no occlusion) yet still '
                             'unreachable if the pocket it sits in is deeper '
                             'than this. Derived geometrically per face+'
                             'direction as the distance from the face to the '
                             'highest point of the part\'s own envelope '
                             'along that direction (see _compute_reach_depth '
                             'in spif_analysis.py). If omitted, this check '
                             'is skipped entirely and deep pockets are '
                             'reported accessible purely on the collision '
                             'check.')
    parser.add_argument('--thinning_weight', type=float, nargs='?',
                        const=0.5, default=None,
                        help='Opt-in, 3-AXIS ONLY: also minimize area-'
                             'averaged sine-law thinning in the work-plane '
                             'orientation search, blended into the score '
                             'alongside wall-violation/draw-distance/'
                             'uniformity. Pass with no value for the default '
                             '50%% weight, or a value in [0, 1]. Has no '
                             'effect on 5-axis: with the sheet clamped once, '
                             'thinning is fixed by the blank plane and does '
                             'not depend on tool tilt.')
    args = parser.parse_args()

    # Tool diameter for forming-force estimation is ALWAYS derived from
    # --tool_radius (dt = 2 x radius) -- not an independent parameter. If
    # --tool_radius is left at its ray-casting default of 0 (meaning "no
    # clearance ring"), that would make dt=0 and predicted force collapse to
    # 0 N everywhere, so fall back to a documented default in that case only.
    if args.tool_radius > 0:
        tool_diameter_mm = 2.0 * args.tool_radius
    else:
        tool_diameter_mm = 10.0  # fallback only -- see note above

    step_path = Path(args.step)
    out_dir   = step_path.parent
    stem      = step_path.stem

    print(f"\n{'#'*60}")
    print(f"SPIF ANALYSIS PIPELINE")
    print(f"{'#'*60}")
    print(f"  STEP            : {args.step}")
    print(f"  Material        : {args.material}")
    print(f"  Mode            : {args.mode}")
    print(f"  Dirs            : {args.dirs}")
    print(f"  Sheet Thickness : {args.sheet_thickness} mm")
    print(f"  Tool Radius     : {args.tool_radius} mm")
    if args.tool_length is not None:
        print(f"  Tool Length     : {args.tool_length} mm  (max reach into pockets)")
    else:
        print(f"  Tool Length     : none  (deep-pocket reach limit NOT checked)")
    if args.thinning_weight is not None:
        print(f"  Thinning Weight : {args.thinning_weight}  (3-axis optimized-direction search)")
    else:
        print(f"  Thinning Weight : none  (3-axis direction search ignores thinning)")
    if args.mode in ('5axis', 'both'):
        print(f"  5-axis note     : wall angle / thinning / force use the FIXED "
              f"blank plane; tool tilt only affects accessibility")
    if args.tool_radius > 0:
        print(f"  Tool Diameter   : {tool_diameter_mm} mm  (= 2 x tool radius; force estimation only)")
    else:
        print(f"  Tool Diameter   : {tool_diameter_mm} mm  (FALLBACK -- --tool_radius is 0, "
              f"2x radius would give 0mm; pass --tool_radius for a real tool diameter)")

    # ── 1. STEP → STL ────────────────────────────────────────────────────
    if args.stl and args.json:
        stl_file  = args.stl
        json_file = args.json
        print(f"\n[1] Using existing files:")
        print(f"    STL : {stl_file}")
        print(f"    JSON: {json_file}")
    else:
        print(f"\n[1] Converting STEP -> STL ...")
        stl_file, json_file = generate_stl_with_face_map(
            args.step, target_triangles=args.triangles)

    # ── 2. Load ───────────────────────────────────────────────────────────
    print(f"\n[2] Loading mesh and face mapping...")
    mesh, mapping = load_mesh_and_mapping(stl_file, json_file)
    n_step_faces  = len(
        mapping['step_surface_tag_to_stl_triangle_indices'])
    print(f"    Triangles  : {len(mesh.faces)}")
    print(f"    Vertices   : {len(mesh.vertices)}")
    print(f"    STEP faces : {n_step_faces}")

    # ── 3. Visualize + select faces ───────────────────────────────────────
    print(f"\n[3] Launching face selection view (PyVista)...")
    selected_tags = visualize_and_select_faces(stl_file, json_file)
    selected_tri  = get_selected_triangle_indices(selected_tags, mapping)

    print(f"\n    Selected STEP tags  : {selected_tags}")
    print(f"    Selected triangles  : {len(selected_tri)}")

    if not selected_tri:
        print("  ERROR: No triangles for selected faces. Exiting.")
        return

    face_normals, face_centroids, face_areas = \
        get_face_data(mesh, selected_tri)
    print(f"    Selected area       : {face_areas.sum():.1f} mm2")

    # ── Build PyVista mesh once for hemisphere visualization ──────────────
    import pyvista as pv
    n_f      = len(mesh.faces)
    pv_faces = np.hstack([
        np.full((n_f, 1), 3, dtype=np.int32),
        mesh.faces.astype(np.int32)
    ]).ravel()
    part_pv  = pv.PolyData(mesh.vertices.astype(np.float32), pv_faces)

    # ── 4. Hemisphere confirmation ────────────────────────────────────────
    print(f"\n[4] Hemisphere of forming directions (PyVista)...")
    directions, hemi_axis = confirm_hemisphere_with_user(
        mesh=mesh,
        part_mesh_pv=part_pv,
        n_directions=args.dirs,
    )
    print(f"    Confirmed hemisphere axis: {hemi_axis.tolist()}")

    # ── 5. Analysis ───────────────────────────────────────────────────────
    forming_limit = get_forming_limit(args.material, thickness_mm=args.sheet_thickness)

    if args.mode in ('3axis', 'both'):
        print(f"\n[5a] 3-AXIS analysis...")
        r3 = analyze_3axis(
            face_normals, face_centroids, face_areas,
            directions, forming_limit, mesh.bounds,
            trimesh_mesh=mesh,
            tool_radius=args.tool_radius,
            face_indices_global=selected_tri,
            material_key=args.material,
            sheet_thickness_mm=args.sheet_thickness,
            tool_diameter_mm=tool_diameter_mm,
            tool_length=args.tool_length,
            thinning_weight=args.thinning_weight,
            blank_normal=hemi_axis)

        print(f"\n[5b] Visualizing 3-AXIS results (2-3 PyVista windows)...")
        visualize_3axis_results(
            stl_file, r3, selected_tri,
            forming_limit, args.material)   # accessibility grey-out is read from r3 itself

    if args.mode in ('5axis', 'both'):
        print(f"\n[6] 5-AXIS database build...")
        db_path = str(out_dir / f"{stem}_5axis_db")
        build_5axis_database(
            face_normals, face_centroids, face_areas,
            face_indices_global=selected_tri,
            directions=directions,
            forming_limit_deg=forming_limit,
            mesh_bounds=mesh.bounds,
            output_path=db_path,
            trimesh_mesh=mesh,
            tool_radius=args.tool_radius,
            max_tilt_deg=args.max_tilt_deg,
            material_key=args.material,
            sheet_thickness_mm=args.sheet_thickness,
            tool_diameter_mm=tool_diameter_mm,
            tool_length=args.tool_length,
            blank_normal=hemi_axis)   # sheet clamped once, normal to the confirmed axis

    if args.mode in ('multipass', 'both'):
        print(f"\n[7] MULTI-PASS SPIF Analysis...")
        multipass_res = analyze_multipass(
            face_normals=face_normals,
            face_areas=face_areas,
            forming_limit_deg=forming_limit,
            max_step_per_pass_deg=args.max_step_pass,
            t0_mm=args.sheet_thickness,
            material_key=args.material,
            tool_diameter_mm=tool_diameter_mm,
            face_centroids=face_centroids,
            trimesh_mesh=mesh,
            tool_radius=args.tool_radius,
            tool_length=args.tool_length,
            nominal_dir=hemi_axis,
        )

        print(f"\n[7b] Visualizing MULTI-PASS results (PyVista)...")
        visualize_multipass_results(
            stl_path=stl_file,
            multipass_results=multipass_res,
            selected_triangle_indices=selected_tri,
            t0_mm=args.sheet_thickness,
            accessible_mask=multipass_res['accessible'],
        )

    print(f"\n{'#'*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()