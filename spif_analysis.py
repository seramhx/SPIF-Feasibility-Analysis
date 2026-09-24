"""
SPIF ANALYSIS CORE
==================
Receives the confirmed hemisphere directions array from hemisphere.py.
All analysis runs ONLY on those directions — no internal resampling.

3-AXIS:
  Per-direction metrics across hemisphere.
  Per-face wall angles at baseline and optimized direction.
  Ray casting accessibility check (Warp) — face blocked by geometry?
  Each candidate direction models a GLOBAL WORK-PLANE ROTATION: the whole
  part is re-clamped so that the blank plane is normal to that direction
  (Vanhove et al., process window extension through optimal work plane
  rotation). Tool axis and blank normal coincide, so the per-direction angle
  IS the physical wall angle for that candidate orientation.

5-AXIS:
  The sheet is clamped ONCE, with its blank plane normal to the confirmed
  hemisphere axis (`blank_normal`); only the tool reorients. Therefore:
    - Wall angle, sine-law thinning and forming force are computed ONCE per
      face, against that fixed blank normal -- they do not depend on which
      tool direction is used.
    - The per-candidate-direction angle (face normal vs. tool axis) is the
      TOOL-APPROACH ANGLE: kept only as a kinematic/diagnostic quantity and
      never fed into the forming limit, sine law or force model.
    - Accessibility (collision / reach / tilt bound) is what legitimately
      varies with tool direction. Each face gets the minimum-tilt direction
      that is collision-clear, within tool reach and within the tilt bound.
  Saved as .npz + .json.

MULTI-PASS:
  Evaluates steep faces exceeding single-pass limits, against the same fixed
  blank normal. Calculates required intermediate pass counts, the pure-shear
  sine-law thickness and a single-pass-equivalent force proxy.

RESULT TIERS
============
  Every reported metric belongs to one of two tiers (see METRIC_TIERS):
    Tier 1 (established models, reused from literature, or direct geometric
            computation): wall angle vs. forming limit, sine-law thickness,
            Aerens steady-state force, ray-cast collision / reach / tilt
            accessibility and its infeasibility attribution.
    Tier 2 (exploratory heuristics, NOT validated): wall-angle gradient,
            Flange Reservoir Ratio, multi-pass pass-count rule.
  Tier 2 numbers are always printed/saved with an explicit flag.

WALL ANGLE REFERENCE  (2026-09 correction)
==============================================================
  The sine law t = t0*cos(a) assumes pure shear along the axis normal to the
  ORIGINAL clamped blank; a is measured against that fixed blank plane. It is
  not a statement about the tool orientation at the moment of contact.
  Earlier versions of build_5axis_database measured a against each candidate
  TOOL direction and fed that into the forming-limit check, sine-law thinning
  and force -- i.e. they modeled re-clamping the blank per face, which is not
  what tool tilting does. That conflation made 5-axis tilt appear to "fix"
  over-steep walls and reduce thinning, force and pass counts. It has been
  removed: tool tilting only changes accessibility here. Any effect of tool
  tilt on local strain or force (contact conditions, friction) is a separate
  mechanism, not modeled.

References:
  Wall angle / sine law:  Duflou et al. 2018 page 752
  Work plane orientation: Duflou et al. 2018 page 754
  5-axis capability:      Duflou et al. 2018 page 744
  Forming limits:         Duflou et al. 2018 Fig.2 and page 749
                          (Int J Mater Form 11:743-773)

MANUFACTURABILITY RULE PROVENANCE  (2026-09 literature pass)
==============================================================
  Wall angle limit, thickness dependence:
    Base FORMING_LIMITS_DEG values kept as previously documented (Duflou et al.
    2018). Reference thickness for those values is NOT stated in any source
    found -- REF_THICKNESS_MM = 1.0 mm is an ASSUMPTION (chosen because 1.0 mm
    is the most common SPIF test-sheet thickness reported in the surveyed
    literature). THICKNESS_SLOPE_DEG_PER_MM is sourced from a single paired
    data point -- NOT a material-matched regression for every material below:
      Wu, Ma, Gao, Zhao, Rashed, Ma. "A novel multi-step strategy of single
      point incremental forming for high wall angle shape." J. Manuf. Process.
      2020;56:697-706. doi:10.1016/j.jmapro.2020.05.009
      -> Al3003-O, 10mm tool: theta_max = 71 deg at t0=1.2mm, 76 deg at t0=2.0mm
      -> slope = (76-71)/(2.0-1.2) = 6.25 deg/mm
    CAVEAT: the thickness-formability relationship is NOT reported as
    monotonic-increasing across all materials in the wider literature (some
    studies report decreasing formability with increasing thickness for other
    material/tool combinations). Applying this one alloy's slope to every
    material here is a best-effort approximation, flagged as such at runtime.

  Sine law / critical (fracture) thickness ratio, material-specific:
    tc_ratio is now DERIVED per material as sin(90 - forming_limit_deg) instead
    of a flat constant. Justification: the geometric sine law thickness at a
    material's own established forming-limit angle is the thickness at which
    that material is documented to fail. Cross-checked against an independent
    source: US Patent 12,358,093 ("Incremental sheet forming systems and
    methods for forming structures having steep walls") states that walls
    steeper than 60 deg are generally infeasible and the sheet has thinned to
    "less than half" of t0 at that point -- sin(90-60) = sin(30) = 0.50,
    matching the pure sine-law figure exactly.

  Multi-pass step angle (delta a) per pass:
    Default max_step_per_pass_deg changed 12.0 -> 10.0 deg, matching the
    5-step truncated-cone strategy (wall angle 50 -> 90 deg in 10 deg
    increments) in:
      Duflou JR, et al. "Process window enhancement for single point
      incremental forming through multi-step toolpaths." CIRP Annals.
      2008;57(1):253-256.
    This is common experimental practice, not a codified universal rule --
    no source was found giving a general closed-form N = f(delta_a).

  Multi-pass thickness (2026-09 correction):
    Earlier versions compounded the sine law over the staged angles,
    t = t0 * prod(cos(a_k)). That is inconsistent with the sine law's own
    derivation: under pure shear along the fixed blank normal, the axial
    thickness stays t0 through every pass, so the normal thickness is
    t0*cos(a_final) regardless of the number of passes (path-independent).
    Compounding treated each intermediate wall as a fresh flat blank and
    grossly over-predicted thinning. The pure-shear value is now reported
    as-is. It is a first-order estimate: real multi-step forming departs
    from pure shear (material is drawn in from base/flange regions), and the
    ESAFORM benchmark reports that multi-stage forming "does not
    automatically result in more uniform thickness distributions, and it
    can even lead to increased thinning in critical areas". Consequence: in
    this geometry-only model, a face steeper than the single-pass limit
    always falls below the critical thickness, i.e. multi-pass success
    cannot be certified from geometry alone.

  5-axis tool tilt bound:
    NOT available in any source surveyed as a SPIF-specific, material- or
    process-derived number -- machine/robot tilt limits are installation-
    specific (workspace, wrist singularities, tool holder geometry). The
    MAX_TILT_DEG_PLACEHOLDER value below is an ARBITRARY placeholder, not a
    literature value. It exists only so the search can be bounded; replace it
    with your machine's real kinematic limit via --max_tilt_deg.

  Wall angle gradient rule:
    No standardized rule or numeric threshold exists in the literature
    surveyed under this name. WALL_ANGLE_GRADIENT_THRESHOLD_DEG_PER_MM below
    is an UNSOURCED HEURISTIC: it flags faces whose wall angle changes sharply
    over a short distance relative to their neighbors, as a proxy for local
    strain-concentration risk (loosely motivated by VWACF fracture-testing
    literature showing that fracture depends on how wall angle varies with
    position, not only its local value -- but that literature does not supply
    a portable numeric limit). Treat as a tunable diagnostic, not a validated
    pass/fail criterion.

FORCE MODEL PROVENANCE  (2026-09)
==============================================================
  Steady-state axial force Fz_s, per face:
    Aerens R, Eyckens P, Van Bael A, Duflou JR. "Force prediction for single
    point incremental forming deduced from experimental and FEM observations."
    Int J Adv Manuf Technol. 2010;46:969-982. doi:10.1007/s00170-009-2160-2

    DC01_steel uses the paper's own dedicated DC01 regression (their Eq. 13):
      Fz_s = 16.26 * t^1.35 * dt^0.48 * dh^0.12 * a^1.11 * cos(a)   [+-13.4%]
    Every other material uses the paper's GENERALIZED formula (their Eq. 30),
    which needs only the tensile strength Rm of the material (derived from
    proportionality between Rm and a "reference force" across their 5 tested
    materials -- see TENSILE_STRENGTH_RM_MPA below):
      Fz_s = 0.0716 * Rm * t^1.57 * dt^0.41 * dh^0.09 * a * cos(a)
    where t = sheet thickness (mm), dt = tool diameter (mm), dh = scallop
    height (mm), a = wall angle (DEGREES, as specified in the source paper --
    note a appears both as a bare degree value AND inside cos(radians(a))).
    Generalized-formula precision, per the paper: <=15% error in 77% of test
    cases, checked against a material (Al 2024) not used to fit it.

    Inputs used everywhere: the PHYSICAL wall angle (against the clamped
    blank plane -- the work-plane-rotated plane for a 3-axis candidate, the
    fixed blank normal for 5-axis and multi-pass) and the initial thickness
    t0, i.e. exactly the conditions the regression was fitted on. Treat the
    output as a relative process-effort proxy. The model was fitted on
    single-pass forming only.

    IMPORTANT non-monotonic behavior (directly from the paper, not a bug):
    Fz_s is proportional to a*cos(a), which INCREASES with wall angle up to
    roughly 50-60 deg and then DECREASES toward 90 deg -- steady-state force
    is not maximal at the steepest walls. The paper's OTHER force, the peak
    force Fz_p, keeps rising with wall angle instead, but was not generalized
    across materials (only given per-material for the 5 tested alloys), so it
    is NOT implemented here.

  Tool diameter:
    In main.py, tool_diameter_mm is ALWAYS DERIVED from --tool_radius as
    2 x tool_radius -- there is no independent --tool_diameter CLI flag, by
    project decision (the two describe the same physical tool). If
    --tool_radius is left at its ray-casting default of 0 ("no clearance
    ring"), main.py falls back to a fixed 10mm for force estimation only
    (2 x 0 would otherwise zero out every predicted force).
    TOOL_DIAMETER_DEFAULT_MM = 10.0 (the paper's own "standard" test value)
    remains only as estimate_forming_force_fz()'s own default for direct
    API use outside main.py -- it is not read by the CLI at all.

  Scallop height (project decision, not a literature gap):
    SCALLOP_HEIGHT_PLACEHOLDER_MM = 0.010 -- fixed at the midpoint of the
    paper's own tested range (0.005-0.015mm) rather than exposed as a new
    parameter, per explicit project decision: all five of the paper's fitted
    scallop-height exponents are small (0.07-0.14), i.e. force is only weakly
    sensitive to it, so a fixed representative value was judged not worth the
    added parameter surface.

  Material tensile strength Rm (TENSILE_STRENGTH_RM_MPA):
    None of this project's 11 materials exactly match the paper's 5 tested
    alloys (AA3003, AA5754/AlMg3, DC01, AISI 304, 65Cr2) except DC01_steel,
    whose Rm=357 N/mm2 is taken directly from the paper's own Table (Section
    5). All other Rm values were sourced separately per-alloy (see inline
    comments on the table) -- typical/handbook values, not measurements of
    the actual sheet stock this pipeline will be run against.
    CAVEAT (heat-assisted materials): "_laser" and "_warm_*" variants reuse
    their room-temperature sibling's Rm because no forming-temperature-
    specific Rm was found for these exact alloys/conditions. The force
    model has no temperature term, so predicted force for these materials is
    a KNOWN OVERESTIMATE -- general elevated-temperature studies found during
    this search show substantial tensile-strength softening with heating
    (e.g. Ti-6Al-4V UTS drops roughly 40% by 500 degC in unrelated elevated-
    temperature tensile studies), so treat heated-variant force predictions
    as a conservative upper bound only.
"""

import json
import numpy as np
import torch
import warp as wp
from pathlib import Path


# Reference forming-limit angle per material at REF_THICKNESS_MM.
# See MANUFACTURABILITY RULE PROVENANCE above for sourcing / caveats.
FORMING_LIMITS_DEG = {
    "AA1050"          : 68,
    "AA5182"          : 55,
    "AA6061_T6"       : 50,
    "DC01_steel"      : 67,
    "Ti_grade2_RT"    : 50,
    "Ti_grade2_laser" : 62,
    "Ti_grade5_RT"    : 32,
    "Ti_grade5_laser" : 56,
    "AZ31_RT"         : 45,
    "AZ31_warm_150C"  : 59,
    "AZ31_warm_300C"  : 60,
}

# ASSUMPTION (undocumented in source material) -- see provenance note above.
REF_THICKNESS_MM = 1.0

# Sourced from ONE paired data point (Al3003-O, Wu et al. 2020) -- applied
# generically to every material as a best-effort approximation. See provenance
# note above for the full caveat.
THICKNESS_SLOPE_DEG_PER_MM = 6.25

# Clip bounds to keep extrapolation for extreme thicknesses non-nonsensical.
_MIN_FORMING_LIMIT_DEG = 10.0
_MAX_FORMING_LIMIT_DEG = 89.0

NOMINAL_TOOL_DIRECTION = np.array([0.0, 1.0, 0.0], dtype=np.float32)

# Bumped whenever the saved 5-axis .npz layout/meaning changes. Version 2 =
# fixed-blank-plane wall angle (see WALL ANGLE REFERENCE in the header).
FIVE_AXIS_DB_SCHEMA_VERSION = 2

# See RESULT TIERS in the file header.
METRIC_TIERS = {
    'wall_angle_vs_forming_limit' : 1,
    'sine_law_thickness'          : 1,
    'predicted_force_N'           : 1,
    'accessibility'               : 1,
    'infeasible_reason'           : 1,
    'passes_required'             : 2,
    'reservoir_ratio'             : 2,
    'wall_angle_gradient'         : 2,
}

# Heat-assisted variants reuse room-temperature Rm -- predicted force for
# these is a KNOWN OVERESTIMATE (see FORCE MODEL PROVENANCE above).
HEAT_ASSISTED_MATERIALS = {
    "Ti_grade2_laser", "Ti_grade5_laser", "AZ31_warm_150C", "AZ31_warm_300C",
}


def force_model_caveat(material_key):
    """Caveat string for a material's force prediction, or None."""
    if material_key in HEAT_ASSISTED_MATERIALS:
        return ("heat-assisted material: room-temperature Rm reused, force "
                "model has no temperature term -> predicted force is an "
                "UPPER BOUND (known overestimate)")
    return None


def blank_plane_wall_angles(face_normals, blank_normal):
    """
    Physical wall angle (deg) per face: angle between the face normal and the
    fixed blank normal (the axis normal to the clamped sheet). This is the
    only angle the forming limit, sine law and force model accept.
    """
    n = np.asarray(blank_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    dots = np.clip(np.abs(np.asarray(face_normals, dtype=np.float64) @ n), 0.0, 1.0)
    return np.degrees(np.arccos(dots)).astype(np.float32)


def sine_law_thickness(wall_angle_deg, t0_mm):
    """Pure-shear sine-law thickness t0*cos(a) = t0*sin(90-a), mm."""
    return (t0_mm * np.cos(np.radians(wall_angle_deg))).astype(np.float32)


def _nearest_direction_index(directions, target, label):
    """Index of the sampled direction closest to `target`, warning if the
    nearest sample is noticeably off (then per-direction quantities such as
    accessibility are only an approximation at `target`)."""
    t = np.asarray(target, dtype=np.float64)
    t = t / np.linalg.norm(t)
    idx = int(np.argmax(directions @ t))
    off = float(np.degrees(np.arccos(np.clip(directions[idx] @ t, -1.0, 1.0))))
    if off > 0.1:
        print(f"  WARNING: no sampled direction coincides with the {label} "
              f"(nearest is {off:.1f} deg off) -- per-direction values at "
              f"the {label} (e.g. accessibility) use that nearest sample.")
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# FORMING FORCE ESTIMATION  (Aerens et al. 2010) -- see FORCE MODEL PROVENANCE
# at the top of this file for full sourcing and caveats.
# ─────────────────────────────────────────────────────────────────────────────

# Ultimate tensile strength Rm, N/mm^2 (MPa). Feeds the generalized force
# formula for every material except DC01_steel, which uses a dedicated
# per-material regression instead (see estimate_forming_force_fz).
TENSILE_STRENGTH_RM_MPA = {
    "AA1050"          : 70,    # EN 573-3 O-temper spec range 60-80 MPa, midpoint
    "AA5182"          : 275,   # O-temper automotive body sheet (typical; alloy spans 280-420 MPa across all tempers)
    "AA6061_T6"       : 310,   # T6 typical (290 MPa minimum spec)
    "DC01_steel"      : 357,   # Aerens et al. 2010 Section 5 -- their own DC01 test material, exact match
    "Ti_grade2_RT"    : 345,   # CP-Ti Grade 2, ASTM minimum spec ~50 ksi
    "Ti_grade2_laser" : 345,   # SAME as RT -- no forming-temperature-specific Rm found; force is a KNOWN OVERESTIMATE, see file header
    "Ti_grade5_RT"    : 950,   # Ti-6Al-4V, annealed, room temperature
    "Ti_grade5_laser" : 950,   # SAME as RT -- see caveat above
    "AZ31_RT"         : 267,   # AZ31 sheet, room temperature
    "AZ31_warm_150C"  : 267,   # SAME as RT -- see caveat above
    "AZ31_warm_300C"  : 267,   # SAME as RT -- see caveat above; larger overestimate expected than at 150C
}

# Aerens et al. 2010's own "standard" test tool diameter. Override via
# --tool_diameter for your actual tool.
TOOL_DIAMETER_DEFAULT_MM = 10.0

# Fixed placeholder, not a CLI parameter -- see FORCE MODEL PROVENANCE above
# (Scallop height) for why this was a deliberate project decision.
SCALLOP_HEIGHT_PLACEHOLDER_MM = 0.010


def estimate_forming_force_fz(wall_angle_deg, thickness_mm, material_key,
                               tool_diameter_mm=TOOL_DIAMETER_DEFAULT_MM,
                               scallop_height_mm=SCALLOP_HEIGHT_PLACEHOLDER_MM):
    """
    Steady-state axial force Fz_s (N), from Aerens et al. 2010 (see FORCE
    MODEL PROVENANCE at the top of this file).

    wall_angle_deg : scalar or array, deg (0 = flat, 90 = vertical wall)
    thickness_mm   : scalar or array (broadcastable with wall_angle_deg),
                     sheet thickness AT THE POINT OF FORMING (mm) -- pass the
                     current (possibly already-thinned) thickness, not
                     necessarily the original t0, for multi-pass use.
    material_key   : one of FORMING_LIMITS_DEG's keys

    Returns Fz_s in Newtons, broadcast shape of wall_angle_deg/thickness_mm.
    Note the result is NOT monotonic in wall_angle_deg -- see the
    non-monotonic-behavior note in the file header before interpreting a
    lower force at a steeper wall as unexpected.
    """
    alpha_deg = np.asarray(wall_angle_deg, dtype=np.float64)
    alpha_rad = np.radians(alpha_deg)
    t  = np.asarray(thickness_mm, dtype=np.float64)
    dt = float(tool_diameter_mm)
    dh = float(scallop_height_mm)

    if material_key == "DC01_steel":
        # Dedicated DC01 regression, Aerens et al. 2010 Eq. 13 (+-13.4%)
        fz = (16.26 * t**1.35 * dt**0.48 * dh**0.12
              * (alpha_deg**1.11) * np.cos(alpha_rad))
    else:
        rm = TENSILE_STRENGTH_RM_MPA.get(material_key, 200.0)
        # Generalized formula, Aerens et al. 2010 Eq. 30
        fz = (0.0716 * rm * t**1.57 * dt**0.41 * dh**0.09
              * alpha_deg * np.cos(alpha_rad))

    fz = np.clip(fz, 0.0, None)
    return float(fz) if np.ndim(fz) == 0 else fz.astype(np.float32)


def get_forming_limit(material_key, thickness_mm=REF_THICKNESS_MM):
    """
    Thickness-adjusted forming limit angle.

    limit(t) = limit_ref + THICKNESS_SLOPE_DEG_PER_MM * (t - REF_THICKNESS_MM)

    See MANUFACTURABILITY RULE PROVENANCE at the top of this file: the slope
    is a single-alloy approximation applied to all materials, not a
    material-matched regression. thickness_mm == REF_THICKNESS_MM reproduces
    the original unscaled table exactly.
    """
    base = FORMING_LIMITS_DEG.get(material_key, 60)
    delta_t = float(thickness_mm) - REF_THICKNESS_MM
    adjusted = base + THICKNESS_SLOPE_DEG_PER_MM * delta_t
    adjusted = float(np.clip(adjusted, _MIN_FORMING_LIMIT_DEG, _MAX_FORMING_LIMIT_DEG))

    print(f"  Material : {material_key}  ->  Forming limit: {adjusted:.1f} deg "
          f"(ref {base} deg @ {REF_THICKNESS_MM:.1f}mm, t0={thickness_mm:.2f}mm, "
          f"slope {THICKNESS_SLOPE_DEG_PER_MM:+.2f} deg/mm -- "
          f"single-alloy approximation, see file header)")
    return adjusted


def angle_between(d1, d2):
    d1 = np.array(d1, dtype=float)
    d2 = np.array(d2, dtype=float)
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    return float(np.degrees(
        np.arccos(np.clip(abs(float(np.dot(d1, d2))), 0.0, 1.0))))


def _wall_angles_gpu(fn_gpu, direction_gpu):
    dot = torch.matmul(fn_gpu, direction_gpu)
    dot = torch.clamp(torch.abs(dot), 0.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


# ─────────────────────────────────────────────────────────────────────────────
# WARP RAY CASTING
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def _ray_cast_kernel(
        mesh       : wp.uint64,
        origins    : wp.array(dtype=wp.vec3),
        directions : wp.array(dtype=wp.vec3),
        normals    : wp.array(dtype=wp.vec3),
        accessible : wp.array(dtype=int),
        ray_length : float,
        epsilon    : float):
    tid = wp.tid()

    ray_origin = origins[tid]
    ray_dir    = directions[tid]
    face_norm  = normals[tid]

    t      = float(0.0)
    u      = float(0.0)
    v      = float(0.0)
    sign   = float(0.0)
    normal = wp.vec3()
    face   = int(0)

    norm_len = wp.length(face_norm)
    if norm_len < 0.001:
        adjusted_origin = ray_origin + ray_dir * epsilon
    else:
        adjusted_origin = ray_origin + face_norm * epsilon + ray_dir * epsilon

    max_query_dist = ray_length * 10.0

    if wp.mesh_query_ray(mesh, adjusted_origin, ray_dir,
                         max_query_dist, t, u, v, sign, normal, face):
        if t > ray_length * 0.9:
            accessible[tid] = 1
        else:
            accessible[tid] = 0
    else:
        accessible[tid] = 1


def _compute_reach_depth(trimesh_mesh, face_centroids, directions):
    """
    (n_faces, n_dirs) insertion depth in mm: for each face and candidate
    direction, the distance from the face back to the highest point of the
    WHOLE part's bounding envelope along that same direction -- i.e. how
    far a tool tip has to travel, starting at the part's outer extent,
    straight down to this face, in the worst case.

    depth[i, j] = max_v( dot(v, directions[j]) )  -  dot(face_centroids[i], directions[j])

    where the max is taken over every vertex of the full mesh (not just the
    selected region) -- the bounding envelope along that direction.

    PREVIOUS APPROACH (ray fired in reverse, from outside the part back
    toward the face) had a real blind spot: for any pocket whose walls run
    parallel to the direction -- which is exactly the common case for an
    "accessible" (ray-clear) face, since a ray never intersects a surface
    it's exactly parallel to -- the reverse ray sails straight through
    without hitting the walls and lands on the face's OWN triangle at
    t == ray_length, reporting depth == 0 regardless of true pocket depth.
    This projection-based bound has no such blind spot: it only depends on
    where the face sits relative to the part's actual extent along the
    direction, never on hitting (or failing to hit) a wall.

    Trade-off: this can OVERESTIMATE depth for a face on an easily-reached
    sub-feature that merely happens to sit "below" some unrelated tall
    feature elsewhere on the part along the same direction -- it's a
    whole-part envelope bound, not a true local pocket depth. Pure numpy,
    no GPU ray casting needed for this part.
    """
    verts = np.asarray(trimesh_mesh.vertices, dtype=np.float32)   # (n_verts, 3)
    dirs  = np.asarray(directions, dtype=np.float32)              # (n_dirs, 3)

    proj_verts = verts @ dirs.T                # (n_verts, n_dirs)
    max_proj   = proj_verts.max(axis=0)        # (n_dirs,)

    proj_face = np.asarray(face_centroids, dtype=np.float32) @ dirs.T   # (n_faces, n_dirs)
    depth = max_proj[np.newaxis, :] - proj_face
    return np.clip(depth, 0.0, None).astype(np.float32)


def _build_warp_mesh(trimesh_mesh):
    """Build a Warp BVH mesh from a trimesh object."""
    verts = wp.array(trimesh_mesh.vertices.astype(np.float32),
                     dtype=wp.vec3, device='cuda')
    inds  = wp.array(trimesh_mesh.faces.flatten().astype(np.int32),
                     dtype=int, device='cuda')
    return wp.Mesh(points=verts, indices=inds)


def _ray_cast_batch(mesh_wp, face_centroids, face_normals,
                    directions_batch, ray_length, epsilon=0.1,
                    tool_radius=0.0, n_ring=8):
    n_faces = len(face_centroids)
    B       = len(directions_batch)

    if tool_radius > 0.0:
        ring_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        n_rays = 1 + n_ring
    else:
        n_rays = 1

    origins_list = []
    dirs_list    = []
    normals_list = []

    for d in directions_batch:
        d_np = np.array(d, dtype=np.float32)

        if tool_radius > 0.0:
            if abs(d_np[0]) < 0.9:
                perp1 = np.cross(d_np, [1, 0, 0])
            else:
                perp1 = np.cross(d_np, [0, 1, 0])
            perp1 /= np.linalg.norm(perp1)
            perp2  = np.cross(d_np, perp1)
            perp2 /= np.linalg.norm(perp2)

            origins_list.append(face_centroids)
            dirs_list.append(np.tile(d_np, (n_faces, 1)))
            normals_list.append(face_normals)

            for angle in ring_angles:
                offset = (perp1 * np.cos(angle) +
                        perp2 * np.sin(angle)) * tool_radius
                # Ring rays start one full tool_radius further along the
                # candidate direction than the center ray, not at the same
                # height -- e.g. tool_radius=2 -> ring origins are +2mm
                # towards the hemisphere direction from the face centroid.
                lift   = d_np * tool_radius
                origins_list.append(face_centroids + offset + lift)
                dirs_list.append(np.tile(d_np, (n_faces, 1)))
                normals_list.append(face_normals)
        else:
            origins_list.append(face_centroids)
            dirs_list.append(np.tile(d_np, (n_faces, 1)))
            normals_list.append(face_normals)

    origins_np = np.concatenate(origins_list, axis=0).astype(np.float32)
    dirs_np    = np.concatenate(dirs_list,    axis=0).astype(np.float32)
    normals_np = np.concatenate(normals_list, axis=0).astype(np.float32)

    origins_wp    = wp.array(origins_np,  dtype=wp.vec3, device='cuda')
    dirs_wp       = wp.array(dirs_np,     dtype=wp.vec3, device='cuda')
    normals_wp    = wp.array(normals_np,  dtype=wp.vec3, device='cuda')
    accessible_wp = wp.zeros(len(origins_np), dtype=int, device='cuda')

    wp.launch(
        kernel=_ray_cast_kernel,
        dim=len(origins_np),
        inputs=[mesh_wp.id, origins_wp, dirs_wp, normals_wp,
                accessible_wp, float(ray_length), float(epsilon)],
        device='cuda',
    )

    result_flat = accessible_wp.numpy()
    del origins_wp, dirs_wp, normals_wp, accessible_wp

    if tool_radius > 0.0:
        result = result_flat.reshape(B, n_rays, n_faces)
        result = result.all(axis=1)
    else:
        result = result_flat.reshape(B, n_faces)

    return result.T.astype(bool)     # (n_faces, B)


def _compute_accessibility(trimesh_mesh, face_centroids, face_normals,
                            directions, batch_size=50,
                            tool_radius=0.0, n_ring=8,
                            tool_length=None, return_details=False):
    """
    tool_length : max tool reach in mm (tip to where the holder/shank would
                  start colliding with the part). A face+direction can pass
                  the collision (occlusion) check yet still be unreachable
                  if the required insertion depth into a pocket exceeds this
                  -- see _compute_reach_depth. None (default) skips the
                  check entirely (a deep, ray-clear pocket is then reported
                  accessible regardless of how deep it actually is).
    return_details : if True, also return a dict with the pre-reach
                  collision-only mask, the reach-blocked mask, and the raw
                  depth matrix (used by build_5axis_database for per-face
                  infeasibility attribution). Default False preserves the
                  original plain-array return for existing callers.
    """
    n_faces   = len(face_centroids)
    n_dirs    = len(directions)
    batch_size = int(batch_size)

    bbox_diag  = np.linalg.norm(
        trimesh_mesh.bounds[1] - trimesh_mesh.bounds[0])
    ray_length = bbox_diag * 2.0

    if tool_radius > 0.0:
        print(f"  Tool radius (clearance) : {tool_radius:.1f} mm  "
              f"({n_ring} ring rays per direction)")
    else:
        print(f"  Tool radius : 0 (single ray, no clearance)")

    print(f"  Building Warp BVH mesh...")
    mesh_wp = _build_warp_mesh(trimesh_mesh)

    collision_ok = np.zeros((n_faces, n_dirs), dtype=bool)
    n_batches  = (n_dirs + batch_size - 1) // batch_size

    print(f"  Ray casting: {n_dirs} directions, "
          f"{n_batches} batches of {batch_size}...")

    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, n_dirs)
        batch = directions[start:end]

        collision_ok[:, start:end] = _ray_cast_batch(
            mesh_wp, face_centroids, face_normals,
            batch, ray_length,
            tool_radius=tool_radius, n_ring=n_ring)

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"    Batch {b+1}/{n_batches} done")

    reach_depth_mm   = None
    blocked_by_reach = np.zeros((n_faces, n_dirs), dtype=bool)

    if tool_length is not None:
        print(f"  Tool length (max reach) : {tool_length:.1f} mm  "
              f"-- checking pocket depth vs. part envelope...")
        reach_depth_mm   = _compute_reach_depth(
            trimesh_mesh, face_centroids, directions)
        blocked_by_reach = reach_depth_mm > tool_length
        n_newly_blocked  = int((collision_ok & blocked_by_reach).sum())
        print(f"    Additional blocks (ray-clear but too deep for tool) : "
              f"{n_newly_blocked} face-direction pairs")
    else:
        print(f"  Tool length : none specified -- deep-pocket reach limit "
              f"NOT checked (a ray-clear path may still be too deep for "
              f"any real, finite-length tool). Pass --tool_length to enable.")

    accessible = collision_ok & ~blocked_by_reach

    if return_details:
        return accessible, {
            'collision_ok'      : collision_ok,
            'blocked_by_reach'  : blocked_by_reach,
            'reach_depth_mm'    : reach_depth_mm,
        }
    return accessible


# UNSOURCED HEURISTIC -- see MANUFACTURABILITY RULE PROVENANCE at the top of
# this file. No literature threshold exists for this; tune to your part/mesh.
WALL_ANGLE_GRADIENT_THRESHOLD_DEG_PER_MM = 15.0


def compute_wall_angle_gradient(trimesh_mesh, face_indices_global, wall_angles_deg,
                                 threshold_deg_per_mm=None):
    """
    HEURISTIC, NOT FROM LITERATURE. See MANUFACTURABILITY RULE PROVENANCE at
    the top of this file (Wall angle gradient rule). Flags faces whose wall
    angle changes sharply, relative to a topologically adjacent face, over a
    short centroid-to-centroid distance -- a proxy for local strain
    concentration. Not validated against experimental fracture data.

    trimesh_mesh          : full trimesh object (for face_adjacency + centroids)
    face_indices_global    : (n_selected,) global triangle indices, same order
                             as wall_angles_deg
    wall_angles_deg        : (n_selected,) wall angle per selected face, deg
    threshold_deg_per_mm   : override for WALL_ANGLE_GRADIENT_THRESHOLD_DEG_PER_MM
    """
    if threshold_deg_per_mm is None:
        threshold_deg_per_mm = WALL_ANGLE_GRADIENT_THRESHOLD_DEG_PER_MM

    face_indices_global = np.asarray(face_indices_global, dtype=int)
    wall_angles_deg     = np.asarray(wall_angles_deg, dtype=np.float32)
    n_sel        = len(face_indices_global)
    n_mesh_faces = len(trimesh_mesh.faces)

    local_idx_map = np.full(n_mesh_faces, -1, dtype=int)
    local_idx_map[face_indices_global] = np.arange(n_sel)
    is_selected = local_idx_map >= 0

    adjacency  = trimesh_mesh.face_adjacency
    pair_mask  = is_selected[adjacency[:, 0]] & is_selected[adjacency[:, 1]]
    pairs      = adjacency[pair_mask]

    gradient_per_face = np.zeros(n_sel, dtype=np.float32)

    if len(pairs) > 0:
        centroids = trimesh_mesh.triangles_center
        la   = local_idx_map[pairs[:, 0]]
        lb   = local_idx_map[pairs[:, 1]]
        dist = np.linalg.norm(centroids[pairs[:, 0]] - centroids[pairs[:, 1]], axis=1)
        dist = np.maximum(dist, 1e-6)
        grad = np.abs(wall_angles_deg[la] - wall_angles_deg[lb]) / dist

        np.maximum.at(gradient_per_face, la, grad)
        np.maximum.at(gradient_per_face, lb, grad)

    gradient_risk = gradient_per_face > threshold_deg_per_mm
    n_risk = int(gradient_risk.sum())
    print(f"  Wall angle gradient (TIER 2 HEURISTIC, unsourced threshold "
          f"{threshold_deg_per_mm:.1f} deg/mm): {n_risk}/{n_sel} faces flagged")

    return {
        'gradient_deg_per_mm'           : gradient_per_face,
        'gradient_risk'                 : gradient_risk,
        'gradient_threshold_deg_per_mm' : threshold_deg_per_mm,
    }


def _hemisphere_pole(directions, dirs_gpu):
    centroid  = directions.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    pole_idx  = int(np.argmax(directions @ centroid))
    pole_dir  = directions[pole_idx]
    pole_gpu  = dirs_gpu[pole_idx]
    pole_label = (f'[{pole_dir[0]:.3f},{pole_dir[1]:.3f},{pole_dir[2]:.3f}]'
                  f'  (hemisphere pole — no tilt)')
    return pole_dir, pole_gpu, pole_label


# ─────────────────────────────────────────────────────────────────────────────
# 3-AXIS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_3axis(face_normals, face_centroids, face_areas,
                  directions, forming_limit_deg, mesh_bounds,
                  trimesh_mesh=None, tool_radius=0.0, n_ring=8,
                  face_indices_global=None, material_key=None,
                  sheet_thickness_mm=1.0,
                  tool_diameter_mm=TOOL_DIAMETER_DEFAULT_MM,
                  tool_length=None, thinning_weight=None, blank_normal=None):
    """
    Every candidate direction d is evaluated as a GLOBAL WORK-PLANE ROTATION:
    the blank is clamped normal to d and the tool axis is d (3-axis). The
    angle between a face normal and d is therefore the physical wall angle
    for that candidate orientation, and the forming limit, sine-law thinning
    and force can legitimately be evaluated per direction here -- unlike in
    build_5axis_database, where the blank is clamped once.

    blank_normal : the confirmed hemisphere axis (main.py's hemi_axis). The
        BASELINE is the sampled direction coinciding with it (hemisphere.py
        includes the exact axis as directions[0]). None: falls back to the
        sample nearest the hemisphere centroid (legacy behavior).

    thinning_weight : if not None (opt-in), the optimized-direction search
        also minimizes local single-pass sine-law thinning
        (1 - cos(wall_angle), area-weighted across the region), not just
        wall-angle violation / draw distance / draw uniformity. The existing
        three terms keep their relative 60/25/15 proportions but are
        rescaled to share (1 - thinning_weight) of the total score, with
        thinning_weight taking the rest. Typical value 0.5. None (default)
        reproduces the original scoring exactly.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device   : {device}")
    print(f"  Faces    : {len(face_normals)}")
    print(f"  Dirs     : {len(directions)}")

    n_faces = len(face_normals)
    n_dirs  = len(directions)

    fn_gpu   = torch.from_numpy(face_normals.copy()).float().to(device)
    fc_gpu   = torch.from_numpy(face_centroids.copy()).float().to(device)
    fa_gpu   = torch.from_numpy(face_areas.copy()).float().to(device)
    dirs_gpu = torch.from_numpy(directions.copy()).float().to(device)

    bbox_center = ((mesh_bounds[0] + mesh_bounds[1]) / 2).astype(np.float32)
    bc_gpu      = torch.from_numpy(bbox_center).to(device)
    total_area  = float(face_areas.sum())

    if trimesh_mesh is not None:
        print(f"\n  Ray casting accessibility check...")
        accessible = _compute_accessibility(
            trimesh_mesh, face_centroids, face_normals, directions,
            tool_radius=tool_radius, n_ring=n_ring, tool_length=tool_length)
    else:
        print(f"  WARNING: No trimesh_mesh provided — skipping ray casting.")
        accessible = np.ones((n_faces, n_dirs), dtype=bool)

    fa_np = face_areas.copy()

    wall_violation_pct     = np.zeros(n_dirs)
    blocked_pct            = np.zeros(n_dirs)
    combined_violation_pct = np.zeros(n_dirs)
    draw_distance_mm       = np.zeros(n_dirs)
    draw_uniformity        = np.zeros(n_dirs)
    thinning_pct           = np.zeros(n_dirs)
    face_viol_count        = torch.zeros(n_faces, device=device)

    for i in range(n_dirs):
        d  = dirs_gpu[i]
        wa = _wall_angles_gpu(fn_gpu, d)

        wall_viol = wa > forming_limit_deg
        viol_area = (fa_gpu * wall_viol.float()).sum()
        wall_violation_pct[i] = (viol_area / total_area * 100).item()

        blocked_mask = ~accessible[:, i]
        blocked_area = float(fa_np[blocked_mask].sum())
        blocked_pct[i] = blocked_area / total_area * 100

        wall_viol_np  = wall_viol.cpu().numpy()
        combined_mask = wall_viol_np | blocked_mask
        combined_area = float(fa_np[combined_mask].sum())
        combined_violation_pct[i] = combined_area / total_area * 100

        face_viol_count += wall_viol.float()

        proj = torch.matmul(fc_gpu - bc_gpu, d)
        draw_distance_mm[i] = (proj.max() - proj.min()).item()
        draw_uniformity[i]  = proj.std().item()

        # Single-pass sine-law thinning fraction (1 - cos(wall_angle)),
        # area-weighted -- same law used in analyze_multipass. Valid per
        # direction here ONLY because each candidate is a whole-part work-
        # plane rotation (blank re-clamped normal to d). Only feeds the
        # score if thinning_weight is set.
        thin_frac = 1.0 - torch.cos(torch.deg2rad(wa))
        thin_area = (fa_gpu * thin_frac).sum()
        thinning_pct[i] = (thin_area / total_area * 100).item()

    nv    = combined_violation_pct / (combined_violation_pct.max() + 1e-9)
    nd    = draw_distance_mm       / (draw_distance_mm.max()       + 1e-9)
    nu    = draw_uniformity        / (draw_uniformity.max()        + 1e-9)

    if thinning_weight is not None:
        w_thin = float(thinning_weight)
        w_rest = 1.0 - w_thin
        nt     = thinning_pct / (thinning_pct.max() + 1e-9)
        score  = w_rest * (0.60 * nv + 0.25 * nd + 0.15 * nu) + w_thin * nt
        print(f"\n  Thinning-aware direction search: ON  "
              f"(thinning_weight={w_thin:.2f}, rest scaled by {w_rest:.2f})")
    else:
        score = 0.60 * nv + 0.25 * nd + 0.15 * nu

    best_idx = int(np.argmin(score))
    best_dir = directions[best_idx]

    if blank_normal is not None:
        baseline_pole_idx = _nearest_direction_index(
            directions, blank_normal, 'confirmed hemisphere axis')
        baseline_dir   = directions[baseline_pole_idx]
        baseline_gpu   = dirs_gpu[baseline_pole_idx]
        baseline_label = (f'[{baseline_dir[0]:.3f},{baseline_dir[1]:.3f},'
                          f'{baseline_dir[2]:.3f}]  (confirmed axis — no rotation)')
    else:
        baseline_dir, baseline_gpu, baseline_label = _hemisphere_pole(
            directions, dirs_gpu)
        baseline_pole_idx = int(np.argmax(directions @ (
            baseline_dir / np.linalg.norm(baseline_dir))))
    baseline_wa = _wall_angles_gpu(fn_gpu, baseline_gpu).cpu().numpy()
    baseline_accessible = accessible[:, baseline_pole_idx]

    best_dir_gpu    = dirs_gpu[best_idx]
    best_wa         = _wall_angles_gpu(fn_gpu, best_dir_gpu).cpu().numpy()
    best_accessible = accessible[:, best_idx]

    face_difficulty = (face_viol_count / n_dirs).cpu().numpy()

    print(f"\n  Best direction         : [{best_dir[0]:.3f},{best_dir[1]:.3f},"
          f"{best_dir[2]:.3f}]")
    print(f"  Combined violation     : {combined_violation_pct[best_idx]:.1f}%  "
          f"(wall + blocked)")
    print(f"    Wall angle only      : {wall_violation_pct[best_idx]:.1f}%")
    print(f"    Blocked (ray cast)   : {blocked_pct[best_idx]:.1f}%")
    if thinning_weight is not None:
        print(f"    Thinning (area-avg)  : {thinning_pct[best_idx]:.1f}%")

    baseline_combined = combined_violation_pct[baseline_pole_idx]
    print(f"  Baseline {baseline_label}")
    print(f"    Combined violation   : {baseline_combined:.1f}%")
    if thinning_weight is not None:
        print(f"    Thinning (area-avg)  : {thinning_pct[baseline_pole_idx]:.1f}%")

    gradient_result = None
    if face_indices_global is not None and trimesh_mesh is not None:
        print(f"\n  Wall angle gradient check (heuristic, at best direction)...")
        gradient_result = compute_wall_angle_gradient(
            trimesh_mesh, face_indices_global, best_wa)

    predicted_force_N = None
    if material_key is not None:
        predicted_force_N = estimate_forming_force_fz(
            best_wa, sheet_thickness_mm, material_key,
            tool_diameter_mm=tool_diameter_mm)
        print(f"\n  Predicted steady-state axial force (Aerens et al. 2010, "
              f"at optimized work-plane orientation, t0):")
        print(f"    Max  : {predicted_force_N.max():.0f} N")
        print(f"    Mean : {predicted_force_N.mean():.0f} N")
        caveat = force_model_caveat(material_key)
        if caveat:
            print(f"    CAVEAT: {caveat}")

    return {
        'directions'              : directions,
        'wall_violation_pct'      : wall_violation_pct,
        'blocked_pct'             : blocked_pct,
        'combined_violation_pct'  : combined_violation_pct,
        'draw_distance_mm'        : draw_distance_mm,
        'draw_uniformity_std'     : draw_uniformity,
        'thinning_pct'            : thinning_pct,
        'thinning_weight_used'    : thinning_weight,   # None if thinning-aware search was off
        'combined_score'          : score,
        'best_idx'                : best_idx,
        'best_direction'          : best_dir,
        'baseline_direction'      : baseline_dir,
        'baseline_label'          : baseline_label,
        'baseline_wall_angles'    : baseline_wa,
        'baseline_accessible'     : baseline_accessible,
        'best_wall_angles'        : best_wa,
        'best_accessible'         : best_accessible,
        'face_difficulty'         : face_difficulty,
        'accessible'              : accessible,
        'forming_limit_deg'       : forming_limit_deg,
        'total_area_mm2'          : total_area,
        'n_faces_analyzed'        : n_faces,
        'baseline_idx'            : baseline_pole_idx,
        'wall_angle_gradient'     : gradient_result,   # None if face_indices_global not passed (TIER 2)
        'predicted_force_N'       : predicted_force_N, # None if material_key not passed
        'force_caveat'            : force_model_caveat(material_key),
        'metric_tiers'            : METRIC_TIERS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-AXIS DATABASE BUILD
# ─────────────────────────────────────────────────────────────────────────────

# PLACEHOLDER -- NOT a literature value. See MANUFACTURABILITY RULE PROVENANCE
# at the top of this file. Real 5-axis/robot tilt limits are machine- and
# tool-holder-specific; override via --max_tilt_deg.
MAX_TILT_DEG_PLACEHOLDER = 45.0


def build_5axis_database(face_normals, face_centroids, face_areas,
                          face_indices_global, directions,
                          forming_limit_deg, mesh_bounds, output_path,
                          trimesh_mesh=None, tool_radius=0.0, n_ring=8,
                          max_tilt_deg=None, material_key=None,
                          sheet_thickness_mm=1.0,
                          tool_diameter_mm=TOOL_DIAMETER_DEFAULT_MM,
                          tool_length=None, blank_normal=None):
    """
    5-axis (tool-tilting) SPIF with the sheet clamped ONCE, its blank plane
    normal to `blank_normal` -- see WALL ANGLE REFERENCE in the file header.
    main.py passes the confirmed hemisphere axis; None falls back to the
    hemisphere pole sample (with a warning).

    This function does the GPU work only: the (n_faces, n_dirs) tool-approach
    angle / depth matrices and the ray-cast accessibility. Everything after
    that (fixed-plane severity, per-face tool direction, infeasibility
    attribution, saving) is done by evaluate_5axis_from_matrices() and
    save_5axis_database(), which are pure numpy so a saved .npz can be
    re-evaluated without a GPU (see regenerate_5axis_db.py).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device   : {device}")

    n_faces = len(face_normals)
    n_dirs  = len(directions)

    fn_gpu   = torch.from_numpy(face_normals.copy()).float().to(device)
    fc_gpu   = torch.from_numpy(face_centroids.copy()).float().to(device)
    dirs_gpu = torch.from_numpy(directions.copy()).float().to(device)

    bbox_center = ((mesh_bounds[0] + mesh_bounds[1]) / 2).astype(np.float32)
    bc_gpu      = torch.from_numpy(bbox_center).to(device)
    rel_c       = fc_gpu - bc_gpu

    BATCH = 50
    direction_angles_all = np.zeros((n_faces, n_dirs), dtype=np.float32)
    depth_proj_all       = np.zeros((n_faces, n_dirs), dtype=np.float32)

    print(f"  Building {n_faces} x {n_dirs} tool-approach angle matrix "
          f"(kinematic only -- not a forming quantity in 5-axis)...")

    for b in range(0, n_dirs, BATCH):
        bd   = dirs_gpu[b:b+BATCH]
        dots = torch.clamp(torch.abs(torch.matmul(fn_gpu, bd.T)), 0.0, 1.0)
        wa   = torch.rad2deg(torch.acos(dots))
        dp   = torch.matmul(rel_c, bd.T)
        direction_angles_all[:, b:b+BATCH] = wa.cpu().numpy()
        depth_proj_all[:, b:b+BATCH]       = dp.cpu().numpy()
        if device == 'cuda':
            torch.cuda.empty_cache()

    reach_details = None
    if trimesh_mesh is not None:
        print(f"\n  Ray casting accessibility check...")
        accessible, reach_details = _compute_accessibility(
            trimesh_mesh, face_centroids, face_normals, directions,
            tool_radius=tool_radius, n_ring=n_ring, tool_length=tool_length,
            return_details=True)
    else:
        print(f"  WARNING: No trimesh_mesh provided — skipping ray casting.")
        accessible = np.ones((n_faces, n_dirs), dtype=bool)

    if blank_normal is None:
        blank_normal, _, _ = _hemisphere_pole(directions, dirs_gpu)
        print(f"  WARNING: blank_normal not given -- using the hemisphere pole "
              f"sample {np.round(blank_normal, 3).tolist()} as the clamping "
              f"normal. Pass the confirmed hemisphere axis for exact results.")

    result = evaluate_5axis_from_matrices(
        face_normals, face_areas, directions,
        direction_angles_all, accessible,
        blank_normal=blank_normal,
        forming_limit_deg=forming_limit_deg,
        max_tilt_deg=max_tilt_deg,
        collision_ok=(reach_details['collision_ok']
                      if reach_details is not None else None),
        reach_depth_mm=(reach_details['reach_depth_mm']
                        if reach_details is not None else None),
        tool_length=tool_length,
        material_key=material_key,
        sheet_thickness_mm=sheet_thickness_mm,
        tool_diameter_mm=tool_diameter_mm)
    result['depth_proj_all'] = depth_proj_all

    result.update(save_5axis_database(result, face_indices_global, output_path))
    return result


INFEASIBLE_REASON_TEXT = {
    0: "feasible",
    1: "wall angle (vs. fixed blank plane) exceeds forming limit -- tool tilt "
       "cannot fix this: redesign, multi-pass, or a different clamping "
       "orientation",
    2: "blocked by collision from every direction (occlusion/undercut)",
    3: "reachable only beyond the tilt bound (--max_tilt_deg)",
    4: "collision-clear but too deep for the tool (--tool_length) everywhere",
}


def evaluate_5axis_from_matrices(face_normals, face_areas, directions,
                                  direction_angles_all, accessible,
                                  blank_normal, forming_limit_deg,
                                  max_tilt_deg=None, collision_ok=None,
                                  reach_depth_mm=None, tool_length=None,
                                  material_key=None, sheet_thickness_mm=1.0,
                                  tool_diameter_mm=TOOL_DIAMETER_DEFAULT_MM,
                                  nominal_accessible=None):
    """
    Pure-numpy 5-axis evaluation (fixed clamping, tilting tool).

    SEVERITY (Tier 1) -- one value per face, against the fixed blank normal,
    independent of tool direction:
        blank_wall_angles, wall_ok, sine_law_thickness_mm, thinning_pct,
        predicted_force_N (Aerens, at t0).

    ACCESSIBILITY (Tier 1, geometric) -- the only thing tool tilt changes:
        each face gets the MINIMUM-TILT direction that is collision-clear,
        within tool reach and within max_tilt_deg (access_ok). Its tool-
        approach angle (face normal vs. that tool axis) is returned purely as
        a kinematic diagnostic.

    is_feasible = wall_ok AND access_ok.

    infeasible_reason (see INFEASIBLE_REASON_TEXT): 1 if the wall fails,
    otherwise access_reason. access_reason is kept separately so a face that
    fails BOTH is still reported as unreachable too.

    collision_ok   : (n_faces, n_dirs) pre-reach collision-only mask; None
                     means no reach check was run (== accessible).
    nominal_accessible : override for accessibility at the blank normal
                     (e.g. from an exact ray cast); None -> nearest sample.
    """
    if max_tilt_deg is None:
        max_tilt_deg = MAX_TILT_DEG_PLACEHOLDER
        print(f"  WARNING: max_tilt_deg not provided -- using PLACEHOLDER "
              f"{max_tilt_deg:.1f} deg. This is NOT sourced from literature; "
              f"it is an arbitrary bound only so the search stays finite. "
              f"Pass --max_tilt_deg with your machine's real kinematic tilt "
              f"limit for meaningful results.")
    else:
        print(f"  Tool tilt bound : {max_tilt_deg:.1f} deg")

    directions = np.asarray(directions, dtype=np.float32)
    accessible = np.asarray(accessible, dtype=bool)
    n_faces, n_dirs = accessible.shape
    bn = np.asarray(blank_normal, dtype=np.float64)
    bn = bn / np.linalg.norm(bn)
    t0 = float(sheet_thickness_mm)

    # ── Severity: fixed blank plane, one value per face ──────────────────
    blank_wa   = blank_plane_wall_angles(face_normals, bn)
    wall_ok    = blank_wa <= forming_limit_deg
    thickness  = sine_law_thickness(blank_wa, t0)
    thinning   = ((1.0 - thickness / t0) * 100.0).astype(np.float32)

    predicted_force_N = None
    if material_key is not None:
        predicted_force_N = estimate_forming_force_fz(
            blank_wa, t0, material_key, tool_diameter_mm=tool_diameter_mm)

    # ── Accessibility: varies with tool direction ─────────────────────────
    nominal_idx = _nearest_direction_index(directions, bn, 'blank normal')
    if nominal_accessible is None:
        nominal_accessible = accessible[:, nominal_idx]

    tilt_angles = np.degrees(np.arccos(np.clip(
        np.abs(directions.astype(np.float64) @ bn), 0.0, 1.0))).astype(np.float32)
    tilt_ok = tilt_angles <= max_tilt_deg

    n_dirs_tilt_excluded = int((~tilt_ok).sum())
    if n_dirs_tilt_excluded:
        print(f"  Tilt bound excludes {n_dirs_tilt_excluded}/{n_dirs} directions "
              f"(> {max_tilt_deg:.1f} deg from the blank normal)")

    valid     = accessible & tilt_ok[np.newaxis, :]
    access_ok = valid.any(axis=1)

    score = np.where(valid, tilt_angles[np.newaxis, :], np.float32(np.inf))
    best_dir_idx = np.argmin(score, axis=1).astype(np.int32)
    del score
    # No valid direction: point at the nominal direction so indexing stays
    # meaningful; tilt / approach angle are NaN there and access_ok is False.
    best_dir_idx[~access_ok] = nominal_idx

    rows = np.arange(n_faces)
    tilt_needed = tilt_angles[best_dir_idx].astype(np.float32)
    tilt_needed[~access_ok] = np.nan
    approach_at_best = np.asarray(direction_angles_all)[rows, best_dir_idx].astype(np.float32)
    approach_at_best[~access_ok] = np.nan

    is_feasible = wall_ok & access_ok

    # ── Root-cause attribution ────────────────────────────────────────────
    if collision_ok is None:
        collision_ok = accessible
    has_collision_ok = np.asarray(collision_ok, dtype=bool).any(axis=1)
    has_access       = accessible.any(axis=1)

    access_reason = np.zeros(n_faces, dtype=np.int32)
    access_reason[~has_collision_ok] = 2
    access_reason[has_collision_ok & ~has_access] = 4
    access_reason[has_access & ~access_ok] = 3

    infeasible_reason = access_reason.copy()
    infeasible_reason[~wall_ok] = 1

    best_dir_reach_depth = None
    if reach_depth_mm is not None:
        best_dir_reach_depth = np.asarray(reach_depth_mm)[rows, best_dir_idx].astype(np.float32)
        best_dir_reach_depth[~access_ok] = np.nan

    # ── Summary ───────────────────────────────────────────────────────────
    n_inf = int((~is_feasible).sum())
    print(f"\n  Fixed blank normal : {np.round(bn, 3).tolist()}  "
          f"(wall angle, thinning, force measured against this)")
    print(f"  Faces             : {n_faces}")
    print(f"  Feasible          : {int(is_feasible.sum())}  "
          f"({is_feasible.sum()/n_faces*100:.1f}%)")
    print(f"  Infeasible total  : {n_inf}  ({n_inf/n_faces*100:.1f}%)")
    print(f"    Wall angle fail : {int((~wall_ok).sum())}  "
          f"(of which also unreachable: {int((~wall_ok & ~access_ok).sum())})")
    print(f"    Ray blocked     : {int((infeasible_reason == 2).sum())}")
    print(f"    Tilt bound only : {int((infeasible_reason == 3).sum())}")
    if tool_length is not None:
        print(f"    Too deep for tool (reach > {tool_length:.1f}mm) : "
              f"{int((infeasible_reason == 4).sum())}")
    print(f"  Reachable at nominal (no tilt) : {int(nominal_accessible.sum())}  |  "
          f"reachable with tilt <= {max_tilt_deg:.0f} deg : {int(access_ok.sum())}")
    if access_ok.any():
        print(f"  Tilt needed (reachable faces)  : mean {np.nanmean(tilt_needed):.1f} deg, "
              f"max {np.nanmax(tilt_needed):.1f} deg")
    print(f"  Sine-law thinning (fixed plane): mean {thinning.mean():.1f}%, "
          f"max {thinning.max():.1f}%")
    if predicted_force_N is not None:
        print(f"  Predicted steady-state axial force (Aerens et al. 2010, "
              f"fixed blank-plane wall angle, t0):")
        print(f"    Max  : {predicted_force_N.max():.0f} N")
        print(f"    Mean : {predicted_force_N.mean():.0f} N")
        caveat = force_model_caveat(material_key)
        if caveat:
            print(f"    CAVEAT: {caveat}")

    return {
        'n_faces'                   : n_faces,
        'n_infeasible'              : n_inf,
        'directions'                : directions,
        'blank_normal'              : bn.astype(np.float32),
        'nominal_idx'               : nominal_idx,
        'direction_angles_all'      : np.asarray(direction_angles_all, dtype=np.float32),
        'accessible'                : accessible,
        'reach_depth_mm'            : reach_depth_mm,
        'collision_ok'              : np.asarray(collision_ok, dtype=bool),
        # collision-clear but too deep for the tool
        'blocked_by_reach'          : (None if reach_depth_mm is None
                                       else (np.asarray(collision_ok, dtype=bool)
                                             & ~accessible)),
        'face_areas'                : np.asarray(face_areas, dtype=np.float32),
        # severity (fixed blank plane)
        'blank_wall_angles'         : blank_wa,
        'wall_ok'                   : wall_ok,
        'sine_law_thickness_mm'     : thickness,
        'thinning_pct'              : thinning,
        'predicted_force_N'         : predicted_force_N,
        # accessibility (tool direction)
        'nominal_accessible'        : np.asarray(nominal_accessible, dtype=bool),
        'access_ok'                 : access_ok,
        'best_dir_indices'          : best_dir_idx,
        'tilt_needed_deg'           : tilt_needed,
        'approach_angle_at_best_deg': approach_at_best,
        'best_dir_reach_depth_mm'   : best_dir_reach_depth,
        # combined
        'is_feasible'               : is_feasible,
        'infeasible_reason'         : infeasible_reason,
        'access_reason'             : access_reason,
        # settings
        'forming_limit_deg'         : float(forming_limit_deg),
        'max_tilt_deg'              : float(max_tilt_deg),
        'tool_length_mm'            : tool_length,
        'material_key'              : material_key,
        'sheet_thickness_mm'        : t0,
        'tool_diameter_mm'          : float(tool_diameter_mm),
        'force_caveat'              : force_model_caveat(material_key),
        'metric_tiers'              : METRIC_TIERS,
    }


def save_5axis_database(result, face_indices_global, output_path):
    """Write an evaluate_5axis_from_matrices() result as .npz + .json."""
    r = result
    n_faces = r['n_faces']
    directions = r['directions']
    face_indices_global = np.asarray(face_indices_global)

    npz_path = str(Path(output_path).with_suffix('.npz'))
    arrays = dict(
        schema_version             = np.array([FIVE_AXIS_DB_SCHEMA_VERSION]),
        face_indices_global        = face_indices_global,
        directions                 = directions,
        blank_normal               = r['blank_normal'],
        direction_angles_all       = r['direction_angles_all'],
        accessible                 = r['accessible'],
        face_areas                 = r['face_areas'],
        blank_wall_angles          = r['blank_wall_angles'],
        wall_ok                    = r['wall_ok'],
        sine_law_thickness_mm      = r['sine_law_thickness_mm'],
        thinning_pct               = r['thinning_pct'],
        nominal_accessible         = r['nominal_accessible'],
        access_ok                  = r['access_ok'],
        best_dir_indices           = r['best_dir_indices'],
        tilt_needed_deg            = r['tilt_needed_deg'],
        approach_angle_at_best_deg = r['approach_angle_at_best_deg'],
        is_feasible                = r['is_feasible'],
        infeasible_reason          = r['infeasible_reason'],
        access_reason              = r['access_reason'],
        forming_limit_deg          = np.array([r['forming_limit_deg']]),
        max_tilt_deg               = np.array([r['max_tilt_deg']]),
        tool_length_mm             = np.array([r['tool_length_mm']
                                               if r['tool_length_mm'] is not None else np.nan]),
        sheet_thickness_mm         = np.array([r['sheet_thickness_mm']]),
        tool_diameter_mm           = np.array([r['tool_diameter_mm']]),
        material_key               = np.array([r['material_key'] or '']),
    )
    if r.get('depth_proj_all') is not None:
        arrays['depth_proj_all'] = r['depth_proj_all']
    if r['predicted_force_N'] is not None:
        arrays['predicted_force_N'] = r['predicted_force_N']
    if r['reach_depth_mm'] is not None:
        arrays['reach_depth_mm']   = r['reach_depth_mm']
        arrays['blocked_by_reach'] = r['blocked_by_reach']
        arrays['collision_ok']     = r['collision_ok']
    np.savez_compressed(npz_path, **arrays)

    def _f(x):
        return None if x is None or not np.isfinite(x) else float(x)

    force = r['predicted_force_N']
    reach = r['best_dir_reach_depth_mm']
    json_path = str(Path(output_path).with_suffix('.json'))
    summary = {
        "schema_version"         : FIVE_AXIS_DB_SCHEMA_VERSION,
        "n_faces"                : n_faces,
        "n_directions"           : len(directions),
        "material_key"           : r['material_key'],
        "sheet_thickness_mm"     : r['sheet_thickness_mm'],
        "tool_diameter_mm"       : r['tool_diameter_mm'],
        "forming_limit_deg"      : r['forming_limit_deg'],
        "max_tilt_deg_bound"     : r['max_tilt_deg'],
        "tool_length_mm"         : r['tool_length_mm'],   # None if not checked
        "blank_normal"           : r['blank_normal'].tolist(),
        "force_caveat"           : r['force_caveat'],
        "metric_tiers"           : r['metric_tiers'],
        "infeasible_reason_codes": INFEASIBLE_REASON_TEXT,
        "notes": [
            "blank_wall_angle_deg, sine_law_thickness_mm, thinning_pct and "
            "predicted_force_N are measured against the FIXED blank normal "
            "and do not depend on the tool direction.",
            "tool_approach_angle_deg is the angle between the face normal and "
            "the chosen tool axis: a kinematic diagnostic only, never used "
            "for forming-limit, thinning or force.",
            "predicted_force_N is a steady-state relative effort proxy "
            "(Aerens et al. 2010); values for wall-angle-infeasible faces "
            "are outside the model's formable range.",
        ],
        "faces": [
            {
                "face_index_in_mesh"      : int(face_indices_global[i]),
                "blank_wall_angle_deg"    : float(r['blank_wall_angles'][i]),
                "wall_ok"                 : bool(r['wall_ok'][i]),
                "sine_law_thickness_mm"   : float(r['sine_law_thickness_mm'][i]),
                "thinning_pct"            : float(r['thinning_pct'][i]),
                "predicted_force_N"       : (float(force[i]) if force is not None else None),
                "nominal_accessible"      : bool(r['nominal_accessible'][i]),
                "access_ok"               : bool(r['access_ok'][i]),
                "best_direction"          : (directions[r['best_dir_indices'][i]].tolist()
                                             if r['access_ok'][i] else None),
                "tilt_from_nominal_deg"   : _f(r['tilt_needed_deg'][i]),
                "tool_approach_angle_deg" : _f(r['approach_angle_at_best_deg'][i]),
                "best_dir_reach_depth_mm" : (_f(reach[i]) if reach is not None else None),
                "is_feasible"             : bool(r['is_feasible'][i]),
                "infeasible_reason"       : int(r['infeasible_reason'][i]),
                "access_reason"           : int(r['access_reason'][i]),
            }
            for i in range(n_faces)
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved : {npz_path}")
    print(f"  Saved : {json_path}")
    return {'npz_path': npz_path, 'json_path': json_path}


def load_5axis_database(npz_path):
    """
    Load a 5-axis .npz, refusing databases written before the fixed-blank-
    plane correction (schema < 2): their wall angle / thinning / force /
    feasibility fields were computed against the tool direction and are not
    physically meaningful. Regenerate them with regenerate_5axis_db.py.
    """
    data = np.load(npz_path)
    version = int(data['schema_version'][0]) if 'schema_version' in data.files else 1
    if version < FIVE_AXIS_DB_SCHEMA_VERSION:
        raise ValueError(
            f"{npz_path} uses 5-axis database schema v{version}, which measured "
            f"wall angle against each candidate TOOL direction (see WALL ANGLE "
            f"REFERENCE in spif_analysis.py). Regenerate it:\n"
            f"  python regenerate_5axis_db.py --stem <part> --material <key> "
            f"--sheet_thickness <t0> --tool_diameter <dt>")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PASS SPIF ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_multipass(face_normals, face_areas, forming_limit_deg,
                      max_step_per_pass_deg=10.0, t0_mm=1.0, tc_ratio=None,
                      material_key=None, tool_diameter_mm=TOOL_DIAMETER_DEFAULT_MM,
                      face_centroids=None, trimesh_mesh=None, tool_radius=0.0,
                      n_ring=8, tool_length=None, nominal_dir=None):
    """
    Evaluates faces requiring Multi-Pass SPIF (fixed tool axis = fixed blank
    normal). Calculates required intermediate pass counts (N, TIER 2), the
    pure-shear sine-law thickness and a single-pass-equivalent force proxy.

    Thickness: t0*cos(a_final), path-independent -- NOT compounded over the
    intermediate stages (see "Multi-pass thickness" in the file header).
    Because tc_ratio is derived as cos(forming_limit), a face below the
    critical thickness is exactly a face steeper than the single-pass limit:
    'below_critical_thickness' means "the geometry-only sine law cannot
    certify this wall", not a calibrated multi-pass fracture prediction.

    Force: Aerens et al. 2010 at the fixed-plane wall angle and t0 -- the
    same inputs as a single pass. The model was fitted on single-pass
    forming only; per-pass multi-pass forces are not modeled.

    nominal_dir : the tool axis wall angles are measured against. Pass the
                  ACTUAL confirmed hemisphere axis (main.py's `hemi_axis`, or
                  a saved .npz's `blank_normal`) -- 3-axis/5-axis already
                  correctly measure wall angle against whatever axis the user
                  confirmed interactively; this function previously always
                  used the hardcoded NOMINAL_TOOL_DIRECTION = [0,1,0]
                  regardless of what the user actually set, silently
                  disagreeing with the rest of the pipeline on any part using
                  a non-default axis. None (default) falls back to
                  NOMINAL_TOOL_DIRECTION for backward compatibility with
                  direct/library callers that have no hemisphere info at all.

    face_centroids, trimesh_mesh : if BOTH given, also ray-casts accessibility
                             at nominal_dir (the same axis this function
                             already uses for target_angles), so multi-pass
                             results can be cross-checked against
                             reachability like 3-axis/5-axis already are.
                             Returned as 'accessible' (None if not
                             computed -- this is opt-in, unlike analyze_3axis/
                             build_5axis_database where it's automatic,
                             because callers with no GPU/mesh on hand, e.g.
                             spif_showcase_all.py, still need to work).

    max_step_per_pass_deg : Max incremental angle change per pass. Default 10
                             deg matches the 5-step truncated-cone strategy
                             (50->90 deg in 10 deg increments) in Duflou et
                             al., "Process window enhancement for single point
                             incremental forming through multi-step
                             toolpaths", CIRP Annals 2008;57(1):253-256. This
                             reflects common experimental practice, not a
                             codified universal rule.
    t0_mm                  : Initial sheet metal thickness (mm)
    tc_ratio               : Critical thickness failure ratio. If None
                             (default), DERIVED per material as
                             sin(90 - forming_limit_deg) -- the sine-law
                             thickness at the material's own forming limit
                             angle, i.e. the thickness ratio at which that
                             material is documented to fail. See
                             MANUFACTURABILITY RULE PROVENANCE at the top of
                             this file for the cross-check against US Patent
                             12,358,093 (60 deg wall -> "less than half" of
                             t0, matching sin(90-60)=0.50 exactly).
    """
    print(f"\n[Multi-Pass Analysis Engine]")
    n_faces = len(face_normals)

    if tc_ratio is None:
        tc_ratio = float(np.sin(np.radians(90.0 - forming_limit_deg)))
        print(f"  Critical thickness ratio : {tc_ratio:.3f}  "
              f"(derived: sin(90 - {forming_limit_deg:.1f} deg) -- sine law "
              f"at this material's own forming limit, see file header)")
    else:
        print(f"  Critical thickness ratio : {tc_ratio:.3f}  (user-supplied override)")

    # Target wall angle relative to the confirmed hemisphere axis (falls
    # back to the hardcoded Y-axis only if the caller passed nothing).
    if nominal_dir is None:
        nominal_dir = NOMINAL_TOOL_DIRECTION
        print(f"  WARNING: no nominal_dir passed -- falling back to hardcoded "
              f"[0,1,0]. If this part used a custom hemisphere axis, pass it "
              f"in (main.py's hemi_axis, or a saved .npz's blank_normal) "
              f"or these results won't match the rest of the pipeline.")
    else:
        nominal_dir = np.asarray(nominal_dir, dtype=np.float64)
        nominal_dir = nominal_dir / np.linalg.norm(nominal_dir)
    target_angles = blank_plane_wall_angles(face_normals, nominal_dir)

    accessible = None
    if trimesh_mesh is not None and face_centroids is not None:
        print(f"  Ray casting accessibility check (nominal direction)...")
        accessible = _compute_accessibility(
            trimesh_mesh, face_centroids, face_normals,
            np.array([nominal_dir], dtype=np.float32),
            tool_radius=tool_radius, n_ring=n_ring, tool_length=tool_length,
        )[:, 0]
        n_blocked = int((~accessible).sum())
        print(f"  Accessible : {n_faces - n_blocked}/{n_faces} faces at nominal "
              f"direction ({n_blocked} blocked)")

    needs_multipass = target_angles > forming_limit_deg

    # TIER 2: pass-count rule from common experimental practice (see header).
    excess_angles = np.maximum(0.0, target_angles - forming_limit_deg)
    passes_required = np.ones(n_faces, dtype=int)
    passes_required[needs_multipass] += np.ceil(
        excess_angles[needs_multipass] / max_step_per_pass_deg
    ).astype(int)

    # Pure-shear sine law at the final angle -- path-independent.
    final_thickness = sine_law_thickness(target_angles, t0_mm)

    critical_thickness = t0_mm * tc_ratio
    below_critical = final_thickness < critical_thickness

    # TIER 2 heuristic: unvalidated material-feeding proxy, arbitrary 20 deg
    # flatness threshold.
    flat_flange_area = float(face_areas[target_angles < 20.0].sum())
    steep_wall_area  = float(face_areas[needs_multipass].sum())
    reservoir_ratio  = flat_flange_area / (steep_wall_area + 1e-9)

    print(f"  Faces analyzed             : {n_faces}")
    print(f"  Faces needing Multi-Pass   : {needs_multipass.sum()} ({needs_multipass.sum()/n_faces*100:.1f}%)")
    print(f"  Max passes required        : {passes_required.max()}  [TIER 2: practice-based rule]")
    print(f"  Sine-law thickness         : min {final_thickness.min():.3f} mm "
          f"(t0 = {t0_mm} mm, pure shear, path-independent)")
    print(f"  Below critical thickness   : {below_critical.sum()} faces "
          f"(= faces beyond the single-pass limit; sine law cannot certify "
          f"multi-pass success)")
    print(f"  Flange Reservoir Ratio     : {reservoir_ratio:.2f} (Flange Area / Steep Area)  "
          f"[TIER 2: unvalidated heuristic]")

    predicted_force_N = None
    if material_key is not None:
        predicted_force_N = estimate_forming_force_fz(
            target_angles, t0_mm, material_key,
            tool_diameter_mm=tool_diameter_mm)
        print(f"  Predicted axial force (Aerens et al. 2010, single-pass-"
              f"equivalent at the fixed-plane wall angle and t0):")
        print(f"    Max  : {predicted_force_N.max():.0f} N")
        print(f"    Mean : {predicted_force_N.mean():.0f} N")
        caveat = force_model_caveat(material_key)
        if caveat:
            print(f"    CAVEAT: {caveat}")

    return {
        'needs_multipass'          : needs_multipass,
        'target_wall_angles'       : target_angles,
        'passes_required'          : passes_required,          # TIER 2
        'final_thickness_mm'       : final_thickness,          # pure-shear sine law, t0*cos(a)
        'critical_thickness_mm'    : critical_thickness,
        'tc_ratio'                 : tc_ratio,
        'below_critical_thickness' : below_critical,
        'reservoir_ratio'          : reservoir_ratio,          # TIER 2
        'predicted_force_N'        : predicted_force_N,   # None if material_key not passed
        'force_caveat'             : force_model_caveat(material_key),
        'accessible'               : accessible,          # None if trimesh_mesh/face_centroids not passed
        'metric_tiers'             : METRIC_TIERS,
    }