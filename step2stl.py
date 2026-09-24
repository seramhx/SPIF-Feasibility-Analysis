"""
STEP -> STL CONVERTER WITH FACE MAPPING
Converts STEP to STL and stores JSON mapping:
  STEP surface tag -> triangle indices
  STEP surface tag -> 3D centroid (area-weighted)
  STEP surface tag -> surface area mm2
"""

import json
from pathlib import Path
import numpy as np
import gmsh


def estimate_mesh_params(target_triangles=50000, minh_ratio=0.05, curvature=10):
    gmsh.option.setNumber('Mesh.MeshSizeMax', 1e6)
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.model.mesh.generate(2)

    total_area = 0.0
    total_tris = 0
    for dim, tag in gmsh.model.getEntities(dim=2):
        elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(
            dim=dim, tag=tag)
        for etype, etags, ntags in zip(elem_types, elem_tags, node_tags):
            if gmsh.model.mesh.getElementProperties(etype)[3] != 3:
                continue
            tris = np.array(ntags, dtype=int).reshape(-1, 3)
            for tri in tris:
                coords = [np.array(gmsh.model.mesh.getNode(n)[0]) for n in tri]
                e1 = coords[1] - coords[0]
                e2 = coords[2] - coords[0]
                total_area += np.linalg.norm(np.cross(e1, e2)) / 2
            total_tris += len(etags)

    avg_tri_area = total_area / max(target_triangles, 1)
    maxh = np.sqrt(avg_tri_area / 0.433)
    minh = maxh * minh_ratio

    print(f"  Coarse mesh: {total_tris} tris, surface area ~{total_area:.1f} mm2")
    print(f"  Auto params: minh={minh:.3f}  maxh={maxh:.3f}  curvature={curvature}")

    gmsh.model.mesh.clear()
    return {'minh': minh, 'maxh': maxh, 'curvature': curvature}


def generate_stl_with_face_map(step_file, stl_file=None,
                                mesh_params=None, target_triangles=50000):
    step_path = Path(step_file)
    if stl_file is None:
        stl_file = str(step_path.with_suffix('.stl'))
    json_file = str(Path(stl_file).with_suffix('.json'))

    gmsh.initialize()
    try:
        gmsh.model.add('Model')
        gmsh.merge(step_file)

        gmsh.option.setNumber('General.NumThreads', 8)
        gmsh.option.setNumber('Geometry.OCCScaling', 1)
        gmsh.option.setNumber('Geometry.Tolerance', 1e-3)
        gmsh.option.setNumber('Geometry.OCCParallel', 1)
        gmsh.option.setNumber('General.Verbosity', 2)

        gmsh.model.occ.healShapes(
            tolerance=1e-3, fixDegenerated=True, fixSmallFaces=True,
            fixSmallEdges=True, sewFaces=True, makeSolids=True)
        gmsh.model.occ.synchronize()

        if mesh_params is None:
            print(f"  Auto-computing mesh params "
                  f"(target: {target_triangles} triangles)...")
            mesh_params = estimate_mesh_params(
                target_triangles=target_triangles)

        gmsh.option.setNumber('Mesh.MeshSizeMin', float(mesh_params['minh']))
        gmsh.option.setNumber('Mesh.MeshSizeMax', float(mesh_params['maxh']))
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',
                              float(mesh_params['curvature']))
        gmsh.option.setNumber('Mesh.Binary', 0)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize('Laplace2D', True)

        final_tris = sum(
            len(etags)
            for dim, tag in gmsh.model.getEntities(dim=2)
            for etype, etags, ntags in zip(
                *gmsh.model.mesh.getElements(dim=dim, tag=tag))
            if gmsh.model.mesh.getElementProperties(etype)[3] == 3
        )
        print(f"  Final mesh: {final_tris} triangles")

        # Node coordinate lookup
        all_node_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        all_coords = np.array(all_coords).reshape(-1, 3)
        node_coord = {int(t): all_coords[i]
                      for i, t in enumerate(all_node_tags)}

        # Build face map
        step_surfaces      = gmsh.model.getEntities(dim=2)
        face_map           = {}
        face_centroids_map = {}
        face_areas_map     = {}
        stl_tri_index      = 0

        for dim, tag in step_surfaces:
            elem_types, elem_tags, node_tags_list = \
                gmsh.model.mesh.getElements(dim=dim, tag=tag)

            tag_tri_indices = []
            centroid_sum    = np.zeros(3)
            total_area      = 0.0

            for etype, etags, ntags in zip(
                    elem_types, elem_tags, node_tags_list):
                if gmsh.model.mesh.getElementProperties(etype)[3] != 3:
                    continue
                tris = np.array(ntags, dtype=int).reshape(-1, 3)
                n    = len(etags)

                for tri_nodes in tris:
                    v0 = node_coord[tri_nodes[0]]
                    v1 = node_coord[tri_nodes[1]]
                    v2 = node_coord[tri_nodes[2]]
                    centroid = (v0 + v1 + v2) / 3.0
                    area = np.linalg.norm(
                        np.cross(v1 - v0, v2 - v0)) / 2.0
                    centroid_sum += centroid * area
                    total_area   += area

                tag_tri_indices.extend(
                    range(stl_tri_index, stl_tri_index + n))
                stl_tri_index += n

            face_map[str(tag)]           = tag_tri_indices
            face_areas_map[str(tag)]     = float(total_area)
            face_centroids_map[str(tag)] = (
                (centroid_sum / total_area).tolist()
                if total_area > 0 else [0.0, 0.0, 0.0])

        gmsh.write(stl_file)

        mapping = {
            "step_file"    : str(step_path.resolve()),
            "stl_file"     : str(Path(stl_file).resolve()),
            "mesh_params"  : mesh_params,
            "total_triangles": stl_tri_index,
            "step_surface_tag_to_stl_triangle_indices": face_map,
            "step_surface_tag_to_centroid"            : face_centroids_map,
            "step_surface_tag_to_area_mm2"            : face_areas_map,
        }
        with open(json_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"STL  saved : {stl_file}")
        print(f"Map  saved : {json_file}")

    finally:
        gmsh.finalize()

    return stl_file, json_file


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python step2stl.py <file.step> [target_triangles]")
        sys.exit(1)
    step = sys.argv[1]
    ntri = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    generate_stl_with_face_map(step, target_triangles=ntri)