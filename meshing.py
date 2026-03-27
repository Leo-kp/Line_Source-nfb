from pathlib import Path
import gmsh
import math

def create_scylindre_mesh(
    filepath: Path,
    radius: float,
    thickness: float,
    mesh_size: float,
    r_well: float = 0.0,      # Starting radius (0 for line, >0 for finite)
    refine_size: float=0.1, 
    center_y: float = 0.0
) -> None:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.add(filepath.stem)

    z = center_y
    r_start = r_well 
    r_end = r_well + radius

    p1 = gmsh.model.occ.addPoint(r_start,      z+thickness/2, 0, mesh_size)
    p2 = gmsh.model.occ.addPoint(r_end,   z+thickness/2, 0, mesh_size)
    p3 = gmsh.model.occ.addPoint(r_end,   z-thickness/2, 0, mesh_size)
    p4 = gmsh.model.occ.addPoint(r_start,      z-thickness/2, 0,  mesh_size)

    l1 = gmsh.model.occ.addLine(p2, p1)
    l2 = gmsh.model.occ.addLine(p3, p2)
    l3 = gmsh.model.occ.addLine(p4, p3)
    l4 = gmsh.model.occ.addLine(p1, p4)
    gmsh.model.occ.synchronize()

    cl   = gmsh.model.occ.addCurveLoop([l4, l3, l2, l1])
    surf = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.synchronize()

    # --- local refinement ---
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [l4])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", refine_size)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r_well + refine_size * 5) 
    gmsh.model.mesh.field.setNumber(2, "DistMax", radius * 0.1) 

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    # -------------------------

    pg_domain = gmsh.model.addPhysicalGroup(2, [surf])
    gmsh.model.setPhysicalName(2, pg_domain, "domain")
    bcs = [("top", l1), ("boundary_R", l2), ("bottom", l3), ("well", l4)]
    for name, line in bcs:
        pg = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, pg, name)

    gmsh.model.mesh.generate(2)
    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()

