import numpy as np
import pygmsh

with pygmsh.geo.Geometry() as geom:
    p = geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, mesh_size=0.1)
    geom.add_physical(p.surface, label="100")
    for i in range(4): geom.add_physical(p.lines[i], label=f"{1000 + i}")
    mesh = geom.generate_mesh()
    print("mesh.cell_sets", mesh.cell_sets)
    print("mesh.cell_sets.keys()", mesh.cell_sets.keys())
    # print("mesh.cell_data_dict", mesh.cell_data_dict)
    # print(mesh.cell_data_dict["gmsh:physical"])
    # mesh.write('heat.vtu')


