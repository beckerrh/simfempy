import numpy as np
import pygmsh

def mesh2d(mesh_size=0.1):
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, mesh_size=mesh_size)
        geom.add_physical(p.surface, label="S:100")
        for i in range(4): geom.add_physical(p.lines[i], label=f"L:{1000 + i}")
        return geom.generate_mesh()

#=================================================================#
if __name__ == '__main__':
    mesh = mesh2d()
    print("cell_sets.keys()", mesh.cell_sets.keys())
    mesh.write('testmesh2d.vtu')
