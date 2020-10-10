import numpy as np
import pygmsh

def mesh2d(mesh_size=0.1):
    with pygmsh.geo.Geometry() as geom:
        circle = geom.add_circle(x0=[0.5, 0.5, 0.0], radius=0.2,
                                 mesh_size=mesh_size, num_sections=4,make_surface=False)
        p = geom.add_rectangle(-1, 1, -1, 1, 0, mesh_size=mesh_size, holes=[circle.curve_loop])
        # p = geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, mesh_size=mesh_size)
        geom.add_physical(p.surface, label="S:100")
        print("p.lines", len(p.lines))
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"L:{1000 + i}")
        return geom.generate_mesh()

#=================================================================#
if __name__ == '__main__':
    mesh = mesh2d()
    print("cell_sets.keys()", mesh.cell_sets.keys())
    mesh.write('testmesh2d.vtu')
