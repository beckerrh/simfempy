from os import sys, path
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)
import simfempy.meshes

geom = pygmsh.built_in.Geometry()
poly = geom.add_polygon([
    [ 0.0,  0.5, 0.0],
    [-0.1,  0.1, 0.0],
    [-0.5,  0.0, 0.0],
    [-0.1, -0.1, 0.0],
    [ 0.0, -0.5, 0.0],
    [ 0.1, -0.1, 0.0],
    [ 0.5,  0.0, 0.0],
    [ 0.1,  0.1, 0.0]
    ],
    lcar=0.1
    )
geom.add_physical_surface(poly.surface, label=11)
axis = [0, 0, 1]
top, vol, ext = geom.extrude(poly.surface, translation_axis=axis, rotation_axis=axis, point_on_axis=[0, 0, 0], angle=2.0 / 6.0 * np.pi)
geom.add_physical_surface(top, label=22)
for i,ex in enumerate(ext):
    geom.add_physical_surface(ex, label=(i+3)*11)
geom.add_physical_volume(vol, label=111)

# data = pygmsh.generate_mesh(geom)
data = simfempy.meshes.gmsh.generate_mesh(geom, msh_filename='toto', bin=False)
mesh = simfempy.meshes.simplexmesh.SimplexMesh(data=data)
simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
plt.show()
