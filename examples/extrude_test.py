import pygmsh
import numpy as np
import simfempy


# ------------------------------------- #
def cube():
    geom = pygmsh.built_in.Geometry()
    x, y, z = [-1, 1], [-1, 1], [-1, 1]
    h = 0.8
    p = geom.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], lcar=h)
    geom.add_physical(p.surface, label=100)
    axis = [0, 0, z[1] - z[0]]
    top, vol, ext = geom.extrude(p.surface, axis)
    geom.add_physical(top, label=101+len(ext))
    for i in range(len(ext)):
        geom.add_physical(ext[i], label=101+i)
    geom.add_physical(vol, label=10)
    mesh = pygmsh.generate_mesh(geom)
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)


# ------------------------------------- #
def pygmshexample():
    geom = pygmsh.built_in.Geometry()
    # Draw a cross.
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
        lcar=0.2
    )
    geom.add_physical(poly.surface, label=100)

    axis = [0, 0, 1]

    top, vol, ext = geom.extrude(
        poly,
        translation_axis=axis,
        rotation_axis=axis,
        point_on_axis=[0, 0, 0],
        angle=2.0 / 6.0 * np.pi
    )
    geom.add_physical(top, label=101+len(ext))
    for i in range(len(ext)):
        geom.add_physical(ext[i], label=101+i)
    geom.add_physical(vol, label=10)
    mesh = pygmsh.generate_mesh(geom)
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)



# ------------------------------------- #
mesh = pygmshexample()
# mesh = cube()
simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
