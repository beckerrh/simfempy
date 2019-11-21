# just to make sure the local simfempy is found first
from os import sys, path
simfempypath = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0,simfempypath)
# just to make sure the local simfempy is found first


import pygmsh
import numpy as np
import simfempy
import matplotlib.pyplot as plt


# ------------------------------------- #
def cube():
    geom = pygmsh.built_in.Geometry()
    x, y, z = [-1, 1], [-1, 1], [-1, 2]
    h = 0.3
    p = geom.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], lcar=h)
    geom.add_physical(p.surface, label=100)
    axis = [0, 0, z[1] - z[0]]
    top, vol, ext = geom.extrude(p.surface, axis, rotation_axis=axis, \
                                 point_on_axis=[0, 0, 0], angle=2 / 12 * np.pi)
    geom.add_physical(top, label=101+len(ext))
    for i in range(len(ext)):
        geom.add_physical(ext[i], label=101+i)
    geom.add_physical(vol, label=10)
    # code = geom.get_code()
    # file = open("toto.geo", 'w')
    # file.write(code)
    return pygmsh.generate_mesh(geom)


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
    return pygmsh.generate_mesh(geom)


def createData(bdrylabels):
    bdrylabels = list(bdrylabels)
    labels_lat = bdrylabels[1:-1]
    firstlabel = bdrylabels[0]
    lastlabel = bdrylabels[-1]
    labels_td = [firstlabel,lastlabel]
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    bdrycond.set("Neumann", labels_lat)
    bdrycond.set("Dirichlet", labels_td)
    bdrycond.fct[firstlabel] = lambda x,y,z: 200
    bdrycond.fct[lastlabel] = lambda x,y,z: 100
    postproc = data.postproc
    postproc.type['bdrymean'] = "bdrymean"
    postproc.color['bdrymean'] = labels_lat
    postproc.type['fluxn'] = "bdrydn"
    postproc.color['fluxn'] = labels_td
    data.params.scal_glob["kheat"] = 0.0001
    return data

# ------------------------------------- #
mesh = pygmshexample()
#mesh = cube()
mesh = simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
#simfempy.meshes.plotmesh.meshWithBoundaries(mesh)

data = createData(mesh.bdrylabels.keys())
print("data", data)
data.check(mesh)

heat = simfempy.applications.heat.Heat(problemdata=data, mesh=mesh)
point_data, cell_data, info = heat.solveLinearProblem()
print(f"{info['timer']}")
print(f"{info['iter']}")
print(f"postproc: {info['postproc']}")
simfempy.meshes.plotmesh.meshWithData(mesh, point_data=point_data, cell_data=cell_data)
plt.show()
