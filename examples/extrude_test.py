# just to make sure the local simfempy is found first
from os import sys, path
simfempypath = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0,simfempypath)
# just to make sure the local simfempy is found first


import pygmsh
import numpy as np
import simfempy


# ------------------------------------- #
def rectangle():
    geom = pygmsh.built_in.Geometry()
    x, y = [-1, 1], [-1, 1]
    h = 0.8
    p = geom.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=0, lcar=h)
    geom.add_physical(p.surface, label=100)
    for i in range(len(p.lines)):
        geom.add_physical(p.lines[i], label=1000 + i)
    return pygmsh.generate_mesh(geom)

# ------------------------------------- #
def cube():
    geom = pygmsh.built_in.Geometry()
    x, y, z = [-1, 1], [-1, 1], [-1, 1]
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


def createData():
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    # bdrycond.type[1000] = "Neumann"
    # bdrycond.type[1001] = "Dirichlet"
    # bdrycond.type[1002] = "Neumann"
    # bdrycond.type[1003] = "Dirichlet"
    # bdrycond.fct[1000] = lambda x,y,z, nx, ny, nz: 0
    # bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz: 100
    # bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    bdrycond.set("Neumann", [101, 102, 103, 104])
    bdrycond.set("Dirichlet", [100, 105])
    bdrycond.fct[100] = lambda x,y,z: 200
    bdrycond.fct[105] = lambda x,y,z: 100
    postproc = data.postproc
    postproc.type['bdrymean'] = "bdrymean"
    postproc.color['bdrymean'] = range(101,105)
    postproc.type['fluxn'] = "bdrydn"
    postproc.color['fluxn'] = [100, 105]
    def kheat(label):
        if label==10: return 0.0001
        return 1000.0
    data.datafct["kheat"] = kheat
    return data

# ------------------------------------- #
#mesh = pygmshexample()
mesh = cube()
#mesh = rectangle()
mesh = simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
#simfempy.meshes.plotmesh.meshWithBoundaries(mesh)

data = createData()
print("data", data)
data.check(mesh)

heat = simfempy.applications.heat.Heat(problemdata=data, fem='p1', plotk=True)
heat.setMesh(mesh)
point_data, cell_data, info = heat.solve()
print(f"time: {info['timer']}")
print(f"postproc: {info['postproc']}")
simfempy.meshes.plotmesh.meshWithData(mesh, point_data=point_data, cell_data=cell_data, title="toto")
