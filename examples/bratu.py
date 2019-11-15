# just to make sure the local simfempy is found first
from os import sys, path
simfempypath = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, simfempypath)
# just to make sure the local simfempy is found first

import matplotlib.pyplot as plt
import numpy as np
import pygmsh
import simfempy


# ------------------------------------- #
def rectangle():
    geom = pygmsh.built_in.Geometry()
    x, y, z = [-1, 1], [-1, 1], [-1, 2]
    h = 0.3
    p = geom.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], lcar=h)
    geom.add_physical(p.surface, label=100)
    for i in range(4): geom.add_physical(p.line_loop.lines[i], label=1000 + i)
    return pygmsh.generate_mesh(geom)


# ------------------------------------- #
def createData(bdrylabels):
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond = data.bdrycond
    bdrycond.set("Dirichlet", [1000, 1002, 1001, 1003], 4*[lambda x, y, z: 0])
    postproc = data.postproc
    postproc.type['bdrymean'] = "bdrymean"
    postproc.color['bdrymean'] = [1000, 1002]
    postproc.type['fluxn'] = "bdrydn"
    postproc.color['fluxn'] = [1000, 1001, 1002, 1003]
    data.rhs = lambda x,y,z: np.ones_like(x)
    data.params.scal_glob["kheat"] = 1.0
    reaction = lambda x,y,z,u: np.exp(u)
    data.params.fct_glob['reaction'] = reaction
    return data


# ------------------------------------- #
mesh = rectangle()
mesh = simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
# simfempy.meshes.plotmesh.meshWithBoundaries(mesh)

data = createData(mesh.bdrylabels.keys())
print("data", data)
data.check(mesh)

heat = simfempy.applications.heat.Heat(problemdata=data, mesh=mesh)
# point_data, cell_data, info = heat.solve()
point_data, cell_data, info = heat.solveNonlinearProblem()
print(f"{info['timer']}")
print(f"{info['iter']}")
print(f"postproc: {info['postproc']}")
simfempy.meshes.plotmesh.meshWithData(mesh, point_data=point_data, cell_data=cell_data)
plt.show()