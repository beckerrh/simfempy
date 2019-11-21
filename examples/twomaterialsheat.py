# just to make sure the local simfempy is found first
from os import sys, path
simfempypath = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, simfempypath)
# just to make sure the local simfempy is found first


import pygmsh
import simfempy
import matplotlib.pyplot as plt


# ------------------------------------- #
def rectangle():
    from simfempy.meshes import geomdefs
    h = 0.2
    holes = []
    holes.append([[-0.5, -0.25], [-0.5, 0.25], [0.5, 0.25], [0.5, -0.25]])
    holes.append([[-0.5, 0.75], [-0.5, 1.25], [0.5, 1.25], [0.5, 0.75]])
    holes.append([[-0.5, -0.75], [-0.5, -1.25], [0.5, -1.25], [0.5, -0.75]])
    geom = geomdefs.unitsquareholes.Unitsquareholes(rect=(-1.1, 1.1, -2, 2), holes=holes, h=h)
    return pygmsh.generate_mesh(geom)


# ------------------------------------- #
def createData(bdrylabels):
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond = data.bdrycond
    bdrycond.set("Neumann", [1000, 1002])
    bdrycond.set("Dirichlet", [1001, 1003])
    bdrycond.fct[1001] = lambda x, y, z: 200
    bdrycond.fct[1003] = lambda x, y, z: 200
    bdrycond.fct[1000] = lambda x, y, z, nx, ny, nz: 10
    postproc = data.postproc
    postproc.type['bdrymean'] = "bdrymean"
    postproc.color['bdrymean'] = [1000, 1002]
    postproc.type['fluxn'] = "bdrydn"
    postproc.color['fluxn'] = [1001, 1003]
    params = data.params
    # params.fct_glob["kheat"] = lambda l,x,y,z: 0.1 if l==100 else 1000.
    params.set_scal_cells("kheat", [100], 0.1)
    params.set_scal_cells("kheat", [200, 201, 202], 100.0)
    return data


# ------------------------------------- #
mesh = rectangle()
mesh = simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
# simfempy.meshes.plotmesh.meshWithBoundaries(mesh)

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