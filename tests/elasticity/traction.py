assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import simfempy.applications
from simfempy.meshes import geomdefs

#================================================================#
def mesh_traction(hmean, geomname="unitcube"):
    postproc = {}
    bdrycond =  simfempy.applications.boundaryconditions.BoundaryConditions()
    if geomname == "unitsquare":
        ncomp = 2
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Neumann"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        bdrycond.fct[1001] = lambda x, y, z, nx, ny, nz, lam, mu: np.array([10,0])
        geometry = geomdefs.unitsquare.Unitsquare()
    elif geomname == "unitcube":
        ncomp = 3
        bdrycond.type[11] = "Neumann"
        bdrycond.type[22] = "Neumann"
        bdrycond.type[33] = "Neumann"
        bdrycond.fct[33] = lambda x, y, z, nx, ny, nz, lam, mu: np.array([10,0,0])
        bdrycond.type[44] = "Neumann"
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Neumann"
        geometry = geomdefs.unitcube.Unitcube()
    else:
        raise ValueError("unknown geomname={}".format(geomname))
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry, hmean=hmean)
    # plotmesh.meshWithBoundaries(mesh)
    elasticity = simfempy.applications.elasticity.Elasticity(bdrycond=bdrycond, postproc=postproc, ncomp=ncomp)
    elasticity.setMesh(mesh)
    b = elasticity.computeRhs()
    A = elasticity.matrix()
    A, b, u = elasticity.boundary(A, b)
    return A, b, u, elasticity

#================================================================#
import simfempy.tools.timer
import matplotlib.pyplot as plt
hmeans = [0.3, 0.15, 0.12, 0.09, 0.07, 0.05, 0.03]
times = {}
ns = np.empty(len(hmeans))
for i,hmean in enumerate(hmeans):
    A, b, u, elasticity = mesh_traction(hmean, geomname="unitsquare")
    n, ncomp = elasticity.mesh.ncells, elasticity.ncomp
    solvers = elasticity.linearsolvers
    # solvers.remove('pyamg')
    # solvers = ['pyamg']
    timer = simfempy.tools.timer.Timer(name="elasticity n={}".format(n))
    for solver in solvers:
        if solver=='umf' and n > 140000: continue
        if not solver in times.keys(): times[solver] = []
        u = elasticity.linearSolver(A, b, u, solver=solver)
        timer.add(solver)
        ns[i] = n
        times[solver].append(timer.data[solver])
    point_data={}
    for icomp in range(ncomp):
        point_data["u{:1d}".format(icomp)] = u[icomp::ncomp]
    elasticity.mesh.plotWithData(point_data=point_data)
    plt.show()
for solver,data in times.items():
    plt.plot(ns[:len(data)], data, '-x', label=solver)
plt.legend()
plt.show()
