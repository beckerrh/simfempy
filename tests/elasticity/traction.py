assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications
from fempy.meshes import plotmesh

#================================================================#
def mesh_traction(hmean, geomname="unitcube"):
    postproc = {}
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    if geomname == "unitsquare":
        ncomp = 2
        bdrycond.type[11] = "Neumann"
        bdrycond.type[22] = "Neumann"
        bdrycond.fct[22] = lambda x, y, z, nx, ny, nz, lam, mu: np.array([1,0])
        bdrycond.type[33] = "Neumann"
        bdrycond.type[44] = "Dirichlet"
    elif geomname == "unitcube":
        ncomp = 3
        bdrycond.type[11] = "Neumann"
        bdrycond.type[22] = "Neumann"
        bdrycond.type[33] = "Neumann"
        bdrycond.fct[33] = lambda x, y, z, nx, ny, nz, lam, mu: np.array([10,0,0])
        bdrycond.type[44] = "Neumann"
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Neumann"
    else:
        raise ValueError("unknown geomname={}".format(geomname))
    mesh = fempy.meshes.simplexmesh.SimplexMesh(geomname=geomname, hmean=hmean)
    # plotmesh.meshWithBoundaries(mesh)
    elasticity = fempy.applications.elasticity.Elasticity(bdrycond=bdrycond, postproc=postproc, ncomp=ncomp)
    elasticity.setMesh(mesh)
    b = elasticity.computeRhs()
    A = elasticity.matrix()
    A, b, u = elasticity.boundary(A, b)
    return A, b, u, elasticity

#================================================================#
import time
import matplotlib.pyplot as plt
hmeans = [0.3, 0.15, 0.12, 0.09, 0.07, 0.06]
times = {}
ns = np.empty(len(hmeans))
for i,hmean in enumerate(hmeans):
    A, b, u, elasticity = mesh_traction(hmean)
    solvers = elasticity.linearsolvers
    solvers.remove('pyamg')
    for solver in solvers:
        t0 = time.time()
        u = elasticity.linearSolver(A, b, u, solver=solver)
        t1 = time.time()
        n = elasticity.mesh.ncells
        print("n={:4d} {:12s} {:10.2e}".format(n, solver, t1 - t0))
        if not solver in times.keys():
            times[solver]=np.empty(len(hmeans))
        ns[i] = n
        times[solver][i] = t1-t0
#plotmesh.meshWithData(mesh, {"u0": u[::ncomp], "u1":u[1::ncomp]})
for solver,data in times.items():
    plt.plot(ns, data, '-x',label=solver)
plt.legend()
plt.show()
