assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications


#================================================================#
geomname = "unitsquare"
# geomname = "unitcube"
bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
bdrycond.type[11] = "Neumann"
bdrycond.type[33] = "Neumann"
# bdrycond.fct[11] = lambda x, y, z: np.array([0.5*(1+x),0])
# bdrycond.fct[33] = lambda x, y, z: np.array([0.5*(1+x),0])
# bdrycond.type[22] = "Neumann"
# bdrycond.fct[22] = lambda x, y, z, nx, ny, nz, lam, mu: np.array([1,0])
bdrycond.type[22] = "Dirichlet"
bdrycond.fct[22] = lambda x, y, z: np.array([1,0])
bdrycond.type[44] = "Dirichlet"

postproc = {}
ncomp = 2
if geomname == "unitcube":
    ncomp = 3
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"


mesh = fempy.meshes.simplexmesh.SimplexMesh(geomname=geomname, hmean=0.5)

from fempy.meshes import plotmesh
plotmesh.meshWithBoundaries(mesh)
elasticity = fempy.applications.elasticity.Elasticity(bdrycond=bdrycond, postproc=postproc, ncomp=ncomp)
elasticity.setMesh(mesh)
print("elasticity.linearsolvers=", elasticity.linearsolvers)
b = elasticity.computeRhs()
A = elasticity.matrix()
A, b, u = elasticity.boundary(A, b)
plotmesh.meshWithData(mesh, {"b0": b[::ncomp], "b1":b[1::ncomp]})
u = elasticity.linearSolver(A, b)
plotmesh.meshWithData(mesh, {"u0": u[::ncomp], "u1":u[1::ncomp]})
