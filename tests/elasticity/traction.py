assert __name__ == '__main__'
from os import sys, path
import numpy as np
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

from simfempy.applications.elasticity import Elasticity
from simfempy.meshes import geomdefs
import simfempy.tools.timer
import matplotlib.pyplot as plt

#================================================================#
def mesh_traction(hmean, geomname="unitcube", x=[-1,1], y=[-1,1], z=[-1,1]):
    postproc = {}
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    if geomname == "unitsquare":
        ncomp = 2
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Neumann"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        bdrycond.fct[1001] = lambda x, y, z, nx, ny, nz: np.array([10,0])
        geometry = geomdefs.unitsquare.Unitsquare(x, y)
    elif geomname == "unitcube":
        ncomp = 3
        bdrycond.type[100] = "Neumann"
        bdrycond.type[101] = "Neumann"
        bdrycond.type[102] = "Neumann"
        bdrycond.fct[102] = lambda x, y, z, nx, ny, nz: np.array([10,0,0])
        bdrycond.type[103] = "Neumann"
        bdrycond.type[104] = "Dirichlet"
        bdrycond.type[105] = "Neumann"
        geometry = geomdefs.unitcube.Unitcube(x, y, z)
    else:
        raise ValueError("unknown geomname={}".format(geomname))
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry, hmean=hmean)
    bdrycond.check(mesh.bdrylabels.keys())
    # mesh.plotWithBoundaries()
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc, ncomp=ncomp)
    elasticity = Elasticity(problemdata=problemdata)
    elasticity.setMesh(mesh)
    A = elasticity.matrix()
    b, u = elasticity.computeRhs()
    return A, b, u, elasticity

#================================================================#
def test_solvers():
    hmeans = [0.3, 0.15, 0.12, 0.09, 0.07, 0.05, 0.03]
    times = {}
    ns = np.empty(len(hmeans))
    for i,hmean in enumerate(hmeans):
        # A, b, u, elasticity = mesh_traction(hmean, geomname="unitsquare")
        A, b, u, elasticity = mesh_traction(hmean)
        n, ncomp = elasticity.mesh.ncells, elasticity.ncomp
        solvers = elasticity.linearsolvers
        # solvers.remove('pyamg')
        # solvers = ['pyamg']
        timer = simfempy.tools.timer.Timer(name="elasticity n={}".format(n))
        for solver in solvers:
            if solver=='umf' and n > 140000: continue
            if not solver in times.keys(): times[solver] = []
            u, niter = elasticity.linearSolver(A, b, u, solver=solver)
            timer.add(solver)
            ns[i] = n
            times[solver].append(timer.data[solver])
        point_data={}
        for icomp in range(ncomp):
            point_data["u{:1d}".format(icomp)] = u[icomp::ncomp]
        # elasticity.mesh.plotWithData(point_data=point_data)
        plt.show()
    for solver,data in times.items():
        plt.plot(ns[:len(data)], data, '-x', label=solver)
    plt.legend()
    plt.show()


#================================================================#
def test_plot():
    hmean = 0.1
    A, b, u, elasticity = mesh_traction(hmean, x=[0,1], y=[0,1], z=[-1,1])
    n, ncomp = elasticity.mesh.ncells, elasticity.ncomp
    u, niter = elasticity.linearSolver(A, b, u, solver="pyamg")
    ncomp = elasticity.ncomp
    point_data={}
    for icomp in range(ncomp):
        point_data["u{:1d}".format(icomp)] = u[icomp::ncomp]
    assert ncomp==3
    elasticity.mesh.plotWithData(point_data=point_data, translate_point_data=True)

#================================================================#

test_solvers()
# test_plot()