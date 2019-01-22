assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications

#----------------------------------------------------------------#
def test_flux():
    import fempy.tools.comparerrors
    geomname = "unitsquare"
    geomname = "unitcube"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Dirichlet"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.fct[11] = lambda x,y,z: 0
    bdrycond.fct[44] = bdrycond.fct[33] = bdrycond.fct[22] = bdrycond.fct[11]
    postproc = {}
    postproc['flux'] = "bdrydn:11,22,33,44"
    if geomname == "unitcube":
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Dirichlet"
        bdrycond.fct[66] = bdrycond.fct[55] = bdrycond.fct[44]
        postproc['flux'] += ",55,66"
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = fempy.applications.heat.Heat(rhs=lambda x,y,z:1, bdrycond=bdrycond, kheat=lambda id:1, postproc=postproc, fem=fem, method=meth)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [2, 1, 0.5, 0.25, 0.125]
    result = comp.compare(geomname=geomname, h=h)

#----------------------------------------------------------------#
def test_analytic():
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'
    geomname = "unitsquare"
    # geomname = "unitcube"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    postproc = {}
    if geomname == "unitsquare":
        bdrycond.type[11] = "Neumann"
        bdrycond.type[33] = "Neumann"
        bdrycond.type[22] = "Dirichlet"
        bdrycond.type[44] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:11,33"
        postproc['bdrydn'] = "bdrydn:22,44"
    if geomname == "unitcube":
        problem += "3d"
        bdrycond.type[11] = "Neumann"
        bdrycond.type[33] = "Dirichlet"
        bdrycond.type[22] = "Dirichlet"
        bdrycond.type[44] = "Dirichlet"
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Neumann"
        postproc['bdrymean'] = "bdrymean:11,66"
        postproc['bdrydn'] = "bdrydn:22,33,44,55"

    methods = {}
    for fem in ['p1', 'cr1']:
        methods[fem] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem, method='new', random=False)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=True)
    h = [0.5, 0.25, 0.125, 0.06, 0.03]
    h = [2.0, 1.0, 0.5]
    result = comp.compare(geomname=geomname, h=h)

# ----------------------------------------------------------------#
def test_solvers():
    import time
    problem = 'Analytic_Sinus3d'
    bdrycond = fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Neumann"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"
    fem = 'p1'
    heat  = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, fem=fem)
    mesh = fempy.meshes.simplexmesh.SimplexMesh(geomname='unitcube', hmean=0.05)
    heat.setMesh(mesh)
    print("heat.linearsolvers=", heat.linearsolvers)
    b = heat.computeRhs()
    A = heat.matrix()
    A, b, u = heat.boundary(A, b)
    for solver in heat.linearsolvers:
        t0 = time.time()
        u = heat.linearSolver(A, b, solver=solver)
        t1 = time.time()
        print("{:4d} {:12s} {:10.2e}".format(mesh.ncells, solver, t1-t0))

#================================================================#

test_analytic()
#test_solvers()
# test_flux()
