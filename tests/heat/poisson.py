from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy

#----------------------------------------------------------------#
def test_flux(geomname = "unitcube"):
    import fempy.tools.comparerrors
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    postproc = {}
    if geomname=='unitsquare':
        problem += '_2d'
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
    elif geomname == "unitcube":
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
    for color in bdrycond.colors():
        bdrycond.fct[color] = lambda x, y, z: 0
    postproc['flux'] = "bdrydn:"+','.join([str(c) for c in bdrycond.colors()])
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = fempy.applications.heat.Heat(rhs=lambda x,y,z:1, bdrycond=bdrycond, kheat=lambda id:1, postproc=postproc, fem=fem, method=meth)
    comp = fempy.tools.comparerrors.CompareErrors(methods, verbose=2)
    h = [2, 1, 0.5, 0.25, 0.125]
    result = comp.compare(geomname=geomname, h=h)

#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Linear", geomname = "unitsquare", verbose=2):
    import fempy.tools.comparerrors
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    postproc = {}
    if geomname == "unitsquare":
        problem += '_2d'
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
    elif geomname == "unitcube":
        problem += "_3d"
        bdrycond.type[100] = "Neumann"
        bdrycond.type[105] = "Neumann"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:100,105"
        postproc['bdrydn'] = "bdrydn:101,102,103,104"
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem, method=meth, random=False)
    comp = fempy.tools.comparerrors.CompareErrors(methods, verbose=verbose)
    h = [0.5, 0.25, 0.125, 0.06, 0.03]
    h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06]
    result = comp.compare(geomname=geomname, h=h)
    return result[3]['error']['L2']

# ----------------------------------------------------------------#
def test_solvers(geomname='unitcube', fem = 'p1'):
    problem = 'Analytic_Sinus'
    import fempy
    bdrycond = fempy.applications.boundaryconditions.BoundaryConditions()
    if geomname=='unitsquare':
        problem += '_2d'
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        hmean = 0.005
    elif geomname=='unitcube':
        problem += '_3d'
        bdrycond.type[100] = "Neumann"
        bdrycond.type[105] = "Neumann"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        hmean = 0.05
    heat  = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, fem=fem)
    mesh = fempy.meshes.simplexmesh.SimplexMesh(geomname=geomname, hmean=hmean)
    # mesh.plotWithBoundaries()
    heat.setMesh(mesh)
    print("heat.linearsolvers=", heat.linearsolvers)
    b = heat.computeRhs()
    A = heat.matrix()
    A, b, u = heat.boundary(A, b)
    import fempy.tools.timer
    timer = fempy.tools.timer.Timer(name=fem+'_'+geomname+'_'+str(mesh.ncells))
    for solver in heat.linearsolvers:
        u = heat.linearSolver(A, b, solver=solver)
        timer.add(solver)

#================================================================#
if __name__ == '__main__':
    # test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare")
    # test_analytic(problem = 'Analytic_Linear', geomname = "unitcube")
    # test_analytic(problem = 'Analytic_Quadratic', geomname = "unitsquare")
    # test_analytic(problem = 'Analytic_Sinus', geomname = "unitcube")
    test_solvers(geomname='unitsquare')
    test_solvers()
    # test_flux()
