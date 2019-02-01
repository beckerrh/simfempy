from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import simfempy
from simfempy.meshes import geomdefs

#----------------------------------------------------------------#
def _getGeometry(geomname = "unitcube"):
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    postproc = {}
    if geomname == "unitsquare":
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
        geometry = geomdefs.unitsquare.Unitsquare()
    elif geomname == "unitcube":
        bdrycond.type[100] = "Neumann"
        bdrycond.type[105] = "Neumann"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:100,105"
        postproc['bdrydn'] = "bdrydn:101,102,103,104"
        geometry = geomdefs.unitcube.Unitcube()
    return geometry, bdrycond, postproc

#----------------------------------------------------------------#
def test_flux(geomname = "unitcube"):
    import simfempy.tools.comparerrors
    geometry, bdrycond, postproc = _getGeometry(geomname)
    for color in bdrycond.colors():
        bdrycond.type[color] = "Dirichlet"
        bdrycond.fct[color] = lambda x, y, z: 0
    postproc = {}
    postproc['flux'] = "bdrydn:"+','.join([str(c) for c in bdrycond.colors()])
    problemdata =  simfempy.applications.problemdata.ProblemData(rhs=lambda x, y, z:1, bdrycond=bdrycond)
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = simfempy.applications.heat.Heat(problemdata=problemdata, kheat=lambda id:1, postproc=postproc, fem=fem, method=meth)
    comp = simfempy.tools.comparerrors.CompareErrors(methods, verbose=2)
    h = [2, 1, 0.5, 0.25, 0.125]
    result = comp.compare(geometry=geometry, h=h)

#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Linear", geomname = "unitsquare", verbose=2):
    import simfempy.tools.comparerrors
    geometry, bdrycond, postproc = _getGeometry(geomname)
    if geomname == "unitsquare":
        h = [0.5, 0.25, 0.125, 0.06, 0.03]
        if problem=="Analytic_Linear": h = h[:-2]
        problem += '_2d'
        geometry = geomdefs.unitsquare.Unitsquare()
    elif geomname == "unitcube":
        h = [2.0, 1.0, 0.5, 0.25, 0.125]
        if problem=="Analytic_Linear": h = h[:-2]
        problem += "_3d"
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = simfempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem, method=meth, random=False)
    comp = simfempy.tools.comparerrors.CompareErrors(methods, verbose=verbose)
    result = comp.compare(geometry=geometry, h=h)
    return result[3]['error']['L2']

# ----------------------------------------------------------------#
def test_solvers(geomname='unitcube', fem = 'p1'):
    problem = 'Analytic_Sinus'
    import simfempy
    geometry, bdrycond, postproc = _getGeometry(geomname)
    if geomname == "unitsquare":
        hmean = 0.06
        problem += '_2d'
        geometry = geomdefs.unitsquare.Unitsquare()
    elif geomname == "unitcube":
        hmean = 0.08
        problem += "_3d"
    heat  = simfempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, fem=fem)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry, hmean=hmean)
    # mesh.plotWithBoundaries()
    heat.setMesh(mesh)
    print("heat.linearsolvers=", heat.linearsolvers)
    b = heat.computeRhs()
    A = heat.matrix()
    A, b, u = heat.boundary(A, b)
    import simfempy.tools.timer
    timer = simfempy.tools.timer.Timer(name=fem + '_' + geomname + '_' + str(mesh.ncells))
    for solver in heat.linearsolvers:
        u = heat.linearSolver(A, b, solver=solver)
        timer.add(solver)

#================================================================#
if __name__ == '__main__':
    # test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare")
    # test_analytic(problem = 'Analytic_Linear', geomname = "unitcube")
    test_analytic(problem = 'Analytic_Quadratic', geomname = "unitsquare")
    # test_analytic(problem = 'Analytic_Sinus', geomname = "unitcube")
    # test_solvers(geomname='unitsquare')
    # test_solvers()
    # test_flux()
