from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import simfempy
from simfempy.meshes import geomdefs
from simfempy.applications.heat import Heat

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
    problemdata =  simfempy.applications.problemdata.ProblemData(rhs=lambda x, y, z:1, bdrycond=bdrycond, postproc=postproc)
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = Heat(problemdata=problemdata, kheat=lambda id:1, fem=fem, method=meth)
    comp = simfempy.tools.comparerrors.CompareErrors(methods, verbose=2)
    h = [2, 1, 0.5, 0.25, 0.125]
    result = comp.compare(geometry=geometry, h=h)

#----------------------------------------------------------------#
def test_analytic(exactsolution="Linear", geomname = "unitsquare", verbose=2):
    import simfempy.tools.comparerrors
    geometry, bdrycond, postproc = _getGeometry(geomname)
    if geomname == "unitsquare":
        h = [0.5, 0.25, 0.125, 0.06, 0.03]
    elif geomname == "unitcube":
        h = [2.0, 1.0, 0.5, 0.25, 0.125]
    if exactsolution == "Linear":  h = [2, 1, 0.5, 0.25]
    heat = Heat(geometry=geometry, showmesh=False)
    problemdata = heat.generatePoblemData(exactsolution=exactsolution, bdrycond=bdrycond, postproc=postproc)
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = Heat(problemdata=problemdata, fem=fem, method=meth)
    comp = simfempy.tools.comparerrors.CompareErrors(methods, verbose=verbose)
    result = comp.compare(geometry=geometry, h=h)
    return result[3]['error']['L2']

# ----------------------------------------------------------------#
def test_solvers(geomname='unitcube', fem = 'p1', method='new'):
    exactsolution = 'Sinus'
    import simfempy
    geometry, bdrycond, postproc = _getGeometry(geomname)
    if geomname == "unitsquare":
        hmean = 0.06
    elif geomname == "unitcube":
        hmean = 0.08
    heat = Heat(geometry=geometry, showmesh=False)
    problemdata = heat.generatePoblemData(exactsolution=exactsolution, bdrycond=bdrycond, postproc=postproc)
    heat  = Heat(problemdata=problemdata, fem=fem, method=method)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry, hmean=hmean)
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
    # test_analytic(exactsolution = 'Linear', geomname = "unitsquare")
    # test_analytic(exactsolution = 'Linear', geomname = "unitcube")
    test_analytic(exactsolution = 'Quadratic', geomname = "unitsquare")
    # test_analytic(exactsolution = 'Sinus', geomname = "unitcube")
    # test_solvers(geomname='unitsquare')
    # test_solvers()
    # test_flux()
