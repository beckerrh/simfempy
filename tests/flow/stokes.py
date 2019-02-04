from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

from simfempy.meshes import geomdefs
from simfempy.applications.stokes import Stokes

#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Sinus", geomname = "unitsquare", verbose=5):
    import simfempy.tools.comparerrors
    postproc = {}
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    if geomname == "unitsquare":
        problem += "_2d"
        ncomp = 2
        h = [2, 1, 0.5, 0.25]
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
        geometry = geomdefs.unitsquare.Unitsquare()
    if geomname == "unitcube":
        problem += "_3d"
        ncomp = 3
        h = [2, 1, 0.5, 0.25]
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:100,105"
        postproc['bdrydn'] = "bdrydn:101,102,103,104"
        geometry = geomdefs.unitcube.Unitcube()
    compares = {}
    for rhsmethod in ['cr','rt']:
        compares[rhsmethod] = Stokes(problem=problem, bdrycond=bdrycond, postproc=postproc,rhsmethod=rhsmethod, ncomp=ncomp, random=False)
    comp = simfempy.tools.comparerrors.CompareErrors(compares, verbose=verbose)
    result = comp.compare(geometry=geometry, h=h)
    res = {}
    res['L2-V-cr'] = result[3]['error']['L2-V']['cr']
    res['L2-P-rt'] = result[3]['error']['L2-P']['rt']
    return res


#================================================================#
if __name__ == '__main__':
    # result = test_analytic(problem="Analytic_Quadratic", geomname = "unitcube", verbose=2)
    # result = test_analytic(problem="Analytic_Quadratic", verbose=2)
    result = test_analytic(problem="Analytic_Linear", verbose=2)
    print("result", result)
