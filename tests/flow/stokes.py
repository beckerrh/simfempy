from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

from simfempy.meshes import geomdefs
from simfempy.applications.stokes import Stokes

#----------------------------------------------------------------#
def test_analytic(exactsolution="Sinus", geomname = "unitsquare", verbose=5):
    import simfempy.tools.comparemethods
    postproc = {}
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    if geomname == "unitsquare":
        h = [2, 1, 0.5, 0.25, 0.12, 0.06]
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
        geometry = geomdefs.unitsquare.Unitsquare()
    if geomname == "unitcube":
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
    stokes = Stokes(geometry=geometry, showmesh=False)
    problemdata = stokes.generatePoblemData(exactsolution=exactsolution, bdrycond=bdrycond, postproc=postproc, random=False)
    print("problemdata", problemdata)
    methods = {}
    for rhsmethod in ['cr','rt']:
        methods[rhsmethod] = Stokes(problemdata=problemdata, rhsmethod=rhsmethod)
    comp = simfempy.tools.comparemethods.CompareMethods(methods, verbose=verbose)
    result = comp.compare(geometry=geometry, h=h)
    res = {}
    res['L2-V-cr'] = result[3]['error']['L2-V']['cr']
    res['L2-P-rt'] = result[3]['error']['L2-P']['rt']
    return res


#================================================================#
if __name__ == '__main__':
    # result = test_analytic(exactsolution="Quadratic", geomname = "unitcube", verbose=2)
    result = test_analytic(exactsolution="Quadratic", verbose=2)
    # result = test_analytic(exactsolution="Linear", verbose=2)
    print("result", result)
