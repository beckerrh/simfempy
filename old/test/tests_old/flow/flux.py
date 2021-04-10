from os import sys, path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

from simfempy import applications
from simfempy.meshes import geomdefs
import simfempy.tools.comparemethods

#----------------------------------------------------------------#
def test_flux(geomname = "unitsquare", verbose=5):
    postproc = {}
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    if geomname == "unitsquare":
        ncomp = 2
        h = [0.5, 0.25, 0.126, 0.06]
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
        bdrycond.fct[1000] = bdrycond.fct[1001] = bdrycond.fct[1002] = bdrycond.fct[1003] = lambda x,y,z: (0,0)
        postproc['bdrydn'] = "bdrydn:1000,1002,1001,1003"
        rhsv = lambda x,y,z,mu: (y,x)
        geometry = geomdefs.unitsquare.Unitsquare()
    if geomname == "unitcube":
        ncomp = 3
        h = [2, 1, 0.5, 0.25]
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrydn'] = "bdrydn:100,105,101,102,103,104"
        geometry = geomdefs.unitcube.Unitcube()
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, rhs=(rhsv,None), postproc=postproc, ncomp=ncomp)
    print("problemdata",problemdata)
    compares = {}
    for rhsmethod in ['cr','rt']:
        compares[rhsmethod] = applications.stokes.StokesStrong(problemdata=problemdata, rhsmethod=rhsmethod)
    comp = simfempy.tools.comparemethods.CompareMethods(compares, verbose=verbose)
    result = comp.compare(geometry=geometry, h=h)
    print("result", result)


#================================================================#
if __name__ == '__main__':
    test_flux(geomname = "unitsquare", verbose=5)
