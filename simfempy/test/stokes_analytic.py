import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.stokes import Stokes
import simfempy.applications.problemdata
from test_analytic import test_analytic

#----------------------------------------------------------------#
def test(dim, **kwargs):
    exactsolution = kwargs.pop('exactsolution', 'Linear')
    data = simfempy.applications.problemdata.ProblemData()
    paramargs = {}
    if dim==2:
        data.ncomp=2
        createMesh = testmeshes.unitsquare
        data.bdrycond.type[1000] = "Neumann"
        data.bdrycond.type[1001] = "Dirichlet"
        data.bdrycond.type[1002] = "Neumann"
        data.bdrycond.type[1003] = "Dirichlet"
        data.postproc.set(name='bdrymean', type='bdry_mean', colors=[1000,1002])
        data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=[1001,1003])
    else:
        data.ncomp=3
        createMesh = testmeshes.unitcube
        raise NotImplementedError("no")
    linearsolver = kwargs.pop('linearsolver', 'pyamg')
    applicationargs= {'problemdata': data, 'exactsolution': exactsolution, 'linearsolver': linearsolver}
    return test_analytic(application=Stokes, createMesh=createMesh, paramargs=paramargs, applicationargs=applicationargs, **kwargs)



#================================================================#
if __name__ == '__main__':
    test(dim=2, exactsolution="Linear")
