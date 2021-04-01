import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.stokes import Stokes
import simfempy.applications.problemdata
from simfempy.test.test_analytic import test_analytic

#----------------------------------------------------------------#
def test(dim, **kwargs):
    exactsolution = kwargs.pop('exactsolution', 'Linear')
    data = simfempy.applications.problemdata.ProblemData()
    paramargs = {}
    if dim==2:
        data.ncomp=2
        createMesh = testmeshes.unitsquare
        colordir = [1001,1003]
        colorneu = [1000,1002]
    else:
        data.ncomp=3
        createMesh = testmeshes.unitcube
        raise NotImplementedError("no")
    data.bdrycond.set("Dirichlet", colordir)
    data.bdrycond.set("Neumann", colorneu)
    data.postproc.set(name='bdrymean', type='bdry_mean', colors=colorneu)
    data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=colordir)
    applicationargs= {'problemdata': data, 'exactsolution': exactsolution}
    return test_analytic(application=Stokes, createMesh=createMesh, paramargs=paramargs, applicationargs=applicationargs, **kwargs)



#================================================================#
if __name__ == '__main__':
    test(dim=2, exactsolution="Linear", niter=1, h1=4)
