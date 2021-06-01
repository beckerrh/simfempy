import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0,simfempypath)

import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.elasticity import Elasticity
import simfempy.applications.problemdata
from simfempy.test.test_analytic import test_analytic

#----------------------------------------------------------------#
def test(dim, **kwargs):
    exactsolution = kwargs.pop('exactsolution', 'Linear')
    paramargs = {'fem': kwargs.pop('fem', ['p1','cr1'])}
    paramargs['dirichletmethod'] = kwargs.pop('dirichletmethod', ['strong','new'])
    data = simfempy.applications.problemdata.ProblemData()
    if dim==2:
        data.ncomp=2
        createMesh = testmeshes.unitsquare
        colordir = [1001,1003]
        colorneu = [1000,1002]
        colordir = [1002,1001,1003]
        colorneu = [1000]
    else:
        data.ncomp=3
        createMesh = testmeshes.unitcube
        colordir = [101,102,103,104]
        colorneu = [100,105]
        # colordir = [100, 105, 101, 102, 103, 104]
        # colorneu = []
    data.bdrycond.set("Dirichlet", colordir)
    data.bdrycond.set("Neumann", colorneu)
    data.postproc.set(name='bdrymean', type='bdry_mean', colors=colorneu)
    data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=colordir)
    linearsolver = kwargs.pop('linearsolver', 'pyamg')
    applicationargs= {'problemdata': data, 'exactsolution': exactsolution, 'linearsolver': linearsolver}
    return test_analytic(application=Elasticity, createMesh=createMesh, paramargs=paramargs, applicationargs=applicationargs, **kwargs)



#================================================================#
if __name__ == '__main__':
    # test(dim=2, exactsolution=["1", "y"], fem=['p1','cr1'], niter=4)
    test(dim=2, exactsolution='Linear', fem=['p1','cr1'], dirichletmethod=['strong','new'], niter=3, linearsolver='pyamg')
    # test(dim=2, exactsolution='Quadratic', fem=['p1','cr1'], niter=6)
