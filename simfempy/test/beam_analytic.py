import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0,simfempypath)

import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.beam import Beam
import simfempy.applications.problemdata
from simfempy.test.test_analytic import test_analytic

#----------------------------------------------------------------#
def test(**kwargs):
    data = simfempy.applications.problemdata.ProblemData()
    exactsolution = kwargs.pop('exactsolution', 'Linear')
    paramargs = {}
    data.params.scal_glob['EI'] = kwargs.pop('EI', 1)
    createMesh = testmeshes.unitline
    data.bdrycond.set("Clamped", [10000,10001])
    linearsolver = kwargs.pop('linearsolver', 'umf')
    applicationargs= {'problemdata': data, 'exactsolution': exactsolution, 'linearsolver': linearsolver}
    return test_analytic(application=Beam, createMesh=createMesh, paramargs=paramargs, applicationargs=applicationargs, **kwargs)

#================================================================#
if __name__ == '__main__':
    test(exactsolution = '(x-0.5)**2', niter=1, h1=0.5, linearsolver='umf', plotsolution=True)
