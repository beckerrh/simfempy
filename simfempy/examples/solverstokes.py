assert __name__ == '__main__'
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

from simfempy.applications.stokes import Stokes
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.tools.comparemethods import CompareMethods
from simfempy.examples import incompflow

#----------------------------------------------------------------#
def test(testcase, **kwargs):
    mu = kwargs.pop("mu", 1)
    testcasefct = eval(f"incompflow.{testcase}")
    mesh, data = testcasefct(mu=mu)
    def createMesh(h): return SimplexMesh(testcasefct(h=h)[0])
    applicationargs = {'problemdata': data}
    paramsdict = {'mu@scal_glob': [1, 1e-3], 'linearsolver':['pyamg_gmres@full@100@0', 'scipy_gmres@full@100@0', 'pyamg_fgmres@full@100@0', 'scipy_lgmres@full@20@0', 'scipy_gcrotmk@full@20@0']}
    niter = kwargs.pop('niter', 3)
    comp =  CompareMethods(niter=niter, createMesh=createMesh, paramsdict=paramsdict, application=Stokes, applicationargs=applicationargs, **kwargs)
    return comp.compare()



#================================================================#
if __name__ == '__main__':
    test(testcase='poiseuille2d', niter=6)
    # test(niter=4, exactsolution=[["x**2-y+z**2","-2*x*y*z+x**2","x**2-y**2+z"],"x*y+x*z"])
