assert __name__ == '__main__'
# in shell
import os, sys
import matplotlib.pyplot as plt
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)
from simfempy.models.stokes import Stokes
from simfempy.models.navierstokes import NavierStokes
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.tools.comparemethods import CompareMethods
from simfempy.examples import incompflow

#----------------------------------------------------------------#
def test_navierstokes(testcase, **kwargs):
    from simfempy.models import app_navierstokes
    mu = kwargs.pop("mu", 1)
    h = kwargs.pop("h", 0.1)
    linearsolver = kwargs.pop("linearsolver", None)
    app = eval(f"app_navierstokes.applications.{testcase}(h={h}, mu=mu)")

    model = NavierStokes(problemdata=app.problemdata, mesh=app.createMesh(), linearsolver=linearsolver, scale_ls=True)
    result, u = model.static()
    model.plot(result.data)
    print(f"{result.info=}")
    plt.show()

#----------------------------------------------------------------#
def test_stokes(testcase, **kwargs):
    from simfempy.models import app_navierstokes
    mu = kwargs.pop("mu", 1)
    app = eval(f"app_navierstokes.applications.{testcase}(h=0.2, mu=mu)")
    # testcasefct = eval(f"incompflow.{testcase}")
    # mesh, data = testcasefct(mu=mu)
    # def createMesh(h): return SimplexMesh(testcasefct(h=h)[0])
    modelargs = {'problemdata': app.problemdata}
    # modelargs['scalels'] = True
    paramsdict = {'mu@scal_glob': [0.01, 1]}
    paramsdict['alpha@scal_glob'] = [0, 1, 1000]
    # paramsdict['linearsolver'] = [['spsolve',{'method': 'spsolve'}]]
    paramsdict['linearsolver'] = []
    gmres = 'scipy_lgmres'
    # gmres = 'idr'
    paramsdict['linearsolver'].append(['BS', {'method': gmres, 'maxiter': 100, 'prec': 'BS', 'alpha':1}])
    paramsdict['linearsolver'].append(['Ch', {'method': gmres, 'maxiter': 100, 'prec': 'Chorin'}])
    # paramsdict['linearsolver'].append(['full', {'method': gmres, 'maxiter': 100, 'prec': 'full'}])
    # paramsdict['linearsolver'].append(['triup', {'method': gmres, 'maxiter': 100, 'prec': 'triup'}])
    # solver_v = {'method':'pyamg', 'pyamgtype': 'aggregation', 'accel': 'none', 'smoother': 'gauss_seidel', 'maxiter': 1, 'disp': 0}
    # linearsolver = {'method': gmres, 'maxiter': 200, 'prec': 'full', 'solver_v':solver_v}
    # solver_p = {'type': 'schur', 'method': gmres, 'maxiter':5}
    # paramsdict['linearsolver'].append(['gmres-schur', linearsolver|{'solver_p':solver_p}])
    # solver_p = {'type': 'scale', 'method': gmres, 'maxiter':20}
    # paramsdict['linearsolver'].append(['gmres-scale', linearsolver|{'solver_p':solver_p}])

    modelargs['singleA'] = True
    modelargs['scale_ls'] = True
    # sinon cata
    niter = kwargs.pop('niter', 3)
    kwargs['only_iter_and_timer'] = True
    comp =  CompareMethods(niter=niter, createMesh=app.createMesh, paramsdict=paramsdict, model=Stokes, modelargs=modelargs, **kwargs)
    return comp.compare()



#================================================================#
if __name__ == '__main__':
    # test(testcase='poiseuille2d', niter=6)
    # test(testcase='poiseuille3d', niter=5)
    # test(niter=4, exactsolution=[["x**2-y+z**2","-2*x*y*z+x**2","x**2-y**2+z"],"x*y+x*z"])

    test_stokes(testcase='BackwardFacingStep2d', niter=4)
    # test(testcase='BackwardFacingStep3d', niter=4)

    # linearsolver = {'method': 'pyamg_fgmres', 'maxiter': 300, 'prec': 'Chorin', 'disp': 0, 'rtol': 1e-10}
    # test_navierstokes(testcase='BackwardFacingStep2d', h=0.1, mu=0.01, linearsolver=None)
