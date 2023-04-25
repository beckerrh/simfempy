assert __name__ == '__main__'
# in shell
import os, sys
import matplotlib.pyplot as plt
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)
from simfempy.models.stokes import Stokes
from simfempy.models.navierstokes import NavierStokes
from simfempy.tools.comparemethods import CompareMethods
import app_navierstokes

#----------------------------------------------------------------#
def test(testcase, **kwargs):
    mu = kwargs.pop("mu", 1)
    h = kwargs.pop("h", 0.1)
    run = kwargs.pop("run", True)
    plot = kwargs.pop("plot", True)
    modelname = kwargs.pop("model", "NavierStokes")
    linearsolver = kwargs.pop("linearsolver", None)
    app = eval(f"app_navierstokes.applications.{testcase}(h={h}, mu=mu)")
    # kwargs = {'problemdata': app.problemdata, 'mesh':app.createMesh(), 'linearsolver':linearsolver}
    kwargs = {'application': app, 'linearsolver':linearsolver}
    if not run and plot:
        kwargs['clean_data'] = False
    if modelname=="NavierStokes":
        model = NavierStokes(**kwargs)
    else:
        model = Stokes(**kwargs)
    if run:
        result, u = model.static()
        print(f"{result.info=}")
        filename = model.__class__.__name__+'.vtu'
        model.mesh.write(filename, data=model.sol_to_data(u, single_vector=False))
    if plot:
        model.plot()
    plt.show()

#----------------------------------------------------------------#
def compare_ls(testcase, **kwargs):
    mu = kwargs.pop("mu", 1)
    app = eval(f"app_navierstokes.applications.{testcase}(h=0.2, mu=mu)")
    # testcasefct = eval(f"incompflow.{testcase}")
    # mesh, data = testcasefct(mu=mu)
    # def createMesh(h): return SimplexMesh(testcasefct(h=h)[0])
    paramsdict = {'mu@scal_glob': [0.01, 1]}
    paramsdict['alpha@scal_glob'] = [0, 1, 1000]
    # paramsdict['linearsolver'] = [['spsolve',{'method': 'spsolve'}]]
    paramsdict['linearsolver'] = []
    gmres = 'scipy_lgmres'
    # gmres = 'pyamg_fgmres'
    # gmres = 'idr'
    # paramsdict['linearsolver'].append(['BS', {'method': gmres, 'maxiter': 100, 'prec': 'BS', 'alpha':1}])
    paramsdict['linearsolver'].append(['Ch', {'method': gmres, 'maxiter': 200, 'prec': 'Chorin'}])
    # paramsdict['linearsolver'].append(['full', {'method': gmres, 'maxiter': 100, 'prec': 'full'}])
    # paramsdict['linearsolver'].append(['triup', {'method': gmres, 'maxiter': 100, 'prec': 'triup'}])
    # solver_v = {'method':'pyamg', 'pyamgtype': 'aggregation', 'accel': 'none', 'smoother': 'gauss_seidel', 'maxiter': 1, 'disp': 0}
    # linearsolver = {'method': gmres, 'maxiter': 200, 'prec': 'full', 'solver_v':solver_v}
    # solver_p = {'type': 'schur', 'method': gmres, 'maxiter':5}
    # paramsdict['linearsolver'].append(['gmres-schur', linearsolver|{'solver_p':solver_p}])
    # solver_p = {'type': 'scale', 'method': gmres, 'maxiter':20}
    # paramsdict['linearsolver'].append(['gmres-scale', linearsolver|{'solver_p':solver_p}])

    modelargs = {}
    modelargs['singleA'] = True
    modelargs['scale_ls'] = True
    modelargs['mode'] = 'linear'
    # sinon cata
    niter = kwargs.pop('niter', 3)
    kwargs['only_iter_and_timer'] = True
    comp =  CompareMethods(niter=niter, application=app, paramsdict=paramsdict, model=Stokes, modelargs=modelargs, **kwargs)
    return comp.compare()



#================================================================#
if __name__ == '__main__':
    # test(testcase='poiseuille2d', niter=6)
    # test(testcase='poiseuille3d', niter=5)
    # test(niter=4, exactsolution=[["x**2-y+z**2","-2*x*y*z+x**2","x**2-y**2+z"],"x*y+x*z"])

    compare_ls(testcase='BackwardFacingStep2d', niter=3)
    # compare_ls(testcase='BackwardFacingStep3d', niter=4)

    # test(testcase='BackwardFacingStep2d', h=0.1, mu=0.005)
    # test(plot=False, testcase='BackwardFacingStep3d', h=0.5, mu=0.005)

    # test(run=False, model="Stokes", testcase='BackwardFacingStep3d', h=0.1, mu=0.1)

    # linearsolver = {'method': 'pyamg_fgmres', 'maxiter': 300, 'prec': 'Chorin', 'disp': 0, 'rtol': 1e-10}
    # test_navierstokes(testcase='BackwardFacingStep2d', h=0.1, mu=0.01, linearsolver=None)
