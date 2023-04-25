# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simfempy.models.stokes import Stokes
from simfempy.models.navierstokes import NavierStokes
from simfempy.meshes import plotmesh, animdata
from simfempy.tools import timer

# ================================================================c#
def getModel(**kwargs):
    femparams = kwargs.pop('femparams', {})
    application = kwargs.pop('application', {})
    model = kwargs.pop('model', 'NavierStokes')
    bdryplot = kwargs.pop('bdryplot', False)
    # mesh, data, appname = application.createMesh(), application.problemdata, application.__class__.__name__
    if bdryplot:
        plotmesh.meshWithBoundaries(model.mesh)
        plt.show()
        return
    # create model
    if model == "Stokes":
        model = Stokes(application=application, femparams=femparams)
    else:
        # model = NavierStokes(mesh=mesh, problemdata=data, hdivpenalty=10)
        model = NavierStokes(application=application, femparams=femparams)
    return model


# ================================================================c#
def static(**kwargs):
    model = getModel(**kwargs)
    newtonmethod = kwargs.pop('newtonmethod', 'newton')
    newtonmaxiter = kwargs.pop('newtonmaxiter', 20)
    appname  = model.application.__class__.__name__
    mesh = model.mesh
    t = timer.Timer("mesh")
    # result, u = model.static(maxiter=30)
    if 'increase_reynolds' in kwargs:
        # model.linearsolver_def['disp'] = 1
        factor = kwargs['increase_reynolds']
        u = None
        newton_failure = 0
        for i in range(100):
            # try:
                result, u = model.static(u=u, maxiter=newtonmaxiter, method=newtonmethod, rtol=1e-3)
                if not result.newtoninfo.success:
                    newton_failure += 1
                    if newton_failure==3:
                        break
                print(f"{result.data['scalar']}")
                model.sol_to_vtu(u=u, suffix=f"_{i:03d}")
                # model.plot(title=f"{model.problemdata.params.scal_glob['mu']}")
                # plt.show()
                model.problemdata.params.scal_glob['mu'] *= factor
                model.new_params()
            # except:
            #     print(f"min viscosity found {model.problemdata.params.scal_glob['mu']}")
            #     break
    else:
        result, u = model.static(maxiter=2*newtonmaxiter, method=newtonmethod, rtol=1e-6)
        print(f"{result}")
        model.plot()
        model.sol_to_vtu()
        plt.show()
# ================================================================c#
def dynamic(**kwargs):
    model = getModel(**kwargs)
    appname  = model.application.__class__.__name__
    stokes = Stokes(application=model.application, femparams=femparams)
    result, u = stokes.solve()
    T = kwargs.pop('T', 200)
    dt = kwargs.pop('dt', 0.52)
    nframes = kwargs.pop('nframes', int(T/2))
    result = model.dynamic(u, t_span=(0, T), nframes=nframes, dt=dt, theta=0.8, rtolnewton=1e-3, output_vtu=True)
    print(f"{model.timer=}")
    print(f"{model.newmatrix=}")
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"{appname}")
    gs = fig.add_gridspec(2, 3)
    nhalf = (nframes - 1) // 2
    for i in range(3):
        model.plot(fig=fig, gs=gs[i], iter = i*nhalf, title=f't={result.time[i*nhalf]}')
    pp = model.get_postprocs_dynamic()
    ax = fig.add_subplot(gs[1, :])
    for k,v in pp['postproc'].items():
        ax.plot(pp['time'], v, label=k)
    ax.legend()
    ax.grid()
    plt.show()
    # def initfct(ax, u):
    #     ax.set_aspect(aspect=1)
    # anim = animdata.AnimData(mesh, v, plotfct=model.plot_v, initfct=initfct)
    # plt.show()

#================================================================#
if __name__ == '__main__':
    from simfempy.examples.app_navierstokes import applications
    femparams = {'dirichletmethod': 'nitsche', 'convmethod': 'none', 'divdivparam': 0., 'hdivpenalty': 0.}
    test = 'st2d'
    stat = True
    if test == 'st2d':
        app = applications.SchaeferTurek2d(h=0.2, mu=10)
        # app = applications.SchaeferTurek2d(h=0.2, mu=0.002, errordrag=False)
    elif test == 'st3d':
        # app = applications.SchaeferTurek3d(h=0.5, mu=0.01)
        app = applications.SchaeferTurek3d(h=0.5, mu=0.01)
    if stat:
        static(increase_reynolds=0.6, application=app, femparams=femparams)
        # static(application=app, femparams=femparams)
    else:
        dynamic(application=app, femparams=femparams)


    # main(model='Stokes', static=False, testcase='schaeferTurek2d', h=0.2, mu=1e-3, femparams=femparams)
    # dynamic(model='NavierStokes', application=app, femparams=femparams)

    # main(model='Stokes', testcase='poiseuille2d', h=0.2, mu=1e-2, femparams=femparams)
    # main(model='NavierStokes', testcase='poiseuille2d', h=0.1, mu=1e-4, femparams=femparams)
    # main(model='Stokes', testcase='schaeferTurek2d', h=0.2, mu=1e-2, femparams=femparams)
    # main(model='Stokes', testcase='drivenCavity2d', h=1, mu=3e-2, femparams=femparams)
    # main(testcase='backwardFacingStep2d', mu=2e-3)
    # main(testcase='backwardFacingStep3d', mu=2e-2)
    # main(testcase='schaeferTurek2d')
    # main(testcase='poiseuille3d', h=0.2, mu=1e-3)
    # main(testcase='drivenCavity3d', mu=0.001, precond_p='schur')
    # main(testcase='schaeferTurek3d', h=0.5, bdryplot=False, model='Stokes', plot=False)
    # main(testcase='schaeferTurek3d', h=0.5, bdryplot=False, linearsolver='gcrotmk_1', model='Stokes', plot=False)
    # main(testcase='schaeferTurek3d', h=0.95, bdryplot=False, linearsolver='spsolve', model='Stokes', plot=False)
