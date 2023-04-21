# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
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
    mesh, data, appname = application.createMesh(), application.problemdata, application.__class__.__name__
    print(f"{mesh=}")
    if bdryplot: 
        plotmesh.meshWithBoundaries(mesh)
        plt.show()
        return
    # create model
    if model == "Stokes":
        model = Stokes(mesh=mesh, problemdata=data, femparams=femparams)
    else:
        # model = NavierStokes(mesh=mesh, problemdata=data, hdivpenalty=10)
        model = NavierStokes(mesh=mesh, problemdata=data, femparams=femparams)
    return model, appname


# ================================================================c#
def static(**kwargs):
    model, appname = getModel(**kwargs)
    mesh = model.mesh
    t = timer.Timer("mesh")
    result = model.solve()
    print(f"{result.info['timer']}")
    print(f"postproc:")
    for k, v in result.data['global'].items(): print(f"{k}: {v}")
    if mesh.dimension==2:
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
        plotmesh.meshWithData(mesh, data=result.data, title="Stokes", fig=fig, outer=outer[0])
        plotmesh.meshWithData(mesh, title="Stokes", fig=fig, outer=outer[1],
                            quiver_data={"V":list(result.data['point'].values())})
        plt.show()
    else:
        filename = appname+'.vtu'
        mesh.write(filename, data=result.data)
        import pyvista as pv
        mesh = pv.read(filename)
        cpos = mesh.plot()
# ================================================================c#
def dynamic(**kwargs):
    model, appname = getModel(**kwargs)
    mesh, data = model.mesh, model.problemdata
    t = timer.Timer("mesh")
    stokes = Stokes(mesh=mesh, problemdata=data, femparams=femparams)
    result, u = stokes.solve()
    # u.fill(0)
    # print(f"StokesStokesStokesStokes {result.data['global']=}")
    T, dt, nframes = 200, 0.2, 200
    result = model.dynamic(u, t_span=(0, T), nframes=nframes, dt=dt, theta=0.8)

    print(f"{model.timer=}")
    print(f"{model.newmatrix=}")
    v = result.data['point']['V']
    p = result.data['cell']['P']
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"{appname}")
    gs = fig.add_gridspec(4,2)
    nhalf = (nframes - 1) // 2
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        model.plot_p(ax, p[i * nhalf], title=f"t={result.time[i * nhalf]}")
        ax = fig.add_subplot(gs[i, 1])
        model.plot_v(ax, v[i * nhalf], title=f"t={result.time[i * nhalf]}")
    postprocs = result.data['global']
    ax = fig.add_subplot(gs[3, :])
    for i, k in enumerate(data.postproc.plot):
        ax.plot(result.time, postprocs[k], label=k)
    ax.legend()
    ax.grid()
    plt.show()
    def initfct(ax, u):
        ax.set_aspect(aspect=1)
    anim = animdata.AnimData(mesh, v, plotfct=model.plot_v, initfct=initfct)
    plt.show()

#================================================================#
if __name__ == '__main__':
    from simfempy.models import app_navierstokes
    femparams = {'dirichletmethod':'nitsche', 'convmethod': 'none', 'divdivparam': 0., 'hdivpenalty': 0.}
    # main(model='NavierStokes', testcase='schaeferTurek2d', h=0.2, mu=1e-2, femparams=femparams)
    # main(model='Stokes', static=False, testcase='schaeferTurek2d', h=0.2, mu=1e-3, femparams=femparams)
    app = app_navierstokes.applications.SchaeferTurek2d(h=0.2, mu=1e-3)
    dynamic(model='NavierStokes', static=False, application=app, femparams=femparams)

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
