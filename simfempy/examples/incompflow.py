import matplotlib.pyplot as plt
import pathlib, sys
simfempypath = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0,simfempypath)
from simfempy.models.navierstokes import NavierStokes
from simfempy.models.stokes import Stokes
from simfempy.examples import navier_stokes

# ================================================================c#
def getModel(**kwargs):
    disc_params = kwargs.pop('disc_params', {})
    application = kwargs.pop('application', {})
    model = kwargs.pop('model', 'NavierStokes')
    if model == "Stokes":
        model = Stokes(application=application, disc_params=disc_params)
    else:
        model = NavierStokes(application=application, disc_params=disc_params)
    return model


# ================================================================c#
def static(**kwargs):
    model = getModel(**kwargs)
    newtonmethod = kwargs.pop('newtonmethod', 'newton')
    newtonmaxiter = kwargs.pop('newtonmaxiter', 20)
    model.linearsolver['disp'] = 0
    model.linearsolver['maxiter'] = 50
    result, u = model.static(maxiter=newtonmaxiter, method=newtonmethod, rtol=1e-6)
    print(f"{result=}")
    model.application.plot(mesh=model.mesh, data=model.sol_to_data(u))
    model.sol_to_vtu()
    plt.show()
# ================================================================c#
def dynamic(**kwargs):
    model = getModel(**kwargs)
    appname  = model.application.__class__.__name__
    stokes = Stokes(application=model.application, stack_storage=False)
    result, u = stokes.solve()
    T = kwargs.pop('T', 200)
    dt = kwargs.pop('dt', 0.52)
    nframes = kwargs.pop('nframes', int(T/2))
    kwargs_dynamic = {'t_span':(0, T), 'nframes':nframes, 'dt':dt, 'theta':0.8, 'output_vtu': False}
    if kwargs.pop('semi_implicit', False):
        kwargs_dynamic['semi_implicit'] = True
    kwargs_dynamic['newton_verbose'] = False
    result = model.dynamic(u, **kwargs_dynamic)
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

#================================================================#
if __name__ == '__main__':
    test = 'dc_3d_stat'
    # test = 'ps_2d_stat'
    # test = 'dc_2d_stat'
    # test = 'bf_2d_stat'
    if test == 'st_2d_stat':
        app = navier_stokes.SchaeferTurek2d(h=0.25)
    elif test == 'st_2d_dyn':
        app = navier_stokes.SchaeferTurek2d(h=0.3, mu=0.002, errordrag=False)
        T, dt = 50, 0.25
    elif test == 'st_3d_stat':
        app = navier_stokes.SchaeferTurek3d(h=0.5)
    elif test == 'st_3d_dyn':
        app = navier_stokes.SchaeferTurek3d(h=0.5, mu=0.01)
        T, dt = 50, 0.25
    elif test == 'dc_2d_stat':
        app = navier_stokes.DrivenCavity2d(h=0.1)
    elif test == 'dc_2d_dyn':
        app = navier_stokes.DrivenCavity2d(h=0.1, mu=1e-4)
        T, dt = 50, 0.25
    elif test == 'dc_3d_stat':
        app = navier_stokes.DrivenCavity3d(h=0.2)
    elif test == 'ps_2d_stat':
        app = navier_stokes.Poiseuille2d(h=0.2, mu=0.1)
    elif test == 'ps_2d_dyn':
        app = navier_stokes.Poiseuille2d(h=0.2, mu=0.001)
        T, dt = 50, 0.25
    elif test == 'ps_3d_stat':
        app = navier_stokes.Poiseuille3d(h=0.5, mu=0.01)
    elif test == 'ps_3d_dyn':
        app = navier_stokes.Poiseuille3d(h=0.5, mu=0.001)
        T, dt = 50, 0.25
    elif test == 'bf_2d_stat':
        app = navier_stokes.BackwardFacingStep2d(h=0.1, mu=0.01)
    elif test == 'bf_2d_dyn':
        app = navier_stokes.BackwardFacingStep2d(h=0.1, mu=0.01)
        T, dt = 50, 0.25
    elif test == 'bf_3d_stat':
        app = navier_stokes.BackwardFacingStep3d(h=0.1, mu=0.01)
    elif test == 'bf_3d_dyn':
        app = navier_stokes.BackwardFacingStep3d(h=0.1, mu=0.001)
        T, dt = 50, 0.25
    todo = test.split('_')[-1]
    if todo == 'stat':
        static(application=app)
    elif todo == 'dyn':
        dynamic(application=app, T=T, dt=dt)
    else:
        raise ValueError("don't know what to do")
