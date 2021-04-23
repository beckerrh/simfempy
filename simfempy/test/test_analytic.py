import itertools
from simfempy.tools.comparemethods import CompareMethods

#----------------------------------------------------------------#
def test_analytic(application, createMesh, paramargs, applicationargs, **kwargs):
    import warnings
    import numpy as np
    warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
    """
    Runs application on a sequence of meshes
    :param createMesh: function creating mesh
    :param paramargs: arguments to application that vary (different fems, e.g.)
    :param applicationargs: arguments passed to application (has to include 'exactsolution' and 'problemdata')
    :param kwargs: paramaters piloting this function
            'plotmesh': just plot first mesh with boundary labels
    :return:
    """
    niter = kwargs.pop('niter', 3)
    h1 = kwargs.pop('h1', 1)
    h = [h1*0.5**i for i in range(niter)]
    if 'plotmesh' in kwargs:
        from simfempy.meshes import plotmesh
        plotmesh.meshWithBoundaries(createMesh(h[0]))
        return
    for pname,params in paramargs.items():
        if isinstance(params, str): paramargs[pname] = [params]
    paramsprod = list(itertools.product(*paramargs.values()))
    paramslist = [{k:params[i] for i,k in enumerate(paramargs)} for params in paramsprod]
    if not 'exactsolution' in applicationargs:
        raise KeyError(f"'exactsolution' should be set in 'applicationkwargs'")
    if not 'problemdata' in applicationargs:
        raise KeyError(f"'problemdata' should be set in 'applicationkwargs'")
    sims = {}
    for p in paramslist:
        name = ''
        for pname, param in p.items():
            if len(paramargs[pname])>1: name += param
            applicationargs[pname] = param
        sims[name] = application(**applicationargs)
    # raise ValueError(f"{kwargs=}")
    comp = CompareMethods(sims, createMesh=createMesh, **kwargs)
    naps, pname, params, infos = comp.compare(h=h)
    errordicts={}
    for k, v in infos.items():
        if k[:3]=='err': errordicts[k] = v
    return errordicts
