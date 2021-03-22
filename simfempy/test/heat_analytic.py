import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)
import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.heat import Heat
from simfempy.tools.comparemethods import CompareMethods
import simfempy.applications.problemdata

#----------------------------------------------------------------#
def test_analytic(createMesh, h, data, exactsolution="Linear", fems=['p1'], methods=['new','trad']):
    if isinstance(fems,str): fems = [fems]
    if isinstance(methods,str): methods = [methods]
    sims = {}
    for fem in fems:
        for method in methods:
            kwargs = {'problemdata':data, 'fem':fem, 'method':method, 'masslumpedbdry':False}
            kwargs['exactsolution'] = exactsolution
            kwargs['random'] = False
            sims[fem+method] = Heat(**kwargs)
    comp = CompareMethods(sims, createMesh=createMesh, plot=False)
    result = comp.compare(h=h)
    # global refine
    # comp = CompareMethods(sims, createMesh=createMesh, h=0.5)
    # result = comp.compare(niter=3)

#----------------------------------------------------------------#
def test(dim, exactsolution='Linear', fems=['p1'], methods=['new','trad']):
    data = simfempy.applications.problemdata.ProblemData()
    data.params.scal_glob['kheat'] = 1
    data.params.fct_glob['beta'] = ["y", "-x"]
    data.params.fct_glob['beta'] = ["1", "1"]
    if dim==1:
        createMesh = testmeshes.unitline
        colors = [10000, 10001]
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03, 0.02]
    elif dim==2:
        createMesh = testmeshes.unitsquare
        colors = [1000, 1001, 1002, 1003]
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03]
    else:
        createMesh = testmeshes.unitcube
        colors = [100, 101, 102, 103, 104, 105]
        h = [1.0, 0.5, 0.25, 0.125]
    if exactsolution == "Constant" or exactsolution == "Linear": h = h[:3]
    data.bdrycond.set("Dirichlet", colors[:])
    data.bdrycond.set("Neumann", colors[0])
    data.bdrycond.set("Robin", colors[1])
    data.bdrycond.param[colors[1]] = 111.
    data.postproc.type['bdrymean'] = "bdry_mean"
    data.postproc.color['bdrymean'] = [colors[1]]
    data.postproc.type['nflux'] = "bdry_nflux"
    data.postproc.color['nflux'] = [*colors[:]]
    test_analytic(createMesh=createMesh, h=h, data=data, exactsolution=exactsolution, fems=fems, methods=methods)

#================================================================#
if __name__ == '__main__':
    test(dim=2, exactsolution = 'Linear', fems=['p1','cr1'])
    # test(dim=2, exactsolution = 'Quadratic', fems=['p1','cr1'])
