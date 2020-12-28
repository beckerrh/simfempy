import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)
import simfempy.meshes.testmeshes as testmeshes
import simfempy.meshes.testgeoms as testgeoms
from simfempy.applications.heat import Heat
from simfempy.tools.comparemethods import CompareMethods

#----------------------------------------------------------------#
def test_analytic(exactsolution="Linear", geomname = "unitsquare", fems=['p1'], methods=['new','trad']):
    # methods = ['trad']
    import simfempy.tools.comparemethods
    data = simfempy.applications.problemdata.ProblemData()
    if geomname == "unitline":
        createMesh = testmeshes.unitline
        geom = testgeoms.unitline
        colors = [10000, 10001]
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03, 0.02]
    elif geomname == "unitsquare":
        colors = [1000, 1001, 1002, 1003]
        createMesh = testmeshes.unitsquare
        geom = testgeoms.unitsquare
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03]
    elif geomname == "unitcube":
        colors = [100, 101, 102, 103, 104, 105]
        createMesh = testmeshes.unitcube
        geom = testgeoms.unitcube
        h = [1.0, 0.5, 0.25, 0.125]
    data.params.scal_glob['kheat'] = 1
    if exactsolution == "Constant" or exactsolution == "Linear": h = h[:3]
    # exactsolution = 'x'
    data.bdrycond.clear()
    data.bdrycond.set("Dirichlet", colors[:])
    data.bdrycond.set("Neumann", colors[0])
    data.bdrycond.set("Robin", colors[1])
    data.bdrycond.param[colors[1]] = 111.
    data.postproc.type['bdrymean'] = "bdry_mean"
    data.postproc.color['bdrymean'] = [colors[1]]
    data.postproc.type['nflux'] = "bdry_nflux"
    data.postproc.color['nflux'] = [*colors[:]]
    sims = {}
    for fem in fems:
        for method in methods:
            kwargs = {'problemdata':data, 'fem':fem, 'method':method, 'masslumpedbdry':False}
            kwargs['exactsolution'] = exactsolution
            kwargs['random'] = False
            sims[fem+method] = Heat(**kwargs)
    comp = CompareMethods(sims, createMesh=createMesh, plot=False)
    result = comp.compare(h=h)
    # comp = CompareMethods(sims, createMesh=createMesh, h=0.5)
    # result = comp.compare(niter=3)
#================================================================#
if __name__ == '__main__':
    exactsolution = 'Constant'
    # exactsolution = 'Linear'
    exactsolution = 'Quadratic'
    # test_analytic(exactsolution = exactsolution, geomname = "unitline")
    test_analytic(exactsolution = exactsolution, geomname = "unitsquare")
    # test_analytic(exactsolution = exactsolution, geomname = "unitcube")
