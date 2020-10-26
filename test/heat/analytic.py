import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)
import simfempy
import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.heat import Heat

#----------------------------------------------------------------#
def test_analytic(exactsolution="Linear", geomname = "unitsquare", fems=['p1'], methods=['new']):
    import simfempy.tools.comparemethods
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    if geomname == "unitline":
        createMesh = testmeshes.unitline
        colors = [10000, 10001]
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03, 0.02]
    elif geomname == "unitsquare":
        colors = [1000, 1001, 1002, 1003]
        createMesh = testmeshes.unitsquare
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03]
    elif geomname == "unitcube":
        colors = [100, 101, 102, 103, 104, 105]
        createMesh = testmeshes.unitcube
        h = [1.0, 0.5, 0.25, 0.125]
    data.params.scal_glob['kheat'] = 1
    if exactsolution == "Constant" or exactsolution == "Linear":
        h = h[:3]
    data.bdrycond.clear()
    data.bdrycond.set("Dirichlet", colors[:])
    data.bdrycond.set("Neumann", colors[2])
    data.bdrycond.set("Robin", colors[1])
    data.bdrycond.param[colors[1]] = 1.
    sims = {}
    for fem in fems:
        for method in methods:
            sims[fem+method] = Heat(problemdata=data, fem=fem, method=method, exactsolution=exactsolution, random=False)
    comp = simfempy.tools.comparemethods.CompareMethods(sims, plot=True)
    result = comp.compare(createMesh=createMesh, h=h)
#================================================================#
if __name__ == '__main__':
    exactsolution = 'Constant'
    exactsolution = 'Linear'
    # exactsolution = 'Quadratic'
    # test_analytic(exactsolution = exactsolution, geomname = "unitline")
    test_analytic(exactsolution = exactsolution, geomname = "unitsquare")
    # test_analytic(exactsolution = exactsolution, geomname = "unitcube")
