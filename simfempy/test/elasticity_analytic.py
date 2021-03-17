import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.elasticity import Elasticity
from simfempy.tools.comparemethods import CompareMethods

#----------------------------------------------------------------#
def test_analytic(dim, exactsolution="Sinus", fems=['p1'], methods=['new','trad'], verbose=5):
    import simfempy.tools.comparemethods
    data = simfempy.applications.problemdata.ProblemData()
    if dim==2:
        createMesh = testmeshes.unitsquare
        h = [0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]
        if exactsolution=="Linear": h = h[:-2]
        data.bdrycond.type[1000] = "Neumann"
        data.bdrycond.type[1001] = "Dirichlet"
        data.bdrycond.type[1002] = "Neumann"
        data.bdrycond.type[1003] = "Dirichlet"
        data.postproc.type['bdrymean'] = "bdrymean"
        data.postproc.color['bdrymean'] = [1000,1002]
        data.postproc.type['bdrydn'] = "bdrydn"
        data.postproc.color['bdrydn'] = [1001,1003]
    else:
        createMesh = testmeshes.unitcube
        h = [2, 1, 0.5, 0.25, 0.125, 0.08]
        if exactsolution=="Linear": h = h[:-2]
        data.bdrycond.type[100] = "Neumann"
        data.bdrycond.type[105] = "Neumann"
        data.bdrycond.type[101] = "Dirichlet"
        data.bdrycond.type[102] = "Dirichlet"
        data.bdrycond.type[103] = "Dirichlet"
        data.bdrycond.type[104] = "Dirichlet"
        data.postproc.type['bdrymean'] = "bdrymean"
        data.postproc.color['bdrymean'] = [100,105]
        data.postproc.type['bdrydn'] = "bdrydn"
        data.postproc.color['bdrydn'] = [101,102,103,104]
    if isinstance(fems, str): fems = [fems]
    if isinstance(methods, str): methods = [methods]
    sims = {}
    for fem in fems:
        for method in methods:
            kwargs = {'problemdata': data, 'fem': fem, 'method': method, 'masslumpedbdry': False}
            kwargs['exactsolution'] = exactsolution
            kwargs['random'] = False
            sims[fem + method] = Elasticity(**kwargs)
    comp = CompareMethods(sims, createMesh=createMesh, plot=False)
    result = comp.compare(h=h)



#================================================================#
if __name__ == '__main__':
    test_analytic(dim=2)
