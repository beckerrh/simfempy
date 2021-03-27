import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.elasticity import Elasticity
from simfempy.tools.comparemethods import CompareMethods

#----------------------------------------------------------------#
def test_analytic(dim, exactsolution="Sinus", fems=['p1'], dirichlets=['new'], verbose=5):
    import simfempy.tools.comparemethods
    data = simfempy.applications.problemdata.ProblemData()
    if dim==2:
        data.ncomp=2
        createMesh = testmeshes.unitsquare
        h = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]
        data.bdrycond.type[1000] = "Neumann"
        data.bdrycond.type[1001] = "Dirichlet"
        data.bdrycond.type[1002] = "Neumann"
        data.bdrycond.type[1003] = "Dirichlet"
        data.postproc.set(name='bdrymean', type='bdry_mean', colors=[1000,1002])
        data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=[1001,1003])
    else:
        data.ncomp=3
        createMesh = testmeshes.unitcube
        h = [2, 1, 0.5, 0.25, 0.125, 0.08]
        data.bdrycond.type[100] = "Neumann"
        data.bdrycond.type[105] = "Neumann"
        data.bdrycond.type[101] = "Dirichlet"
        data.bdrycond.type[102] = "Dirichlet"
        data.bdrycond.type[103] = "Dirichlet"
        data.bdrycond.type[104] = "Dirichlet"
        data.postproc.set(name='bdrymean', type='bdry_mean', colors=[100,105])
        data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=[101,102,103,104])
    if isinstance(fems, str): fems = [fems]
    if isinstance(dirichlets, str): dirichlets = [dirichlets]
    if exactsolution == "Constant" or exactsolution == "Linear": h = h[:3]
    sims = {}
    for fem in fems:
        for dirichlet in dirichlets:
            kwargs = {'problemdata': data, 'fem': fem, 'dirichlet': dirichlet}
            kwargs['exactsolution'] = exactsolution
            kwargs['random'] = False
            kwargs['linearsolver'] = 'pyamg'
            # kwargs['linearsolver'] = 'umf'
            sims[fem + dirichlet] = Elasticity(**kwargs)
    comp = CompareMethods(sims, createMesh=createMesh, plot=False)
    result = comp.compare(h=h)



#================================================================#
if __name__ == '__main__':
    # test_analytic(dim=3, exactsolution="Linear")
    test_analytic(dim=2, exactsolution="Linear", fems=['cr1'])
    # test_analytic(dim=2, exactsolution="Quadratic")
