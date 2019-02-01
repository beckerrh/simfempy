from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

from simfempy.applications.elasticity import Elasticity


#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Sinus", geomname = "unitsquare", verbose=5):
    import simfempy.tools.comparerrors
    postproc = {}
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    if geomname == "unitsquare":
        h = [0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]
        if problem=="Analytic_Linear": h = h[:-2]
        problem += "_2d"
        ncomp = 2
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
    if geomname == "unitcube":
        h = [2, 1, 0.5, 0.25, 0.125, 0.08]
        if problem=="Analytic_Linear": h = h[:-2]
        problem += "_3d"
        ncomp = 3
        bdrycond.type[100] = "Neumann"
        bdrycond.type[105] = "Neumann"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:100,105"
        postproc['bdrydn'] = "bdrydn:101,102,103,104"
    compares = {}
    app = Elasticity(problem=problem, bdrycond=bdrycond, ncomp=ncomp)
    for fem in ['p1']:
        for bdry in ['trad','new']:
            compares[fem+bdry] = Elasticity(solexact=app.solexact, bdrycond=bdrycond, postproc=postproc,\
                                            fem=fem, ncomp=ncomp, method=bdry, problemname=app.problemname)
    comp = simfempy.tools.comparerrors.CompareErrors(compares, verbose=verbose)
    if problem.split('_')[1] == "Linear":
        h = [2, 1, 0.5, 0.25]
    result = comp.compare(geomname=geomname, h=h)
    return result[3]['error']['L2']


#================================================================#
if __name__ == '__main__':
    test_analytic()
