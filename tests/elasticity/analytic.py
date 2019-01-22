from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

from fempy.applications.elasticity import Elasticity


#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Sinus", geomname = "unitsquare", verbose=5):
    import fempy.tools.comparerrors
    postproc = {}
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    if geomname == "unitsquare":
        ncomp = 2
        bdrycond.type[11] = "Neumann"
        bdrycond.type[33] = "Neumann"
        bdrycond.type[22] = "Dirichlet"
        bdrycond.type[44] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:11,33"
        postproc['bdrydn'] = "bdrydn:22,44"
    if geomname == "unitcube":
        ncomp = 3
        bdrycond.type[11] = "Neumann"
        bdrycond.type[33] = "Dirichlet"
        bdrycond.type[22] = "Dirichlet"
        bdrycond.type[44] = "Dirichlet"
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Neumann"
        postproc['bdrymean'] = "bdrymean:11,66"
        postproc['bdrydn'] = "bdrydn:22,33,44,55"
    compares = {}
    app = Elasticity(problem=problem, bdrycond=bdrycond, ncomp=ncomp)
    for fem in ['p1']:
        for bdry in ['trad','new']:
            compares[fem+bdry] = Elasticity(solexact=app.solexact, bdrycond=bdrycond, postproc=postproc,\
                                            fem=fem, ncomp=ncomp, method=bdry, problemname=app.problemname)
    comp = fempy.tools.comparerrors.CompareErrors(compares, verbose=verbose)
    h = [0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]
    if geomname == "unitcube":
        h = [2, 1, 0.5, 0.25, 0.125, 0.08]
    if problem == 'Analytic_Linear':
        h = [1, 0.5, 0.25, 0.125]
    result = comp.compare(geomname=geomname, h=h)
    return result[3]['error']['L2']


#================================================================#
if __name__ == '__main__':
    test_analytic()
