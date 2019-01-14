assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications


#----------------------------------------------------------------#
def test_analytic():
    import fempy.tools.comparerrors
    # problem = 'Analytic_Constant'
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'
    geomname = "unitsquare"
    geomname = "unitcube"
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
    app = fempy.applications.elasticity.Elasticity(problem=problem, bdrycond=bdrycond, ncomp=ncomp)
    for fem in ['p1']:
        for bdry in ['trad','new']:
            compares[fem+bdry] = fempy.applications.elasticity.Elasticity(solexact=app.solexact, bdrycond=bdrycond, postproc=postproc, fem=fem, ncomp=ncomp, method=bdry, problemname=app.problemname)
    comp = fempy.tools.comparerrors.CompareErrors(compares, plot=False)
    h = [0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]
    if geomname == "unitcube":
        h = [2, 1, 0.5, 0.25, 0.125, 0.08]
    if problem == 'Analytic_Linear':
        h = [1, 0.5, 0.25, 0.125]
    result = comp.compare(geomname=geomname, h=h)


#================================================================#

test_analytic()
