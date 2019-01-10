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
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[33] = "Neumann"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    postproc = {}
    ncomp = 2
    if geomname == "unitcube":
        ncomp = 3
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Dirichlet"
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Dirichlet"
    compares = {}
    app = fempy.applications.elasticity.Elasticity(problem=problem, bdrycond=bdrycond, ncomp=ncomp)
    for fem in ['p1']:
        for bdry in ['trad']:
            compares[fem+bdry] = fempy.applications.elasticity.Elasticity(solexact=app.solexact, bdrycond=bdrycond, postproc=postproc, fem=fem, ncomp=ncomp, method=bdry, problemname=app.problemname)
    comp = fempy.tools.comparerrors.CompareErrors(compares, plot=False)
    h = [0.5, 0.25, 0.125, 0.6]
    if geomname == "unitcube":
        h = [2, 1, 0.5, 0.25, 0.125]
    result = comp.compare(geomname=geomname, h=h)


#================================================================#

test_analytic()
