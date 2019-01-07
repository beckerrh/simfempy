assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications

#----------------------------------------------------------------#
def test_analytic():
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'
    geomname = "unitsquare"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[33] = "Neumann"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    postproc = {}
    postproc['mean'] = "mean:11,33"
    postproc['flux'] = "flux:22,44"
    methods = {}
    for fem in ['p1']:
        methods[fem] = fempy.applications.elasticity.Elasticity(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [0.5, 0.25, 0.125]
    # h = [2.0, 1.0]
    result = comp.compare(geomname=geomname, h=h)

#================================================================#

test_analytic()

