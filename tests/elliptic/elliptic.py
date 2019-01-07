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
    bdrycond0 =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond0.type[11] = "Neumann"
    bdrycond0.type[33] = "Neumann"
    bdrycond0.type[22] = "Dirichlet"
    bdrycond0.type[44] = "Dirichlet"
    bdrycond1 =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond1.type[11] = "Dirichlet"
    bdrycond1.type[33] = "Dirichlet"
    bdrycond1.type[22] = "Neumann"
    bdrycond1.type[44] = "Neumann"
    postproc0 = {}
    postproc0['bdrymean'] = "bdrymean:11,33"
    postproc0['bdrydn'] = "bdrydn:22,44"
    postproc1 = {}
    postproc1['bdrymean'] = "bdrymean:22,44"
    postproc1['bdrydn'] = "bdrydn:11,33"
    bdrycond = [bdrycond0, bdrycond1]
    postproc = [postproc0, postproc1]
    compares = {}
    app = fempy.applications.elliptic.Elliptic(problem=problem, bdrycond=bdrycond, ncomp=2)
    for fem in ['p1']:
        for bdry in ['trad','new']:
            compares[fem+bdry] = fempy.applications.elliptic.Elliptic(solexact=app.solexact, bdrycond=bdrycond, postproc=postproc, fem=fem, ncomp=2, method=bdry)
    # femmethods = ['p1-trad', 'p1-new']
    # for femmethod in femmethods:
    #     fem, method  = femmethod.split('-')
    #     methods[femmethod] = fempy.applications.elliptic.Elliptic(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem, ncomp=2, method=method)
    comp = fempy.tools.comparerrors.CompareErrors(compares, plot=False)
    h = [0.5, 0.25, 0.125, 0.06, 0.03]
    h = [2.0, 1.0, 0.5, 0.25]
    result = comp.compare(geomname=geomname, h=h)

#----------------------------------------------------------------#
def test_analytic3d():
    import matplotlib.pyplot as plt
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear3d'
    problem = 'Analytic_Quadratic3d'
    problem = 'Analytic_Sinus3d'
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Neumann"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"
    postproc = {}
    postproc['mean'] = "mean:11,22"
    postproc['flux'] = "flux:33,44,55,66"
    methods = {}
    for fem in ['p1', 'cr1']:
        methods[fem] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [np.power(i*20,-2/3) for i in range(1,6)]
    h = [1.0, 0.5, 0.25, 0.13, 0.08]
    print("h", h)
    result = comp.compare(geomname="unitcube", h=h)


#================================================================#

test_analytic()
#test_analytic3d()
