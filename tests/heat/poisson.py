assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.tools.comparerrors
import fempy.applications

problem = 'Analytic_Linear'
# problem = 'Analytic_Quadratic'
# problem = 'Analytic_Sinus'

geomname="unitsquare"
bdrycond={}
bdrycond[11] = "Neumann"
bdrycond[22] = "Neumann"
bdrycond[33] = "Dirichlet"
bdrycond[44] = "Dirichlet"

methods = {}
methods['p1'] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond)

comp = fempy.tools.comparerrors.CompareErrors(methods, latex=True, vtk=True, plot=True)
# comp.compare(h=[1.0, 0.5, 0.25, 0.125])
comp.compare(geomname=geomname, h=[2, 1.0, 0.5, 0.25, 0.125])
