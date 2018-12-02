assert __name__ == '__main__'

import fempy.tools.comparerrors
import fempy.applications

# problem = 'Analytic_Linear'
problem = 'Analytic_Quadratic'
problem = 'Analytic_Sinus'

methods = {}
methods['p1'] = fempy.applications.heat.Heat(problem=problem)

comp = fempy.tools.comparerrors.CompareErrors(methods, latex=True, vtk=True, plot=True)
# comp.compare(h=[1.0, 0.5, 0.25, 0.125])
comp.compare(h=[2, 1.0, 0.5, 0.25, 0.125])
