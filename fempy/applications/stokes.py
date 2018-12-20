import time
import numpy as np
import fempy.tools.analyticalsolution
from fempy import solvers
from fempy import fems

#=================================================================#
class Stokes(solvers.newtonsolver.NewtonSolver):
    """
    """
    def __init__(self, **kwargs):
        solvers.newtonsolver.NewtonSolver.__init__(self)
        self.dirichlet = None
        self.neumann = None
        self.rhs = None
        self.solexact = None
        self.bdrycond = kwargs.pop('bdrycond')
        self.kheat = None
        fem = 'p1'
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        if fem == 'p1':
            self.fem = fems.femp1.FemP1()
        elif fem == 'cr1':
            self.fem = fems.femcr1.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'mu' in kwargs:
            self.mu = np.vectorize(kwargs.pop('mu'))
        else:
            self.mu = np.vectorize(lambda i: 0.123)
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        if 'rhs' in kwargs:
            rhs = kwargs.pop('rhs')
            assert rhs is not None
            self.rhs = np.vectorize(rhs)
        if 'postproc' in kwargs:
            self.postproc = kwargs.pop('postproc')
        else:
            self.postproc={}
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="trad"
    def defineProblem(self, problem):
        self.problem = problem
        problemsplit = problem.split('_')
        if problemsplit[0] != 'Analytic':
            raise ValueError("unownd problem {}".format(problem))
        function = problemsplit[1]
        if function == 'Linear2d':
            self.solexact = {}
            self.solexact['V'] = []
            self.solexact['V'].append(fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.7 * y'))
            self.solexact['V'].append(fempy.tools.analyticalsolution.AnalyticalSolution('0.7 * x - 0.3 * y'))
            self.solexact['P'] = []
            self.solexact['0'].append(fempy.tools.analyticalsolution.AnalyticalSolution('0'))
        elif function == 'Linear3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.2 * y + 0.4*z')
        else:
            raise ValueError("unknown analytic solution: {}".format(function))
        for color, bc in self.bdrycond.type.items():
            if bc == "Dirichlet":
                self.bdrycond.fct[color] = {}
                self.bdrycond.fct[color]['V'] = []
                for sol in self.solexact['V']:
                    self.bdrycond.fct[color]['V'].append(sol)
            elif bc == "Neumann":
                self.bdrycond.fct[color] = None
            else:
                raise ValueError("unownd boundary condition {} for color {}".format(bc,color))
    def setMesh(self, mesh):
        t0 = time.time()
        self.mesh = mesh
        self.fem.setMesh(self.mesh)
        colorsdir = []
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.fem.prepareBoundary(colorsdir, self.postproc)
        self.mucell = self.mu(self.mesh.cell_labels)
        t1 = time.time()
        self.timer['setmesh'] = t1-t0
    def solvestatic(self):
        return self.solveLinear()
    def solve(self, iter, dirname):
        return self.solveLinear()
    def computeRhs(self):
        return self.fem.computeRhs(self.rhs, self.solexact, self.mucell, self.bdrycond)
    def matrix(self):
        return self.fem.matrixDiffusion(self.mucell)
    def boundary(self, A, b):
        return self.fem.boundary(A, b, self.bdrycond)

    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = self.fem.tonode(u)
        if self.solexact:
            info['error'] = {}
            info['error']['L2'], e = self.fem.computeErrorL2(self.solexact, u)
            point_data['E'] = self.fem.tonode(e)
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        info['postproc'] = {}
        for key, val in self.postproc.items():
            type,data = val.split(":")
            if type == "mean":
                info['postproc'][key] = self.fem.computeMean(u, key, data)
            elif type == "flux":
                info['postproc'][key] = self.fem.computeFlux(u, key, data)
            else:
                raise ValueError("unknown postprocess {}".format(key))
        cell_data['k'] = self.kheatcell
        return point_data, cell_data, info

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
