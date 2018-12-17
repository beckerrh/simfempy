import time
import numpy as np
import scipy.linalg as linalg
import fempy.tools.analyticalsolution
import scipy.sparse as sparse
from fempy import solvers
from fempy import fems

#=================================================================#
class Heat(solvers.newtonsolver.NewtonSolver):
    """
    """
    def __init__(self, **kwargs):
        solvers.newtonsolver.NewtonSolver.__init__(self)
        self.fem = fems.femp1.FemP1()
        self.dirichlet = None
        self.neumann = None
        self.rhs = None
        self.solexact = None
        self.bdrycond = kwargs.pop('bdrycond')
        self.kheat = None
        if 'rhocp' in kwargs:
            self.rhocp = np.vectorize(kwargs.pop('rhocp'))
        else:
            self.rhocp = np.vectorize(lambda i: 1234.56)
        if 'kheat' in kwargs:
            self.kheat = np.vectorize(kwargs.pop('kheat'))
        else:
            self.kheat = np.vectorize(lambda i: 0.123)
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        else:
            self.rhs = np.vectorize(kwargs.pop('rhs'))
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
        if function == 'Linear':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.7 * y')
        elif function == 'Linear3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.2 * y + 0.4*z')
        elif function == 'Quadratic':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y')
        elif function == 'Quadratic3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y+3*z*z')
        elif function == 'Hubbel':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
        elif function == 'Exponential':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
        elif function == 'Sinus':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
        elif function == 'Sinus3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y+0.5*z)')
        else:
            raise ValueError("unknown analytic solution: {}".format(function))
        # class NeummannExact():
        #     def __init__(self, ex):
        #         self.ex = ex
        #     def __call__(self, x, y, z, nx, ny, nz, k):
        #         return k*(self.ex.x(x, y, z)*nx + self.ex.y(x, y, z)*ny + self.ex.z(x, y, z)*nz)
        # class RhsExact():
        #     def __init__(self, ex, k):
        #         self.ex = ex
        #         self.k = k
        #     # def __call__(self, x, y, z):
        #     #     return -self.k*(self.ex.xx(x, y, z) + self.ex.yy(x, y, z) + self.ex.zz(x, y, z))
        # k0 = self.kheat(0)
        # def rhs(x, y, z):
        #     return -k0 * (self.solexact.xx(x, y, z) + self.solexact.yy(x, y, z) + self.solexact.zz(x, y, z))
        #
        # neumannex = np.vectorize(NeummannExact(self.solexact).__call__)
        # rhsclass = RhsExact(self.solexact, self.kheat(0))
        # self.rhs = np.vectorize(rhs)
        if self.solexact:
            for color, bc in self.bdrycond.type.items():
                if bc == "Dirichlet":
                    self.bdrycond.fct[color] = self.solexact
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
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.rhocpcell = self.rhocp(self.mesh.cell_labels)
        t1 = time.time()
        self.timer['setmesh'] = t1-t0
    def solvestatic(self):
        return self.solveLinear()
    def solve(self, iter, dirname):
        return self.solveLinear()
    def computeRhs(self):
        return self.fem.computeRhs(self.rhs, self.solexact, self.kheatcell, self.bdrycond)
    def matrix(self):
        return self.fem.matrixDiffusion(self.kheatcell)
    def boundary(self, A, b):
        return self.fem.boundary(A, b, self.bdrycond)
    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = u
        if self.solexact:
            info['error'] = {}
            info['error']['L2'], point_data['E'] = self.fem.computeErrorL2(self.solexact, u)
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
