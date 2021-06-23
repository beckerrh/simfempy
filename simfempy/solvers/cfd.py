import numpy as np
import pyamg
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from simfempy import tools
import time
import simfempy.solvers.linalg as linalg

#=================================================================#
class VelcoitySolver():
    def _selectsolver(self, solvername, A, **kwargs):
        if solvername in linalg.scipysolvers:
            return linalg.ScipySolve(matrix=A, method=solvername, **kwargs)
        elif solvername == "pyamg":
            return linalg.Pyamg(A, **kwargs)
        elif solvername == "umf":
            return linalg.ScipySpSolve(matrix=A)
        else:
            raise ValueError(f"unknwown {solvername=}")
    def __init__(self, A, **kwargs):
        solvernames = kwargs.pop('solver',  ['pyamg','lgmres', 'umf', 'gcrotmk', 'bicgstab'])
        if isinstance(solvernames, str):
            self.solver = self._selectsolver(solvernames, A, **kwargs)
            self.maxiter = kwargs.pop('maxiter', 1)
        else:
            if 'maxiter' in kwargs: print(f"??? maxiter unused")
            self.reduction = kwargs.pop('reduction', 0.01)
            self.solvers = {}
            for solvername in solvernames:
                self.solvers[solvername] = self._selectsolver(solvername, A, **kwargs)
            b = np.random.random(A.shape[0])
            solverbest, self.maxiter = linalg.selectBestSolver(self.solvers, self.reduction, b, maxiter=100, tol=1e-6, verbose=1)
            print(f"{solverbest=}")
            self.solver = self.solvers[solverbest]
    def solve(self, b):
        return self.solver.solve(b, maxiter=self.maxiter, tol=1e-16)



#=================================================================#
class PressureSolverScale():
    def __init__(self, mesh, mu):
        self.BP = sparse.diags(1/mesh.dV*mu, offsets=(0), shape=(mesh.ncells, mesh.ncells))
    def solve(self, b):
        return self.BP.dot(b)
#=================================================================#
class PressureSolverDiagonal():
    def __init__(self, mesh, mu, ncomp, A, B, AP, **kwargs):
        AD = sparse.diags(1/A.diagonal(), offsets=(0), shape=A.shape)
        self.mat = B@AD@B.T
        self.prec = pyamg.smoothed_aggregation_solver(self.mat)
        self.M = splinalg.LinearOperator(shape=self.mat.shape, matvec=lambda u: self.prec.solve(u))
        self.matvec = splinalg.LinearOperator(shape=self.mat.shape, matvec=lambda u: self.mat.dot(u))
        self.maxiter = kwargs.pop('maxiter',1)
        solvername = kwargs.pop('solver',0)
        assert solvername in linalg.scipysolvers
        self.solver =  linalg.ScipySolve(matvec=self.matvec, method=solvername, M=self.M, counter="pschur", n = self.mat.shape[0], **kwargs)
    def solve(self, b):
        return self.prec.solve(b, maxiter=self.maxiter, tol=1e-12)
        return self.solver.solve(b, maxiter=self.maxiter, tol=1e-12)
#=================================================================#
class PressureSolverSchur():
    def __init__(self, mesh, mu, ncomp, A, B, AP, **kwargs):
        ncells, nfaces = mesh.ncells, mesh.nfaces
        self.A, self.B, self.AP = A, B, AP
        self.BP = sparse.diags(1/mesh.dV*mu, offsets=(0), shape=(mesh.ncells, mesh.ncells))
        self.M = splinalg.LinearOperator(shape=(mesh.ncells, mesh.ncells), matvec=lambda u: self.BP.dot(u))
        self.maxiter = kwargs.pop('maxiter',1)
        solvername = kwargs.pop('solver',0)
        assert solvername in linalg.scipysolvers
        self.solver =  linalg.ScipySolve(matvec=self.matvec, method=solvername, M=self.M, counter="pschur", n = ncells, **kwargs)

    def matvec(self, x):
        v = self.B.T.dot(x)
        v2 = self.AP.solve(v)
        return self.B.dot(v2)
    def solve(self, b):
        self.solver.counter.reset()
        u = self.solver.solve(b, x0=None, maxiter=self.maxiter, tol=1e-12)
        # print(f"### {np.linalg.norm(b)=}")
        # u, info = splinalg.gmres(self.solver, b, x0=None, M=self.M, callback=self.counter, maxiter=self.maxiter, atol=1e-12, tol=tol)
        # print(f"### {np.linalg.norm(u)=}")
        # u, info = splinalg.gcrotmk(self.solver, b, x0=None, M=None, m=5, truncate='smallest', callback=self.counter, maxiter=self.maxiter, atol=1e-12, tol=1e-10)
        # self.counter.niter=0
        # u, info = splinalg.lgmres(self.solver, b, x0=None, M=None, maxiter=3, atol=1e-12, tol=1e-10, callback=self.counter)
        # print(f"{info=}")
        # u, info = pyamg.krylov.bicgstab(self.solver, b, maxiter=3, callback=self.counter, tol=1e-10)
        # if info: raise ValueError(f"no convergence {info=}")
        return u

#=================================================================#
class SystemSolver():
    def __init__(self, n, matvec, matvecprec, **kwargs):
        self.method = kwargs.pop('method','gmres')
        self.atol = kwargs.pop('atol',1e-14)
        self.rtol = kwargs.pop('rtol',1e-10)
        self.disp = kwargs.pop('disp',0)
        self.counter = tools.iterationcounter.IterationCounter(name=self.method, disp=self.disp)
        self.Amult = splinalg.LinearOperator(shape=(n, n), matvec=matvec)
        self.M = splinalg.LinearOperator(shape=(n, n), matvec=matvecprec)
    def solve(self, b, x0):
        if self.method=='lgmres':
            u, info = splinalg.lgmres(self.Amult, b, x0=x0, M=self.M, callback=self.counter, atol=self.atol, tol=self.rtol)
        elif self.method=='gmres':
            u, info = splinalg.gmres(self.Amult, b, x0=x0, M=self.M, callback=self.counter, atol=self.atol, tol=self.rtol)
        elif self.method=='gcrotmk':
            u, info = splinalg.gcrotmk(self.Amult, b, x0=x0, M=self.M, callback=self.counter, atol=self.atol, tol=self.rtol, m=10, truncate='smallest')
        elif self.method=='bicgstab':
            u, info = splinalg.bicgstab(self.Amult, b, x0=x0, M=self.M, callback=self.counter, atol=self.atol, tol=self.rtol)
        elif self.method=='cgs':
            u, info = splinalg.cgs(self.Amult, b, x0=x0, M=self.M, callback=self.counter, atol=self.atol, tol=self.rtol)
        else:
            raise ValueError(f"unknown {self.method=}")
        if info: raise ValueError(f"no convergence in {self.method=} {info=}")
        return u, self.counter.niter

