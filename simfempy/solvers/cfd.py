import numpy as np
from numpy.random import random
import pyamg
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from simfempy import tools

#=================================================================#
class VelcoitySolver():
    def __init__(self, A, **kwargs):
        self.maxiter = kwargs.pop('maxiter', 1)
        self.nsmooth = kwargs.pop('nsmooth', 2)
        self.smoother = kwargs.pop('smoother', 'schwarz')
        # self.smoother = kwargs.pop('smoother', 'strength_based_schwarz')
        # self.smoother = kwargs.pop('smoother', 'block_gauss_seidel')
        smooth = ('energy', {'krylov': 'fgmres'})
        smoother = (self.smoother, {'sweep': 'symmetric', 'iterations': self.nsmooth})
        pyamgargs = {'B': pyamg.solver_configuration(A, verb=False)['B'], 'smooth': smooth, 'presmoother':smoother, 'postsmoother':smoother}
        pyamgargs['symmetry'] = 'nonsymmetric'
        pyamgargs['coarse_solver'] = 'splu'
        self.solver = pyamg.smoothed_aggregation_solver(A, **pyamgargs)
        monotone, rho = self.checkConvergence(n=A.shape[0])
        if not monotone: raise ValueError(f"VelcoitySolver not monotone {rho=}")
        if rho > 0.5: raise ValueError(f"VelcoitySolver bad {rho=}")
        self.maxiter = int(np.log(0.1)/np.log(rho))+1
        print(f"{rho=} {self.maxiter=}")
    def checkConvergence(self, n):
        res=[]
        self.solver.solve(np.random.random(n), maxiter=100, tol=1e-6, residuals=res)
        # print(f"{res=}")
        res = np.asarray(res)
        monotone = np.all(np.diff(res) < 0)
        rho = np.power(res[-1]/res[0], 1/len(res))
        return monotone, rho
    def solve(self, b):
        return self.solver.solve(b, maxiter=self.maxiter, tol=1e-16)
#=================================================================#
class PressureSolverDiagonal():
    def __init__(self, mesh, mu):
        self.BP = sparse.diags(1/mesh.dV*mu, offsets=(0), shape=(mesh.ncells, mesh.ncells))
    def solve(self, b):
        return self.BP.dot(b)
 #=================================================================#
class PressureSolverSchur():
    def __init__(self, mesh, ncomp, A, B, AP, **kwargs):
        self.A, self.B, self.AP = A, B, AP
        self.maxiter = kwargs.pop('maxiter',30)
        ncells, nfaces = mesh.ncells, mesh.nfaces
        self.solver = splinalg.LinearOperator(shape=(ncells,ncells), matvec=self.matvec)
        self.counter = tools.iterationcounter.IterationCounter(name="pschur", disp=1)
        Ainv = sparse.diags(1/A.diagonal(), offsets=(0), shape=(nfaces*ncomp, nfaces*ncomp))
        # self.spilu = splinalg.spilu(B*Ainv*B.T)
        # self.M = splinalg.LinearOperator(shape=(ncells,ncells), matvec=self.spilu.solve)
        self.M = sparse.diags( 1/(B*Ainv*B.T).diagonal(), offsets=(0), shape=(ncells, ncells) )
        self.M = None

    def matvec(self, x):
        v = self.B.T.dot(x)
        v2 = self.AP.solve(v)
        return self.B.dot(v2)
    def solve(self, b):
        tol = 0.1
        # u, info = splinalg.lgmres(self.solver, b, x0=None, M=self.M, maxiter=self.maxiter, atol=1e-12, tol=tol)
        self.counter.niter=0
        u, info = splinalg.gcrotmk(self.solver, b, x0=None, M=None, callback=self.counter, maxiter=self.maxiter, atol=1e-12, tol=1e-10)
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

