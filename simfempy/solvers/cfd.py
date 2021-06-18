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
        self.maxiter = kwargs.pop('maxiter', None)
        solvernames = kwargs.pop('solvers',  ['pyamg','lgmres', 'umf', 'gcrotmk', 'bicgstab'])
        self.reduction = kwargs.pop('reduction', 0.001)
        self.solvers = {}
        # self.analysis = {}
        # print(f"{solvernames=}")
        for solvername in solvernames:
            solver = self._selectsolver(solvername, A, **kwargs)
            # if solvername in linalg.scipysolvers:
            #     solver = linalg.ScipySolve(matrix=A, method=solvername, **kwargs)
            # elif solvername == "pyamg":
            #     solver = linalg.Pyamg(A, **kwargs)
            # elif solvername == "umf":
            #     solver = linalg.ScipySpSolve(matrix=A)
            # else:
            #     raise ValueError(f"unknwown {solvername=}")
            self.solvers[solvername] = solver
        b = np.random.random(A.shape[0])
        solverbest, self.maxiter = linalg.selectBestSolver(self.solvers, self.reduction, b, maxiter=100, tol=1e-6, verbose=1)
        print(f"{solverbest=}")
        self.solver = self.solvers[solverbest]
        # self.maxiter = self.analysis[solverbest][0]
        # for solvername in solvernames:
        #     t0 = time.time()
        #     res = solver.testsolve(b=b, maxiter=100, tol=1e-6)
        #     t = time.time() - t0
        #     monotone = np.all(np.diff(res) < 0)
        #     if len(res)==1:
        #         if res[0] > 1e-6: raise ValueError(f"no convergence")
        #         maxiter = 1
        #     else:
        #         rho = np.power(res[-1]/res[0], 1/len(res))
        #         if not monotone:
        #             print(f"***VelcoitySolver {solvername} not monotone {rho=}")
        #             continue
        #         if rho > 0.8: 
        #             print(f"***VelcoitySolver {solvername} bad {rho=}")
        #             continue
        #         maxiter = int(np.log(self.reduction)/np.log(rho))+1
        #     treq = t/len(res)*maxiter
        #     self.analysis[solvername] = (maxiter, treq)
        # # print(f"{self.analysis=}")
        # for solvername, val in self.analysis.items():
        #     print(f"{solvername=} {val=}")
        # ibest = np.argmin([v[1] for v in self.analysis.values()])
        # solverbest = list(self.analysis.keys())[ibest]
        # print(f"{solverbest=}")
        # self.solver = self.solvers[solverbest]
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
        self.maxiter = kwargs.pop('maxiter',1)
        disp = kwargs.pop('disp',0)
        ncells, nfaces = mesh.ncells, mesh.nfaces
        self.solver = splinalg.LinearOperator(shape=(ncells,ncells), matvec=self.matvec)
        self.counter = tools.iterationcounter.IterationCounter(name="pschur", disp=0)
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

