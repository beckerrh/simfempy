import numpy as np
import pyamg
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from simfempy import tools
import time
import simfempy.solvers.linalg as linalg

#=================================================================#
# class VelcoitySolver():
#     def __init__(self, A, **kwargs):
#         defsolvers = ['lgmres', 'spsolve']
#         defsolvers.append('pyamg@aggregation@none@gauss_seidel')
#         defsolvers.append('pyamg@aggregation@none@schwarz')
#         defsolvers.append('pyamg@aggregation@fgmres@schwarz')
#         # defsolvers.append('pyamg@rootnode@gcrotmk@gauss_seidel')
#         solvernames = kwargs.pop('solver',  None)
#         if solvernames is None: solvernames = defsolvers
#         if isinstance(solvernames, str):
#             self.solver = linalg.getSolverFromName(solvernames, A, **kwargs)
#             print(f"{kwargs=}")
#             self.maxiter = kwargs.pop('maxiter', 1)
#         else:
#             if 'maxiter' in kwargs: print(f"??? maxiter unused")
#             self.reduction = kwargs.pop('reduction', 0.1)
#             self.solver, self.maxiter = linalg.selectBestSolver(solvernames, self.reduction, A, maxiter=20, verbose=1, **kwargs)
#             self.solver.maxiter = self.maxiter
#
#     def solve(self, b):
#         return self.solver.solve(b, maxiter=self.maxiter, rtol=1e-16)



#=================================================================#
class PressureSolverScale():
    def __init__(self, mesh, mu):
        self.BP = sparse.diags(mu/mesh.dV, offsets=(0), shape=(mesh.ncells, mesh.ncells))
    def solve(self, b):
        return self.BP.dot(b)
#=================================================================#
class PressureSolverDiagonal():
    def __init__(self, A, B, **kwargs):
        AD = sparse.diags(1/A.diagonal(), offsets=(0), shape=A.shape)
        self.mat = B@AD@B.T
        self.maxiter = kwargs.pop('maxiter',1)
        kwargs['symmetric'] = True
        self.prec = linalg.Pyamg(self.mat, **kwargs)
    def solve(self, b):
        return self.prec.solve(b, maxiter=self.maxiter, rtol=1e-16)
#=================================================================#
prec_PressureSolverSchur = ['none', 'diag', 'scale']
class PressureSolverSchur():
    def __init__(self, mesh, mu, A, B, AP, **kwargs):
        ncells, nfaces = mesh.ncells, mesh.nfaces
        self.A, self.B, self.AP = A, B, AP
        prec = kwargs.pop("prec", None)
        if prec is None or prec == 'none' or prec == '':
            self.M = None
        elif prec == 'scale':
            self.BP = sparse.diags(1/mesh.dV*mu, offsets=(0), shape=(mesh.ncells, mesh.ncells))
            self.M = splinalg.LinearOperator(shape=(mesh.ncells, mesh.ncells), matvec=lambda u: self.BP.dot(u))
        elif prec == 'diag':
            AD = sparse.diags(1/A.diagonal(), offsets=(0), shape=A.shape)
            self.mat = B@AD@B.T
            self.prec = linalg.Pyamg(self.mat, symmetric=True)
            self.M = splinalg.LinearOperator(shape=(mesh.ncells, mesh.ncells), matvec=lambda u: self.prec.solve(u, maxiter=1, rtol=1e-14))
        else:
            raise ValueError(f"unknown {prec=}")
        self.maxiter = kwargs.pop('maxiter',1)
        solvername = kwargs.pop('solver',0)
        assert solvername in linalg.scipysolvers
        self.solver =  linalg.ScipySolve(matvec=self.matvec, method=solvername, M=self.M, counter="pschur", n = ncells, **kwargs)

    def matvec(self, x):
        v = self.B.T.dot(x)
        v2 = self.AP.solve(v)
        v3 = self.B.dot(v2)
        # print(f"{np.linalg.norm(x)=} {np.linalg.norm(v)=} {np.linalg.norm(v2)=} {np.linalg.norm(v3)=}")
        return v3
    def solve(self, b):
        self.solver.counter.reset()
        u = self.solver.solve(b, x0=None, maxiter=self.maxiter, rtol=1e-12)
        return u

