import numpy as np
from numpy.lib.function_base import copy
from simfempy.applications.stokes import Stokes
from simfempy import fems, meshes, solvers
import scipy.sparse as sparse

class NavierStokes(Stokes):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode='nonlinear'
        self.convdata = fems.data.ConvectionData()
        self.convmethod = 'supg'
        # self.convmethod = 'upwalg'
    def solve(self, dirname="Run"):
        sdata = solvers.newtondata.StoppingData(maxiter=200, steptype='rb', nbase=2)
        return self.static(dirname=dirname, mode='nonlinear',sdata=sdata)
    def computeForm(self, u):
        self.timer.add('form')
        if not hasattr(self,'Astokes'): self.Astokes = super().computeMatrix()
        d = super().matrixVector(self.Astokes,u)
        # d = super().computeForm(u)
        v = self._split(u)[0]
        dv = self._split(d)[0]
        self.computeFormConvection(dv, v)
        return d
    def computeMatrix(self, u=None):
        self.timer.add('matrix')
        if not hasattr(self,'Astokes'): self.Astokes = super().computeMatrix()
        X = [A.copy() for A in self.Astokes]
        # X = super().computeMatrix(u)
        if u is None: return X
        v = self._split(u)[0]
        X[0] += self.computeMatrixConvection(v)
        return X
    def computeFormConvection(self, dv, v):
        rt = fems.rt0.RT0(self.mesh)
        self.convdata.betart = rt.interpolateCR1(v)
        self.convdata.beta = rt.toCell(self.convdata.betart)
        # meshes.plotmesh.meshWithData(self.mesh, title="Stokes", quiver_data={"V":[self.convdata.beta[:,0],self.convdata.beta[:,1]]})
        # import matplotlib.pyplot as plt
        # plt.show()

        dim = self.mesh.dimension
        if not hasattr(self.mesh,'innerfaces'): self.mesh.constructInnerFaces()
        self.convdata.md = meshes.move.move_midpoints(self.mesh, self.convdata.beta, bound=1/dim)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        vdir = self.femv.interpolateBoundary(colorsdirichlet, self.problemdata.bdrycond.fct).ravel()
        self.femv.massDotBoundary(dv, vdir, colors=colorsdirichlet, ncomp=self.ncomp, coeff=np.minimum(self.convdata.betart, 0))
        for icomp in range(dim):
            self.femv.fem.computeFormConvection(dv[icomp::dim], v[icomp::dim], self.convdata, method=self.convmethod)

    def computeMatrixConvection(self, v):
        A = self.femv.fem.computeMatrixConvection(self.convdata, method=self.convmethod)
        return self.femv.matrix2systemdiagonal(A, self.ncomp).tocsr()

    def computeDx(self, b, u, info):
        # it,rhor,dx, step, y = info
        if info.iter>2: rtol = 0.1*info.rhor
        else: rtol = 0.01
        self.A = self.computeMatrix(u=u) 
        # if dx is not None and it>2:
        #     dv = self._split(dx)[0]
        #     yv = self._split(y)[0]
        #     self.A[0] = tools.matrix.addRankOne(self.A[0], step*dv, yv, relax=1)          
        try:
            u, niter = self.linearSolver(self.A, bin=b, uin=None, solver=self.linearsolver, rtol=rtol)
        except Warning:
            raise ValueError(f"matrix is singular {self.A.shape=} {self.A.diagonal()=}")
        self.timer.add('solve')
        return u, niter

    def getVelocitySolver(self, A):
        return solvers.cfd.VelcoitySolver(A)
    def getPressureSolver(self, A, B, AP):
        mu = self.problemdata.params.scal_glob['mu']
        # return solvers.cfd.PressureSolverDiagonal(self.mesh, mu)    
        return solvers.cfd.PressureSolverSchur(self.mesh, self.ncomp, A, B, AP)    
