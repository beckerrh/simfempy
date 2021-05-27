import numpy as np
from simfempy.applications.stokes import Stokes
from simfempy import fems, meshes, tools
import scipy.sparse as sparse

class NavierStokes(Stokes):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode='nonlinear'
        self.convdata = fems.data.ConvectionData()
        self.convmethod = 'supg'
    def solve(self, dirname='Run'):
        return self.static(dirname=dirname, mode='nonlinear',maxiter=200)
    def computeForm(self, u):
        d = super().computeForm(u)
        v = self._split(u)[0]
        dv = self._split(d)[0]
        self.computeFormConvection(dv, v)
        return d
    def computeMatrix(self, u=None):
        # étrange, j'arrive pas à récuperer le boulot de stokes...
        A = self.femv.computeMatrixLaplace(self.mucell)
        B = self.femv.computeMatrixDivergence()
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        if self.dirichletmethod == 'strong':
            A, B = self.matrixBoundary(A, B, self.bdrydata, self.dirichletmethod)
        else:
            A, B = self.computeMatrixBdryNitsche(A, B, colorsdir, self.mucell)
        if u is not None:
            v = self._split(u)[0]
            A += self.computeMatrixConvection(v)
        if not self.pmean:
            return A, B
        ncells = self.mesh.ncells
        rows = np.zeros(ncells, dtype=int)
        cols = np.arange(0, ncells)
        C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
        return A,B,C
    def computeFormConvection(self, dv, v):
        rt = fems.rt0.RT0(self.mesh)
        self.convdata.betart = rt.interpolateCR1(v)
        self.convdata.beta = rt.toCell(self.convdata.betart)
        dim = self.mesh.dimension
        if not hasattr(self.mesh,'innerfaces'): self.mesh.constructInnerFaces()
        self.convdata.md = meshes.move.move_midpoints(self.mesh, self.convdata.beta, bound=1/dim)
        for icomp in range(dim):
            self.femv.fem.computeFormConvection(dv[icomp::dim], v[icomp::dim], self.convdata, method=self.convmethod)

    def computeMatrixConvection(self, v):
        A = self.femv.fem.computeMatrixConvection(self.convdata, method=self.convmethod)
        return self.femv.matrix2systemdiagonal(A, self.ncomp).tocsr()

    def getVelocityPreconditioner(self, A):
        import pyamg
        B = pyamg.solver_configuration(A, verb=False)['B']
        smoother = ('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})
        build_args = {'symmetry': 'nonsymmetric', 'presmoother': smoother, 'postsmoother':smoother}       
        # return pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy', presmoother=smoother, postsmoother=smoother)
        return pyamg.smoothed_aggregation_solver(A, B=B, smooth='energy', **build_args)
