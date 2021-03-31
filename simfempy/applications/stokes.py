import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.tools.analyticalfunction import analyticalSolution

#=================================================================#
class Stokes(Application):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.femv = fems.cr1sys.CR1sys(self.ncomp)
        self.femp = fems.d0.D0()
        self.mu = kwargs.pop('mu', 1)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.mucell = np.full(self.mesh.ncells, self.mu)
    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        # print(f"defineAnalyticalSolution: {dim=} {self.ncomp=}")
        v = analyticalSolution(exactsolution[0], dim, dim, random)
        p = analyticalSolution(exactsolution[1], dim, 1, random)
        return v,p
    def defineRhsAnalyticalSolution(self, solexact):
        v,p = solexact
        def _fctrhsv(x, y, z):
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            mu = self.mu
            for i in range(self.ncomp):
                rhsv[i] -= mu * v[i].dd(i, i, x, y, z)
                rhsv[i] += p.d(i, x, y, z)
            return rhsv
        def _fctrhsp(x, y, z):
            rhsp = np.zeros(x.shape[0])
            for i in range(self.ncomp):
                rhsp = v[i].d(i, x, y, z)
            return rhsp
        return _fctrhsv, _fctrhsp
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctneumannv(x, y, z, nx, ny, nz):
            v, p = solexact
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            mu = self.mu
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhsv[i] -= mu  * v[i].d(j, x, y, z) * normals[j]
                rhsv[i] += p(x, y, z) * normals[i]
            return rhsv
        def _fctneumannp(x, y, z, nx, ny, nz):
            v, p = solexact
            rhsp = np.zeros(shape=x.shape[0])
            normals = nx, ny, nz
            for i in range(self.ncomp):
                rhsp -= v[i](x, y, z) * normals[i]
            return rhsp
        return _fctneumannv, _fctneumannp
    def solve(self, iter, dirname): return self.static(iter, dirname)
    def computeRhs(self, b=None, u=None, coeff=1, coeffmass=None):
        bv = np.zeros(self.femv.nunknowns() * self.ncomp)
        bp = np.zeros(self.femp.nunknowns())
        rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
        if rhsv: self.femv.computeRhsCells(bv, rhsv)
        if rhsp: self.femp.computeRhsCells(bp, rhsp)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        self.femv.computeRhsBoundary(bv, colorsneu, self.problemdata.bdrycond.fct)
        self.femp.computeRhsBoundary(bp, colorsdir, self.problemdata.bdrycond.fct)
        return self.femv.vectorDirichlet(self.problemdata, 'new', bv, u), bp
    def computeMatrix(self):
        A = self.femv.computeMatrixLaplace(self.mucell)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        B = self.femv.computeMatrixDivergence(colorsdir)
        return self.femv.matrixDirichlet('new', A).tobsr(), B

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
