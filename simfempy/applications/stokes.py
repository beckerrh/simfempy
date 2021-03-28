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
        def _fctu(x, y, z):
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            rhsp = np.zeros(x.shape[0])
            mu = self.mu
            # print(f"{solexact[0](x,y,z)=}")
            for i in range(self.ncomp):
                rhsv[i] -= mu * v[i].dd(i, i, x, y, z)
                rhsv[i] += p.d(i, x, y, z)
                rhsp += v[i].d(i, x, y, z)
            return rhsv, rhsp
        return _fctu
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctneumann(x, y, z, nx, ny, nz):
            v, p = solexact
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            mu = self.mu
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhsv[i] += mu  * v[i].d(j, x, y, z) * normals[j]
                rhsv[i] -= p(x, y, z) * normals[i]
            return rhsv
        return _fctneumann
    def solve(self, iter, dirname): return self.static(iter, dirname)
    def computeMatrix(self):
        A = self.femv.computeMatrixLaplace(self.mucell)
        B = self.femv.computeMatrixDivergence()
        return A,B
    def computeRhs(self, u=None):
        b = self.fem.computeRhs(self.problemdata)

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
