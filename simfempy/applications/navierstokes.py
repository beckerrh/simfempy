import numpy as np
from simfempy.applications.stokes import Stokes
from simfempy import fems, meshes, tools
import scipy.sparse as sparse

class NavierStokes(Stokes):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode='nonlinear'
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
        self.betart = rt.interpolateCR1(v)
        self.beta = rt.toCell(self.betart)
        d = self.mesh.dimension
        if not hasattr(self.mesh,'innerfaces'): self.mesh.constructInnerFaces()
        self.md = meshes.move.move_midpoints(self.mesh, self.beta, bound=1/d)
        #supg
        nfaces, dim, dV = self.mesh.nfaces, self.mesh.dimension, self.mesh.dV
        cellgrads, foc = self.femv.fem.cellgrads[:,:,:dim], self.mesh.facesOfCells
        for icomp in range(dim):
            r = np.einsum('n,njk,nk,ni,nj -> ni', dV, cellgrads, self.beta, 1-dim*self.md.mus, v[icomp::dim][foc])
            np.add.at(dv[icomp::dim], foc, r)
        #bdry
        faces = self.mesh.bdryFaces()
        normalsS = self.mesh.normals[faces]
        dS = -np.linalg.norm(normalsS, axis=1)*np.minimum(self.betart[faces],0)
        for icomp in range(dim):
            dv[icomp::dim][faces] += dS*v[icomp::dim][faces]
        massloc = tools.barycentric.crbdryothers(dim)
        ci = self.mesh.cellsOfFaces[faces][:,0]
        foc = self.mesh.facesOfCells[ci]
        mask = foc != faces[:, np.newaxis]
        fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
        for icomp in range(dim):
            r = np.einsum('n,ij,nj -> ni', dS, massloc, v[icomp::dim][fi])
            np.add.at(dv[icomp::dim], fi, r)


    def computeMatrixConvection(self, v):
        A = self.femv.fem.computeMatrixTransportCellWise(type='supg', data=(self.beta, self.betart, self.md.mus))
        return self.femv.matrix2systemdiagonal(A, self.ncomp).tocsr()

    def getVelocityPreconditioner(self, A):
        import pyamg
        config = pyamg.solver_configuration(A, verb=False)
        smoother = ('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 2})
        build_args = {'symmetry': 'nonsymmetric', 'presmoother': smoother, 'postsmoother':smoother}       
        return pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy', presmoother=smoother, postsmoother=smoother)
        return pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy', **build_args)
