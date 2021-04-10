import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.stokesbase import StokesBase
from simfempy.tools.analyticalfunction import analyticalSolution

#=================================================================#
class StokesStrong(StokesBase):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, dirichletmethod='new')
    def setMesh(self, mesh):
        super().setMesh(mesh)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        self.bdrydata = self.femv.prepareBoundary(colorsdirichlet, colorsflux)
    def computeRhs(self, b=None, u=None, coeffmass=None):
        bv = np.zeros(self.femv.nunknowns() * self.ncomp)
        bp = np.zeros(self.femp.nunknowns())
        rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
        if rhsv: self.femv.computeRhsCells(bv, rhsv)
        if rhsp: self.femp.computeRhsCells(bp, rhsp)
        # print(f"{bv=}")
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        # bdryfctv = {k:v[0] for k,v in self.problemdata.bdrycond.fct.items()}
        # bdryfctp = {k:v[1] for k,v in self.problemdata.bdrycond.fct.items()}
        self.femv.computeRhsBoundary(bv, colorsneu, self.problemdata.bdrycond.fct)
        # self.femp.computeRhsBoundary(bp, colorsdir, bdryfctp)
        b, u, self.bdrydata = self.vectorBoundary((bv, bp), u, self.problemdata.bdrycond.fct)
        if not self.pmean: return b,u
        if hasattr(self.problemdata,'solexact'):
            p = self.problemdata.solexact[1]
            pmean = self.femp.computeMean(p)
        else: pmean=0
        print(f"{pmean=}")
        return (bv,bp,pmean), (u[0], u[1], 0)
    def computeMatrix(self):
        A = self.femv.computeMatrixLaplace(self.mucell)
        B = self.femv.computeMatrixDivergence()
        A, B, self.bdrydata = self.matrixBoundary(A, B)
        # print(f"A\n{A.todense()}")
        # print(f"B\n{B.todense()}")
        if not self.pmean:
            return A, B
        ncells = self.mesh.ncells
        rows = np.zeros(ncells, dtype=int)
        cols = np.arange(0, ncells)
        C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
        return A,B,C
    def computeBdryNormalFlux(self, v, p, colors):
        nfaces, ncells, ncomp, bdrydata  = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.bdrydata
        flux, omega = np.zeros(shape=(ncomp,len(colors))), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            As = bdrydata.Asaved[color]
            Bs = bdrydata.Bsaved[color]
            res = bdrydata.bsaved[color] - As * v + Bs.T * p
            for icomp in range(ncomp):
                flux[icomp, i] = np.sum(res[icomp::ncomp])
            # print(f"{flux=}")
            #TODO flux Stokes Dirichlet strong wrong
        return flux
    def vectorBoundary(self, b, u, bdryfctv):
        bv, bp = b
        if u is None:
            uv = np.zeros_like(bv)
            up = np.zeros_like(bp)
        else:
            uv, up = u
            assert uv.shape == bv.shape
            assert up.shape == bp.shape
        bv, uv, self.bdrydata = self.femv.vectorBoundary(bv, uv, bdryfctv, self.bdrydata)
        facesdirall, facesinner, colorsdir, facesdirflux = self.bdrydata.facesdirall, self.bdrydata.facesinner, self.bdrydata.colorsdir, self.bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        self.bdrydata.bsaved = {}
        for key, faces in facesdirflux.items():
            indfaces = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
            self.bdrydata.bsaved[key] = bv[indfaces]
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        bp -= self.bdrydata.B_inner_dir * uv[inddir]
        return (bv,bp), (uv,up), self.bdrydata
    def matrixBoundary(self, A, B):
        A, self.bdrydata = self.femv.matrixBoundary(A, self.bdrydata)
        facesdirall, facesinner, colorsdir, facesdirflux = self.bdrydata.facesdirall, self.bdrydata.facesinner, self.bdrydata.colorsdir, self.bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        self.bdrydata.Bsaved = {}
        for key, faces in facesdirflux.items():
            nb = faces.shape[0]
            helpB = sparse.dok_matrix((ncomp*nfaces, ncomp*nb))
            for icomp in range(ncomp):
                for i in range(nb): helpB[icomp + ncomp*faces[i], icomp + ncomp*i] = 1
            self.bdrydata.Bsaved[key] = B.dot(helpB)
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        self.bdrydata.B_inner_dir = B[:,:][:,inddir]
        help = np.ones((ncomp * nfaces))
        help[inddir] = 0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        B = B.dot(help)
        return A,B, self.bdrydata

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
