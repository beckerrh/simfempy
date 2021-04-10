import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.stokesbase import StokesBase
from simfempy.tools.analyticalfunction import analyticalSolution
from simfempy.tools import npext

#=================================================================#
class Stokes(StokesBase):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.femv = fems.cr1sys.CR1sys(self.ncomp)
        self.femp = fems.d0.D0()
        self.dirichlet_nitsche = 4
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.mucell = self.compute_cell_vector_from_params('mu', self.problemdata.params)
        self.pmean = list(self.problemdata.bdrycond.type.values()) == len(self.problemdata.bdrycond.type)*['Dirichlet']
    def computeRhs(self, b=None, u=None, coeffmass=None):
        bv = np.zeros(self.femv.nunknowns() * self.ncomp)
        bp = np.zeros(self.femp.nunknowns())
        if u is None:
            uv = np.zeros_like(bv)
            up = np.zeros_like(bp)
            u = (uv,up)
        if 'rhs' in self.problemdata.params.fct_glob:
            rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
            if rhsv: self.femv.computeRhsCells(bv, rhsv)
            if rhsp: self.femp.computeRhsCells(bp, rhsp)
        # print(f"{bv=}")
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        # print(f"{self.problemdata.bdrycond=}")
        # bdryfctv = {k:v[0] if len(v)>1 else None for k,v in self.problemdata.bdrycond.fct.items()}
        # bdryfctv = {c:self.problemdata.bdrycond.fct[c][0] for c in colorsneu if c in self.problemdata.bdrycond.fct}
        # bdryfctp = {k:v[1] for k,v in self.problemdata.bdrycond.fct.items()}
        # self.femv.computeRhsBoundary(bv, colorsneu, bdryfctv)
        self.femv.computeRhsBoundary(bv, colorsneu, self.problemdata.bdrycond.fct)
        # self.femp.computeRhsBoundary(bp, colorsdir, bdryfctp)
        self.computeRhsNitsche((bv,bp), colorsdir, self.problemdata.bdrycond.fct, self.mucell)
        if not self.pmean: return (bv,bp), u
        # if hasattr(self.problemdata,'solexact'):
        if self.problemdata.solexact is not None:
            p = self.problemdata.solexact[1]
            pmean = self.femp.computeMean(p)
        else: pmean=0
        print(f"{pmean=}")
        return (bv,bp,pmean), (u[0], u[1], 0)
    def computeMatrix(self):
        A = self.femv.computeMatrixLaplace(self.mucell)
        B = self.femv.computeMatrixDivergence()
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        A, B = self.computeMatrixNitsche(A, B, colorsdir, self.mucell)
        if not self.pmean:
            return A, B
        ncells = self.mesh.ncells
        rows = np.zeros(ncells, dtype=int)
        cols = np.arange(0, ncells)
        C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
        return A,B,C
    def computeBdryNormalFlux(self, v, p, colors):
        nfaces, ncells, ncomp = self.mesh.nfaces, self.mesh.ncells, self.ncomp
        bdryfct = self.problemdata.bdrycond.fct
        flux, omega = np.zeros(shape=(len(colors),ncomp)), np.zeros(len(colors))
        xf, yf, zf = self.mesh.pointsf.T
        cellgrads = self.femv.fem.cellgrads
        facesOfCell = self.mesh.facesOfCells
        mucell = self.mucell
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces,0]
            normalsS = self.mesh.normals[faces][:,:ncomp]
            dS = np.linalg.norm(normalsS, axis=1)
            if color in bdryfct:
                # bfctv, bfctp = bdryfct[color]
                bfctv = bdryfct[color]
                dirichv = np.hstack([bfctv(xf[faces], yf[faces], zf[faces])])
            flux[i] -= np.einsum('f,fk->k', p[cells], normalsS)
            indfaces = self.mesh.facesOfCells[cells]
            ind = npext.positionin(faces, indfaces).astype(int)
            for icomp in range(ncomp):
                vicomp = v[icomp+ ncomp*facesOfCell[cells]]
                flux[i,icomp] = np.einsum('fj,f,fi,fji->', vicomp, mucell[cells], normalsS, cellgrads[cells, :, :ncomp])
                vD = v[icomp+ncomp*faces]
                if color in bdryfct:
                    vD -= dirichv[icomp]
                flux[i,icomp] -= self.dirichlet_nitsche*np.einsum('f,fi,fi->', vD * mucell[cells], normalsS, cellgrads[cells, ind, :ncomp])
            # for icomp in range(ncomp):
            #     mat2 = np.einsum('fj,f->fj', mat, dirichv[icomp])
            #     np.add.at(bv, icomp + ncomp * indfaces, -mat2)
            # ind = npext.positionin(faces, indfaces).astype(int)
            # for icomp in range(ncomp):
            #     bv[icomp + ncomp * faces] += self.dirichlet_nitsche * np.choose(ind, mat.T) * dirichv[icomp]
        return flux.T
    def computeRhsNitsche(self, b, colorsdir, bdryfct, mucell, coeff=1):
        bv, bp = b
        xf, yf, zf = self.mesh.pointsf.T
        nfaces, ncells, dim, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.ncomp
        cellgrads = self.femv.fem.cellgrads
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces,0]
            normalsS = self.mesh.normals[faces][:,:ncomp]
            dS = np.linalg.norm(normalsS,axis=1)
            # normalsS = normalsS/dS[:,np.newaxis]
            if not color in bdryfct.keys(): continue
            bfctv = bdryfct[color]
            dirichv = np.hstack([bfctv(xf[faces], yf[faces], zf[faces])])
            bp[cells] -= np.einsum('kn,nk->n', coeff*dirichv, normalsS)
            mat = np.einsum('f,fi,fji->fj', coeff*mucell[cells], normalsS, cellgrads[cells, :, :dim])
            indfaces = self.mesh.facesOfCells[cells]
            for icomp in range(ncomp):
                mat2 = np.einsum('fj,f->fj', mat, dirichv[icomp])
                np.add.at(bv, icomp+ncomp*indfaces, -mat2)
            ind = npext.positionin(faces, indfaces).astype(int)
            for icomp in range(ncomp):
                bv[icomp+ncomp*faces] += self.dirichlet_nitsche * np.choose(ind, mat.T)*dirichv[icomp]
        # print(f"{bv.shape=} {bp.shape=}")
    def computeMatrixNitsche(self, A, B, colorsdir, mucell):
        nfaces, ncells, ncomp, dim  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp, self.mesh.dimension
        nloc = dim+1
        nlocncomp = ncomp * nloc
        cellgrads = self.femv.fem.cellgrads
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        indfaces = np.repeat(ncomp * faces, ncomp)
        for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
        cols = indfaces.ravel()
        rows = cells.repeat(ncomp).ravel()
        mat = normalsS.ravel()
        BN = sparse.coo_matrix((mat, (rows, cols)), shape=(ncells, ncomp*nfaces)).tocsr()

        mat = np.einsum('f,fi,fji->fj', mucell[cells], normalsS, cellgrads[cells, :, :dim])
        mats = mat.repeat(ncomp)
        dofs = self.mesh.facesOfCells[cells, :]
        cols = np.repeat(ncomp * dofs, ncomp).reshape(dofs.shape[0], nloc, ncomp)
        for icomp in range(ncomp): cols[:,:,icomp::ncomp] += icomp
        indfaces = np.repeat(ncomp * faces, ncomp*nloc).reshape(dofs.shape[0],nloc,ncomp)
        for icomp in range(ncomp): indfaces[:,:,icomp::ncomp] += icomp
        cols = cols.reshape(dofs.shape[0],nloc*ncomp)
        rows = indfaces.reshape(dofs.shape[0],nloc*ncomp)
        AN = sparse.coo_matrix((mats, (rows.ravel(), cols.ravel())), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
        AD = sparse.diags(AN.diagonal(), offsets=(0), shape=(ncomp*nfaces, ncomp*nfaces))
        A = A- AN -AN.T + self.dirichlet_nitsche*AD
        # print(f"{AN.todense()=}")
        # print(f"{AD.toarray()=}")
        # print(f"{A.toarray()=}")
        # print(f"{A=}")
        B -= BN
        return A,B

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
