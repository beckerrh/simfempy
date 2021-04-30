# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import simfempy.fems.bdrydata
from simfempy.tools import barycentric,npext
from simfempy import fems
from simfempy.meshes import move

#=================================================================#
class CR1(fems.fem.Fem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dirichlet_al = 1
        self.dirichlet_nitsche = 2
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.computeStencilCell(self.mesh.facesOfCells)
        self.cellgrads = self.computeCellGrads()
    def nlocal(self): return self.mesh.dimension+1
    def nunknowns(self): return self.mesh.nfaces
    def dofspercell(self): return self.mesh.facesOfCells
    def tonode(self, u):
        # print(f"{u=}")
        unodes = np.zeros(self.mesh.nnodes)
        if u.shape[0] != self.mesh.nfaces: raise ValueError(f"{u.shape=} {self.mesh.nfaces=}")
        scale = self.mesh.dimension
        np.add.at(unodes, self.mesh.simplices.T, np.sum(u[self.mesh.facesOfCells], axis=1))
        np.add.at(unodes, self.mesh.simplices.T, -scale*u[self.mesh.facesOfCells].T)
        countnodes = np.zeros(self.mesh.nnodes, dtype=int)
        np.add.at(countnodes, self.mesh.simplices.T, 1)
        unodes /= countnodes
        # print(f"{unodes=}")
        return unodes
    def prepareAdvection(self, beta, scale, method):
        rt = fems.rt0.RT0(self.mesh)
        self.betart = scale*rt.interpolate(beta)
        self.beta = rt.toCell(self.betart)
        d = self.mesh.dimension
        if method == 'supg':
            self.mesh.constructInnerFaces()
            self.md = move.move_midpoints(self.mesh, self.beta, bound=1/d)
            # self.md.plot(self.mesh, self.beta, type='midpoints')
        elif method == 'supg2':
            self.mesh.constructInnerFaces()
            self.md = move.move_midpoints(self.mesh, self.beta, candidates='all')
            # print(f"{self.md.mus=}")
            self.md.plot(self.mesh, self.beta, type='midpoints')
        elif method == 'upw':
            self.mesh.constructInnerFaces()
            self.md = move.move_midpoints(self.mesh, self.beta, bound=1/d)
            self.md.plot(self.mesh, self.beta, type='midpoints')
        elif method == 'upw2':
            assert 0
        elif method == 'lps':
            self.mesh.constructInnerFaces()
        else:
            raise ValueError(f"don't know {method=}")
    def computeCellGrads(self):
        normals, facesOfCells, dV = self.mesh.normals, self.mesh.facesOfCells, self.mesh.dV
        return (normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
    def prepareStab(self):
        self.computeStencilInnerSidesCell(self.mesh.facesOfCells)
    # strong bc
    def prepareBoundary(self, colorsdir, colorsflux=[]):
        bdrydata = simfempy.fems.bdrydata.BdryData()
        bdrydata.facesdirall = np.empty(shape=(0), dtype=np.uint32)
        bdrydata.colorsdir = colorsdir
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.facesdirall = np.unique(np.union1d(bdrydata.facesdirall, facesdir))
        bdrydata.facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int), bdrydata.facesdirall)
        bdrydata.facesdirflux = {}
        for color in colorsflux:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.facesdirflux[color] = facesdir
        return bdrydata
    def computeRhsNitscheDiffusion(self, b, diffcoff, colorsdir, bdrycond, coeff=1):
        nfaces, ncells, dim, nlocal  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        x, y, z = self.mesh.pointsf.T
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces,0]
            normalsS = self.mesh.normals[faces][:,:dim]
            dS = np.linalg.norm(normalsS,axis=1)
            if not color in bdrycond.fct: continue
            dirichlet = bdrycond.fct[color]
            u = dirichlet(x[faces], y[faces], z[faces])
            mat = np.einsum('f,fi,fji->fj', coeff*u*diffcoff[cells], normalsS, self.cellgrads[cells, :, :dim])
            np.add.at(b, self.mesh.facesOfCells[cells], -mat)
            ind = npext.positionin(faces, self.mesh.facesOfCells[cells]).astype(int)
            if not np.all(faces == self.mesh.facesOfCells[cells,ind]):
                print(f"{faces=}")
                print(f"{self.mesh.facesOfCells[cells]=}")
                print(f"{self.mesh.facesOfCells[cells,ind]=}")
            b[faces] += self.dirichlet_nitsche * np.choose(ind,mat.T)
        return b
    def computeMatrixNitscheDiffusion(self, A, diffcoff, colorsdir, coeff=1):
        nfaces, ncells, dim, nlocal  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :dim]
        cols = self.mesh.facesOfCells[cells, :].ravel()
        rows = faces.repeat(nlocal)
        mat = np.einsum('f,fi,fji->fj', coeff * diffcoff[cells], normalsS, self.cellgrads[cells, :, :dim]).ravel()
        AN = sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
        AD = sparse.diags(AN.diagonal(), offsets=(0), shape=(nfaces, nfaces))
        return A- AN -AN.T + self.dirichlet_nitsche*AD
    def computeBdryNormalFluxNitsche(self, u, colors, bdrycond, diffcoff):
        flux= np.zeros(len(colors))
        nfaces, ncells, dim, nlocal  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        facesOfCell = self.mesh.facesOfCells
        x, y, z = self.mesh.pointsf.T
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces, 0]
            normalsS = self.mesh.normals[faces,:dim]
            cellgrads = self.cellgrads[cells, :, :dim]
            # print(f"{u[facesOfCell[cells]].shape=}")
            # print(f"{normalsS.shape=}")
            flux[i] = np.einsum('fj,f,fi,fji->', u[facesOfCell[cells]], diffcoff[cells], normalsS, cellgrads)
            dirichlet = bdrycond.fct[color]
            uD = u[faces]
            if color in bdrycond.fct:
                uD -= dirichlet(x[faces], y[faces], z[faces])
            ind = npext.positionin(faces, self.mesh.facesOfCells[cells]).astype(int)
            # print(f"{self.cellgrads[cells, ind, :dim].shape=}")
            flux[i] -= self.dirichlet_nitsche*np.einsum('f,fi,fi->', uD * diffcoff[cells], normalsS, self.cellgrads[cells, ind, :dim])
        return flux
    def vectorBoundaryZero(self, du, bdrydata):
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        du[facesdirall] = 0
        return du
    def vectorBoundary(self, b, bdrycond, bdrydata, method):
        assert method != 'nitsche'
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        for color, faces in facesdirflux.items():
            bdrydata.bsaved[color] = b[faces]
        help = np.zeros_like(b)
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            if color in bdrycond.fct:
                dirichlet = bdrycond.fct[color]
                help[faces] = dirichlet(x[faces], y[faces], z[faces])
        b[facesinner] -= bdrydata.A_inner_dir * help[facesdirall]
        if method == 'strong':
            b[facesdirall] = help[facesdirall]
        else:
            b[facesdirall] += bdrydata.A_dir_dir * help[facesdirall]
        return b
    def matrixBoundary(self, A, bdrydata, method):
        assert method != 'nitsche'
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        nfaces = self.mesh.nfaces
        for color, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        bdrydata.A_inner_dir = A[facesinner, :][:, facesdirall]
        help = np.ones((nfaces))
        help[facesdirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
        diag = np.zeros((nfaces))
        if method == 'strong':
            diag[facesdirall] = 1.0
            diag = sparse.dia_matrix((diag, 0), shape=(nfaces, nfaces))
        else:
            bdrydata.A_dir_dir = self.dirichlet_al*A[facesdirall, :][:, facesdirall]
            diag[facesdirall] = np.sqrt(self.dirichlet_al)
            diag = sparse.dia_matrix((diag, 0), shape=(nfaces, nfaces))
            diag = diag.dot(A.dot(diag))
        A = help.dot(A.dot(help))
        A += diag
        return A
    # interpolate
    def interpolate(self, f):
        x, y, z = self.mesh.pointsf.T
        return f(x, y, z)
    def interpolateBoundary(self, colors, f, lumped=False):
        """
        :param colors: set of colors to interpolate
        :param f: ditct of functions
        :return:
        """
        b = np.zeros(self.mesh.nfaces)
        if lumped:
            for color in colors:
                if not color in f or not f[color]: continue
                faces = self.mesh.bdrylabels[color]
                ci = self.mesh.cellsOfFaces[faces][:, 0]
                foc = self.mesh.facesOfCells[ci]
                mask = foc != faces[:, np.newaxis]
                fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                normalsS = normalsS/dS[:,np.newaxis]
                nx, ny, nz = normalsS.T
                x, y, z = self.mesh.pointsf[faces].T
                try:
                    b[faces] = f[color](x, y, z, nx, ny, nz)
                except:
                    b[faces] = f[color](x, y, z)
            return b
        massloc = barycentric.crbdryothers(self.mesh.dimension)
        for color in colors:
            if not color in f or not f[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            ci = self.mesh.cellsOfFaces[faces][:, 0]
            foc = self.mesh.facesOfCells[ci]
            x, y, z = self.mesh.pointsf[foc].T
            ff = f[color](x, y, z)
            np.put(b, foc, ff.T)
        return b
    # matrices
    def computeMassMatrix(self, coeff=1, lumped=False):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        nfaces, dim = self.mesh.nfaces, self.mesh.dimension
        if lumped:
            mass = coeff/(dim+1)*dV.repeat(dim+1)
            rows = self.mesh.facesOfCells.ravel()
            return sparse.coo_matrix((mass, (rows, rows)), shape=(nfaces, nfaces)).tocsr()
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        mass = np.einsum('n,kl->nkl', dV, massloc).ravel()
        return sparse.coo_matrix((mass, (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
    def computeBdryMassMatrix(self, colors=None, coeff=1, lumped=False):
        nfaces, d = self.mesh.nfaces, self.mesh.dimension
        massloc = barycentric.crbdryothers(d)
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        lumped = False
        #! sinon très mauvais !!
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            if isinstance(coeff, dict):
                dS = linalg.norm(normalsS, axis=1)*coeff[color]
            else:
                dS = linalg.norm(normalsS, axis=1)*coeff[faces]
            cols = np.append(cols, faces)
            rows = np.append(rows, faces)
            mat = np.append(mat, dS)
            if not lumped:
                ci = self.mesh.cellsOfFaces[faces][:,0]
                foc = self.mesh.facesOfCells[ci]
                mask = foc != faces[:, np.newaxis]
                fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
                # print(f"{massloc=}")
                cols = np.append(cols, np.tile(fi, d).ravel())
                rows = np.append(rows, np.repeat(fi, d).ravel())
                mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).ravel())
        # print(f"{mat=}")
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
    def massDotBoundary(self, b, f, colors=None, coeff=1, lumped=True):
        if colors is None: colors = self.mesh.bdrylabels.keys()
        massloc = barycentric.crbdryothers(self.mesh.dimension)
        lumped=False
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            if isinstance(coeff, (int,float)): dS *= coeff
            elif isinstance(coeff, dict): dS *= coeff[color]
            else:
                assert coeff.shape[0]==self.mesh.nfaces
                dS *= coeff[faces]
            b[faces] += dS*f[faces]
            if not lumped:
                ci = self.mesh.cellsOfFaces[faces][:, 0]
                foc = self.mesh.facesOfCells[ci]
                mask = foc!=faces[:,np.newaxis]
                fi = foc[mask].reshape(foc.shape[0],foc.shape[1]-1)
                r = np.einsum('n,kl,nl->nk', dS, massloc, f[fi])
                # print(f"{np.linalg.norm(f[fi])=}")
                np.add.at(b, fi, r)
        return b
    def computeMassMatrixSupg(self, xd, coeff=1):
        raise NotImplemented(f"computeMassMatrixSupg")
    def computeMatrixTransportUpwind(self, bdrylumped, colors):
        dim, dV, nfaces = self.mesh.dimension, self.mesh.dV, self.mesh.nfaces
        innerfaces, foc = self.mesh.innerfaces, self.mesh.facesOfCells
        print(f"{innerfaces.shape=} {dim=} {self.mesh.nfaces=}")
        ci0 = self.mesh.cellsOfInteriorFaces[:,0]
        ci1 = self.mesh.cellsOfInteriorFaces[:,1]
        normalsS = self.mesh.normals[innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        A = sparse.coo_matrix((nfaces, nfaces))
        centered=True
        if centered:
            rows0 = foc[ci0].ravel()
            rows1 = foc[ci1].ravel()
            cols = np.repeat(np.arange(nfaces)[innerfaces],dim+1).ravel()
            matloc = np.full(shape=(dim+1),fill_value=1/(dim+1))
            mat = np.einsum('n,k->nk', self.betart[innerfaces]*dS, matloc).ravel()
            # print(f"{rows0.shape=} {cols.shape=} {mat.shape=}")
            A += sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
            A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))
        else:
            rows0 = foc[ci0].ravel()
            rows1 = foc[ci1].ravel()
            cols = np.repeat(np.arange(nfaces)[innerfaces],dim+1).ravel()
            matloc = np.full(shape=(dim+1),fill_value=1/(dim+1))
            mat = np.einsum('n,nk->nk', self.betart[innerfaces]*dS, self.md.mus[ci0]).ravel()
            # print(f"{rows0.shape=} {cols.shape=} {mat.shape=}")
            A += sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
            mat = np.einsum('n,nk->nk', self.betart[innerfaces]*dS, self.md.mus[ci1]).ravel()
            A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))



            # rows0 = foc[ci0].ravel()
            # rows1 = foc[ci1].ravel()
            # faceind = np.arange(nfaces)[innerfaces]
            # cols = np.repeat(faceind, dim+1).ravel()
            # matloc = np.full(shape=(dim+1),fill_value=1/(dim+1))
            #
            # betadS = np.maximum(self.betart[self.mesh.innerfaces], 0)*dS
            # mat = np.einsum('n,k->nk', betadS, matloc).ravel()
            # A += sparse.coo_matrix((betadS, (faceind, faceind)), shape=(nfaces, nfaces))
            # A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))
            #
            # betadS = np.minimum(self.betart[self.mesh.innerfaces], 0)*dS
            # mat = np.einsum('n,k->nk', betadS, matloc).ravel()
            # A += sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
            # A -= sparse.coo_matrix((betadS, (faceind, faceind)), shape=(nfaces, nfaces))




            # rows0 = np.repeat(foc[ci0],dim+1).ravel()
            # cols0 = np.tile(foc[ci0],dim+1).ravel()
            # rows1 = np.repeat(foc[ci1],dim+1).ravel()
            # cols1 = np.tile(foc[ci1],dim+1).ravel()
            # matloc = np.full(shape=(dim+1,dim+1),fill_value=1/(dim+1))
            # mat = np.einsum('n,kl->nkl', np.maximum(self.betart[self.mesh.innerfaces], 0)*dS, matloc).ravel()
            # A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(nfaces, nfaces))
            # A -= sparse.coo_matrix((mat, (rows1, cols0)), shape=(nfaces, nfaces))
            # mat = np.einsum('n,kl->nkl', np.minimum(self.betart[self.mesh.innerfaces], 0)*dS, matloc).ravel()
            # # print(f"{A.diagonal()=}")
            # A += sparse.coo_matrix((mat, (rows0, cols1)), shape=(nfaces, nfaces))
            # A -= sparse.coo_matrix((mat, (rows1, cols1)), shape=(nfaces, nfaces))
        #bdry
        faces = self.mesh.bdryFaces()
        normalsS = self.mesh.normals[faces]
        dS = linalg.norm(normalsS, axis=1)
        rows = foc[self.mesh.cellsOfFaces[faces,0]].ravel()
        cols = np.repeat(faces,dim+1).ravel()
        matloc = np.full(shape=(dim + 1), fill_value=1 / (dim + 1))
        mat = np.einsum('n,k->nk', self.betart[faces]*dS, matloc).ravel()
        A += sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))
        # A += self.computeMatrixJump(self.betart)

        A += self.computeMatrixJump(self.betart)
        A += self.computeBdryMassMatrix(coeff=-np.minimum(self.betart, 0), colors=colors, lumped=bdrylumped)

        B = self.computeMatrixTransportSupg(bdrylumped, colors)
        if not np.allclose(B.A, A.tocsr().A):
            raise ValueError(f"{B.todense()=}\n{A.todense()=}")
        # print(f"{A.diagonal()=}")
        return A.tocsr()
    def computeMatrixTransportSupg(self, bdrylumped, colors):
        beta, mus, deltas = self.beta, self.md.mus, self.md.deltas
        nfaces, dim, dV = self.mesh.nfaces, self.mesh.dimension, self.mesh.dV
        cellgrads = self.cellgrads[:,:,:dim]
        # betagrad = np.einsum('njk,nk -> nj', cellgrads, beta)
        # mat = np.einsum('n,nj,i -> nij', dV, betagrad, 1/(dim+1)*np.ones(dim+1))
        # mat += np.einsum('n,nj,ni -> nij', dV*deltas, betagrad, betagrad)
        mat = np.einsum('n,njk,nk,ni -> nij', dV, cellgrads, beta, 1-dim*mus)
        A =  sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        A += self.computeMatrixJump(self.betart)
        A += self.computeBdryMassMatrix(coeff=-np.minimum(self.betart, 0), colors=colors, lumped=bdrylumped)
        return A
    def computeMatrixJump(self, betart):
        dimension, dV, ndofs = self.mesh.dimension, self.mesh.dV, self.nunknowns()
        nloc, dofspercell = self.nlocal(), self.dofspercell()
        ci0 = self.mesh.cellsOfInteriorFaces[:,0]
        ci1 = self.mesh.cellsOfInteriorFaces[:,1]
        assert np.all(ci1>=0)
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        faces = self.mesh.faces[self.mesh.innerfaces]
        ind0 = npext.positionin(faces, self.mesh.simplices[ci0])
        ind1 = npext.positionin(faces, self.mesh.simplices[ci1])
        fi0 = np.take_along_axis(self.mesh.facesOfCells[ci0], ind0, axis=1)
        fi1 = np.take_along_axis(self.mesh.facesOfCells[ci1], ind1, axis=1)
        d = self.mesh.dimension
        massloc = barycentric.crbdryothers(d)
        A = sparse.coo_matrix((ndofs, ndofs))
        rows0 = fi0.repeat(nloc-1)
        cols0 = np.tile(fi0,nloc-1).reshape(-1)
        rows1 = fi1.repeat(nloc-1)
        cols1 = np.tile(fi1,nloc-1).reshape(-1)
        # mat = np.einsum('n,kl->nkl', 0.5*np.absolute(betart[self.mesh.innerfaces])*dS, massloc).ravel()
        mat = np.einsum('n,kl->nkl', -np.minimum(betart[self.mesh.innerfaces], 0)*dS, massloc).ravel()
        A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(ndofs, ndofs))
        A -= sparse.coo_matrix((mat, (rows0, cols1)), shape=(ndofs, ndofs))
        mat = np.einsum('n,kl->nkl', np.maximum(betart[self.mesh.innerfaces], 0)*dS, massloc).ravel()
        A -= sparse.coo_matrix((mat, (rows1, cols0)), shape=(ndofs, ndofs))
        A += sparse.coo_matrix((mat, (rows1, cols1)), shape=(ndofs, ndofs))
        return A
    def computeMatrixTransportLps(self, bdrylumped, colors):
        nfaces, ncells, nfaces, dim = self.mesh.nfaces, self.mesh.ncells, self.mesh.nfaces, self.mesh.dimension
        beta, mus = self.beta, np.full(dim+1,1.0/(dim+1))[np.newaxis,:]
        mat = np.einsum('n,njk,nk,ni -> nij', self.mesh.dV, self.cellgrads[:,:,:dim], beta, mus)
        A =  sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        A += self.computeMatrixLps(beta)
        A += self.computeBdryMassMatrix(coeff=-np.minimum(self.betart, 0), colors=colors, lumped=bdrylumped)
        return A
    def massDotSupg(self, b, f, coeff=1):
        dim, facesOfCells, dV = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.dV
        # beta, mus, deltas = self.beta, self.md.mus, self.md.deltas
        # cellgrads = self.cellgrads[:,:,:dim]
        # betagrad = np.einsum('njk,nk -> nj', cellgrads, beta)
        # r = np.einsum('n,ni->ni', deltas*dV*f[facesOfCells].mean(axis=1), betagrad)
        r = np.einsum('n,nk->nk', coeff*dV*f[facesOfCells].mean(axis=1), dim/(dim+1)-dim*self.md.mus)
        np.add.at(b, facesOfCells, r)
        return b
    # dotmat
    def massDotCell(self, b, f, coeff=1):
        assert f.shape[0] == self.mesh.ncells
        dimension, facesOfCells, dV = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.dV
        massloc = 1/(dimension+1)
        np.add.at(b, facesOfCells, (massloc*coeff*dV*f)[:, np.newaxis])
        return b
    def massDot(self, b, f, coeff=1):
        dim, facesOfCells, dV = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.dV
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        r = np.einsum('n,kl,nl->nk', coeff*dV, massloc, f[facesOfCells])
        np.add.at(b, facesOfCells, r)
        return b
    # rhs
    # postprocess
    def computeErrorL2Cell(self, solexact, uh):
        xc, yc, zc = self.mesh.pointsc.T
        ec = solexact(xc, yc, zc) - np.mean(uh[self.mesh.facesOfCells], axis=1)
        return np.sqrt(np.sum(ec**2* self.mesh.dV)), ec
    def computeErrorL2(self, solexact, uh):
        x, y, z = self.mesh.pointsf.T
        en = solexact(x, y, z) - uh
        Men = np.zeros_like(en)
        return np.sqrt( np.dot(en, self.massDot(Men,en)) ), en
    def computeErrorFluxL2(self, solexact, diffcell, uh):
        xc, yc, zc = self.mesh.pointsc.T
        graduh = np.einsum('nij,ni->nj', self.cellgrads, uh[self.mesh.facesOfCells])
        errv = 0
        for i in range(self.mesh.dimension):
            solxi = solexact.d(i, xc, yc, zc)
            errv += np.sum( diffcell*(solxi-graduh[:,i])**2* self.mesh.dV)
        return np.sqrt(errv)
    def computeBdryMean(self, u, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*u[faces])
        return mean/omega
    def comuteFluxOnRobin(self, u, faces, dS, uR, cR):
        uhmean =  np.sum(dS * u[faces])
        xf, yf, zf = self.mesh.pointsf[faces].T
        nx, ny, nz = np.mean(self.mesh.normals[faces], axis=0)
        if uR:
            try:
                uRmean =  np.sum(dS * uR(xf, yf, zf, nx, ny, nz))
            except:
                uRmean =  np.sum(dS * uR(xf, yf, zf))
        else: uRmean=0
        return cR*(uRmean-uhmean)
    def computeBdryNormalFlux(self, u, colors, bdrydata, bdrycond, diffcoff):
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            if color in bdrydata.bsaved.keys():
                bs, As = bdrydata.bsaved[color], bdrydata.Asaved[color]
                flux[i] = np.sum(As * u - bs)
            else:
                flux[i] = self.comuteFluxOnRobin(u, faces, dS, bdrycond.fct[color], bdrycond.param[color])
        return flux
    def formDiffusion(self, du, u, coeff):
        raise NotImplemented(f"formDiffusion")
    def computeRhsMass(self, b, rhs, mass):
        raise NotImplemented(f"computeRhsMass")
    def computeRhsCell(self, b, rhscell):
        raise NotImplemented(f"computeRhsCell")
    def computeRhsPoint(self, b, rhspoint):
        raise NotImplemented(f"computeRhsPoint")
    def computeRhsBoundary(self, b, bdryfct, colors):
        normals =  self.mesh.normals
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            if not color in bdryfct or bdryfct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            b[faces] += bdryfct[color](xf, yf, zf, nx, ny, nz) * dS
        return b
    def computeRhsBoundaryMass(self, b, bdrycond, types, mass):
        raise NotImplemented(f"")
    def computeBdryFct(self, u, colors):
        raise NotImplemented(f"")
    def computePointValues(self, u, colors):
        raise NotImplemented(f"")
    def computeMeanValues(self, u, colors):
        raise NotImplemented(f"")
    def computeMeanValue(self, u, color):
        raise NotImplemented(f"")
    #------------------------------
    def test(self):
        import scipy.sparse.linalg as splinalg
        colors = mesh.bdrylabels.keys()
        bdrydata = self.prepareBoundary(colorsdir=colors)
        A = self.computeMatrixDiffusion(coeff=1)
        A = self.matrixBoundary(A, bdrydata=bdrydata, method='strong')
        b = np.zeros(self.nunknowns())
        rhs = np.vectorize(lambda x,y,z: 1)
        fp1 = self.interpolateCell(rhs)
        self.massDotCell(b, fp1, coeff=1)
        b = self.vectorBoundaryZero(b, bdrydata)
        return self.tonode(splinalg.spsolve(A, b))


# ------------------------------------- #

if __name__ == '__main__':
    from simfempy.meshes import testmeshes
    from simfempy.meshes import plotmesh
    import matplotlib.pyplot as plt

    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = CR1(mesh=mesh)
    u = fem.test()
    plotmesh.meshWithBoundaries(mesh)
    plotmesh.meshWithData(mesh, point_data={'u':u}, title="P1 Test", alpha=1)
    plt.show()
