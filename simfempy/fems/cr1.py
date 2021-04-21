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
    # def __init__(self, mesh=None, dirichletmethod='trad'):
        super().__init__(**kwargs)
        # super().__init__(mesh=mesh, dirichletmethod=dirichletmethod)
        self.dirichlet_al = 10
        self.dirichlet_nitsche = 4
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.computeStencilCell(self.mesh.facesOfCells)
        self.cellgrads = self.computeCellGrads()
    def nlocal(self): return self.mesh.dimension+1
    def nunknowns(self): return self.mesh.nfaces
    def dofspercell(self): return self.mesh.facesOfCells
    def tonode(self, u):
        unodes = np.zeros(self.mesh.nnodes)
        if u.shape[0] != self.mesh.nfaces: raise ValueError(f"{u.shape=} {self.mesh.nfaces=}")
        scale = self.mesh.dimension
        np.add.at(unodes, self.mesh.simplices.T, np.sum(u[self.mesh.facesOfCells], axis=1))
        np.add.at(unodes, self.mesh.simplices.T, -scale*u[self.mesh.facesOfCells].T)
        countnodes = np.zeros(self.mesh.nnodes, dtype=int)
        np.add.at(countnodes, self.mesh.simplices.T, 1)
        unodes /= countnodes
        return unodes
    def prepareAdvection(self, beta, scale):
        rt = fems.rt0.RT0(self.mesh)
        self.betart = scale*rt.interpolate(beta)
        self.beta = rt.toCell(self.betart)
        if self.stab == 'supg':
            self.md = move.move_midpoints(self.mesh, self.beta)
            # self.md.plot(self.mesh, self.beta, type='midpoints')
        elif self.stab == 'supg2':
            self.md = move.move_midpoints(self.mesh, self.beta, extreme=True)
            # self.md.plot(self.mesh, self.beta, type='midpoints')
        elif self.stab == 'upw':
            self.md = move.move_sides(self.mesh, -self.beta)
            self.md.plot(self.mesh, self.beta, type='sides')
        elif self.stab == 'upw2':
            self.md = move.move_sides(self.mesh, -self.beta, second=True)
            self.md.plot(self.mesh, self.beta, type='sides')
        else:
            raise ValueError(f"don't know {self.stab=}")
    def computeCellGrads(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        return (normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
    def prepareStab(self):
        self.computeStencilInnerSidesCell(self.mesh.facesOfCells)
    # strong bc
    def prepareBoundary(self, colorsdir, colorsflux=[]):
        if self.dirichletmethod == 'nitsche': return
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
        if self.dirichletmethod != 'nitsche': return
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
        if self.dirichletmethod != 'nitsche': return A
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
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            # print(f"{u[facesOfCell[cells]].shape=}")
            flux[i] = np.einsum('fj,f,fi,fji->', u[facesOfCell[cells]], diffcoff[cells], normalsS, self.cellgrads[cells, :, :dim])
            dirichlet = bdrycond.fct[color]
            uD = u[faces]
            if color in bdrycond.fct:
                uD -= dirichlet(x[faces], y[faces], z[faces])
            ind = npext.positionin(faces, self.mesh.facesOfCells[cells]).astype(int)
            # print(f"{self.cellgrads[cells, ind, :dim].shape=}")
            flux[i] -= self.dirichlet_nitsche*np.einsum('f,fi,fi->', uD * diffcoff[cells], normalsS, self.cellgrads[cells, ind, :dim])
        return flux
    def vectorBoundaryZero(self, du, bdrydata):
        if self.dirichletmethod == 'nitsche': return du
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        du[facesdirall] = 0
        return du
    def vectorBoundary(self, b, bdrycond, bdrydata):
        if self.dirichletmethod == 'nitsche': return b
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
        if self.dirichletmethod == 'trad':
            b[facesdirall] = help[facesdirall]
        else:
            b[facesdirall] += bdrydata.A_dir_dir * help[facesdirall]
        return b
    def matrixBoundary(self, A, bdrydata):
        if self.dirichletmethod == 'nitsche': return A, bdrydata
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        nfaces = self.mesh.nfaces
        for color, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        bdrydata.A_inner_dir = A[facesinner, :][:, facesdirall]
        if self.dirichletmethod == 'trad':
            help = np.ones((nfaces))
            help[facesdirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help))
            help = np.zeros((nfaces))
            help[facesdirall] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A += help
        else:
            bdrydata.A_dir_dir = self.dirichlet_al*A[facesdirall, :][:, facesdirall]
            help = np.ones((nfaces))
            help[facesdirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            help2 = np.zeros((nfaces))
            help2[facesdirall] = np.sqrt(self.dirichlet_al)
            help2 = sparse.dia_matrix((help2, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A
    # interpolate
    def interpolate(self, f):
        x, y, z = self.mesh.pointsf.T
        return f(x, y, z)
    def interpolateBoundary(self, colors, f):
        """
        :param colors: set of colors to interpolate
        :param f: ditct of functions
        :return:
        """
        b = np.zeros(self.mesh.nfaces)
        for color in colors:
            if not color in f or not f[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            nx, ny, nz = normalsS.T
            x, y, z = self.mesh.pointsf[faces].T
            # constant normal on whole boundary part !!
            # nx, ny, nz = np.mean(normalsS, axis=0)
            try:
                b[faces] = f[color](x, y, z, nx, ny, nz)
            except:
                b[faces] = f[color](x, y, z)
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
        nfaces = self.mesh.nfaces
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            if isinstance(coeff, dict):
                scalemass = coeff[color]
                dS = linalg.norm(normalsS, axis=1)
            else:
                scalemass = 1
                dS = linalg.norm(normalsS, axis=1)*coeff[faces]
            if lumped:
                rows, cols, mat=self.computeBdryMassMatrixColorLumped(rows, cols, mat, faces, scalemass*dS)
            else:
                rows, cols, mat=self.computeBdryMassMatrixColor(rows, cols, mat, faces, scalemass*dS)
        # print(f"{mat=}")
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
    def computeBdryMassMatrixColorLumped(self, rows, cols, mat, faces, dS):
        cols = np.append(cols, faces)
        rows = np.append(rows, faces)
        mat = np.append(mat, dS)
        return rows, cols, mat
    def computeBdryMassMatrixColor(self, rows, cols, mat, faces, dS):
        raise NotImplemented(f"is wrong")
        self.computeBdryMassMatrixColorLumped(rows, cols, mat, faces, dS)
        ci = self.mesh.cellsOfFaces[faces][:,0]
        assert np.all(faces == self.mesh.facesOfCells[ci][:,-1])
        # print(f"{faces=}")
        fi = self.mesh.facesOfCells[ci][:,:-1]
        # print(f"{fi=}")
        d = self.mesh.dimension
        massloc = barycentric.crbdryothers(d)
        # print(f"{massloc=}")
        cols = np.append(cols, fi.repeat(d))
        rows = np.append(rows, np.tile(fi,d).ravel())
        mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).ravel())
        return rows, cols, mat
    def massDotBoundary(self, b, f, colors=None, coeff=1, lumped=False):
        if colors is None: colors = self.mesh.bdrylabels.keys()
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            if isinstance(coeff, (int,float)): scalemass = coeff
            elif isinstance(coeff, dict): scalemass = coeff[color]
            else:
                assert coeff.shape[0]==self.mesh.nfaces
                scalemass = 1
                dS *= coeff[faces]
            b[faces] += scalemass *dS*f[faces]
            return b
            # le suivant ne peut marcher...
            print(f"{lumped=}")
            if not lumped:
                ci = self.mesh.cellsOfFaces[faces][:, 0]
                fi = self.mesh.facesOfCells[ci][:, :-1]
                d = self.mesh.dimension
                massloc = barycentric.crbdryothers(d)
                r = np.einsum('n,kl,nl->nk', dS, massloc, f[fi])
                np.add.at(b, fi, r)
        return b
    def computeMassMatrixSupg(self, xd, coeff=1):
        raise NotImplemented(f"computeMassMatrixSupg")
    def computeMatrixTransportUpwind(self, bdrylumped, colors):
        self.masslumped = self.computeMassMatrix(coeff=1, lumped=True)
        beta, mus, cells, deltas = self.beta, self.md.mus, self.md.cells, self.md.deltas
        nfaces, fOC, d = self.mesh.nfaces, self.mesh.facesOfCells, self.mesh.dimension
        m = self.md.mask()
        if hasattr(self.md,'cells2'):
            m2 =  self.md.mask2()
            m = self.md.maskonly1()
            print(f"{nfaces=} {np.sum(self.md.mask())=} {np.sum(m2)=} {np.sum(m)=}")
        ml = self.masslumped.diagonal()[m]/deltas[m]
        rows = np.arange(nfaces)[m]
        A = sparse.coo_matrix((ml,(rows,rows)), shape=(nfaces, nfaces))
        mat = (d/(d-1)-d*mus[m])*ml[:,np.newaxis]
        # mat = ((mus[m]-1)/d)*ml[:,np.newaxis]
        rows = rows.repeat(fOC.shape[1])
        cols = fOC[cells[m]]
        A -=  sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(nfaces, nfaces))
        if hasattr(self.md,'cells2'):
            raise NotImplemented()
            cells2 = self.md.cells2
            delta1 = self.md.deltas[m2]
            delta2 = self.md.deltas2[m2]
            mus2 = self.md.mus2
            c0 = (1+delta1/(delta1+delta2))/delta1
            c1 = -(1+delta1/delta2)/delta1
            c2 = -c0-c1
            ml = self.masslumped.diagonal()[m2]
            rows = np.arange(nfaces)[m2]
            A += sparse.coo_matrix((c0*ml,(rows,rows)), shape=(nfaces, nfaces))
            mat = mus[m2]*ml[:,np.newaxis]*c1[:,np.newaxis]
            rows1 = rows.repeat(simp.shape[1])
            cols = simp[cells[m2]]
            A +=  sparse.coo_matrix((mat.ravel(), (rows1.ravel(), cols.ravel())), shape=(nfaces, nfaces))
            mat = mus2[m2] * ml[:, np.newaxis] * c2[:, np.newaxis]
            rows2 = rows.repeat(simp.shape[1])
            cols = simp[cells2[m2]]
            A += sparse.coo_matrix((mat.ravel(), (rows2.ravel(), cols.ravel())), shape=(nfaces, nfaces))
        print(f"{A.diagonal()=}")
        A += self.computeBdryMassMatrix(coeff=-np.minimum(self.betart, 0), colors=colors, lumped=bdrylumped)
        print(f"{A.diagonal()=}")
        return A.tocsr()
    def computeMatrixTransportSupg(self, bdrylumped, colors):
        beta, mus = self.beta, self.md.mus
        nfaces, dim, dV = self.mesh.nfaces, self.mesh.dimension, self.mesh.dV
        mat = np.einsum('n,njk,nk,ni -> nij', dV, self.cellgrads[:,:,:dim], beta, 1-dim*mus)
        mat -= np.einsum('n,njk,nk,i -> nij', dV, self.cellgrads[:,:,:dim], beta, 1/(dim+1)*np.ones(dim+1))
        # mat = np.einsum('n,njk,nk,i -> nij', dV, self.cellgrads[:,:,:dim], beta, 1/(dim+1)*np.ones(dim+1))
        A =  sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        if self.stab =='lps':
            A += self.computeMatrixLps(beta)
        A += self.computeBdryMassMatrix(coeff=-np.minimum(self.betart, 0), colors=colors, lumped=bdrylumped)
        return A
    # def computeMatrixTransport(self, lps):
    #     beta, betaC, ld = self.supdata['convection'], self.supdata['convectionC'], self.supdata['lam']
    #     ncells, nfaces, dim = self.mesh.ncells, self.mesh.nfaces, self.mesh.dimension
    #     if beta.shape != (nfaces,): raise TypeError(f"beta has wrong dimension {beta.shape=} expected {nfaces=}")
    #     if ld.shape != (ncells, dim+1): raise TypeError(f"ld has wrong dimension {ld.shape=}")
    #     mat = np.einsum('n,njk,nk,ni -> nij', self.mesh.dV, self.cellgrads[:,:,:dim], betaC, -dim*ld+1)
    #     A =  sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
    #     if lps:
    #         A += self.computeMatrixLps(betaC)
    #     return A
    #     # print(f"transport {A.toarray()=}")
    #     # B = self.computeBdryMassMatrix(coeff=-np.minimum(beta,0), colors=colors)
    #     # print(f"transport {B.toarray()=}")
    #     # return A+B
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
    def massDotSupg(self, b, f, coeff=1):
        # return b
        if self.stab[:4] != 'supg': return b
        dim, facesOfCells, dV = self.mesh.dimension, self.mesh.simplices, self.mesh.dV
        r = np.einsum('n,nk,n->nk', coeff*dV, 1-dim*self.md.mus-1/(dim+1), f[facesOfCells].mean(axis=1))
        np.add.at(b, facesOfCells, r)
        return b
        if self.stab[:4] != 'supg': return b
        xd = self.supdata['xd']
        dim, facesOfCells, points, dV, xK = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.points, self.mesh.dV, self.mesh.pointsc
        fm = f[facesOfCells].mean(axis=1)
        # marche si xd = xK + delta*betaC
        # r = np.einsum('n,nik,nk -> ni', delta*dV*fm, self.cellgrads[:,:,:dim], betaC)
        r = np.einsum('n,nik,nk -> ni', coeff*dV*fm, self.cellgrads[:,:,:dim], xd[:,:dim]-xK[:,:dim])
        # print(f"{r=}")
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
        if self.dirichletmethod == 'nitsche':
            return self.computeBdryNormalFluxNitsche(u, colors, bdrycond, diffcoff)
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
        scale = 1
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            if not color in bdryfct or bdryfct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            b[faces] += scale * bdryfct[color](xf, yf, zf, nx, ny, nz) * dS
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
        A = self.matrixBoundary(A, bdrydata=bdrydata)
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
