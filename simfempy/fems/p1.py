# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from simfempy import fems, tools, meshes

#=================================================================#
class P1(fems.fem.Fem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dirichlet_strong = 1
        self.dirichlet_nitsche = 4
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.computeStencilCell(self.mesh.simplices)
        self.cellgrads = self.computeCellGrads()
    def prepareAdvection(self, beta, scale, method):
        rt = fems.rt0.RT0(self.mesh)
        betart = scale*rt.interpolate(beta)
        beta = rt.toCell(betart)
        convdata = fems.data.ConvectionData(beta=beta, betart=betart)
        if method == 'upwalg':
             return convdata 
        elif method == 'lps':
            self.mesh.constructInnerFaces()
            return convdata 
        elif method == 'supg':
            md = meshes.move.move_midpoints(self.mesh, beta)
            # self.md.plot(self.mesh, beta, type='midpoints')
        elif method == 'supg2':
            md = meshes.move.move_midpoints(self.mesh, beta, extreme=True)
            # self.md.plot(self.mesh, beta, type='midpoints')
        elif method == 'upwalg':
            pass
        elif method == 'upw':
            md = meshes.move.move_nodes(self.mesh, -beta)
            # self.md.plot(self.mesh, beta)
        elif method == 'upw2':
            md = meshes.move.move_nodes(self.mesh, -beta, second=True)
            # self.md.plot(self.mesh, beta)
        elif method == 'upwsides':
            self.mesh.constructInnerFaces()
            md = meshes.move.move_midpoints(self.mesh, -beta)
        else:
            raise ValueError(f"don't know {method=}")
        convdata.md = md
        return convdata
    def nlocal(self): return self.mesh.dimension+1
    def nunknowns(self): return self.mesh.nnodes
    def dofspercell(self): return self.mesh.simplices
    def computeCellGrads(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        scale = -1/self.mesh.dimension
        return scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
    def tonode(self, u): return u
    #  bc
    def prepareBoundary(self, colorsdir, colorsflux=[]):
        bdrydata = fems.data.BdryData()
        bdrydata.nodesdir={}
        # bdrydata.nodedirall = np.empty(shape=(0), dtype=np.int)
        bdrydata.nodedirall = np.empty(shape=(0), dtype=int)
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.nodesdir[color] = np.unique(self.mesh.faces[facesdir].flat[:])
            bdrydata.nodedirall = np.unique(np.union1d(bdrydata.nodedirall, bdrydata.nodesdir[color]))
        # print(f"{bdrydata.nodedirall=}")
        # bdrydata.nodesinner = np.setdiff1d(np.arange(self.mesh.nnodes, dtype=np.int),bdrydata.nodedirall)
        bdrydata.nodesinner = np.setdiff1d(np.arange(self.mesh.nnodes, dtype=int),bdrydata.nodedirall)
        bdrydata.nodesdirflux={}
        for color in colorsflux:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.nodesdirflux[color] = np.unique(self.mesh.faces[facesdir].ravel())
        return bdrydata
    def matrixBoundaryStrong(self, A, bdrydata, method):
        assert method != 'nitsche'
        nodesdir, nodedirall, nodesinner, nodesdirflux = bdrydata.nodesdir, bdrydata.nodedirall, bdrydata.nodesinner, bdrydata.nodesdirflux
        nnodes = self.mesh.nnodes
        for color, nodes in nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((nb, nnodes))
            for i in range(nb): help[i, nodes[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        bdrydata.A_inner_dir = A[nodesinner, :][:, nodedirall]
        help = np.ones((nnodes))
        help[nodedirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
        # A = help.dot(A.dot(help))
        diag = np.zeros((nnodes))
        if method == 'strong':
            diag[nodedirall] = 1.0
            diag = sparse.dia_matrix((diag, 0), shape=(nnodes, nnodes))
        else:
            bdrydata.A_dir_dir = self.dirichlet_strong*A[nodedirall, :][:, nodedirall]
            diag[nodedirall] = np.sqrt(self.dirichlet_strong)
            diag = sparse.dia_matrix((diag, 0), shape=(nnodes, nnodes))
            diag = diag.dot(A.dot(diag))
        A = help.dot(A)
        A += diag
        return A
    def vectorBoundaryStrong(self, b, bdrycond, bdrydata, method):
        assert method != 'nitsche'
        nodesdir, nodedirall, nodesinner, nodesdirflux = bdrydata.nodesdir, bdrydata.nodedirall, bdrydata.nodesinner, bdrydata.nodesdirflux
        x, y, z = self.mesh.points.T
        for color, nodes in nodesdirflux.items():
            bdrydata.bsaved[color] = b[nodes]
        if method == 'strong':
            for color, nodes in nodesdir.items():
                if color in bdrycond.fct:
                    dirichlet = bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    b[nodes] = dirichlet
                else:
                    b[nodes] = 0
            # b[nodesinner] -= bdrydata.A_inner_dir * b[nodedirall]
        else:
            help = np.zeros_like(b)
            for color, nodes in nodesdir.items():
                if color in bdrycond.fct:
                    dirichlet = bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    help[nodes] = dirichlet
            # b[nodesinner] -= bdrydata.A_inner_dir * help[nodedirall]
            b[nodedirall] = bdrydata.A_dir_dir * help[nodedirall]
        return b
    def vectorBoundaryStrongEqual(self, du, u, bdrydata):
        nodedirall = bdrydata.nodedirall
        du[nodedirall] = u[nodedirall]
    def vectorBoundaryStrongZero(self, du, bdrydata):
        du[bdrydata.nodedirall] = 0
    def formBoundary(self, du, u, bdrydata, dirichletmethod):
        assert(dirichletmethod=='new')
        nodedirall = bdrydata.nodedirall
        # du[nodedirall] += bdrydata.A_dir_dir.dot(u[nodedirall])
        du[nodedirall] += bdrydata.A_dir_dir*u[nodedirall]
    def computeRhsNitscheDiffusion(self, b, diffcoff, colorsdir, fp1, coeff=1):
        assert fp1.shape[0]==self.mesh.nnodes
        dim  = self.mesh.dimension
        massloc = tools.barycentric.tensor(d=dim - 1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        faces = self.mesh.bdryFaces(colorsdir)
        nodes, cells, normalsS = self.mesh.faces[faces], self.mesh.cellsOfFaces[faces,0], self.mesh.normals[faces,:dim]
        dS = linalg.norm(normalsS, axis=1)
        simp, dV = self.mesh.simplices[cells], self.mesh.dV[cells]
        dS *= self.dirichlet_nitsche * coeff * diffcoff[cells] * dS / dV
        r = np.einsum('n,kl,nl->nk', dS, massloc, fp1[nodes])
        np.add.at(b, nodes, r)
        cellgrads = self.cellgrads[cells, :, :dim]
        u = fp1[nodes].mean(axis=1)
        mat = np.einsum('f,fk,fik->fi', coeff*u*diffcoff[cells], normalsS, cellgrads)
        np.add.at(b, simp, -mat)
    def computeFormNitscheDiffusion(self, du, u, diffcoff, colorsdir):
        assert u.shape[0]==self.mesh.nnodes
        dim  = self.mesh.dimension
        massloc = simfempy.tools.barycentric.tensor(d=dim - 1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        faces = self.mesh.bdryFaces(colorsdir)
        nodes, cells, normalsS = self.mesh.faces[faces], self.mesh.cellsOfFaces[faces,0], self.mesh.normals[faces,:dim]
        dS = linalg.norm(normalsS, axis=1)
        simp, dV = self.mesh.simplices[cells], self.mesh.dV[cells]
        dS *= self.dirichlet_nitsche * diffcoff[cells] * dS / dV
        r = np.einsum('n,kl,nl->nk', dS, massloc, u[nodes])
        np.add.at(du, nodes, r)
        cellgrads = self.cellgrads[cells, :, :dim]
        um = u[nodes].mean(axis=1)
        mat = np.einsum('f,fk,fik->fi', um*diffcoff[cells], normalsS, cellgrads)
        np.add.at(du, simp, -mat)
        mat = np.einsum('f,fk,fjk,fj->f', diffcoff[cells]/dim, normalsS, cellgrads,u[simp]).repeat(dim).reshape(faces.shape[0],dim)
        np.add.at(du, nodes, -mat)
    def computeMatrixNitscheDiffusion(self, diffcoff, colorsdir, nitsche_param=None, lumped=True):
        if nitsche_param==None: nitsche_param=self.dirichlet_nitsche
        nnodes, ncells, dim, nlocal  = self.mesh.nnodes, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        dV = self.mesh.dV[cells]
        cellgrads = self.cellgrads[cells, :, :dim]
        simp = self.mesh.simplices[cells]
        facenodes = self.mesh.faces[faces]
        # ind = npext.positionin(facenodes, simp).astype(int)
        # fnind = np.take_along_axis(simp,ind,axis=1)
        # if not np.all(facenodes == fnind):
        #     raise ValueError(f"***not found***\n{facenodes=}\n{fnind=} {ind=}")
        cols = np.tile(simp,dim)
        rows = facenodes.repeat(dim+1)
        mat = np.einsum('f,fk,fjk,i->fij', coeff * diffcoff[cells]/dim, normalsS, cellgrads, np.ones(dim))
        # mat = np.repeat(mat,dim)
        # print(f"{cols.shape=} {rows.shape=} {mat.shape=}")
        AN = sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(nnodes, nnodes)).tocsr()
        massloc = tools.barycentric.tensor(d=dim-1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        # print(f"{massloc=}")
        mat = np.einsum('f,ij->fij', nitsche_param * dS**2/dV*diffcoff[cells], massloc)
        # mat = np.repeat(coeff * diffcoff[cells]/dS, dim)
        rows = np.repeat(facenodes,dim)
        cols = np.tile(facenodes,dim)
        AD = sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(nnodes, nnodes)).tocsr()
        return  - AN - AN.T + AD
    def computeBdryNormalFluxNitsche(self, u, colors, udir, diffcoff):
        flux= np.zeros(len(colors))
        nnodes, dim  = self.mesh.nnodes, self.mesh.dimension
        massloc = tools.barycentric.tensor(d=dim - 1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces,:dim]
            dS = linalg.norm(normalsS, axis=1)
            nodes = self.mesh.faces[faces]
            cells = self.mesh.cellsOfFaces[faces,0]
            simp = self.mesh.simplices[cells]
            cellgrads = self.cellgrads[cells, :, :dim]
            dV = self.mesh.dV[cells]
            flux[i] = np.einsum('nj,n,ni,nji->', u[simp], diffcoff[cells], normalsS, cellgrads)
            uD = u[nodes]-udir[nodes]
            dV = self.mesh.dV[cells]
            dS *= self.dirichlet_nitsche * diffcoff[cells] * dS / dV
            flux[i] -= np.einsum('n,kl,nl->', dS, massloc, uD)
        return flux
    # interpolate
    def interpolate(self, f):
        x, y, z = self.mesh.points.T
        return f(x, y, z)
    def interpolateBoundary(self, colors, f, lumped=False):
        """
        :param colors: set of colors to interpolate
        :param f: ditct of functions
        :return:
        """
        b = np.zeros(self.mesh.nnodes)
        for color in colors:
            if not color in f or not f[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            nx, ny, nz = normalsS.T
            nodes = np.unique(self.mesh.faces[faces].reshape(-1))
            x, y, z = self.mesh.points[nodes].T
            # constant normal !!
            nx, ny, nz = np.mean(normalsS, axis=0)
            try:
                b[nodes] = f[color](x, y, z, nx, ny, nz)
            except:
                b[nodes] = f[color](x, y, z)
        return b
    # matrices
    def computeMassMatrix(self, coeff=1, lumped=False):
        dim, dV, nnodes = self.mesh.dimension, self.mesh.dV, self.mesh.nnodes
        if lumped:
            mass = coeff/(dim+1)*dV.repeat(dim+1)
            rows = self.mesh.simplices.ravel()
            return sparse.coo_matrix((mass, (rows, rows)), shape=(nnodes, nnodes)).tocsr()
        massloc = barycentric.tensor(d=dim, k=2)
        mass = np.einsum('n,kl->nkl', coeff*dV, massloc).ravel()
        return sparse.coo_matrix((mass, (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
    def computeBdryMassMatrix(self, colors=None, coeff=1, lumped=False):
        nnodes = self.mesh.nnodes
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        if colors is None: colors = self.mesh.bdrylabels.keys()
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            if isinstance(coeff, dict):
                dS = linalg.norm(normalsS, axis=1)*coeff[color]
            else:
                dS = linalg.norm(normalsS, axis=1)*coeff[faces]
            nodes = self.mesh.faces[faces]
            if lumped:
                dS /= self.mesh.dimension
                rows = np.append(rows, nodes)
                cols = np.append(cols, nodes)
                mass = np.repeat(dS, self.mesh.dimension)
                mat = np.append(mat, mass)
            else:
                nloc = self.mesh.dimension
                rows = np.append(rows, np.repeat(nodes, nloc).ravel())
                cols = np.append(cols, np.tile(nodes, nloc).ravel())
                massloc = simfempy.tools.barycentric.tensor(d=self.mesh.dimension-1, k=2)
                mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).ravel())
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nnodes, nnodes)).tocsr()
    def computeMatrixTransportUpwindAlg(self, data):
        A =  self.computeMatrixTransportCellWise(data, type='centered')
        return tools.checkmmatrix.makeMMatrix(A)
    def computeMatrixTransportUpwindSides(self, betart):
        nnodes, nfaces, ncells, dim, dV = self.mesh.nnodes, self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.mesh.dV
        normalsS, cof, simp = self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.simplices
        dbS = linalg.norm(normalsS, axis=1)*betart/dim/(dim+1)

        innerfaces = self.mesh.innerfaces
        infaces = np.arange(nfaces)[innerfaces]
        ci0 = self.mesh.cellsOfInteriorFaces[:, 0]
        ci1 = self.mesh.cellsOfInteriorFaces[:, 1]
        rows0 = np.repeat(simp[ci0],dim).ravel()
        rows1 = np.repeat(simp[ci1],dim).ravel()
        cols = np.tile(self.mesh.faces[infaces], dim + 1).ravel()
        matloc = np.ones(shape=(dim,dim+1))
        mat = np.einsum('n,kl->nkl', dbS[infaces], matloc).ravel()
        A = sparse.coo_matrix((mat, (rows1, cols)), shape=(nnodes, nnodes))
        A -= sparse.coo_matrix((mat, (rows0, cols)), shape=(nnodes, nnodes))

        faces = self.mesh.bdryFaces()
        ci0 = self.mesh.cellsOfFaces[faces, 0]
        rows0 = np.repeat(simp[ci0],dim).ravel()
        cols = np.tile(self.mesh.faces[infaces], dim + 1).ravel()
        mat = np.einsum('n,kl->nkl', dbS[faces], matloc).ravel()
        A -= sparse.coo_matrix((mat, (rows0,cols)), shape=(nnodes, nnodes))

        A -= self.computeBdryMassMatrix(coeff=np.minimum(betart, 0), lumped=True)

        B = self.computeMatrixTransportCellWise(type='centered')

        A = A.tocsr()
        B = B.tocsr()
        if not np.allclose(A.A,B.A):
            raise ValueError(f"{A.diagonal()=}\n{B.diagonal()=}\n{A.todense()=}\n{B.todense()=}")

        return A.tocsr()
    def computeMatrixTransportUpwind(self, data, method):
        if method=='upwsides': return self.computeMatrixTransportUpwindSides()
        self.masslumped = self.computeMassMatrix(coeff=1, lumped=True)
        beta, mus, cells, deltas = data.beta, data.md.mus, data.md.cells, data.md.deltas
        nnodes, simp= self.mesh.nnodes, self.mesh.simplices
        m = data.md.mask()
        if hasattr(data.md,'cells2'):
            m2 =  data.md.mask2()
            m = data.md.maskonly1()
            print(f"{nnodes=} {np.sum(data.md.mask())=} {np.sum(m2)=} {np.sum(m)=}")
        ml = self.masslumped.diagonal()[m]/deltas[m]
        rows = np.arange(nnodes)[m]
        A = sparse.coo_matrix((ml,(rows,rows)), shape=(nnodes, nnodes))
        mat = mus[m]*ml[:,np.newaxis]
        rows = rows.repeat(simp.shape[1])
        cols = simp[cells[m]]
        A -=  sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(nnodes, nnodes))
        if hasattr(data.md,'cells2'):
            cells2 = data.md.cells2
            delta1 = data.md.deltas[m2]
            delta2 = data.md.deltas2[m2]
            mus2 = data.md.mus2
            c0 = (1+delta1/(delta1+delta2))/delta1
            c1 = -(1+delta1/delta2)/delta1
            c2 = -c0-c1
            ml = self.masslumped.diagonal()[m2]
            rows = np.arange(nnodes)[m2]
            A += sparse.coo_matrix((c0*ml,(rows,rows)), shape=(nnodes, nnodes))
            mat = mus[m2]*ml[:,np.newaxis]*c1[:,np.newaxis]
            rows1 = rows.repeat(simp.shape[1])
            cols = simp[cells[m2]]
            A +=  sparse.coo_matrix((mat.ravel(), (rows1.ravel(), cols.ravel())), shape=(nnodes, nnodes))
            mat = mus2[m2] * ml[:, np.newaxis] * c2[:, np.newaxis]
            rows2 = rows.repeat(simp.shape[1])
            cols = simp[cells2[m2]]
            A += sparse.coo_matrix((mat.ravel(), (rows2.ravel(), cols.ravel())), shape=(nnodes, nnodes))
        A += self.computeBdryMassMatrix(coeff=-np.minimum(data.betart, 0), lumped=True)
        # A = checkmmatrix.makeMMatrix(A)
        w1, w2 = tools.checkmmatrix.checkMmatrix(A)
        print(f"A {w1=}\n{w2=}")
        return A.tocsr()
    def computeMatrixTransportSupg(self, data, method):
        return self.computeMatrixTransportCellWise(data, type='supg')
    def computeMatrixTransportLps(self, data):
        A = self.computeMatrixTransportCellWise(data, type='centered')
        A += self.computeMatrixLps(data.beta)
        return A
    def computeMatrixTransportCellWise(self, data, type):
        nnodes, ncells, nfaces, dim = self.mesh.nnodes, self.mesh.ncells, self.mesh.nfaces, self.mesh.dimension
        if type=='centered':
            beta, mus = data.beta, np.full(dim+1,1.0/(dim+1))
            mat = np.einsum('n,njk,nk,i -> nij', self.mesh.dV, self.cellgrads[:,:,:dim], beta, mus)
            A =  sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
        elif type=='supg':
            beta, mus = data.beta, data.md.mus
            mat = np.einsum('n,njk,nk,ni -> nij', self.mesh.dV, self.cellgrads[:,:,:dim], beta, mus)
            A =  sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
        else: raise ValueError(f"unknown type {type=}")
        A -= self.computeBdryMassMatrix(coeff=np.minimum(data.betart, 0), lumped=True)
        return A
    def computeMassMatrixSupg(self, xd, data, coeff=1):
        dim, dV, nnodes, xK = self.mesh.dimension, self.mesh.dV, self.mesh.nnodes, self.mesh.pointsc
        massloc = tools.barycentric.tensor(d=dim, k=2)
        mass = np.einsum('n,ij->nij', coeff*dV, massloc)
        massloc = tools.barycentric.tensor(d=dim, k=1)
        # marche si xd = xK + delta*betaC
        # mass += np.einsum('n,nik,nk,j -> nij', coeff*delta*dV, self.cellgrads[:,:,:dim], betaC, massloc)
        mass += np.einsum('n,nik,nk,j -> nij', coeff*dV, self.cellgrads[:,:,:dim], xd[:,:dim]-xK[:,:dim], massloc)
        return sparse.coo_matrix((mass.ravel(), (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
    # dotmat
    def formDiffusion(self, du, u, coeff):
        graduh = np.einsum('nij,ni->nj', self.cellgrads, u[self.mesh.simplices])
        graduh = np.einsum('ni,n->ni', graduh, self.mesh.dV*coeff)
        # du += np.einsum('nj,nij->ni', graduh, self.cellgrads)
        raise ValueError(f"graduh {graduh.shape} {du.shape}")
        return du
    def massDotCell(self, b, f, coeff=1):
        assert f.shape[0] == self.mesh.ncells
        dimension, simplices, dV = self.mesh.dimension, self.mesh.simplices, self.mesh.dV
        massloc = 1/(dimension+1)
        np.add.at(b, simplices, (massloc*coeff*dV*f)[:, np.newaxis])
        return b
    def massDot(self, b, f, coeff=1):
        dim, simplices, dV = self.mesh.dimension, self.mesh.simplices, self.mesh.dV
        massloc = tools.barycentric.tensor(d=dim, k=2)
        r = np.einsum('n,kl,nl->nk', coeff * dV, massloc, f[simplices])
        np.add.at(b, simplices, r)
        return b
    def massDotSupg(self, b, f, data, coeff=1):
        dim, simplices, dV = self.mesh.dimension, self.mesh.simplices, self.mesh.dV
        r = np.einsum('n,nk,n->nk', coeff*dV, data.md.mus-1/(dim+1), f[simplices].mean(axis=1))
        np.add.at(b, simplices, r)
        return b
    def massDotBoundary(self, b, f, colors=None, coeff=1, lumped=True):
        if colors is None: colors = self.mesh.bdrylabels.keys()
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            nodes = self.mesh.faces[faces]
            if isinstance(coeff, (int,float)): dS *= coeff
            elif isinstance(coeff, dict): dS *= coeff[color]
            else:
                assert coeff.shape[0]==self.mesh.nfaces
                dS *= coeff[faces]
            # print(f"{scalemass=}")
            if lumped:
                np.add.at(b, nodes, f[nodes]*dS[:,np.newaxis]/self.mesh.dimension)
            else:
                massloc = tools.barycentric.tensor(d=self.mesh.dimension-1, k=2)
                r = np.einsum('n,kl,nl->nk', dS, massloc, f[nodes])
                np.add.at(b, nodes, r)
        return b
    # rhs
    def computeRhsMass(self, b, rhs, mass):
        if rhs is None: return b
        x, y, z = self.mesh.points.T
        b += mass * rhs(x, y, z)
        return b
    def computeRhsCell(self, b, rhscell):
        if rhscell is None: return b
        scale = 1 / (self.mesh.dimension + 1)
        for label, fct in rhscell.items():
            if fct is None: continue
            cells = self.mesh.cellsoflabel[label]
            xc, yc, zc = self.mesh.pointsc[cells].T
            bC = scale * fct(xc, yc, zc) * self.mesh.dV[cells]
            # print("bC", bC)
            np.add.at(b, self.mesh.simplices[cells].T, bC)
        return b
    def computeRhsPoint(self, b, rhspoint):
        if rhspoint is None: return b
        for label, fct in rhspoint.items():
            if fct is None: continue
            points = self.mesh.verticesoflabel[label]
            xc, yc, zc = self.mesh.points[points].T
            # print("xc, yc, zc, f", xc, yc, zc, fct(xc, yc, zc))
            b[points] += fct(xc, yc, zc)
        return b
    def computeRhsBoundary(self, b, bdryfct, colors):
        normals =  self.mesh.normals
        scale = 1 / self.mesh.dimension
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            if not color in bdryfct or bdryfct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            assert(dS.shape[0] == len(faces))
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            bS = scale * bdryfct[color](xf, yf, zf, nx, ny, nz) * dS
            np.add.at(b, self.mesh.faces[faces].T, bS)
        return b
    def computeRhsBoundaryMass(self, b, bdrycond, types, mass):
        normals =  self.mesh.normals
        help = np.zeros(self.mesh.nnodes)
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] not in types: continue
            if not color in bdrycond.fct or bdrycond.fct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            nx, ny, nz = normalsS.T
            assert(dS.shape[0] == len(faces))
            nodes = np.unique(self.mesh.faces[faces].reshape(-1))
            x, y, z = self.mesh.points[nodes].T
            # constant normal !!
            nx, ny, nz = np.mean(normalsS, axis=0)
            help[nodes] = bdrycond.fct[color](x, y, z, nx, ny, nz)
        # print("help", help)
        b += mass*help
        return b
    # postprocess
    def computeErrorL2Cell(self, solexact, uh):
        xc, yc, zc = self.mesh.pointsc.T
        ec = solexact(xc, yc, zc) - np.mean(uh[self.mesh.simplices], axis=1)
        return np.sqrt(np.sum(ec**2* self.mesh.dV)), ec
    def computeErrorL2(self, solexact, uh):
        x, y, z = self.mesh.points.T
        en = solexact(x, y, z) - uh
        Men = np.zeros_like(en)
        return np.sqrt( np.dot(en, self.massDot(Men,en)) ), en
    def computeErrorFluxL2(self, solexact, uh, diffcell=None):
        xc, yc, zc = self.mesh.pointsc.T
        graduh = np.einsum('nij,ni->nj', self.cellgrads, uh[self.mesh.simplices])
        errv = 0
        for i in range(self.mesh.dimension):
            solxi = solexact.d(i, xc, yc, zc)
            if diffcell is None: errv += np.sum((solxi - graduh[:, i]) ** 2 * self.mesh.dV)
            else: errv += np.sum( diffcell*(solxi-graduh[:,i])**2* self.mesh.dV)
        return np.sqrt(errv)
    def computeBdryMean(self, u, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*np.mean(u[self.mesh.faces[faces]],axis=1))
        return mean/omega
    def comuteFluxOnRobin(self, u, faces, dS, uR, cR):
        uhmean =  np.sum(dS * np.mean(u[self.mesh.faces[faces]], axis=1))
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
    def computeBdryFct(self, u, colors):
        nodes = np.empty(shape=(0), dtype=int)
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            nodes = np.unique(np.union1d(nodes, self.mesh.faces[faces].ravel()))
        return self.mesh.points[nodes], u[nodes]
    def computePointValues(self, u, colors):
        up = np.empty(len(colors))
        for i,color in enumerate(colors):
            nodes = self.mesh.verticesoflabel[color]
            up[i] = u[nodes]
        return up
    def computeMeanValues(self, u, colors):
        up = np.empty(len(colors))
        for i, color in enumerate(colors):
            up[i] = self.computeMeanValue(u,color)
        return up
    def computeMeanValue(self, u, color):
        cells = self.mesh.cellsoflabel[color]
        # print("umean", np.mean(u[self.mesh.simplices[cells]],axis=1))
        return np.sum(np.mean(u[self.mesh.simplices[cells]],axis=1)*self.mesh.dV[cells])

    #------------------------------
    def test(self):
        import scipy.sparse.linalg as splinalg
        colors = mesh.bdrylabels.keys()
        bdrydata = self.prepareBoundary(colorsdir=colors)
        A = self.computeMatrixDiffusion(coeff=1)
        A = self.matrixBoundaryStrong(A, bdrydata=bdrydata, method='strong')
        b = np.zeros(self.nunknowns())
        rhs = np.vectorize(lambda x,y,z: 1)
        fp1 = self.interpolateCell(rhs)
        self.massDotCell(b, fp1, coeff=1)
        b = self.vectorBoundaryStrongZero(b, bdrydata)
        return splinalg.spsolve(A, b)


# ------------------------------------- #

if __name__ == '__main__':
    from simfempy.meshes import testmeshes
    from simfempy.meshes import plotmesh
    import matplotlib.pyplot as plt

    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = P1(mesh=mesh)
    u = fem.test()
    plotmesh.meshWithBoundaries(mesh)
    plotmesh.meshWithData(mesh, point_data={'u':u}, title="P1 Test", alpha=1)
    plt.show()
