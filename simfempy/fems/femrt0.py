# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
try:
    from simfempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh
import simfempy.fems.bdrydata

#=================================================================#
class FemRT0(object):
    """
    on suppose que  self.mesh.edgesOfCell[ic, kk] et oppose à elem[ic,kk] !!!
    """
    def __init__(self, mesh=None, massproj=None):
        if mesh is not None:
            self.setMesh(mesh)
        self.massproj=massproj

    def setMesh(self, mesh):
        self.mesh = mesh
        self.nloc = self.mesh.dimension+1
        self.Mtocell = self.toCellMatrix()

    def toCellMatrix(self):
        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        dim, dV, nloc, p, pc, simp = self.mesh.dimension, self.mesh.dV, self.nloc, self.mesh.points, self.mesh.pointsc, self.mesh.simplices
        dS = sigma * linalg.norm(normals[facesofcells], axis=2)/dim
        ps = p[simp][:,:,:dim]
        ps2 = np.transpose(ps, axes=(2,0,1))
        pc2 = np.repeat(pc[:,:dim].T[:, :, np.newaxis], nloc, axis=2)
        pd = pc2 -ps2
        rows = np.repeat((np.repeat(dim * np.arange(ncells), dim).reshape(ncells,dim) + np.arange(dim)).swapaxes(1,0),nloc)
        cols = np.tile(facesofcells.ravel(), dim)
        mat = np.einsum('ni, jni, n->jni', dS, pd, 1/dV)
        return  sparse.coo_matrix((mat.flatten(), (rows.flatten(), cols.flatten())), shape=(dim*ncells, nfaces))

    def toCell(self, v):
        return self.Mtocell.dot(v)

    def constructMass(self, diffinvcell=None):
        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        dim, dV, nloc, simp = self.mesh.dimension, self.mesh.dV, self.nloc, self.mesh.simplices
        p, pc, pf = self.mesh.points, self.mesh.pointsc, self.mesh.pointsf


        if self.massproj is None:
            # RT
            scalea = 1 / dim / dim / (dim + 2) / (dim + 1)
            scaleb = 1 / dim / dim / (dim + 2) * (dim + 1)
            scalec = 1 / dim / dim
            dS = sigma * linalg.norm(normals[facesofcells], axis=2)
            x1 = scalea *np.einsum('nij,nij->n', p[simp], p[simp]) + scaleb* np.einsum('ni,ni->n', pc, pc)
            mat = np.einsum('ni,nj, n->nij', dS, dS, x1)
            x2 = scalec *np.einsum('nik,njk->nij', p[simp], p[simp])
            mat += np.einsum('ni,nj,nij->nij', dS, dS, x2)
            x3 = - scalec * np.einsum('nik,nk->ni', p[simp], pc)
            mat += np.einsum('ni,nj,ni->nij', dS, dS, x3)
            mat += np.einsum('ni,nj,nj->nij', dS, dS, x3)
            if diffinvcell is None:
                mat = np.einsum("nij, n -> nij", mat, 1/dV)
            else:
                mat = np.einsum("nij, n -> nij", mat, diffinvcell / dV  )
            rows = np.repeat(facesofcells, self.nloc).flatten()
            cols = np.tile(facesofcells, self.nloc).flatten()
            A = sparse.coo_matrix((mat.flatten(), (rows, cols)), shape=(nfaces, nfaces)).tocsr()
            # print("A (RT)", A)
            return A


        elif self.massproj=="L2":
            # RT avec projection L2
            dS = sigma * linalg.norm(normals[facesofcells], axis=2)/dim
            ps = p[simp][:,:,:dim]
            ps2 = np.transpose(ps, axes=(2,0,1))
            pc2 = np.repeat(pc[:,:dim].T[:, :, np.newaxis], nloc, axis=2)
            pd = pc2 -ps2
            mat = np.einsum('kni,knj, ni, nj, n->nij', pd, pd, dS, dS, diffinvcell / dV)
            rows = np.repeat(facesofcells, self.nloc).flatten()
            cols = np.tile(facesofcells, self.nloc).flatten()
            A = sparse.coo_matrix((mat.flatten(), (rows, cols)), shape=(nfaces, nfaces)).tocsr()
            # print("A (RT)", A)
            return A


        elif self.massproj=="Tilde-RT":
            # PG de type Tilde-RT
            dS = sigma * linalg.norm(normals[facesofcells], axis=2)
            ps = p[simp][:, :, :dim]
            ps2 = np.transpose(ps, axes=(2, 0, 1))
            pc2 = np.repeat(pc[:, :dim].T[:, :, np.newaxis], nloc, axis=2)
            pd = pc2 - ps2
            pf2 = pf[facesofcells][:, :, :dim]
            # print("pd", pd.shape)
            # print("pf2", pf2.shape)
            # print("dS", dS.shape)
            scale = 1 / dim / dim
            mat = np.einsum('kni, nik, nj, ni, n->nij', pd, pf2, dS, dS, 1 / dV)
            mat -= np.einsum('kni, njk, nj, ni, n->nij', pd, ps, dS, dS, 1 / dV)
            mat *= scale
            rows = np.repeat(facesofcells, self.nloc).flatten()
            cols = np.tile(facesofcells, self.nloc).flatten()
            A = sparse.coo_matrix((mat.flatten(), (rows, cols)), shape=(nfaces, nfaces)).tocsr()
            # print("A (BV)", A)
            return A


        elif self.massproj=="RT-Tilde":
            # PG de type Tilde-RT
            dS = sigma * linalg.norm(normals[facesofcells], axis=2)
            ps = p[simp][:, :, :dim]
            ps2 = np.transpose(ps, axes=(2, 0, 1))
            pc2 = np.repeat(pc[:, :dim].T[:, :, np.newaxis], nloc, axis=2)
            pd = pc2 - ps2
            pf2 = pf[facesofcells][:, :, :dim]
            scale = 1 / dim / dim
            mat = np.einsum('kni, nik, nj, ni, n->nij', pd, pf2, dS, dS, 1 / dV)
            mat -= np.einsum('kni, njk, nj, ni, n->nij', pd, ps, dS, dS, 1 / dV)
            mat *= scale
            rows = np.repeat(facesofcells, self.nloc).flatten()
            cols = np.tile(facesofcells, self.nloc).flatten()
            A = sparse.coo_matrix((mat.flatten(), (rows, cols)), shape=(nfaces, nfaces)).tocsr()
            # print("A (BV)", A)
            return A.T


        elif self.massproj=="Tilde-Tilde":
            # G de type Tilde-Tilde
            dS = sigma * linalg.norm(normals[facesofcells], axis=2)
            ps = p[simp][:, :, :dim]
            ps2 = np.transpose(ps, axes=(2, 0, 1))
            pc2 = np.repeat(pc[:, :dim].T[:, :, np.newaxis], nloc, axis=2)
            pd = pc2 - ps2
            scale = (dim + 1) / dim**3
            mat = scale * np.einsum('ni, ni, kni, kni, n->ni', dS, dS, pd, pd, diffinvcell / dV)
            rows = facesofcells.flatten()
            A = sparse.coo_matrix((mat.flatten(), (rows, rows)), shape=(nfaces, nfaces)).tocsr()
            # print("A", A)
            return A


        elif self.massproj=="Hat-RT":
            # G de type Tilde-Tilde
            scale = 1/(dim+1)*dim
            mat = np.tile(scale * diffinvcell*dV, nloc)
            rows = facesofcells.flatten()
            A = sparse.coo_matrix((mat.flatten(), (rows, rows)), shape=(nfaces, nfaces)).tocsr()
            # print("A", A)
            return A

        else:
            raise ValueError("unknown type self.massproj={}".format(self.massproj))

    def constructDiv(self):
        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        rows = np.repeat(np.arange(ncells), self.nloc)
        cols = facesofcells.flatten()
        mat =  (sigma*linalg.norm(normals[facesofcells],axis=2)).flatten()
        return  sparse.coo_matrix((mat, (rows, cols)), shape=(ncells, nfaces)).tocsr()

    def reconstruct(self, p, vc, diffinv):
        nnodes, ncells, dim = self.mesh.nnodes, self.mesh.ncells, self.mesh.dimension
        if len(diffinv.shape) != 1:
            raise NotImplemented("only scalar diffusion the time being")
        counts = np.bincount(self.mesh.simplices.reshape(-1))
        pn2 = np.zeros(nnodes)
        xdiff = self.mesh.points[self.mesh.simplices, :dim] - self.mesh.pointsc[:, np.newaxis,:dim]
        rows = np.repeat(self.mesh.simplices,dim)
        cols = np.repeat(dim*np.arange(ncells),dim*(dim+1)).reshape(ncells * (dim+1), dim) + np.arange(dim)
        mat = np.einsum("nij, n -> nij", xdiff, diffinv)
        A = sparse.coo_matrix((mat.reshape(-1), (rows.reshape(-1), cols.reshape(-1))), shape=(nnodes, dim*ncells)).tocsr()
        np.add.at(pn2, self.mesh.simplices.T, p)
        pn2 += A*vc
        pn2 /= counts
        return pn2


    def rhsDirichlet(self, faces, ud):
        return linalg.norm(self.mesh.normals[faces],axis=1) * ud

    def constructRobin(self, bdrycond, type):
        nfaces = self.mesh.nfaces
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] != type: continue
            if not bdrycond.param[color]: continue
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            cols = np.append(cols, faces)
            rows = np.append(rows, faces)
            mat = np.append(mat, 1/bdrycond.param[color] * dS)
        A = sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
        return A

    def matrixNeumann(self, A, B, bdrycond):
        nfaces = self.mesh.nfaces
        bdrydata = simfempy.fems.bdrydata.BdryData()
        bdrydata.facesneumann = np.empty(shape=(0), dtype=int)
        bdrydata.colorsneum = bdrycond.colorsOfType("Neumann")
        for color in bdrydata.colorsneum:
            bdrydata.facesneumann = np.unique(np.union1d(bdrydata.facesneumann, self.mesh.bdrylabels[color]))
        bdrydata.facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int), bdrydata.facesneumann)

        bdrydata.B_inner_neum = B[:, :][:, bdrydata.facesneumann]
        help = np.ones(nfaces)
        help[bdrydata.facesneumann] = 0
        help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
        B = B.dot(help)

        bdrydata.A_inner_neum = A[bdrydata.facesinner, :][:, bdrydata.facesneumann]
        bdrydata.A_neum_neum = A[bdrydata.facesneumann, :][:, bdrydata.facesneumann]
        help2 = np.zeros((nfaces))
        help2[bdrydata.facesneumann] = 1
        help2 = sparse.dia_matrix((help2, 0), shape=(nfaces, nfaces))
        A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))

        return bdrydata, A, B
