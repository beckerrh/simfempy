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
class FemBV0(object):
    """
    on suppose que  self.mesh.edgesOfCell[ic, kk] et oppose à elem[ic,kk] !!!
    """
    def __init__(self, mesh=None, massproj=False):
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

        # pour RT
        dS = sigma * linalg.norm(normals[facesofcells], axis=2)/dim
        ps = p[simp][:,:,:dim]
        ps2 = np.transpose(ps, axes=(2,0,1))
        pc2 = np.repeat(pc[:,:dim].T[:, :, np.newaxis], nloc, axis=2)
        pd = pc2 -ps2
        rows = np.repeat((np.repeat(dim * np.arange(ncells), dim).reshape(ncells,dim) + np.arange(dim)).swapaxes(1,0),nloc)
        cols = np.tile(facesofcells.ravel(), dim)
        mat = np.einsum('ni, jni, n->jni', dS, pd, 1/dV)
        return  sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(dim*ncells, nfaces))


        # pour les tilde
        # rows = np.repeat((np.repeat(dim * np.arange(ncells), dim).reshape(ncells,dim) + np.arange(dim)).swapaxes(1,0),nloc)
        # cols = np.tile(facesofcells.ravel(), dim)
        # dS = linalg.norm(normals[facesofcells], axis=2)
        # mat = np.einsum('nij, ni, n->jni', normals[facesofcells][:,:,:dim], 1/dS, dV/(dim+1))
        # return  sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(dim*ncells, nfaces))

        # pour les hat
        dS = sigma * linalg.norm(normals[facesofcells], axis=2)
        ps = p[simp][:,:,:dim]
        ps2 = np.transpose(ps, axes=(2,0,1))
        pc2 = np.repeat(pc[:,:dim].T[:, :, np.newaxis], nloc, axis=2)
        pd = pc2 -ps2
        rows = np.repeat((np.repeat(dim * np.arange(ncells), dim).reshape(ncells,dim) + np.arange(dim)).swapaxes(1,0),nloc)
        cols = np.tile(facesofcells.ravel(), dim)
        scale = 1/dim
        mat = scale*np.einsum('ni, jni->jni', dS, pd)
        return  sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(dim*ncells, nfaces))

    def toCell(self, v):
        return self.Mtocell.dot(v)

    def constructMass(self, diffinvcell=None):
        # ncells, nfaces, normals = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals
        # cellsOfFaces, facesOfCells, dV = self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        # dim, nloc, p, pc, simp = self.mesh.dimension, self.nloc, self.mesh.points, self.mesh.pointsc, self.mesh.simplices
        #
        # assert dim == 2
        # dS2 = np.einsum('nik,nik->ni', normals[facesOfCells], normals[facesOfCells])
        # ps = p[simp][:, :, :dim]
        # ps2 = np.transpose(ps, axes=(2, 0, 1))
        # pc2 = np.repeat(pc[:, :dim].T[:, :, np.newaxis], nloc, axis=2)
        # pd = pc2 - ps2
        # # print("pd.shape", pd.shape)
        # a = (dim + 1) / dim ** 3
        # mat = a * dS2 * np.einsum('kni,kni,n->ni', pd, pd, diffinvcell / dV)
        # # print("pd**2", np.einsum('kni,kni->ni', pd, pd))
        # # print("dS2", dS2)
        # # print("mat", mat)
        # rows = facesOfCells.ravel()
        # A = sparse.coo_matrix((mat.ravel(), (rows, rows)), shape=(nfaces, nfaces)).tocsr()
        # print("A", A)
        # # self.mesh.plotWithNumbering()
        # return A

        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        dim, dV, nloc, simp = self.mesh.dimension, self.mesh.dV, self.nloc, self.mesh.simplices
        p, pc, pf = self.mesh.points, self.mesh.pointsc, self.mesh.pointsf

        dS = sigma*linalg.norm(normals[facesofcells], axis=2)
        ps = p[simp][:, :, :dim]
        ps2 = np.transpose(ps, axes=(2, 0, 1))
        pc2 = np.repeat(pc[:, :dim].T[:, :, np.newaxis], nloc, axis=2)
        pd = pc2 - ps2
        pf2 = pf[facesofcells][:,:,:dim]
        # print("pd", pd.shape)
        # print("pf2", pf2.shape)
        # print("dS", dS.shape)
        scale = 1/dim/dim

        mat  = np.einsum('kni, nik, nj, ni, n->nij', pd, pf2, dS, dS, 1/dV)
        mat -= np.einsum('kni, njk, nj, ni, n->nij', pd, ps, dS, dS, 1/dV)
        # mat  = np.einsum('kni, knj, ni, nj, n->nij', pd, pd, dS, dS, 1/dV)

        mat *= scale

        rows = np.repeat(facesofcells, self.nloc).ravel()
        cols = np.tile(facesofcells, self.nloc).ravel()
        A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(nfaces, nfaces)).tocsr()
        # print("A (BV)", A)
        return A

        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        dim, dV, nloc, p, pc, simp = self.mesh.dimension, self.mesh.dV, self.nloc, self.mesh.points, self.mesh.pointsc, self.mesh.simplices
        pf = self.mesh.pointsf[facesofcells][:,:,:dim]
        ps = p[simp][:, :, :dim]
        pd = pf - ps
        # print("pd", pd)
        # print("normal", normals[facesofcells][:,:,:dim])
        scale = 1/dim/(dim+1)
        mat = scale*np.einsum('nij, nij, ni->ni', pd, normals[facesofcells][:,:,:dim], sigma)
        rows = facesofcells.ravel()
        A = sparse.coo_matrix((mat.ravel(), (rows, rows)), shape=(nfaces, nfaces)).tocsr()
        # print("A", A)
        # stop
        return A



        dS = sigma * linalg.norm(normals[facesofcells], axis=2)
        ps = p[simp][:,:,:dim]
        ps2 = np.transpose(ps, axes=(2,0,1))
        pc2 = np.repeat(pc[:,:dim].T[:, :, np.newaxis], nloc, axis=2)
        pd = pc2 -ps2
        rows = np.repeat((np.repeat(dim * np.arange(ncells), dim).reshape(ncells,dim) + np.arange(dim)).swapaxes(1,0),nloc)
        cols = np.tile(facesofcells.ravel(), dim)
        # scale = (dim+1)/dim**3 * 4/5
        scale = (dim+1)/dim**2
        mat = scale*np.einsum('ni, ni, jni, jni, n->ni', dS, dS, pd, pd, diffinvcell/dV)
        rows = facesofcells.ravel()
        A = sparse.coo_matrix((mat.ravel(), (rows, rows)), shape=(nfaces, nfaces)).tocsr()
        # print("A", A)
        return A


    def constructDiv(self):
        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        rows = np.repeat(np.arange(ncells), self.nloc)
        cols = facesofcells.ravel()
        mat =  (sigma*linalg.norm(normals[facesofcells],axis=2)).ravel()
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
        scale = 14/15
        scale = 1
        return linalg.norm(self.mesh.normals[faces],axis=1) * ud * scale

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
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()

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
