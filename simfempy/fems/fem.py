# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import numpy.linalg as linalg
from simfempy import fems
from simfempy.meshes.simplexmesh import SimplexMesh
import scipy.sparse as sparse


#=================================================================#
class Fem(object):
    def __init__(self, mesh=None):
        if mesh is not None: self.setMesh(mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
        self.nloc = self.nlocal()
    def computeStencilCell(self, dofspercell):
        self.cols = np.tile(dofspercell, self.nloc).reshape(-1)
        self.rows = np.repeat(dofspercell, self.nloc).reshape(-1)
        #Alternative
        # self.rows = dofspercell.repeat(self.nloc).reshape(self.mesh.ncells, self.nloc, self.nloc)
        # self.cols = self.rows.swapaxes(1, 2)
        # self.cols = self.cols.reshape(-1)
        # self.rows = self.rows.reshape(-1)

    # def computeStencilInnerSidesCell(self, dofspercell):
    #     nloc, faces, cellsOfFaces = self.nloc, self.mesh.faces, self.mesh.cellsOfFaces
    #     # print(f"{faces=}")
    #     # print(f"{cellsOfFaces=}")
    #     innerfaces = cellsOfFaces[:,1]>=0
    #     cellsOfInteriorFaces= cellsOfFaces[innerfaces]
    #     self.cellsOfInteriorFaces = cellsOfInteriorFaces
    #     self.innerfaces = innerfaces
    #     return
    #     # print(f"{innerfaces=}")
    #     print(f"{cellsOfInteriorFaces=}")
    #     raise NotImplementedError(f"no")
    #     ncells, nloc = dofspercell.shape[0], dofspercell.shape[1]
    #     print(f"{ncells=} {nloc=}")
    #     print(f"{dofspercell[cellsOfInteriorFaces,:].shape=}")
    #     rows = dofspercell[cellsOfInteriorFaces,:].repeat(nloc)
    #     cols = np.tile(dofspercell[cellsOfInteriorFaces,:],nloc)
    #     print(f"{rows=}")
    #     print(f"{cols=}")

    def interpolateCell(self, f):
        if isinstance(f, dict):
            b = np.zeros(self.mesh.ncells)
            for label, fct in f.items():
                if fct is None: continue
                cells = self.mesh.cellsoflabel[label]
                xc, yc, zc = self.mesh.pointsc[cells].T
                b[cells] = fct(xc, yc, zc)
            return b
        else:
            xc, yc, zc = self.mesh.pointsc.T
            return f(xc, yc, zc)
    def plotBetaDownwind(self):
        import matplotlib.pyplot as plt
        from simfempy.meshes import plotmesh
        beta, betaC, mesh = self.supdata['convection'], self.supdata['convectionC'], self.mesh
        celldata = {f"beta": [betaC[:, i] for i in range(mesh.dimension)]}
        fig, axs = plotmesh.meshWithData(mesh, quiver_cell_data=celldata, plotmesh=True)
        xd, ld, delta = self.downWind(beta)
        axs[0, 0].plot(xd[:, 0], xd[:, 1], 'or')
        xd, ld, delta = self.downWind(beta, method='supg2')
        axs[0, 0].plot(xd[:, 0], xd[:, 1], 'xb')
        plt.show()
    def downWind(self, beta, method='supg'):
        # beta is supposed RT0
        dim, ncells, fofc, sigma = self.mesh.dimension, self.mesh.ncells, self.mesh.facesOfCells, self.mesh.sigma
        normals, dV = self.mesh.normals, self.mesh.dV
        # method = 'centered'
        if method=='centered':
            lamd = np.ones((ncells,dim+1)) / (dim + 1)
        elif method=='supg':
            dS = linalg.norm(normals[fofc],axis=2)
            # print(f"{dS.shape=}")
            vs = beta[fofc]*sigma*dS/dV[:,np.newaxis]/dim
            vp = np.maximum(vs, 0)
            ips = vp.argmax(axis=1)
            # print(f"{ips=}")
            vps = np.choose(ips, vp.T)
            # print(f"{vps=}")
            delta = 1/vps
            # print(f"{delta.shape=} {sigma.shape=} {v[fofc].shape=}")
            if not np.all(delta > 0): raise ValueError(f"{delta=}\n{vp[ips]=}")
            lamd = (np.ones(ncells)[:,np.newaxis] - delta[:,np.newaxis]*vs)/(dim+1)
            if not np.allclose(lamd.sum(axis=1),1):
                raise ValueError(f"{lamd=}\n{(beta[fofc]*sigma).sum(axis=1)}")
            delta /= (dim+1)
        elif method=='supg2':
            # vp = np.maximum(v[fofc]*sigma, 0)
            # vps = vp.sum(axis=1)
            # if not np.all(vps > 0): raise ValueError(f"{vps=}\n{vp=}")
            # vp /= vps[:,np.newaxis]
            # lamd = (np.ones(ncells)[:,np.newaxis] - vp)/dim
            vm = np.minimum(beta[fofc]*sigma, 0)
            vms = vm.sum(axis=1)
            if not np.all(vms < 0): raise ValueError(f"{vms=}\n{vm=}")
            vm /= vms[:,np.newaxis]
            lamd = vm
        else:
            raise ValueError(f"unknown method {method}")
        # print(f"{v[fofc].shape} {lamd2.shape=}")
        points, simplices = self.mesh.points, self.mesh.simplices
        xd = np.einsum('nji,nj -> ni', points[simplices], lamd)
        delta = np.linalg.norm(np.einsum('nji,nj -> ni', points[simplices], lamd-1/(dim+1)),axis=1)
        # if not np.allclose(lamd, lamd2): print(f"{lamd=} {lamd2=}")
        return xd, lamd, delta
    def supgPoints(self, beta, scale, method):
        rt = fems.rt0.RT0(self.mesh)
        self.supdata={}
        convection = scale*rt.interpolate(beta)
        self.supdata['convection'] = convection
        self.supdata['xd'], self.supdata['lam'], self.supdata['delta'] = self.downWind(convection, method=method)
        self.supdata['convectionC'] = rt.toCell(convection)
    def computeMatrixDiffusion(self, coeff):
        ndofs = self.nunknowns()
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = ( (matxx+matyy+matzz).T*self.mesh.dV*coeff).T.ravel()
        return sparse.coo_matrix((mat, (self.rows, self.cols)), shape=(ndofs, ndofs)).tocsr()
    def computeMatrixLps(self, betaC):
        dimension, dV, ndofs = self.mesh.dimension, self.mesh.dV, self.nunknowns()
        nloc, dofspercell = self.nlocal(), self.dofspercell()
        ci = self.mesh.cellsOfInteriorFaces
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        scale = 0.5*(dV[ci[:,0]]+ dV[ci[:,1]])
        betan = 0.5*(np.linalg.norm(betaC[ci[:,0]],axis=1)+ np.linalg.norm(betaC[ci[:,1]],axis=1))
        scale *= 0.01*dS*betan
        cg0 = self.cellgrads[ci[:,0], :, :]
        cg1 = self.cellgrads[ci[:,1], :, :]
        mat00 = np.einsum('nki,nli,n->nkl', cg0, cg0, scale)
        mat01 = np.einsum('nki,nli,n->nkl', cg0, cg1, -scale)
        mat10 = np.einsum('nki,nli,n->nkl', cg1, cg0, -scale)
        mat11 = np.einsum('nki,nli,n->nkl', cg1, cg1, scale)
        rows0 = dofspercell[ci[:,0],:].repeat(nloc)
        cols0 = np.tile(dofspercell[ci[:,0],:],nloc).reshape(-1)
        rows1 = dofspercell[ci[:,1],:].repeat(nloc)
        cols1 = np.tile(dofspercell[ci[:,1],:],nloc).reshape(-1)
        A00 = sparse.coo_matrix((mat00.reshape(-1), (rows0, cols0)), shape=(ndofs, ndofs))
        A01 = sparse.coo_matrix((mat01.reshape(-1), (rows0, cols1)), shape=(ndofs, ndofs))
        A10 = sparse.coo_matrix((mat10.reshape(-1), (rows1, cols0)), shape=(ndofs, ndofs))
        A11 = sparse.coo_matrix((mat11.reshape(-1), (rows1, cols1)), shape=(ndofs, ndofs))
        return A00+A01+A10+A11


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
