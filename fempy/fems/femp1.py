# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
try:
    from fempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh


#=================================================================#
class FemP1(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
        self.nloc = self.mesh.dimension+1
        simps = self.mesh.simplices
        self.cols = np.tile(simps, self.nloc).flatten()
        self.rows = np.repeat(simps, self.nloc).flatten()
        self.computeFemMatrices()
        self.massmatrix = self.massMatrix()
    def prepareBoundary(self, colorsdir, postproc):
        self.nodesdir={}
        self.nodedirall = np.empty(shape=(0), dtype=int)
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            self.nodesdir[color] = np.unique(self.mesh.faces[facesdir].flat[:])
            self.nodedirall = np.unique(np.union1d(self.nodedirall, self.nodesdir[color]))
        self.nodesinner = np.setdiff1d(np.arange(self.mesh.nnodes, dtype=int),self.nodedirall)
        # print("colorsdir", colorsdir)
        # print("nodesdir", self.nodesdir)
        # print("self.nodesinner", self.nodesinner)
        self.bsaved={}
        self.Asaved={}
        self.nodesdirflux={}
        for key, val in postproc.items():
            type,data = val.split(":")
            if type != "flux": continue
            colors = [int(x) for x in data.split(',')]
            self.nodesdirflux[key] = np.empty(shape=(0), dtype=int)
            for color in colors:
                facesdir = self.mesh.bdrylabels[color]
                self.nodesdirflux[key] = np.unique(np.union1d(self.nodesdirflux[key], np.unique(self.mesh.faces[facesdir].flatten())))
    def computeRhs(self, rhs, solexact, kheatcell, bdrycond):
        if solexact or rhs:
            x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
            if solexact:
                bnodes = -solexact.xx(x, y, z) - solexact.yy(x, y, z)- solexact.zz(x, y, z)
                bnodes *= kheatcell[0]
            else:
                bnodes = rhs(x, y, z)
            b = self.massmatrix*bnodes
        else:
            b = np.zeros(self.mesh.nnodes)
        normals =  self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            condition = bdrycond.type[color]
            if condition == "Neumann":
                neumann = bdrycond.fct[color]
                scale = 1/self.mesh.dimension
                normalsS = normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                xS = np.mean(self.mesh.points[self.mesh.faces[faces]], axis=1)
                kS = kheatcell[self.mesh.cellsOfFaces[faces,0]]
                assert(dS.shape[0] == len(faces))
                assert(xS.shape[0] == len(faces))
                assert(kS.shape[0] == len(faces))
                x1, y1, z1 = xS[:,0], xS[:,1], xS[:,2]
                nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
                if solexact:
                    bS = scale*dS*kS*(solexact.x(x1, y1, z1)*nx + solexact.y(x1, y1, z1)*ny + solexact.z(x1, y1, z1)*nz)
                else:
                    bS = scale * neumann(x1, y1, z1, nx, ny, nz, kS) * dS
                np.add.at(b, self.mesh.faces[faces].T, bS)
        return b
    def massMatrix(self):
        nnodes = self.mesh.nnodes
        self.massmatrix = sparse.coo_matrix((self.mass, (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
        return self.massmatrix
    def matrixDiffusion(self, k):
        nnodes = self.mesh.nnodes
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = ( (matxx+matyy+matzz).T*self.mesh.dV*k).T.flatten()
        return sparse.coo_matrix((mat, (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
    def computeFemMatrices(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        scale = 1/self.mesh.dimension
        self.cellgrads = scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
        scalemass = 1 / self.nloc / (self.nloc+1);
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] *= 2
        self.mass = np.einsum('n,kl->nkl', dV, massloc).flatten()
    def boundary(self, A, b, u, bdrycond, method):
        x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
        nnodes = self.mesh.nnodes
        for key, nodes in self.nodesdirflux.items():
            self.bsaved[key] = b[nodes]
        for key, nodes in self.nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((nb, nnodes))
            for i in range(nb): help[i, nodes[i]] = 1
            self.Asaved[key] = help.dot(A)
        if method == 'trad':
            for color, nodes in self.nodesdir.items():
                dirichlet = bdrycond.fct[color]
                b[nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
                u[nodes] = b[nodes]
            b[self.nodesinner] -= A[self.nodesinner, :][:, self.nodedirall] * b[self.nodedirall]
            help = np.ones((nnodes))
            help[self.nodedirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
            A = help.dot(A.dot(help))
            help = np.zeros((nnodes))
            help[self.nodedirall] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
            A += help
        else:
            for color, nodes in self.nodesdir.items():
                dirichlet = bdrycond.fct[color]
                u[nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
                b[nodes] = 0
            b -= A*u
            b[self.nodedirall] += 2*A[self.nodedirall, :][:, self.nodedirall] * u[self.nodedirall]
            help = np.ones((nnodes))
            help[self.nodedirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
            help2 = np.zeros((nnodes))
            help2[self.nodedirall] = 1
            help2 = sparse.dia_matrix((help2, 0), shape=(nnodes, nnodes))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A, b, u
    def tonode(self, u):
        return u
    def grad(self, ic):
        normals = self.mesh.normals[self.mesh.facesOfCells[ic,:]]
        grads = 0.5*normals/self.mesh.dV[ic]
        chsg =  (ic == self.mesh.cellsOfFaces[self.mesh.facesOfCells[ic,:],0])
        # print("### chsg", chsg, "normals", normals)
        grads[chsg] *= -1.
        return grads
    def phi(self, ic, x, y, z, grad):
        return 1./3. + np.dot(grad, np.array([x-self.mesh.pointsc[ic,0], y-self.mesh.pointsc[ic,1], z-self.mesh.pointsc[ic,2]]))
    def testgrad(self):
        for ic in range(fem.mesh.ncells):
            grads = fem.grad(ic)
            for ii in range(3):
                x = self.mesh.points[self.mesh.simplices[ic,ii], 0]
                y = self.mesh.points[self.mesh.simplices[ic,ii], 1]
                z = self.mesh.points[self.mesh.simplices[ic,ii], 2]
                for jj in range(3):
                    phi = self.phi(ic, x, y, z, grads[jj])
                    if ii == jj:
                        test = np.abs(phi-1.0)
                        if test > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            print('x,y', x, y)
                            print('x-xc,y-yc', x-self.mesh.pointsc[ic,0], y-self.mesh.pointsc[ic,1])
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic,ii,jj, test))
                    else:
                        test = np.abs(phi)
                        if np.abs(phi) > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic,ii,jj, test))
    def computeErrorL2(self, solex, uh):
        x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
        e = solex(x, y, z) - uh
        return np.sqrt( np.dot(e, self.massmatrix*e) ), e
    def computeMean(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            mean += np.sum(dS*np.mean(u[self.mesh.faces[faces]],axis=1))
        return mean
    def computeFlux(self, u, key, data):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(self.bsaved[key] - self.Asaved[key]*u )
        return flux

# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemP1(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
