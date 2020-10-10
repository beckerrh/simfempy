# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from simfempy.fems import femcr1
try:
    from simfempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh

#=================================================================#
class FemCR1(femcr1.FemCR1):

    def __init__(self, mesh=None):
        super().__init__(mesh)

    def setMesh(self, mesh, ncomp):
        super().setMesh(mesh)
        self.ncomp = ncomp
        ncells, facesOfCells = self.mesh.ncells, self.mesh.facesOfCells
        nlocncomp = ncomp * self.nloc
        self.rowssys = np.repeat(ncomp * facesOfCells, ncomp).reshape(ncells * self.nloc, ncomp) + np.arange(ncomp)
        self.rowssys = self.rowssys.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.colssys = self.rowssys.swapaxes(1, 2)
        self.colssys = self.colssys.reshape(-1)
        self.rowssys = self.rowssys.reshape(-1)

    def prepareBoundary(self, bdrycond, postproc):
        if not isinstance(bdrycond, (list, tuple)):
            return super().prepareBoundary(bdrycond.colorsOfType("Dirichlet"), postproc)
        bdrydata = []
        for icomp in range(self.ncomp):
            bdrydata.append(super().prepareBoundary(bdrycond[icomp].colorsOfType("Dirichlet"), postproc[icomp]))
        return bdrydata

    def computeRhs(self, u, rhs, diff, bdrycond, method, bdrydata):
        b = np.zeros(self.mesh.nfaces * self.ncomp)
        x, y, z = self.mesh.pointsf.T
        if rhs:
            for icomp in range(self.ncomp):
                bfaces = rhs[icomp](x, y, z, diff[icomp][0])
                b[icomp::self.ncomp] = self.massmatrix * bfaces
        normals =  self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            for icomp in range(self.ncomp):
                if bdrycond[icomp].type[color] != "Neumann": continue
                normalsS = normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                kS = diff[icomp][self.mesh.cellsOfFaces[faces,0]]
                x1, y1, z1 = self.mesh.pointsf[faces].T
                nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
                bS = bdrycond[icomp].fct[color](x1, y1, z1, nx, ny, nz, kS) * dS
                b[icomp+self.ncomp*faces] += bS
        return self.vectorDirichlet(b, u, bdrycond, method, bdrydata)

    def matrixDiffusion(self, k, bdrycond, method, bdrydata):
        nfaces, ncells, ncomp = self.mesh.nfaces, self.mesh.ncells, self.ncomp
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        nlocncomp = ncomp * self.nloc
        mat = np.zeros(shape=(ncells, nlocncomp, nlocncomp))
        for icomp in range(ncomp):
            mat[:, icomp::ncomp, icomp::ncomp] = ((matxx + matyy + matzz).T * self.mesh.dV * k[icomp]).T
        A = sparse.coo_matrix((mat.ravel(), (self.rowssys, self.colssys)), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
        return self.matrixDirichlet(A, bdrycond, method, bdrydata)

    def vectorDirichlet(self, b, u, bdrycond, method, bdrydata):
        x, y, z = self.mesh.pointsf.T
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        for icomp in range(ncomp):
            facesdirall, facesinner, colorsdir, facesdirflux = bdrydata[icomp].facesdirall, bdrydata[icomp].facesinner, bdrydata[icomp].colorsdir, bdrydata[icomp].facesdirflux
            for key, faces in facesdirflux.items():
                bdrydata[icomp].bsaved[key] = b[icomp + ncomp * faces]
            indin = icomp + ncomp * facesinner
            inddir = icomp + ncomp * facesdirall
            if method == 'trad':
                for color in colorsdir:
                    faces = self.mesh.bdrylabels[color]
                    dirichlet = bdrycond[icomp].fct[color]
                    b[icomp + ncomp * faces] = dirichlet(x[faces], y[faces], z[faces])
                    u[icomp + ncomp * faces] = b[icomp + ncomp * faces]
                b[indin] -= bdrydata[icomp].A_inner_dir * b[inddir]
            else:
                for color in colorsdir:
                    faces = self.mesh.bdrylabels[color]
                    dirichlet = bdrycond[icomp].fct[color]
                    u[icomp + ncomp * faces] = dirichlet(x[faces], y[faces], z[faces])
                    b[icomp + ncomp * faces] = 0
                b[indin] -= bdrydata[icomp].A_inner_dir * u[inddir]
                b[inddir] = bdrydata[icomp].A_dir_dir * u[inddir]
        return b, u, bdrydata

    def matrixDirichlet(self, A, bdrycond, method, bdrydata):
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        for icomp in range(ncomp):
            facesdirall, facesinner, colorsdir, facesdirflux = bdrydata[icomp].facesdirall, bdrydata[icomp].facesinner, bdrydata[icomp].colorsdir, bdrydata[icomp].facesdirflux
            for key, faces in facesdirflux.items():
                nb = faces.shape[0]
                help = sparse.dok_matrix((nb, ncomp * nfaces))
                for i in range(nb): help[i, icomp + ncomp * faces[i]] = 1
                bdrydata[icomp].Asaved[key] = help.dot(A)
            indin = icomp + ncomp * facesinner
            inddir = icomp + ncomp * facesdirall
            bdrydata[icomp].A_inner_dir = A[indin, :][:, inddir]
            if method == 'trad':
                help = np.ones((ncomp * nfaces))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A = help.dot(A.dot(help))
                help = np.zeros((ncomp * nfaces))
                help[inddir] = 1.0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A += help
            else:
                bdrydata[icomp].A_dir_dir = A[inddir, :][:, inddir]
                help = np.ones((ncomp * nfaces))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                help2 = np.zeros((ncomp * nfaces))
                help2[inddir] = 1
                help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A, bdrydata

    def boundary(self, A, b, u, bdrycond, bdrydata, method):
        x, y, z = self.mesh.pointsf.T
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        self.bsaved = []
        self.Asaved = []
        for icomp in range(ncomp):
            facesdirall, facesinner, colorsdir, facesdirflux = bdrydata[icomp]
            self.bsaved.append({})
            self.Asaved.append({})
            for key, faces in facesdirflux.items():
                self.bsaved[icomp][key] = b[icomp + ncomp * faces]
            for key, faces in facesdirflux.items():
                nb = faces.shape[0]
                help = sparse.dok_matrix((nb, ncomp * nfaces))
                for i in range(nb): help[i, icomp + ncomp * faces[i]] = 1
                self.Asaved[icomp][key] = help.dot(A)
            if method == 'trad':
                for color in colorsdir:
                    faces = self.mesh.bdrylabels[color]
                    dirichlet = bdrycond[icomp].fct[color]
                    b[icomp + ncomp * faces] = dirichlet(x[faces], y[faces], z[faces])
                    u[icomp + ncomp * faces] = b[icomp + ncomp * faces]
                indin = icomp + ncomp *facesinner
                inddir = icomp + ncomp *facesdirall
                b[indin] -= A[indin, :][:,inddir] * b[inddir]
                help = np.ones((ncomp * nfaces))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A = help.dot(A.dot(help))
                help = np.zeros((ncomp * nfaces))
                help[inddir] = 1.0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A += help
            else:
                for color in colorsdir:
                    faces = self.mesh.bdrylabels[color]
                    dirichlet = bdrycond[icomp].fct[color]
                    u[icomp + ncomp * faces] = dirichlet(x[faces], y[faces], z[faces])
                    b[icomp + ncomp * faces] = 0
                indin = icomp + ncomp *facesinner
                inddir = icomp + ncomp *facesdirall
                b[indin] -= A[indin, :][:, inddir] * u[inddir]
                b[inddir] = A[inddir, :][:, inddir] * u[inddir]
                help = np.ones((ncomp * nfaces))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                help2 = np.zeros((ncomp * nfaces))
                help2[inddir] = 1
                help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A, b, u

    def grad(self, ic):
        normals = self.mesh.normals[self.mesh.facesOfCells[ic,:]]
        grads = -normals/self.mesh.dV[ic]
        chsg =  (ic == self.mesh.cellsOfFaces[self.mesh.facesOfCells[ic,:],0])
        # print("### chsg", chsg, "normals", normals)
        grads[chsg] *= -1.
        return grads

    def phi(self, ic, x, y, z, grad):
        return 1./3. + np.dot(grad, np.array([x-self.mesh.pointsc[ic,0], y-self.mesh.pointsc[ic,1], z-self.mesh.pointsc[ic,2]]))

    def testgrad(self):
        for ic in range(self.mesh.ncells):
            grads = self.grad(ic)
            for ii in range(3):
                x = self.pointsf[self.mesh.facesOfCells[ic,ii], 0]
                y = self.pointsf[self.mesh.facesOfCells[ic,ii], 1]
                z = self.pointsf[self.mesh.facesOfCells[ic,ii], 2]
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
        x, y, z = self.mesh.pointsf.T
        e = []
        err = []
        for icomp in range(self.ncomp):
            e.append(solex[icomp](x, y, z) - uh[icomp::self.ncomp])
            err.append(np.sqrt(np.dot(e[icomp], self.massmatrix * e[icomp])))
        return err, e

    def computeBdryMean(self, u, data, icomp=None):
        colors = [int(x) for x in data.split(',')]
        if icomp is None:
            mean, omega = np.zeros(shape=(self.ncomp,len(colors))), np.zeros(len(colors))
            for i, color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                for icomp in range(self.ncomp):
                    mean[icomp,i] = np.sum(dS * u[icomp + self.ncomp * faces])
            return mean / omega
        else:
            mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                mean[i] = np.sum(dS * u[icomp + self.ncomp * faces])
            return mean/omega


    def computeBdryDn(self, u, data, bdrydata, bdrycond, icomp=None):
        colors = [int(x) for x in data.split(',')]
        if icomp is None:
            flux, omega = np.zeros(shape=(self.ncomp,len(colors))), np.zeros(len(colors))
            for i, color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                for icomp in range(self.ncomp):
                    if bdrycond[icomp].type[color] == "Dirichlet":
                        bs, As = bdrydata[icomp].bsaved[color], bdrydata[icomp].Asaved[color]
                        flux[icomp,i] = np.sum(As * u - bs)
                    else:
                        raise NotImplementedError("computeBdryDn for condition '{}'".format(bdrycond.type[color]))
            return flux
        else:
            flux, omega = np.zeros(shape=(len(colors))), np.zeros(len(colors))
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                if bdrycond[icomp].type[color] == "Dirichlet":
                    bs, As = bdrydata[icomp].bsaved[color], bdrydata[icomp].Asaved[color]
                    flux[i] = np.sum(As * u - bs)
                else:
                    raise NotImplementedError("computeBdryDn for condition '{}'".format(bdrycond.type[color]))
            return flux

    def tonode(self, u):
        unodes = np.zeros(self.mesh.nnodes)
        scale = self.mesh.dimension
        np.add.at(unodes, self.mesh.simplices.T, np.sum(u[self.mesh.facesOfCells], axis=1))
        np.add.at(unodes, self.mesh.simplices.T, -scale*u[self.mesh.facesOfCells].T)
        countnodes = np.zeros(self.mesh.nnodes, dtype=int)
        np.add.at(countnodes, self.mesh.simplices.T, 1)
        unodes /= countnodes
        return unodes


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemCR1(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
