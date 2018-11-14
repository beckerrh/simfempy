# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import matplotlib.pyplot as plt
from mesh.trimesh import TriMesh

class FemP12D(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)

    def setMesh(self, mesh):
        assert isinstance(mesh, TriMesh)
        self.mesh = mesh
        self.xedges = self.mesh.x[self.mesh.edges].mean(axis=1)
        self.yedges = self.mesh.y[self.mesh.edges].mean(axis=1)

    def phi(self, ic, x, y, grad):
        return 1./3. + np.dot(grad, np.array([x-self.mesh.centersx[ic], y-self.mesh.centersy[ic]]))


    def grad(self, ic):
        normals = self.mesh.normals[self.mesh.edgesOfCell[ic,:]]
        grads = 0.5*normals/self.mesh.area[ic]
        chsg =  (ic == self.mesh.cellsOfEdge[self.mesh.edgesOfCell[ic,:],0])
        grads[chsg] *= -1.
        return grads

    def testgrad(self):
        for ic in range(fem.mesh.ncells):
            grads = fem.grad(ic)
            for ii in range(3):
                x = self.mesh.x[self.mesh.triangles[ic,ii]]
                y = self.mesh.y[self.mesh.triangles[ic,ii]]
                for jj in range(3):
                    phi = self.phi(ic, x, y, grads[jj])
                    if ii == jj:
                        if np.abs(phi-1.0) > 1e-15:
                            print('ic=', ic, 'grad=', grads)
                            print('x,y', x, y)
                            print('x-xc,y-yc', x-self.xcells[ic], y-self.ycells[ic])
                            raise ValueError('wrong in cell=%d, ii,jj=%d,%d phi: 1!=%g' %(ic,ii,jj, phi))
                    else:
                        if np.abs(phi) > 1e-15:
                            print('ic=', ic, 'grad=', grads)
                            raise ValueError('wrong in cell=%d, ii,jj=%d,%d phi: 0!=%g' %(ic,ii,jj, phi))



# ------------------------------------- #

if __name__ == '__main__':
    filename = 'test2.vtu'
    trimesh = TriMesh(filename=filename)
    fem = FemP12D(trimesh)
    trimesh.plot(plt)
    fem.testgrad()
    plt.show()
