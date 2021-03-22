# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

#=================================================================#
class Femsys():
    def __init__(self, fem, ncomp, mesh=None):
        self.ncomp = ncomp
        self.fem = fem
    def setMesh(self, mesh):
        self.mesh = mesh
        self.fem.setMesh(mesh)
        ncomp, nloc, ncells = self.ncomp, self.fem.nloc, self.mesh.ncells
        dofs = self.fem.dofs_cells()
        nlocncomp = ncomp * nloc
        self.rowssys = np.repeat(ncomp * dofs, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
        self.rowssys = self.rowssys.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.colssys = self.rowssys.swapaxes(1, 2)
        self.colssys = self.colssys.reshape(-1)
        self.rowssys = self.rowssys.reshape(-1)
    def prepareBoundary(self, colorsdirichlet, colorsflux=[]):
        self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
    def computeErrorL2(self, solex, uh):
        eall, ecall = [], []
        for icomp in range(self.ncomp):
            e, ec = self.fem.computeErrorL2(solex[icomp], uh[icomp::self.ncomp])
            eall.append(e)
            ecall.append(ec)
        return eall, ecall
    def computeBdryMean(self, u, colors):
        all = []
        for icomp in range(self.ncomp):
            a = self.fem.computeBdryMean(u[icomp::self.ncomp], colors)
            all.append(a)
        return all

# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError(f"pas de test")
