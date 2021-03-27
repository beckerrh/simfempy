# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from simfempy.fems import fem
from simfempy.tools import barycentric

#=================================================================#
class D0(fem.Fem):
    def __init__(self, mesh=None):
        super().__init__(mesh)
        self.dirichlet_al = 2
    def setMesh(self, mesh):
        super().setMesh(mesh)
    def nlocal(self): return 1
    def nunknowns(self): return self.mesh.ncells

# ------------------------------------- #
if __name__ == '__main__':
    raise NotImplementedError("no test")
