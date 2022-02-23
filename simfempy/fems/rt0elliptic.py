import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from simfempy import fems

#=================================================================#
class RTelliptic(fems.fem.Fem):
    def __init__(self, mesh=None):
        super(RTelliptic, self).__init__()
        self.rt = fems.rt0.RT0()
        self.d0 = fems.d0.D0()
    def setMesh(self, mesh):
        self.rt.setMesh(mesh)
        self.d0.setMesh(mesh)
    def prepareBoundary(self, colorsdirichlet, colorsflux):
        pass
    def computeMatrixDiffusion(self, diffcoff):
        raise NotImplementedError
    def computeMatrixNitscheDiffusion(self, diffcoff, colors):
        pass
    def computeBdryMassMatrix(self, colors=None, coeff=1, lumped=False):
        raise NotImplementedError
