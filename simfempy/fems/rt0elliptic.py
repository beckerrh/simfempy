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
    def nunknowns(self):
        return self.rt.nunknowns() + self.d0.nunknowns()
    # def prepareBoundary(self, bdrycond, postproc):
    #     colorsneumann = bdrycond.colorsOfType("Neumann")
    #     return self.rt.prepareBoundary(colorsneumann)
    # def computeMatrixDiffusion(self, diffcoff):
    #     A = self.rt.constructMass(1/diffcoff)
    #     B = self.rt.constructDiv()
    #     return A,B
    #
    # def computeBdryMassMatrix(self, colorsrobin, param, lumped=False):
    #     return  self.rt.constructRobin(colorsrobin, param), None
    #     raise NotImplementedError
    # def computeMatrixNeumann(self, colorsneumann, param):
    #     self.bdrydata, A,B = self.rt.matrixNeumann(A, B, self.problemdata.bdrycond)
    #     raise NotImplementedError

    # def computeMatrixNitscheDiffusion(self, diffcoff, colors):
    #     return None,None


    # def matrixBoundaryStrong(self, A, bdrydata):
    #     A,B = A[0],A[1]
    #     A,B,bdrydata = self.rt.matrixNeumann(A, B, bdrydata)
    #     return (A,B), bdrydata


    def interpolate(self, f):
        return self.d0.interpolate(f)
    def massDot(self, b, fp):
        nfaces = self.d0.mesh.nfaces
        return self.d0.massDot(b[nfaces:], fp)
    def computeRhsNitscheDiffusion(self, b, diffcoff, colorsdir, udir, bdrycondfct):
        return b
    def vectorBoundaryStrong(self, b, bdrycond, bdrydata):
        return b


