import time
import numpy as np
from simfempy import solvers
from simfempy import fems

#=================================================================#
class Heat(solvers.solver.Solver):
    """
    """
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z, diff):
            rhs = np.zeros(x.shape[0])
            for i in range(self.mesh.dimension):
                rhs -= diff * solexact.dd(i, i, x, y, z)
            return rhs
        return _fctu

    def defineNeumannAnalyticalSolution(self, solexact):
        def _fctneumann(x, y, z, nx, ny, nz, diff):
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += diff * solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann

    def defineRobinAnalyticalSolution(self, solexact, param=None):
        if param is None: raise NotImplementedError("defineRobinAnalyticalSolution needs Robin parameter")
        def _fctrobin(x, y, z, nx, ny, nz, diff):
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += diff * solexact.d(i, x, y, z) * normals[i]
            rhs += param*solexact(x,y,z)
            return rhs
        return _fctrobin

    def setParameter(self, paramname, param):
        if paramname == "dirichlet_al": self.fem.dirichlet_al = param
        else:
            if not hasattr(self, self.paramname):
                raise NotImplementedError("{} has no paramater '{}'".format(self, self.paramname))
            cmd = "self.{} = {}".format(self.paramname, param)
            eval(cmd)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linearsolver = 'pyamg'
        fem = 'p1'
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        if fem == 'p1':
            self.fem = fems.femp1.FemP1()
        elif fem == 'cr1':
            self.fem = fems.femcr1.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'rhocp' in kwargs:
            self.rhocp = np.vectorize(kwargs.pop('rhocp'))
        else:
            self.rhocp = np.vectorize(lambda i: 1234.56)
        if 'kheat' in kwargs:
            self.kheat = np.vectorize(kwargs.pop('kheat'))
        else:
            self.kheat = np.vectorize(lambda i: 0.123)
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="trad"
        if 'plotk' in kwargs:
            self.plotk = kwargs.pop('plotk')
        else:
            self.plotk = False

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.fem.setMesh(self.mesh)
        self.bdrydata = self.fem.prepareBoundary(self.bdrycond.colorsOfType("Dirichlet"), self.postproc)
        # print("bdrydata = ", self.bdrydata)
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.rhocpcell = self.rhocp(self.mesh.cell_labels)

    def solve(self, iter=0, dirname=None):
        return self.solveLinear()

    def computeRhs(self, u=None):
        b, u, self.bdrydata = self.fem.computeRhs(u, self.rhs, self.kheatcell, self.bdrycond, self.method, self.bdrydata)
        return b,u

    def matrix(self):
        A, self.bdrydata = self.fem.matrixDiffusion(self.kheatcell, self.bdrycond, self.method, self.bdrydata)
        return A

    def boundaryvec(self, b, u=None):
        if u is None: u = np.zeros_like(b)
        b, u, self.bdrydata = self.fem.boundaryvec(b, u, self.bdrycond, self.method, self.bdrydata)
        return b,u

    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = self.fem.tonode(u)
        if self.problemdata.solexact:
            info['error'] = {}
            info['error']['L2'], e = self.fem.computeErrorL2(self.problemdata.solexact, u)
            point_data['E'] = self.fem.tonode(e)
        info['postproc'] = {}
        if self.postproc:
            for key, val in self.postproc.items():
                type,data = val.split(":")
                if type == "bdrymean":
                    info['postproc'][key] = self.fem.computeBdryMean(u, key, data)
                elif type == "bdryfct":
                    info['postproc'][key] = self.fem.computeBdryFct(u, key, data)
                elif type == "bdrydn":
                    bs, As = self.bdrydata.bsaved[key], self.bdrydata.Asaved[key]
                    info['postproc'][key] = self.fem.computeBdryDn(u, key, data, bs, As)
                elif type == "pointvalues":
                    info['postproc'][key] = self.fem.computePointValues(u, key, data)
                else:
                    raise ValueError("unknown postprocess '{}' for key '{}'".format(type, key))
        assert self.kheatcell.shape[0] == self.mesh.ncells
        if self.plotk: cell_data['k'] = self.kheatcell
        return point_data, cell_data, info

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
