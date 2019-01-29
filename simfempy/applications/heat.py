import time
import numpy as np
from simfempy import solvers
from simfempy import fems

#=================================================================#
class Heat(solvers.solver.Solver):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        t0 = time.time()
        self.mesh = mesh
        self.fem.setMesh(self.mesh)
        colorsdir = []
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.fem.prepareBoundary(colorsdir, self.postproc)
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.rhocpcell = self.rhocp(self.mesh.cell_labels)
        t1 = time.time()
        self.timer['setmesh'] = t1-t0

    def solve(self, iter=0, dirname=None):
        return self.solveLinear()
    def computeRhs(self):
        return self.fem.computeRhs(self.rhs, self.solexact, self.kheatcell, self.bdrycond)
    def matrix(self):
        return self.fem.matrixDiffusion(self.kheatcell)
    def boundary(self, A, b, u=None):
        if u is None: u = np.asarray(b)
        return self.fem.boundary(A, b, u, self.bdrycond, self.method)
    def boundaryvec(self, b, u=None):
        if u is None: u = np.asarray(b)
        return self.fem.boundaryvec(b, u, self.bdrycond, self.method)

    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = self.fem.tonode(u)
        if self.solexact:
            info['error'] = {}
            info['error']['L2'], e = self.fem.computeErrorL2(self.solexact, u)
            point_data['E'] = self.fem.tonode(e)
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        info['postproc'] = {}
        for key, val in self.postproc.items():
            type,data = val.split(":")
            if type == "bdrymean":
                info['postproc'][key] = self.fem.computeMean(u, key, data)
            elif type == "bdryfct":
                info['postproc'][key] = self.fem.computeBdryFct(u, key, data)
            elif type == "bdrydn":
                info['postproc'][key] = self.fem.computeFlux(u, key, data)
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
