import time
import numpy as np
from simfempy import solvers
from simfempy import fems

#=================================================================#
class Heat(solvers.solver.Solver):
    """
    Class for the heat equation
    After initialization, the function setMesh(mesh) has to be called
    Then, solve() solves the stationary problem
    Parameters in the coçnstructor:
        fem: only p1 or cr1
        problemdata
        method
        plotk
    Paramaters used from problemdata:
        rhocp
        kheat
        reaction
        they can either be given as global constant, cell-wise constants, or global function
        - global constant is taken from problemdata.paramglobal
        - cell-wise constants are taken from problemdata.paramcells
        - problemdata.paramglobal is taken from problemdata.datafct and are called with arguments (color, xc, yc, zc)
    """
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            rhs = np.zeros(x.shape[0])
            for i in range(self.mesh.dimension):
                rhs -= self.problemdata.paramglobal['kheat'] * solexact.dd(i, i, x, y, z)
            return rhs
        return _fctu

    def defineNeumannAnalyticalSolution(self, solexact):
        def _fctneumann(x, y, z, nx, ny, nz):
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += self.problemdata.paramglobal['kheat'] * solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann

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
        if 'reaction' in kwargs:
            self.reaction = np.vectorize(kwargs.pop('reaction'))
        else:
            self.reaction = None
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="trad"
        if 'plotk' in kwargs:
            self.plotk = kwargs.pop('plotk')
        else:
            self.plotk = False
        if 'mesh' in kwargs:
            self.setMesh(kwargs.pop('mesh'))

    # def _computekheatcell_(self):
    #     if 'kheat' in self.problemdata.datafct:
    #         assert 'kheat' not in self.problemdata.paramglobal
    #         assert 'kheat' not in self.problemdata.paramcells
    #         xc, yc, zc = self.mesh.pointsc.T
    #         kheat = np.vectorize(self.problemdata.datafct['kheat'])
    #         self.kheatcell = kheat(self.mesh.cell_labels, xc, yc, zc)
    #     elif 'kheat' in self.problemdata.paramglobal:
    #         assert 'kheat' not in self.problemdata.datafct
    #         assert 'kheat' not in self.problemdata.paramcells
    #         self.kheatcell = np.full(self.mesh.ncells, self.problemdata.paramglobal['kheat'])
    #     elif 'kheat' in self.problemdata.paramcells:
    #         assert 'kheat' not in self.problemdata.paramglobal
    #         assert 'kheat' not in self.problemdata.datafct
    #         self.kheatcell = np.empty(self.mesh.ncells)
    #         for color in self.problemdata.paramcells['kheat']:
    #             self.kheatcell[self.mesh.cellsoflabel[color]] = self.problemdata.paramcells['kheat'][color]
    #     else:
    #         raise ValueError("'kheat' should be given in 'problemdata.datafct' or 'problemdata.paramglobal' or 'problemdata.paramcells'")

    def _computearrcell_(self, name):
        if name in self.problemdata.datafct:
            assert name not in self.problemdata.paramglobal
            assert name not in self.problemdata.paramcells
            xc, yc, zc = self.mesh.pointsc.T
            fct = np.vectorize(self.problemdata.datafct[name])
            arr = fct(self.mesh.cell_labels, xc, yc, zc)
        elif name in self.problemdata.paramglobal:
            assert name not in self.problemdata.datafct
            assert name not in self.problemdata.paramcells
            arr = np.full(self.mesh.ncells, self.problemdata.paramglobal[name])
        elif name in self.problemdata.paramcells:
            assert name not in self.problemdata.paramglobal
            assert name not in self.problemdata.datafct
            arr = np.empty(self.mesh.ncells)
            for color in self.problemdata.paramcells[name]:
                arr[self.mesh.cellsoflabel[color]] = self.problemdata.paramcells[name][color]
        else:
            msg = f"{name} should be given in 'problemdata.datafct' or 'problemdata.paramglobal' or 'problemdata.paramcells'"
            raise ValueError(msg)
        return arr

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.fem.setMesh(self.mesh, self.problemdata.bdrycond)
        self.bdrydata = self.fem.prepareBoundary(self.problemdata.bdrycond.colorsOfType("Dirichlet"), self.problemdata.postproc)
        self.kheatcell = self._computearrcell_('kheat')
        self.rhocpcell = self._computearrcell_('rhocp')

    def solve(self, iter=0, dirname=None):
        if not hasattr(self,'mesh'): raise ValueError("*** no mesh given ***")
        return self.solveLinear()

    def computeRhs(self, u=None):
        b, u, self.bdrydata = self.fem.computeRhs(u, self.problemdata, self.kheatcell, self.method, self.bdrydata)
        return b,u

    def matrix(self):
        A, self.bdrydata = self.fem.matrixDiffusion(self.kheatcell, self.problemdata.bdrycond, self.method, self.bdrydata)
        return A

    def boundaryvec(self, b, u=None):
        if u is None: u = np.zeros_like(b)
        b, u, self.bdrydata = self.fem.boundaryvec(b, u, self.problemdata.bdrycond, self.method, self.bdrydata)
        return b,u

    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = self.fem.tonode(u)
        if self.problemdata.solexact:
            info['error'] = {}
            info['error']['pnL2'], info['error']['pcL2'], e = self.fem.computeErrorL2(self.problemdata.solexact, u)
            info['error']['vcL2'] = self.fem.computeErrorFluxL2(self.problemdata.solexact, self.kheatcell, u)
            point_data['E'] = self.fem.tonode(e)
        info['postproc'] = {}
        if self.problemdata.postproc:
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == "bdrymean":
                    info['postproc'][name] = self.fem.computeBdryMean(u, colors)
                elif type == "bdryfct":
                    info['postproc'][name] = self.fem.computeBdryFct(u, colors)
                elif type == "bdrydn":
                    info['postproc'][name] = self.fem.computeBdryDn(u, colors, self.bdrydata, self.problemdata.bdrycond)
                elif type == "pointvalues":
                    info['postproc'][name] = self.fem.computePointValues(u, colors)
                elif type == "meanvalues":
                    info['postproc'][name] = self.fem.computeMeanValues(u, colors)
                else:
                    raise ValueError("unknown postprocess '{}' for key '{}'".format(type, name))
        if self.kheatcell.shape[0] != self.mesh.ncells:
            raise ValueError(f"self.kheatcell.shape[0]={self.kheatcell.shape[0]} but self.mesh.ncells={self.mesh.ncells}")
        if self.plotk: cell_data['k'] = self.kheatcell
        return point_data, cell_data, info

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
