import numpy as np
from simfempy import fems
from simfempy.applications.application import Application

#=================================================================#
class Transport(Application):
    """
    Class for the (stationary) transport equation
    $$
    alpha u + \div(beta u) = f   domain
    beta\cdot n = g              bdry
    $$
    After initialization, the function setMesh(mesh) has to be called
    Then, solve() solves the stationary problem
    Parameters in the constructor:
        fem: only p1 or cr1
        problemdata
        method
        plotk
    Paramaters used from problemdata:
        alpha : global constant from problemdata.paramglobal
        beta : RT0 field
    Possible parameters for computaion of postprocess:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'linearsolver' in kwargs: self.linearsolver = kwargs.pop('linearsolver')
        else: self.linearsolver = 'pyamg'
        fem = 'p1'
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        if fem == 'p1':
            self.fem = fems.p1.P1()
        elif fem == 'cr1':
            self.fem = fems.femcr1.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="trad"
    def _checkProblemData(self):
        self.problemdata.check(self.mesh)
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            kheat = self.problemdata.params.scal_glob['kheat']
            rhs = np.zeros(x.shape[0])
            for i in range(self.mesh.dimension):
                rhs -= kheat * solexact.dd(i, i, x, y, z)
            return rhs
        return _fctu
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctneumann(x, y, z, nx, ny, nz):
            kheat = self.problemdata.params.scal_glob['kheat']
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann
    def defineRobinAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        alpha = problemdata.bdrycond.param[color]
        # alpha = 1
        def _fctrobin(x, y, z, nx, ny, nz):
            kheat = self.problemdata.params.scal_glob['kheat']
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            # print(f"{alpha=}")
            rhs += alpha*solexact(x, y, z)
            for i in range(self.mesh.dimension):
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctrobin
    def setParameter(self, paramname, param):
        if paramname == "dirichlet_al": self.fem.dirichlet_al = param
        else:
            if not hasattr(self, self.paramname):
                raise NotImplementedError("{} has no paramater '{}'".format(self, self.paramname))
            cmd = "self.{} = {}".format(self.paramname, param)
            eval(cmd)

    def setMesh(self, mesh):
        super().setMesh(mesh)
        # if mesh is not None: self.mesh = mesh
        self._checkProblemData()
        self.fem.setMesh(self.mesh)
        dircols = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        self.bdrydata = self.fem.prepareBoundary(dircols, colorsflux)
        params = self.problemdata.params
        self.kheatcell = self.compute_cell_vector_from_params('kheat', params)
        # if not params.paramdefined('rhocp'): params.scal_glob['rhocp'] = 1
        # self.rhocpcell = self._computearrcell_('rhocp')

    def computeMatrix(self):
        bdrycond, method, bdrydata = self.problemdata.bdrycond, self.method, self.bdrydata
        A = self.fem.matrixDiffusion(self.kheatcell)
        lumped = False
        colors = bdrycond.colorsOfType("Robin")
        self.Arobin = self.fem.computeBdryMassMatrix(colors, bdrycond.param, lumped=lumped)
        A += self.Arobin
        A, self.bdrydata = self.fem.matrixBoundary(A, method, bdrydata)
        return A

    def computeRhs(self, u=None):
        if not hasattr(self.bdrydata,"A_inner_dir"):
            raise ValueError("matrix() has to be called befor computeRhs()")
        bdrycond, method, bdrydata = self.problemdata.bdrycond, self.method, self.bdrydata
        b = np.zeros(self.fem.nunknowns())

        # b = self.fem.computeRhsMass(b, self.problemdata.rhs, self.M)
        fp1 = self.fem.interpolate(self.problemdata.rhs)
        self.fem.massDot(b, fp1)
        if self.problemdata.rhscell:
            fp1 = self.fem.interpolateCell(self.problemdata.rhscell)
            self.fem.massDotCell(b, fp1)

        colors = bdrycond.colorsOfType("Robin")
        fp1 = self.fem.interpolateBoundary(colors, bdrycond.fct)
        self.fem.massDotBoundary(b, fp1, colors)


        colors = bdrycond.colorsOfType("Neumann")
        fp1 = self.fem.interpolateBoundary(colors, bdrycond.fct)
        self.fem.massDotBoundary(b, fp1, colors)

        # self.fem.computeRhsBoundary(b, bdrycond, ["Neumann"])
        # self.fem.computeRhsBoundaryMass(b, bdrycond, ["Robin"], self.Arobin)


        self.fem.computeRhsPoint(b, self.problemdata.rhspoint)
        # self.fem.computeRhsBoundary(b, bdrycond, ["Neumann", "Robin"])
        b, u, self.bdrydata = self.fem.vectorBoundary(b, u, bdrycond, method, bdrydata)
        return b,u


    def postProcess(self, u):
        point_data, side_data, cell_data, global_data = {}, {}, {}, {}
        # point_data['U'] = self.fem.tonode(u)
        # if self.problemdata.solexact:
        #     global_data['error'] = {}
        #     global_data['error']['pcL2'], ec = self.fem.computeErrorL2Cell(self.problemdata.solexact, u)
        #     global_data['error']['pnL2'], en = self.fem.computeErrorL2Node(self.problemdata.solexact, u)
        #     global_data['error']['vcL2'] = self.fem.computeErrorFluxL2(self.problemdata.solexact, self.kheatcell, u)
        #     cell_data['E'] = ec
        # global_data['postproc'] = {}
        # if self.problemdata.postproc:
        #     types = ["bdry_mean", "bdry_fct", "bdry_nflux", "pointvalues", "meanvalues"]
        #     for name, type in self.problemdata.postproc.type.items():
        #         colors = self.problemdata.postproc.colors(name)
        #         if type == types[0]:
        #             global_data['postproc'][name] = self.fem.computeBdryMean(u, colors)
        #         elif type == types[1]:
        #             global_data['postproc'][name] = self.fem.computeBdryFct(u, colors)
        #         elif type == types[2]:
        #             global_data['postproc'][name] = self.fem.computeBdryNormalFlux(u, colors, self.bdrydata, self.problemdata.bdrycond)
        #         elif type == types[3]:
        #             global_data['postproc'][name] = self.fem.computePointValues(u, colors)
        #         elif type == types[4]:
        #             global_data['postproc'][name] = self.fem.computeMeanValues(u, colors)
        #         else:
        #             raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        # if self.kheatcell.shape[0] != self.mesh.ncells:
        #     raise ValueError(f"self.kheatcell.shape[0]={self.kheatcell.shape[0]} but self.mesh.ncells={self.mesh.ncells}")
        # cell_data['k'] = self.kheatcell
        return point_data, side_data, cell_data, global_data


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
