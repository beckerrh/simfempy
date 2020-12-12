import numpy as np
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.fems.rt0 import RT0
from simfempy.tools.analyticalfunction import AnalyticalFunction

#=================================================================#
class Heat(Application):
    """
    Class for the (stationary) heat equation
    $$
    rhoCp (T_t + v\cdot\nabla T) -\div(kheat \nabla T) = f         domain
    kheat\nabla\cdot n + alpha T = g  bdry
    $$
    After initialization, the function setMesh(mesh) has to be called
    Then, solve() solves the stationary problem
    Parameters in the constructor:
        fem: only p1 or cr1
        problemdata
        method
        masslumpedbdry, masslumpedvol
    Paramaters used from problemdata:
        rhocp
        kheat
        reaction
        alpha
        they can either be given as global constant, cell-wise constants, or global function
        - global constant is taken from problemdata.paramglobal
        - cell-wise constants are taken from problemdata.paramcells
        - problemdata.paramglobal is taken from problemdata.datafct and are called with arguments (color, xc, yc, zc)
    Possible parameters for computaion of postprocess:
        errors
        bdry_mean: computes mean temperature over boundary parts according to given color
        bdry_nflux: computes mean normal flux over boundary parts according to given color
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linearsolver = kwargs.pop('linearsolver', 'pyamg')
        self.method = kwargs.pop('method', "trad")
        self.masslumpedbdry = kwargs.pop('masslumpedbdry', True)
        self.masslumpedvol = kwargs.pop('masslumpedvol', True)

        fem = kwargs.pop('fem','p1')
        if fem == 'p1': self.fem = fems.p1.P1()
        elif fem == 'cr1': self.fem = fems.femcr1.FemCR1()
        else: raise ValueError("unknown fem '{}'".format(fem))
        if 'convection' in self.problemdata.params.fct_glob.keys():
            convection_given = self.problemdata.params.fct_glob['convection']
            if not isinstance(convection_given, list):
                p = "problemdata.params.fct_glob['convection']"
                raise ValueError(f"need '{p}' as a list of length dim of str or AnalyticalSolution")
            elif isinstance(convection_given[0],str):
                self.problemdata.params.fct_glob['convection'] = [AnalyticalFunction(expr=e) for e in convection_given]
            else:
                if not isinstance(convection_given[0], AnalyticalFunction):
                    raise ValueError(f"convection should be given as 'str' and not '{type(convection_given[0])}'")
    def _checkProblemData(self):
        self.problemdata.check(self.mesh)
        bdrycond = self.problemdata.bdrycond
        for color in self.mesh.bdrylabels:
            if not color in bdrycond.type: raise ValueError(f"color={color} not in bdrycond={bdrycond}")
            if bdrycond.type[color] in ["Robin"]:
                if not color in bdrycond.param:
                    raise ValueError(f"Robin condition needs paral 'alpha' color={color} bdrycond={bdrycond}")
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
            # rhs += alpha*solexact(x, y, z)
            rhs += solexact(x, y, z)
            for i in range(self.mesh.dimension):
                # rhs += kheat * solexact.d(i, x, y, z) * normals[i]
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]/alpha
            return rhs
        return _fctrobin
    def setParameter(self, paramname, param):
        if paramname == "dirichlet_al": self.fem.dirichlet_al = param
        else:
            if not hasattr(self, self.paramname):
                raise NotImplementedError("{} has no paramater '{}'".format(self, self.paramname))
            cmd = "self.{} = {}".format(self.paramname, param)
            eval(cmd)
    def solve(self, iter, dirname): return self.static(iter, dirname)
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
        self.problemdata.params.scal_glob.setdefault('rhocp',1)
        # TODO: non-constant rhocp
        rhocp = self.problemdata.params.scal_glob.setdefault('rhocp', 1)
        # if not params.paramdefined('rhocp'): params.scal_glob['rhocp'] = 1
        # self.rhocpcell = self.compute_cell_vector_from_params('rhocp', params)
        if 'convection' in self.problemdata.params.fct_glob.keys():
            convectionfct = self.problemdata.params.fct_glob['convection']
            rt = RT0(mesh)
            self.convection = rt.interpolate(convectionfct)
            self.convection *= rhocp
            self.xd, self.lamdconvection, self.delta = self.fem.downWind(self.convection, method="supg")
            self.convectionC = rt.toCell(self.convection)
            conv = self.problemdata.params.fct_glob['convection']
            if len(conv) != self.mesh.dimension: raise ValueError(f"convection has wrong length")
            # for i, c in enumerate(conv):
            #     print(f"{c.fct_x[i]=}")
    def computeMatrix(self, coeffmass=None):
        bdrycond, method, bdrydata = self.problemdata.bdrycond, self.method, self.bdrydata
        A = self.fem.computematrixDiffusion(self.kheatcell)
        lumped = True
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        self.Arobin = self.fem.computeBdryMassMatrix(colorsrobin, bdrycond.param, lumped=self.masslumpedbdry)
        A += self.Arobin
        if 'convection' in self.problemdata.params.fct_glob.keys():
            A +=  self.fem.comptuteMatrixTransport(self.convection, self.convectionC, self.lamdconvection, colorsdir+colorsrobin)
        if coeffmass is not None:
            A += self.fem.computeMassMatrix(coeff=coeffmass)
        A, self.bdrydata = self.fem.matrixBoundary(A, method, bdrydata)
        return A
    def computeRhs(self, b=None, u=None, coeffmass=None, fillzeros=True):
        if not hasattr(self.bdrydata,"A_inner_dir"):
            raise ValueError("matrix() has to be called befor computeRhs()")
        bdrycond, method, bdrydata = self.problemdata.bdrycond, self.method, self.bdrydata
        if b is None:
            b = np.zeros(self.fem.nunknowns())
        else:
            if b.shape[0] != self.fem.nunknowns(): raise ValueError(f"{b.shape=} {self.fem.nunknowns()=}")
            if fillzeros: b.fill(0)
        if 'rhs' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolate(self.problemdata.params.fct_glob['rhs'])
            self.fem.massDot(b, fp1)
            if 'convection' in self.problemdata.params.fct_glob.keys():
                self.fem.massDotSupg(b, fp1, self.xd)
        if 'rhscell' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolateCell(self.problemdata.params.fct_glob['rhscell'])
            self.fem.massDotCell(b, fp1)
        if 'rhspoint' in self.problemdata.params.fct_glob:
            self.fem.computeRhsPoint(b, self.problemdata.params.fct_glob['rhspoint'])
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        colorsneu = bdrycond.colorsOfType("Neumann")
        if 'convection' in self.problemdata.params.fct_glob.keys():
            fp1 = self.fem.interpolateBoundary(colorsdir, bdrycond.fct)
            self.fem.massDotBoundary(b, fp1, coeff=-np.minimum(self.convection, 0), colors=colorsdir+colorsrobin)
        fp1 = self.fem.interpolateBoundary(colorsrobin, bdrycond.fct)
        self.fem.massDotBoundary(b, fp1, colorsrobin, lumped=self.masslumpedbdry, coeff=bdrycond.param)
        fp1 = self.fem.interpolateBoundary(colorsneu, bdrycond.fct)
        self.fem.massDotBoundary(b, fp1, colorsneu)
        if coeffmass is not None:
            assert u is not None
            self.fem.massDot(b, u, coeff=coeffmass)
        # print(f"***{id(u)=} {type(u)=}")
        b, u, self.bdrydata = self.fem.vectorBoundary(b, u, bdrycond, method, bdrydata)
        # print(f"***{id(u)=} {type(u)=}")
        return b,u

    # def residualNewton(self, u):


    #     if not hasattr(self, 'du'): self.du = np.empty_like(u)
    #     self.du[:] = 0
    #     self.fem.formDiffusion(self.du, u, self.kheatcell)
    #     self.du -= self.b
    #     self.du = self.vectorDirichletZero(self.du, self.bdrydata)
    #     return self.du

    def postProcess(self, u):
        # TODO: virer 'error' et 'postproc'
        data = {'point':{}, 'cell':{}, 'global':{}}
        # point_data, side_data, cell_data, global_data = {}, {}, {}, {}
        data['point']['U'] = self.fem.tonode(u)
        if self.problemdata.solexact:
            data['global']['err_pcL2'], ec = self.fem.computeErrorL2Cell(self.problemdata.solexact, u)
            data['global']['err_pnL2'], en = self.fem.computeErrorL2Node(self.problemdata.solexact, u)
            data['global']['err_vcL2'] = self.fem.computeErrorFluxL2(self.problemdata.solexact, self.kheatcell, u)
            data['cell']['err'] = ec
        if self.problemdata.postproc:
            types = ["bdry_mean", "bdry_fct", "bdry_nflux", "pointvalues", "meanvalues"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['global'][name] = self.fem.computeBdryMean(u, colors)
                elif type == types[1]:
                    data['global'][name] = self.fem.computeBdryFct(u, colors)
                elif type == types[2]:
                    data['global'][name] = self.fem.computeBdryNormalFlux(u, colors, self.bdrydata, self.problemdata.bdrycond)
                elif type == types[3]:
                    data['global'][name] = self.fem.computePointValues(u, colors)
                elif type == types[4]:
                    data['global'][name] = self.fem.computeMeanValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        if self.kheatcell.shape[0] != self.mesh.ncells:
            raise ValueError(f"self.kheatcell.shape[0]={self.kheatcell.shape[0]} but self.mesh.ncells={self.mesh.ncells}")
        data['cell']['k'] = self.kheatcell
        return data
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
        # return point_data, side_data, cell_data, global_data


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
