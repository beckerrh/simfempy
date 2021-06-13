import numpy as np
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.tools.analyticalfunction import AnalyticalFunction

# ================================================================= #
class Heat(Application):
    """
    Class for the (stationary) heat equation
    $$
    rhoCp (T_t + beta\cdot\nabla T) -\div(kheat \nabla T) = f         domain
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
    def __format__(self, spec):
        if spec=='-':
            repr = super(Heat, self).__format__(spec)
            if self.convection: repr += f"\tstab={self.convmethod}"
            repr += f"\tdirichletmethod={self.dirichletmethod}"
            repr += f"\tmasslumpedbdry={self.masslumpedbdry}"
            repr += f"\tmasslumpedvol={self.masslumpedvol}"
            return repr
        return self.__repr__()
    def __repr__(self):
        repr = super(Heat, self).__repr__()
        repr += f"\nfem={self.fem}"
        repr += f"\tstab={self.convmethod}"
        repr += f"\tdirichletmethod={self.dirichletmethod}"
        repr += f"\tmasslumpedbdry={self.masslumpedbdry}"
        return repr
    def __init__(self, **kwargs):
        self.masslumpedvol = kwargs.pop('masslumpedvol', True)
        self.masslumpedbdry = kwargs.pop('masslumpedbdry', True)
        fem = kwargs.pop('fem','p1')
        self.dirichletmethod = kwargs.pop('dirichletmethod', 'strong')
        self.convmethod = kwargs.pop('convmethod', 'supg')
        femargs = {'dirichletmethod':self.dirichletmethod, 'stab': self.convmethod}
        if fem == 'p1': self.fem = fems.p1.P1(**femargs)
        elif fem == 'cr1': self.fem = fems.cr1.CR1(**femargs)
        else: raise ValueError("unknown fem '{}'".format(fem))
        self.convection = 'convection' in kwargs['problemdata'].params.fct_glob.keys()
        super().__init__(**kwargs)
        #fct_glob
    def setMesh(self, mesh):
        super().setMesh(mesh)
        # if mesh is not None: self.mesh = mesh
        self._checkProblemData()
        self.fem.setMesh(self.mesh)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        if self.dirichletmethod != 'nitsche':
            self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
        self.kheatcell = self.compute_cell_vector_from_params('kheat', self.problemdata.params)
        self.problemdata.params.scal_glob.setdefault('rhocp',1)
        # TODO: non-constant rhocp
        rhocp = self.problemdata.params.scal_glob.setdefault('rhocp', 1)
        if self.convection:
            convectionfct = self.problemdata.params.fct_glob['convection']
            self.convdata = self.fem.prepareAdvection(convectionfct, rhocp, self.convmethod)
            colorsinflow = self.findInflowColors()
            colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
            if not set(colorsinflow).issubset(set(colorsdir)):
                raise ValueError(f"Inflow boundaries nees to be subset of Dirichlet boundaries {colorsinflow=} {colorsdir=}")
    def findInflowColors(self):
        colors=[]
        for color in self.mesh.bdrylabels.keys():
            faces = self.mesh.bdrylabels[color]
            if np.any(self.convdata.betart[faces]<-1e-10): colors.append(color)
        return colors
    def _checkProblemData(self):
        if self.verbose: print(f"checking problem data {self.problemdata=}")
        self.problemdata.check(self.mesh)
        if self.convection:
            convection_given = self.problemdata.params.fct_glob['convection']
            if not isinstance(convection_given, list):
                p = "problemdata.params.fct_glob['convection']"
                raise ValueError(f"need '{p}' as a list of length dim of str or AnalyticalSolution")
            elif isinstance(convection_given[0],str):
                self.problemdata.params.fct_glob['convection'] = [AnalyticalFunction(expr=e) for e in convection_given]
            else:
                if not isinstance(convection_given[0], AnalyticalFunction):
                    raise ValueError(f"convection should be given as 'str' and not '{type(convection_given[0])}'")
            if len(self.problemdata.params.fct_glob['convection']) != self.mesh.dimension:
                raise ValueError(f"{self.mesh.dimension=} {self.problemdata.params.fct_glob['convection']=}")
        bdrycond = self.problemdata.bdrycond
        for color in self.mesh.bdrylabels:
            if not color in bdrycond.type: raise ValueError(f"color={color} not in bdrycond={bdrycond}")
            if bdrycond.type[color] in ["Robin"]:
                if not color in bdrycond.param:
                    raise ValueError(f"Robin condition needs paral 'alpha' color={color} bdrycond={bdrycond}")
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            kheat = self.problemdata.params.scal_glob['kheat']
            beta = self.problemdata.params.fct_glob['convection']
            rhs = np.zeros(x.shape)
            for i in range(self.mesh.dimension):
                rhs += beta[i](x,y,z) * solexact.d(i, x, y, z)
                rhs -= kheat * solexact.dd(i, i, x, y, z)
            return rhs
        def _fctu2(x, y, z):
            kheat = self.problemdata.params.scal_glob['kheat']
            rhs = np.zeros(x.shape)
            for i in range(self.mesh.dimension):
                rhs -= kheat * solexact.dd(i, i, x, y, z)
            return rhs
        if self.convection: return _fctu
        return _fctu2
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctneumann(x, y, z, nx, ny, nz):
            kheat = self.problemdata.params.scal_glob['kheat']
            rhs = np.zeros(x.shape)
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
            rhs = np.zeros(x.shape)
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
        if paramname == "dirichlet_strong": self.fem.dirichlet_strong = param
        else:
            if not hasattr(self, self.paramname):
                raise NotImplementedError("{} has no paramater '{}'".format(self, self.paramname))
            cmd = "self.{} = {}".format(self.paramname, param)
            eval(cmd)
    def computeForm(self, u, coeffmass=None):
        du2 = self.A@u
        du = np.zeros_like(u)
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        self.fem.computeFormDiffusion(du, u, self.kheatcell)
        if self.dirichletmethod=="new":
            self.fem.formBoundary(du, u, self.bdrydata, self.dirichletmethod)
        elif self.dirichletmethod=="nitsche":
            self.fem.computeFormNitscheDiffusion(du, u, self.kheatcell, colorsdir)
        # elif self.dirichletmethod=="strong":
        #     self.fem.vectorBoundaryStrongEqual(u, self.b, self.bdrydata, self.dirichletmethod)
        if self.convection:
            self.fem.computeFormConvection(du, u, self.convdata, self.convmethod)
        if coeffmass is not None:
            self.fem.massDot(du, u, coeff=coeffmass)
        if self.dirichletmethod=="strong":
            self.fem.vectorBoundaryStrongEqual(du, u, self.bdrydata)
        if not np.allclose(du,du2):
            # f = (f"\n{du[self.bdrydata.facesdirall]}\n{du2[self.bdrydata.facesdirall]}")
            raise ValueError(f"\n{du=}\n{du2=}")
        return du
    def computeMatrix(self, u=None, coeffmass=None):
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        A = self.fem.computeMatrixDiffusion(self.kheatcell)
        if self.dirichletmethod=="new":
            A = self.fem.matrixBoundaryStrong(A, self.bdrydata, self.dirichletmethod)
        elif self.dirichletmethod=="nitsche":
            A += self.fem.computeMatrixNitscheDiffusion(diffcoff=self.kheatcell, colorsdir=colorsdir)
        if self.verbose: print(f"{A.diagonal()=}")
        A += self.fem.computeBdryMassMatrix(colorsrobin, bdrycond.param, lumped=True)
        if self.convection:
            A += self.fem.computeMatrixConvection(self.convdata, self.convmethod)
        if coeffmass is not None:
            A += self.fem.computeMassMatrix(coeff=coeffmass)
        if self.dirichletmethod=="strong":
            A = self.fem.matrixBoundaryStrong(A, self.bdrydata, self.dirichletmethod)
        return A
    def computeRhs(self, b=None, coeffmass=None, u=None):
        if b is None:
            b = np.zeros(self.fem.nunknowns())
        else:
            if b.shape[0] != self.fem.nunknowns(): raise ValueError(f"{b.shape=} {self.fem.nunknowns()=}")
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        colorsneu = bdrycond.colorsOfType("Neumann")
        if 'rhs' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolate(self.problemdata.params.fct_glob['rhs'])
            self.fem.massDot(b, fp1)
            if self.convection and self.convmethod[:4] == 'supg':
                self.fem.massDotSupg(b, fp1, self.convdata)
        if 'rhscell' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolateCell(self.problemdata.params.fct_glob['rhscell'])
            self.fem.massDotCell(b, fp1)
        if 'rhspoint' in self.problemdata.params.fct_glob:
            self.fem.computeRhsPoint(b, self.problemdata.params.fct_glob['rhspoint'])
        if self.dirichletmethod in ['strong','new'] and not hasattr(self.bdrydata,"A_inner_dir"):
            raise ValueError("matrix() has to be called befor computeRhs()")
        if self.dirichletmethod=="new":
            self.fem.vectorBoundaryStrong(b, bdrycond, self.bdrydata, self.dirichletmethod)
        elif self.dirichletmethod=="nitsche":
            fp1 = self.fem.interpolateBoundary(colorsdir, bdrycond.fct)
            self.fem.computeRhsNitscheDiffusion(b, self.kheatcell, colorsdir, fp1)
        if self.convection:
            fp1 = self.fem.interpolateBoundary(self.mesh.bdrylabels.keys(), bdrycond.fct)
            self.fem.massDotBoundary(b, fp1, coeff=-np.minimum(self.convdata.betart, 0), lumped=self.masslumpedbdry)
        #Fourier-Robin
        fp1 = self.fem.interpolateBoundary(colorsrobin, bdrycond.fct, lumped=True)
        self.fem.massDotBoundary(b, fp1, colors=colorsrobin, lumped=True, coeff=bdrycond.param)
        #Neumann
        fp1 = self.fem.interpolateBoundary(colorsneu, bdrycond.fct, lumped=True)
        if self.dirichletmethod == "new":
            b2 = np.zeros_like(b)
            self.fem.massDotBoundary(b2, fp1, colorsneu)
            self.fem.vectorBoundaryStrongZero(b2, self.bdrydata)
            b += b2
        else:
            self.fem.massDotBoundary(b, fp1, colorsneu)
        if coeffmass is not None:
            assert u is not None
            self.fem.massDot(b, u, coeff=coeffmass)
        if self.dirichletmethod == "strong" or self.dirichletmethod == "new":
            self.fem.vectorBoundaryStrong(b, bdrycond, self.bdrydata, self.dirichletmethod)
        return b
    def postProcess(self, u):
        # TODO: virer 'error' et 'postproc'
        data = {'point':{}, 'cell':{}, 'global':{}}
        # point_data, side_data, cell_data, global_data = {}, {}, {}, {}
        data['point']['U'] = self.fem.tonode(u)
        if self.problemdata.solexact:
            data['global']['err_L2c'], ec = self.fem.computeErrorL2Cell(self.problemdata.solexact, u)
            data['global']['err_L2n'], en = self.fem.computeErrorL2(self.problemdata.solexact, u)
            data['global']['err_H1'] = self.fem.computeErrorFluxL2(self.problemdata.solexact, u)
            data['global']['err_Flux'] = self.fem.computeErrorFluxL2(self.problemdata.solexact, u, self.kheatcell)
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
                    if self.dirichletmethod == 'nitsche':
                        udir = self.fem.interpolateBoundary(colors, self.problemdata.bdrycond.fct)
                        data['global'][name] = self.fem.computeBdryNormalFluxNitsche(u, colors, udir, self.kheatcell)
                    else:
                        data['global'][name] = self.fem.computeBdryNormalFlux(u, colors, self.bdrydata, self.problemdata.bdrycond, self.kheatcell)
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
    def pyamg_solver_args(self, maxiter):
        return {'cycle': 'W', 'maxiter': maxiter, 'tol': 1e-12, 'accel': 'bicgstab'}
    def own_gs(A, b):
        x = np.zeros_like(b)
        return x
    def pyamg_solver_args(self, maxiter):
        if self.convection:
            return {'cycle': 'V', 'maxiter': maxiter, 'tol': 1e-13, 'accel': 'bicgstab'}
        return {'cycle': 'V', 'maxiter': maxiter, 'tol': 1e-13, 'accel': 'cg'}
    def build_pyamg(self, A):
        import pyamg
        # return pyamg.smoothed_aggregation_solver(A)
        B = np.ones((A.shape[0], 1))
        B = pyamg.solver_configuration(A, verb=False)['B']
        if self.convection:
            symmetry = 'nonsymmetric'
            # smoother = 'gauss_seidel_nr'
            smoother = 'gauss_seidel'
            smoother = 'block_gauss_seidel'
            # smoother = 'strength_based_schwarz'
            smoother = 'schwarz'

            # global setup_own_gs
            # def setup_own_gs(lvl, iterations=2, sweep='forward'):
            #     def smoother(A, x, b):
            #         pyamg.relaxation.gauss_seidel(A, x, b, iterations=iterations, sweep=sweep)
            #
            #     return smoother

            # smoother = 'own_gs'
            # smooth = ('energy', {'krylov': 'fgmres'})
            smooth = ('energy', {'krylov': 'bicgstab'})
            # improve_candidates =[ ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}), None]
            improve_candidates = None
        else:
            symmetry = 'hermitian'
            smooth = ('energy', {'krylov': 'cg'})
            smoother = 'gauss_seidel'
            # improve_candidates =[ ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None]
            improve_candidates = None
        strength = [('evolution', {'k': 2, 'epsilon': 10.0})]
        presmoother = (smoother, {'sweep': 'symmetric', 'iterations': 2})
        postsmoother = (smoother, {'sweep': 'symmetric', 'iterations': 2})
        SA_build_args = {
            'max_levels': 10,
            'max_coarse': 25,
            'coarse_solver': 'pinv2',
            'symmetry': symmetry
        }
        return pyamg.smoothed_aggregation_solver(A, B, smooth=smooth, strength=strength, presmoother=presmoother,
                                                 postsmoother=postsmoother, improve_candidates=improve_candidates,
                                                **SA_build_args)
        # return pyamg.rootnode_solver(A, B, **SA_build_args)


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
