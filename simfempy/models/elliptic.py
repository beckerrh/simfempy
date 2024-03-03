import numpy as np
from simfempy import fems
from simfempy.models.model import Model
from simfempy.tools.analyticalfunction import AnalyticalFunction
import scipy
import scipy.sparse.linalg as splinalg
import simfempy.tools.iterationcounter
from simfempy.linalg import saddle_point, matrixsystem

# ================================================================= #
def Elliptic(**kwargs):
    if kwargs['fem'] == 'rt0': return EllipticMixed(**kwargs)
    elif kwargs['fem'] in ['p1', 'cr1']: return EllipticPrimal(**kwargs)
    else: raise NotImplementedError(f"unknown {kwargs['fem']=}")
# ================================================================= #
class EllipticBase(Model):
    """
    Class for the elliptic equation
    $$
    -\div(A \nabla T) + b\cdot\nabla u + c u= f         domain
    A\nabla\cdot n + alpha T = g  bdry
    $$
    After initialization, the function setMesh(mesh) has to be called
    Then, solve() solves the stationary problem
    Parameters in the constructor:
        fem: only p1, cr1, or rt0
        problemdata
        method
        masslumpedbdry, masslumpedvol
    Paramaters used from problemdata:
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
            repr = super().__format__(spec)
            repr += f"\nfem={self.fem}"
            return repr
        return self.__repr__()
    def __repr__(self):
        repr = super().__repr__()
        repr += f"\nfem={self.fem}"
        return repr
    def __init__(self, **kwargs):
        # print(f"{kwargs=}")
        self.fem = kwargs.pop('fem','p1')
        super().__init__(**kwargs)
    def createFem(self):
        self.convection = 'convection' in self.disc_params
        self.dirichletmethod = self.disc_params.pop('dirichletmethod','nitsche')
        if self.dirichletmethod=='nitsche':
            self.nitscheparam = self.disc_params.pop('nitscheparam', 10)
        if self.fem == 'p1': self.fem = fems.p1.P1()
        elif self.fem == 'cr1': self.fem = fems.cr1.CR1()
        else:
            self.rt = fems.rt0.RT0()
            self.d0 = fems.d0.D0()
            self.fem = "RT0-D0"
    def meshSet(self):
        if hasattr(self, 'A'):
            del self.A
        self._checkProblemData()
        self.kheatcell = self.compute_cell_vector_from_params('kheat', self.problemdata.params)
        if self.convection:
            convectionfct = self.problemdata.params.fct_glob['convection']
            self.convdata = self.fem.prepareAdvection(convectionfct, 1)
            colorsinflow = self.findInflowColors()
            colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
            if not set(colorsinflow).issubset(set(colorsdir)):
                raise ValueError(f"Inflow boundaries need to be subset of Dirichlet boundaries {colorsinflow=} {colorsdir=}")
    def findInflowColors(self):
        colors=[]
        for color in self.mesh.bdrylabels.keys():
            faces = self.mesh.bdrylabels[color]
            if np.any(self.convdata.betart[faces]<-1e-10): colors.append(color)
        return colors
    def _checkProblemData(self):
        if self.verbose: print(f"checking problem data {self.problemdata=}")
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
        kheat = self.problemdata.params.scal_glob['kheat']
        def _fctrobin(x, y, z, nx, ny, nz):
            rhs = np.zeros(x.shape)
            normals = nx, ny, nz
            rhs += alpha*solexact(x, y, z)
            for i in range(self.mesh.dimension):
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctrobin
    def setParameter(self, paramname, param):
        if paramname == "dirichlet_strong": self.fem.dirichlet_strong = param
        else:
            if not hasattr(self, self.paramname):
                raise NotImplementedError("{} has no paramater '{}'".format(self, self.paramname))
            cmd = "self.{} = {}".format(self.paramname, param)
            eval(cmd)
# ================================================================= #
class EllipticPrimal(EllipticBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def meshSet(self):
        super().meshSet()
        self.fem.setMesh(self.mesh)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        if self.dirichletmethod!="nitsche":
            self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
    def getSystemSize(self):
        ncomps = [1]
        ns = [self.fem.nunknowns()]
        return ncomps, ns
    def computeMassMatrix(self):
        lumped = self.disc_params.get('masslumped', False)
        return self.fem.computeMassMatrix(lumped=lumped)
    def computeForm(self, u):
        if not hasattr(self, 'A'):
            self.A = self.computeMatrix()
        # du2 = self.A@u
        du = np.zeros_like(u)
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        self.fem.computeFormDiffusion(du, u, self.kheatcell)
        if self.convection:
            self.fem.computeFormConvection(du, u, self.convdata)
        # if coeffmass is not None:
        #     self.fem.massDot(du, u, coeff=coeffmass)
        self.fem.massDotBoundary(du, u, colorsrobin, bdrycond.param, lumped=True)
        if self.dirichletmethod!="nitsche":
            self.fem.vectorBoundaryStrongEqual(du, u, self.bdrydata)
        else:
            self.fem.computeFormNitscheDiffusion(self.nitscheparam, du, u, self.kheatcell, colorsdir)
        # if not np.allclose(du,du2):
        #     # f = (f"\n{du[self.bdrydata.facesdirall]}\n{du2[self.bdrydata.facesdirall]}")
        #     raise ValueError(f"{np.linalg.norm(du-du2)}\n{du=}\n{du2=}")
        return du
    def computeMatrix(self, u=None, coeffmass=None):
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        A = self.fem.computeMatrixDiffusion(self.kheatcell)
        if self.dirichletmethod=="nitsche":
            A += self.fem.computeMatrixNitscheDiffusion(self.nitscheparam, diffcoff=self.kheatcell, colors=colorsdir)
        A += self.fem.computeBdryMassMatrix(colorsrobin, bdrycond.param, lumped=True)
        if self.convection:
            A += self.fem.computeMatrixConvection(self.convdata)
        if coeffmass is not None:
            A += self.fem.computeMassMatrix(coeff=coeffmass)
        # if hasattr(self, 'bdrydata'):
        if hasattr(self, 'bdrydata'):
            # print(f"{self.bdrydata=}")
            assert self.dirichletmethod!="nitsche"
            A = self.fem.matrixBoundaryStrong(A, self.bdrydata)
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
            if hasattr(self, 'convdata'): self.fem.massDotSupg(b, fp1, self.convdata)
        if 'rhscell' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolateCell(self.problemdata.params.fct_glob['rhscell'])
            self.fem.massDotCell(b, fp1)
        if 'rhspoint' in self.problemdata.params.fct_glob:
            self.fem.computeRhsPoint(b, self.problemdata.params.fct_glob['rhspoint'])
        if self.dirichletmethod=="nitsche":
            self.fem.computeRhsNitscheDiffusion(self.nitscheparam, b, self.kheatcell, colorsdir, udir=None, bdrycondfct=bdrycond.fct)
        else:
            self.fem.vectorBoundaryStrong(b, bdrycond, self.bdrydata)
        if self.convection:
            fp1 = self.fem.interpolateBoundary(self.mesh.bdrylabels.keys(), bdrycond.fct)
            self.fem.massDotBoundary(b, fp1, coeff=-np.minimum(self.convdata.betart, 0))
        #Fourier-Robin
        fp1 = self.fem.interpolateBoundary(colorsrobin, bdrycond.fct, lumped=True)
        # self.fem.massDotBoundary(b, fp1, colors=colorsrobin, lumped=True, coeff=bdrycond.param)
        self.fem.massDotBoundary(b, fp1, colors=colorsrobin, lumped=True, coeff=1)
        #Neumann
        fp1 = self.fem.interpolateBoundary(colorsneu, bdrycond.fct)
        self.fem.massDotBoundary(b, fp1, colorsneu)
        if coeffmass is not None:
            assert u is not None
            self.fem.massDot(b, u, coeff=coeffmass)
        if hasattr(self, 'bdrydata'):
            self.fem.vectorBoundaryStrong(b, bdrycond, self.bdrydata)
        return b
    def sol_to_data(self, u):
        data = {'point': {}, 'cell': {}, 'global': {}}
        # point_data, side_data, cell_data, global_data = {}, {}, {}, {}
        data['point']['U'] = self.fem.tonode(u)
        return data
    def postProcess(self, u):
        data = {'scalar':{}}
        solexact = self.application.exactsolution
        if solexact:
            data['scalar']['err_L2c'], ec = self.fem.computeErrorL2Cell(solexact, u)
            data['scalar']['err_L2n'], en = self.fem.computeErrorL2 (solexact, u)
            data['scalar']['err_H1'] = self.fem.computeErrorFluxL2  (solexact, u)
            data['scalar']['err_Flux'] = self.fem.computeErrorFluxL2(solexact, u, self.kheatcell)
            data['cell'] = {}
            data['cell']['err'] = ec
        if self.problemdata.postproc:
            types = ["bdry_mean", "bdry_fct", "bdry_nflux", "pointvalues", "meanvalues", "linemeans"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['scalar'][name] = self.fem.computeBdryMean(u, colors)
                elif type == types[1]:
                    data['scalar'][name] = self.fem.computeBdryFct(u, colors)
                elif type == types[2]:
                    if self.dirichletmethod == 'nitsche':
                        udir = self.fem.interpolateBoundary(colors, self.problemdata.bdrycond.fct)
                        data['scalar'][name] = self.fem.computeBdryNormalFluxNitsche(self.nitscheparam, u, colors, udir, self.kheatcell)
                    else:
                        data['scalar'][name] = self.fem.computeBdryNormalFlux(u, colors, self.bdrydata, self.problemdata.bdrycond, self.kheatcell)
                elif type == types[3]:
                    data['scalar'][name] = self.fem.computePointValues(u, colors)
                elif type == types[4]:
                    data['scalar'][name] = self.fem.computeMeanValues(u, colors)
                elif type == types[5]:
                    data['scalar'][name] = self.fem.computeLineValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data
    def pyamg_solver_args(self, maxiter):
        if self.convection:
            return {'cycle': 'V', 'maxiter': maxiter, 'tol': 1e-10, 'accel': 'bicgstab'}
        return {'cycle': 'V', 'maxiter': maxiter, 'tol': 1e-10, 'accel': 'cg'}
    def build_pyamg(self, A):
        try:
            import pyamg
        except:
            raise ImportError(f"*** pyamg not found {self.linearsolver=} ***")
        # return pyamg.smoothed_aggregation_solver(A)
        B = np.ones((A.shape[0], 1))
        B = pyamg.solver_configuration(A, verb=False)['B']
        if self.convection:
            symmetry = 'nonsymmetric'
            # smoother = 'gauss_seidel_nr'
            smoother = 'gauss_seidel'
            smoother = 'block_gauss_seidel'
            smoother = 'strength_based_schwarz'
            smoother = 'schwarz'
            # smooth = ('energy', {'krylov': 'fgmres'})
            smooth = ('energy', {'krylov': 'bicgstab'})
            # improve_candidates =[ (smoother, {'sweep': 'symmetric', 'iterations': 4}), None]
            improve_candidates = None
        else:
            symmetry = 'hermitian'
            smooth = ('energy', {'krylov': 'cg'})
            smoother = 'gauss_seidel'
            # improve_candidates =[ ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None]
            improve_candidates = None
        # strength = [('evolution', {'k': 2, 'epsilon': 10.0})]
        strength = [('symmetric', {'theta': 0.05})]
        psmoother = (smoother, {'sweep': 'symmetric', 'iterations': 1})
        # psmoother = (smoother, {'maxiter': 10})
        SA_build_args = {
            'max_levels': 10,
            'max_coarse': 25,
            'coarse_solver': 'pinv2',
            'symmetry': symmetry,
            'smooth': smooth,
            'strength': strength,
            'presmoother': psmoother,
            'presmoother': psmoother,
            'improve_candidates': improve_candidates,
            'diagonal_dominance': False
        }
        # return pyamg.smoothed_aggregation_solver(A, B, **SA_build_args)
        return pyamg.rootnode_solver(A, B, **SA_build_args)
# ================================================================= #
class EllipticMixed(EllipticBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def meshSet(self):
        super().meshSet()
        self.pureneumann = not ('Dirichlet' in self.problemdata.bdrycond.type.values() or 'Robin' in self.problemdata.bdrycond.type.values())
        # print(f"{self.pureneumann=} {self.dirichletmethod=}")
        self.rt.setMesh(self.mesh)
        self.d0.setMesh(self.mesh)
        colorsneumann = self.problemdata.bdrycond.colorsOfType("Neumann")
        self.bdrydata = self.rt.prepareBoundary(colorsneumann)
        self.divcoeffinv = 1/self.kheatcell
    def getSystemSize(self):
        ncomps = [1, 1]
        ns = [self.rt.nunknowns(), self.d0.nunknowns()]
        return ncomps, ns
    # def computeForm(self, u, coeffmass=None):
    #     raise NotImplementedError(f"computeForm for rt")
    def computeMatrix(self, u=None, coeffmass=None):
        # print("Hallo computeMatrix")
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        A = self.rt.constructMass(diffinvcell=self.divcoeffinv)
        B = self.rt.constructDiv()
        A += self.rt.computeBdryMassMatrix(colorsrobin, bdrycond.param)
        if self.convection:
            raise NotImplementedError(f"convection for rt")
        if coeffmass is not None:
            raise NotImplementedError(f"recation for rt")
        if self.pureneumann:
            A, B, self.bdrydata = self.rt.matrixNeumann(A, B, self.bdrydata)
        positions = [[{'pos':(0,0)}], [{'pos':(1,0)}, {'pos':(0,1), 'trp':True, 'scl':1}]]
        return matrixsystem.MatrixSystem(self.vectorview, [A,B], positions)
        raise ValueError(f"{M=}")
        return saddle_point.SaddlePointSystem(self.vectorview, [A, B], singleA=False)
        # return A, B
    def computeRhs(self, b=None, coeffmass=None, u=None):
        assert b == None
        b = np.zeros(self.vectorview.n())
        bsides = self.vectorview.get_part(0, b)
        bcells = self.vectorview.get_part(1, b)
        # bsides = np.zeros(self.mesh.nfaces)
        # bcells = np.zeros(self.mesh.ncells)
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdirrobin = bdrycond.colorsOfType(["Dirichlet","Robin"])
        colorsneu = bdrycond.colorsOfType("Neumann")
        if 'rhs' in self.problemdata.params.fct_glob:
            fp1 = self.d0.interpolate(self.problemdata.params.fct_glob['rhs'])
            self.d0.massDot(bcells, fp1)
        for color in colorsdirrobin:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            normalsS = normalsS / dS[:, np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            try:
                ud = bdrycond.fct[color](xf, yf, zf, nx, ny, nz)
            except:
                ud = bdrycond.fct[color](xf, yf, zf)
            if color in colorsrobin: dS /= bdrycond.param[color]
            bsides[faces] += dS * ud
            if self.convection:
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = np.linalg.norm(normalsS, axis=1)
                normalsS = normalsS / dS[:, np.newaxis]
                xf, yf, zf = self.mesh.pointsf[faces].T
                beta = np.array(self.convection(xf, yf, zf))
                # print("beta", beta)
                # print("normalsS", normalsS.T)
                bn = np.einsum("ij,ij->j", beta, normalsS.T)
                # print("bn", bn)
                bn[bn<=0] = 0
                cells = self.mesh.cellsOfFaces[faces,0]
                bcells[cells] += bn*ud*dS
        help = np.zeros(self.mesh.nfaces)
        for color in colorsneu:
            if not color in bdrycond.fct or not bdrycond.fct[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            normalsS = normalsS / dS[:, np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            help[faces] += bdrycond.fct[color](xf, yf, zf, nx, ny, nz)
        # print(f"{self.bdrydata=}")
        # print(f"{type(self.bdrydata.A_inner_neum)=}")
        if self.pureneumann:
            bsides[self.bdrydata.facesinner] -= self.bdrydata.A_inner_neum*help[self.bdrydata.facesneumann]
            bsides[self.bdrydata.facesneumann] += self.bdrydata.A_neum_neum*help[self.bdrydata.facesneumann]
            bcells -= self.bdrydata.B_inner_neum*help[self.bdrydata.facesneumann]
        return b
        # return bsides, bcells
    def sol_to_data(self, u):
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        v = self.vectorview.get_part(0, u)
        p = self.vectorview.get_part(1, u)
        # sigma, u = uin[0], uin[1]
        # sigma, u = u[:nfaces], u[nfaces:]
        data = {'point':{}, 'cell':{}, 'global':{}}
        # point_data, side_data, cell_data, global_data = {}, {}, {}, {}
        data['cell']['p'] = p
        vc = self.rt.toCell(v)
        pn = self.rt.reconstruct(p, vc, self.divcoeffinv)
        data['point']['pn'] = pn
        for i in range(dim):
            data['cell']['v{:1d}'.format(i)] = vc[:, i]
        return data
    def postProcess(self, u):
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        v = self.vectorview.get_part(0, u)
        p = self.vectorview.get_part(1, u)
        vc = self.rt.toCell(v)
        pn = self.rt.reconstruct(p, vc, self.divcoeffinv)
        data = {}
        if self.application.exactsolution:
            errp, errs, errpn, pe, ve = self.computeError(self.application.exactsolution, p, vc, pn)
            data['cell']={}
            data['cell']['err_p'] = np.abs(pe - p)
            for i in range(dim):
                data['cell']['err_v{:1d}'.format(i)] = np.abs(ve[i] - vc[:,i])
            data['scalar']={}
            data['scalar']['err_L2c'] = errp
            data['scalar']['err_L2n'] = errpn
            data['scalar']['err_Flux'] = errs
        if self.problemdata.postproc:
            types = ["bdry_mean", "bdry_fct", "bdry_nflux", "pointvalues", "meanvalues", "linemeans"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['scalar'][name] = self.computeBdryMean(pn, colors)
                elif type == types[1]:
                    data['scalar'][name] = self.fem.computeBdryFct(u, colors)
                elif type == types[2]:
                    data['scalar'][name] = self.computeBdryDn(v, colors)
                elif type == types[3]:
                    data['scalar'][name] = self.fem.computePointValues(u, colors)
                elif type == types[4]:
                    data['scalar'][name] = self.fem.computeMeanValues(u, colors)
                elif type == types[5]:
                    data['scalar'][name] = self.fem.computeLineValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data
    def computeBdryDn(self, u, colors):
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            flux[i] = np.sum(dS*u[faces])
        return flux
        return flux/omega
    def computeBdryMean(self, pn, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*np.mean(pn[self.mesh.faces[faces]],axis=1))
        return mean/omega
    def computeError(self, solexact, p, vc, pn):
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        xc, yc, zc = self.mesh.pointsc.T
        pex = solexact(xc, yc, zc)
        errp = np.sqrt(np.sum((pex-p)**2* self.mesh.dV))
        errv = 0
        vexx=[]
        for i in range(dim):
            solxi = self.kheatcell*solexact.d(i, xc, yc, zc)
            errv += np.sum( (solxi-vc[:,i])**2* self.mesh.dV)
            vexx.append(solxi)
        errv = np.sqrt(errv)
        x, y, z = self.mesh.points.T
        errpn = solexact(x, y, z) - pn
        errpn = errpn**2
        errpn= np.mean(errpn[self.mesh.simplices], axis=1)
        errpn = np.sqrt(np.sum(errpn* self.mesh.dV))
        return errp, errv, errpn, pex, vexx

    def _to_single_matrix(self, Ain):
        A, B = Ain
        ncells = self.mesh.ncells
        help = np.zeros((ncells))
        help = scipy.sparse.dia_matrix((help, 0), shape=(ncells, ncells))
        A1 = scipy.sparse.hstack([A, B.T])
        A2 = scipy.sparse.hstack([B, help])
        Aall = scipy.sparse.vstack([A1, A2])
        return Aall.tocsr()
    # def computelinearSolver(self, A):
    #     if isinstance(self.linearsolver,str):
    #         args = {'method': self.linearsolver}
    #     else:
    #         args = self.linearsolver.copy()
    #     return simfempy.linalg.linalg.getLinearSolver(**args)
    # def solvelinear(self, Ain, bin, u=None, solver = None, verbose=0):
    #     if solver is None: solver = self.linearsolver
    #     if solver == 'spsolve':
    #         # print("bin", bin)
    #         Aall = self._to_single_matrix(Ain)
    #         b = np.concatenate((bin[0], bin[1]))
    #         u =  splinalg.spsolve(Aall, b, permc_spec='COLAMD')
    #         # print("u", u)
    #         return u, 1
    #     elif solver == 'gmres':
    #         nfaces, ncells = self.mesh.nfaces, self.mesh.ncells
    #         counter = simfempy.tools.iterationcounter.IterationCounter(name=solver, verbose=verbose)
    #         # Aall = self._to_single_matrix(Ain)
    #         # M2 = splinalg.spilu(Aall, drop_tol=0.2, fill_factor=2)
    #         # M_x = lambda x: M2.solve(x)
    #         # M = splinalg.LinearOperator(Aall.shape, M_x)
    #         A, B = Ain
    #         A, B = A.tocsr(), B.tocsr()
    #         D = scipy.sparse.diags(1/A.diagonal(), offsets=(0), shape=(nfaces,nfaces))
    #         S = -B*D*B.T
    #         import pyamg
    #         config = pyamg.solver_configuration(S, verb=False)
    #         ml = pyamg.rootnode_solver(S, B=config['B'], smooth='energy')
    #         # Ailu = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
    #         def amult(x):
    #             v,p = x[:nfaces],x[nfaces:]
    #             return np.hstack( [A.dot(v) + B.T.dot(p), B.dot(v)])
    #         Amult = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=amult)
    #         def pmult(x):
    #             v,p = x[:nfaces],x[nfaces:]
    #             w = D.dot(v)
    #             # w = Ailu.solve(v)
    #             q = ml.solve(p - B.dot(w), maxiter=1, tol=1e-16)
    #             w = w - D.dot(B.T.dot(q))
    #             # w = w - Ailu.solve(B.T.dot(q))
    #             return np.hstack( [w, q] )
    #         P = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=pmult)
    #         u,info = splinalg.lgmres(Amult, bin, M=P, callback=counter, atol=1e-12, tol=1e-12, inner_m=10, outer_k=4)
    #         if info: raise ValueError("no convergence info={}".format(info))
    #         # print("u", u)
    #         return u, counter.niter
    #     else:
    #         raise NotImplementedError("solver '{}' ".format(solver))


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
