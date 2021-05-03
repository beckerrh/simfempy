import numpy as np
from scipy import sparse
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.tools.analyticalfunction import AnalyticalFunction
import scipy.sparse.linalg as splinalg

#=================================================================#
class Beam(Application):
    """
    Class for the (stationary) 1D beam equation
    $$
    (EI w'')'' = f         domain
    w = w' = 0  clamped bdry
    w = w'' = 0  simply supported bdry
    w'' = w''' = 0  free bdry
    $$
    After initialization, the function setMesh(mesh) has to be called
    Then, solve() solves the stationary problem
    Parameters in the constructor:
        fem: only p1 or cr1
        problemdata
    Paramaters used from problemdata:
        EI
    Possible parameters for computaion of postprocess:
        errors
    """
    def __repr__(self):
        repr = super(Beam, self).__repr__()
        return repr
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fem = fems.p1.P1()
    def _checkProblemData(self):
        if self.verbose: print(f"checking problem data {self.problemdata=}")
        self.problemdata.check(self.mesh)
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            EI = self.problemdata.params.scal_glob['EI']
            rhs = EI * solexact.dddd(x, y, z)
            return rhs
        return _fctu
    def defineClampedAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctneumann(x, y, z, nx, ny, nz):
            rhs = solexact.d(0, x, y, z) * nx
            return rhs
        return solexact, _fctneumann
    def solve(self, iter, dirname): return self.static(iter, dirname)
    def setMesh(self, mesh):
        assert mesh.dimension == 1
        super().setMesh(mesh)
        # if mesh is not None: self.mesh = mesh
        self._checkProblemData()
        self.fem.setMesh(self.mesh)
        self.EIcell = self.compute_cell_vector_from_params('EI', self.problemdata.params)
    def computeMatrix(self, coeffmass=None):
        A = self.fem.computeMatrixDiffusion(coeff=1)
        n = self.fem.nunknowns()
        mats = np.ones(2)
        rows = np.array([0,1])
        # l'extrémité droite et le deuxième points !
        cols = np.array([0,1])
        C = sparse.coo_matrix((mats, (rows, cols)), shape=(2, n)).tocsr()
        dV = self.mesh.dV
        D = dV / self.EIcell / 4
        # D = np.arange(1,n)
        # print(f"{D=}")
        E = np.empty(n)
        E[:-1] = D
        E[1:] += D
        # print(f"{E=}")
        B = sparse.diags((D, E, D), offsets=(-1,0,1), shape=(n, n))
        # raise ValueError(f"B=\n{B.toarray()}\n")
        return A, B, C
    def computeRhs(self, b=None, u=None, coeffmass=None):
        if b is None:
            a = np.zeros(self.fem.nunknowns())
            b = np.zeros(self.fem.nunknowns())
            c = np.zeros(2)
        if 'rhs' in self.problemdata.params.fct_glob:
            xc, yc, zc = self.mesh.pointsc.T
            dV, simplices = self.mesh.dV, self.mesh.simplices
            fc = self.problemdata.params.fct_glob['rhs'](xc, yc, zc)
            # print(f"{fc=}")
            self.fem.massDotCell(a, fc)
            Dmub = -dV**3/self.EIcell/24*fc
            np.add.at(b, simplices, Dmub[:, np.newaxis])
        colors = self.problemdata.bdrycond.colorsOfType("Clamped")
        x, y, z = self.mesh.pointsf.T
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS
            nx, ny, nz = normalsS.T
            if not color in self.problemdata.bdrycond.fct: continue
            fct1, fct2 = self.problemdata.bdrycond.fct[color]
            c[faces[0]] = fct1(x[faces], y[faces], z[faces])
            dn = fct2(x[faces], y[faces], z[faces], nx, ny, nz)
            print(f"{dn=} {x[faces]=} {nx=} {faces=}")
            b[faces] += dn
        return (a,b,c), u
    def postProcess(self, uin):
        data = {'point':{}, 'cell':{}, 'global':{}}
        u,w,l = uin
        print(f"{l=} {u[0]=} {u[1]=}")
        data['point']['U'] = self.fem.tonode(u)
        data['point']['W'] = self.fem.tonode(w)
        if self.problemdata.solexact:
            data['global']['err_L2c'], ec = self.fem.computeErrorL2Cell(self.problemdata.solexact, u)
            data['global']['err_L2n'], en = self.fem.computeErrorL2(self.problemdata.solexact, u)
            data['cell']['err'] = ec
        return data
    def _to_single_matrix(self, Ain):
        n = self.fem.nunknowns()
        A, B, C = Ain
        null1 = sparse.coo_matrix(([], ([], [])), shape=(n, n))
        null2 = sparse.coo_matrix(([], ([], [])), shape=(2, n))
        null3 = sparse.coo_matrix(([], ([], [])), shape=(2, 2))
        A1 = sparse.hstack([null1, A.T, C.T])
        A2 = sparse.hstack([A, B, null2.T])
        A3 = sparse.hstack([C, null2, null3])
        Aall = sparse.vstack([A1, A2, A3]).tocsr()
        assert np.allclose(Aall.data, Aall.T.data)
        print(f"A=\n{Aall.toarray()}")
        return Aall
    def linearSolver(self, Ain, bin, uin=None, solver='umf', verbose=0):
        n = self.fem.nunknowns()
        if solver == 'umf':
            Aall = self._to_single_matrix(Ain)
            ball = np.hstack((bin[0], bin[1], bin[2]))
            uall =  splinalg.spsolve(Aall, ball, permc_spec='COLAMD')
            return (uall[:n], uall[n:2*n], uall[2*n:]), 1
        elif solver == 'gmres':
            raise NotImplemented()


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
