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
            rhs = EI * solexact.xxxx(x, y, z)
            return rhs
        return _fctu
    def defineClampedAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctclamped(x, y, z, nx, ny, nz):
            rhs = solexact.d(0, x, y, z) * nx
            return rhs
        return solexact, _fctclamped
    def defineSimplySupportedAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        EI = self.problemdata.params.scal_glob['EI']
        def _fctsimsupp2(x, y, z, nx, ny, nz):
            rhs = EI*solexact.xx(x, y, z) * nx
            return rhs
        return solexact, _fctsimsupp2
    def defineForcesAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        EI = self.problemdata.params.scal_glob['EI']
        def _fctsimsupp2(x, y, z, nx, ny, nz):
            rhs = EI*solexact.xx(x, y, z) * nx
            return rhs
        def _fctsimsupp3(x, y, z, nx, ny, nz):
            rhs = EI*solexact.xxx(x, y, z) * nx
            return rhs
        return _fctsimsupp2, _fctsimsupp3

    def setMesh(self, mesh):
        assert mesh.dimension == 1
        super().setMesh(mesh)
        # if mesh is not None: self.mesh = mesh
        self._checkProblemData()
        self.fem.setMesh(self.mesh)
        self.EIcell = self.compute_cell_vector_from_params('EI', self.problemdata.params)
        self.prepareBoundary()
    def prepareBoundary(self):
        self.facesDir = []
        colors = self.problemdata.bdrycond.colorsOfType("Clamped")
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            self.facesDir.append(faces[0])
        colors = self.problemdata.bdrycond.colorsOfType("SimplySupported")
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            self.facesDir.append(faces[0])
        self.faceNotNeu = []
        colors = self.problemdata.bdrycond.colorsOfType("SimplySupported")
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            self.faceNotNeu.append(faces[0])
        colors = self.problemdata.bdrycond.colorsOfType("Forces")
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            self.faceNotNeu.append(faces[0])

    def computeMatrix(self, coeffmass=None):
        A = self.fem.computeMatrixDiffusion(coeff=1)
        n = self.fem.nunknowns()
        ndir = len(self.facesDir)
        C1 = sparse.csr_matrix((np.ones(ndir), (np.arange(ndir), self.facesDir)), shape=(ndir, n)).tocsr()
        nnotn = len(self.faceNotNeu)
        C2 = sparse.csr_matrix((np.ones(nnotn), (np.arange(nnotn), self.faceNotNeu)), shape=(nnotn, n)).tocsr()
        dV = self.mesh.dV
        D = dV / self.EIcell / 4
        E = np.empty(n)
        E[:-1] = D
        E[1:] += D
        B = sparse.diags((D, E, D), offsets=(-1,0,1), shape=(n, n))
        return A, B, C1, C2
    def computeRhs(self, b=None, u=None, coeffmass=None):
        ndir = len(self.facesDir)
        nnotn = len(self.faceNotNeu)
        if b is None:
            a = np.zeros(self.fem.nunknowns())
            b = np.zeros(self.fem.nunknowns())
            c = np.zeros(ndir)
            d = np.zeros(nnotn)
        if 'rhs' in self.problemdata.params.fct_glob:
            xc, yc, zc = self.mesh.pointsc.T
            dV, simplices = self.mesh.dV, self.mesh.simplices
            fc = self.problemdata.params.fct_glob['rhs'](xc, yc, zc)
            self.fem.massDotCell(a, fc)
            Dmub = -dV**3/self.EIcell/24*fc
            np.add.at(b, simplices, Dmub[:, np.newaxis])
        x, y, z = self.mesh.pointsf.T
        idir=0
        colors = self.problemdata.bdrycond.colorsOfType("Clamped")
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS
            nx, ny, nz = normalsS.T
            if not color in self.problemdata.bdrycond.fct: continue
            fct1, fct2 = self.problemdata.bdrycond.fct[color]
            c[idir] = fct1(x[faces], y[faces], z[faces])
            idir += 1
            dn = fct2(x[faces], y[faces], z[faces], nx, ny, nz)
            cell = self.mesh.cellsOfFaces[faces[0], 0]
            # print(f"{nx=} {faces=} {self.mesh.simplices[cell]=}")
            b[faces] += dn
        colors = self.problemdata.bdrycond.colorsOfType("SimplySupported")
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS
            nx, ny, nz = normalsS.T
            if not color in self.problemdata.bdrycond.fct: continue
            fct1, fct2 = self.problemdata.bdrycond.fct[color]
            c[idir] = fct1(x[faces], y[faces], z[faces])
            idir += 1
            ddn = fct2(x[faces], y[faces], z[faces], nx, ny, nz)
            cell = self.mesh.cellsOfFaces[faces[0], 0]
            # print(f"{faces=} {self.mesh.simplices[cell]=} {self.fem.cellgrads[cell]=}")
            a[self.mesh.simplices[cell]] -= ddn*self.fem.cellgrads[cell][:,0]
        colors = self.problemdata.bdrycond.colorsOfType("Forces")
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS
            nx, ny, nz = normalsS.T
            if not color in self.problemdata.bdrycond.fct: continue
            fct1, fct2 = self.problemdata.bdrycond.fct[color]
            ddn = fct1(x[faces], y[faces], z[faces], nx, ny, nz)
            dddn = fct2(x[faces], y[faces], z[faces], nx, ny, nz)
            cell = self.mesh.cellsOfFaces[faces[0], 0]
            # print(f"{faces=} {self.mesh.simplices[cell]=} {self.fem.cellgrads[cell]=}")
            a[self.mesh.simplices[cell]] -= ddn*self.fem.cellgrads[cell][:,0]
            a[faces] += dddn
        return a,b,c,d
    def postProcess(self, uin):
        data = {'point':{}, 'cell':{}, 'global':{}}
        u,w,l = uin
        # print(f"{l=} {u[0]=} {u[1]=}")
        data['point']['U'] = self.fem.tonode(u)
        data['point']['W'] = self.fem.tonode(w)
        if self.problemdata.solexact:
            data['global']['err_L2c'], ec = self.fem.computeErrorL2Cell(self.problemdata.solexact, u)
            data['global']['err_L2n'], en = self.fem.computeErrorL2(self.problemdata.solexact, u)
            data['cell']['err'] = ec
        return data
    def _to_single_matrix(self, Ain):
        n = self.fem.nunknowns()
        A, B, C1, C2 = Ain
        n1, n2 = C1.shape[0], C2.shape[0]
        # print(f"{n1=} {n2=}")
        null1 = sparse.csr_matrix(([], ([], [])), shape=(n, n))
        null2 = sparse.csr_matrix(([], ([], [])), shape=(n1, n))
        null3 = sparse.csr_matrix(([], ([], [])), shape=(n1, n1))
        null4 = sparse.csr_matrix(([], ([], [])), shape=(n2, n))
        null5 = sparse.csr_matrix(([], ([], [])), shape=(n2, n2))
        null6 = sparse.csr_matrix(([], ([], [])), shape=(n1, n2))
        A1 = sparse.hstack([null1, A.T, C1.T, null4.T])
        A2 = sparse.hstack([A, B, null2.T, C2.T])
        A3 = sparse.hstack([C1, null2, null3, null6])
        A4 = sparse.hstack([null4, C2, null6.T, null5])
        Aall = sparse.vstack([A1, A2, A3, A4]).tocsr()
        assert np.allclose(A.data, A.T.data)
        assert np.allclose(Aall.data, Aall.T.data)
        # print(f"A=\n{Aall.toarray()}")
        return Aall.tocsr()
    def linearSolver(self, Ain, bin, uin=None, verbose=0):
        n = self.fem.nunknowns()
        if self.linearsolver == 'spsolve':
            Aall = self._to_single_matrix(Ain)
            ball = np.hstack((bin[0], bin[1], bin[2], bin[3]))
            uall =  splinalg.spsolve(Aall, ball, permc_spec='COLAMD')
            return (uall[:n], uall[n:2*n], uall[2*n:]), 1
        else:
            raise NotImplemented()


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
