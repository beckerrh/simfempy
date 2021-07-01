# assert __name__ == '__main__'
from os import sys, path
# simfempypath = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),'simfempy')
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

print("sys.path",sys.path)
import simfempy.applications
import numpy as np
import matplotlib.pyplot as plt
import copy


#----------------------------------------------------------------#
# class Plotter:
#     def __init__(self, solver):
#         self.solver = solver
#     def plot(self, **kwargs):
#         if not 'point_data' in kwargs: point_data = self.solver.point_data
#         else: point_data = kwargs.pop('point_data')
#         if not 'cell_data' in kwargs: cell_data = self.solver.cell_data
#         else: cell_data = kwargs.pop('cell_data')
#         quiver_cell_data={'v': (cell_data['v0'],cell_data['v1'])}
#         cell_data={'u':cell_data['p'], 'diff':cell_data['diff']}
#         point_data={}
#         kwargs['point_data'] = point_data
#         kwargs['cell_data'] = cell_data
#         kwargs['quiver_cell_data'] = quiver_cell_data
#         fig, axs = simfempy.meshes.plotmesh.meshWithData(self.solver.mesh, **kwargs)
#         plt.show()

#----------------------------------------------------------------#
class EIT(simfempy.applications.laplacemixed.LaplaceMixed):

    def conductivityinv(self, label):
        if label == 100:
            return self.diffglobalinv
        # print("conductivity", label, self.param[self.param_labels_inv[label]], self.param)
        return self.param[self.param_labels_inv[label]]

    def dconductivityinv(self, label):
        if label == self.dlabel:
            return self.dcoeff
        return 0

    def __init__(self, **kwargs):
        kwargs['plotdiff'] = True
        super().__init__(**kwargs)
        # self.plotter = Plotter(self)
        self.linearsolver = "spsolve"
        self.linearsolver = "gmres"
        self.measure_labels = kwargs.pop('measure_labels')
        self.measure_labels_inv = {}
        for i in range(len(self.measure_labels)):
            self.measure_labels_inv[int(self.measure_labels[i])] = i
        self.param_labels = kwargs.pop('param_labels')
        self.param_labels_inv = {}
        for i in range(len(self.param_labels)):
            self.param_labels_inv[int(self.param_labels[i])] = i
        self.nparam = len(self.param_labels)
        self.nmeasures = len(self.measure_labels)
        self.diffglobalinv = kwargs.pop('diffglobalinv')
        self.param = self.diffglobalinv*np.ones(self.nparam )
        self.diffinv = np.vectorize(self.conductivityinv)
        self.ddiffinv = np.vectorize(self.dconductivityinv)

    def diffinv2param(self, diffinv):
        return diffinv

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.Ais = [np.empty(shape=(0,0)) for i in range(self.nparam)]
        self.computeAis()

    def computeAis(self):
        bdrycond_bu = copy.deepcopy(self.problemdata.bdrycond)
        for color in self.problemdata.bdrycond.fct:
            self.problemdata.bdrycond.fct[color] = None
            self.problemdata.bdrycond.param[color] = 0
        for i in range(self.nparam):
            self.dlabel = self.param_labels[i]
            self.dcoeff = 1
            # self.dcoeff = -1/param[i]**2
            self.diffcellinv = self.ddiffinv(self.mesh.cell_labels)
            Ai, B = self.matrix()
            self.Ais[i] = Ai
        self.problemdata.bdrycond = bdrycond_bu

    def plot(self, **kwargs):
        if not 'point_data' in kwargs: point_data = self.point_data
        else: point_data = kwargs.pop('point_data')
        if not 'cell_data' in kwargs: cell_data = self.cell_data
        else: cell_data = kwargs.pop('cell_data')
        quiver_cell_data={'v': (cell_data['v0'],cell_data['v1'])}
        cell_data={'u':cell_data['p'], 'diff':cell_data['diff']}
        point_data={}
        kwargs['point_data'] = point_data
        kwargs['cell_data'] = cell_data
        kwargs['quiver_cell_data'] = quiver_cell_data
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.mesh, **kwargs)
        plt.show()
        # self.plotter.plot(**kwargs)

    def getData(self, infopp):
        # print("infopp", infopp)
        return infopp['measured']

    def computeRes(self, param, u=None):
        # print("#")
        self.param = param
        if not np.all(param>0):
            print(10*"#", param)
            # self.param = np.fmax(1/param, 1/self.diffglobal)
        self.diffcellinv = self.diffinv(self.mesh.cell_labels)
        self.diffcell = 1/self.diffcellinv
        # print("self.param", self.param)
        # print("self.diffcellinv", self.diffcellinv)
        # print("self.diffcell", self.diffcell)
        # self.diffcellinv = 1/self.diffcell
        A = self.matrix()
        b, u = self.computeRhs(u)
        self.A = A
        u, iter = self.linearSolver(A, b, u, verbose=0)
        # print("state iter", iter)
        self.point_data, self.cell_data, self.info = self.postProcess(u)
        data = self.getData(self.info['postproc'])
        return data, u

    def computeDRes(self, param, u, du):
        self.param = param
        if du is None: du = self.nparam*[np.empty(0)]
        jac = np.zeros(shape=(self.nmeasures, self.nparam))
        b = np.zeros_like(u)
        for i in range(self.nparam):
            b[:self.mesh.nfaces] = -self.Ais[i].dot(u[:self.mesh.nfaces])
            du[i] = np.zeros_like(b)
            du[i], iter = self.linearSolver(self.A, b, du[i], verbose=0)
            point_data, cell_data, info = self.postProcess(du[i])
            jac[:self.nmeasures,i] = self.getData(info['postproc'])
        return jac, du

    def computeAdj(self, param, r, u, z):
        self.param = param
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if z is None: z = np.zeros(self.mesh.nfaces+self.mesh.ncells)
        assert r.shape[0] == self.nmeasures
        for i, label in enumerate(self.measure_labels):
            self.problemdata.bdrycond.fct[label] = simfempy.solvers.optimize.RhsParam(r[i])
        self.diffcellinv = self.diffinv(self.mesh.cell_labels)
        self.diffcell = 1/self.diffcellinv
        b, z = self.computeRhs(z)
        z, iter = self.linearSolver(self.A, b, z, solver=self.linearsolver, verbose=0)
        # point_data, cell_data, info = self.postProcess(z)
        # self.plotter.plot(point_data, cell_data)
        self.problemdata = problemdata_bu
        return z

    def computeM(self, param, du, z):
        self.param = param
        M = np.zeros(shape=(self.nparam,self.nparam))
        assert z is not None
        for i in range(self.nparam):
            for j in range(self.nparam):
                M[j, i] -= self.Ais[i].dot(du[j][:self.mesh.nfaces]).dot(z[:self.mesh.nfaces])
                M[i, j] -= self.Ais[i].dot(du[j][:self.mesh.nfaces]).dot(z[:self.mesh.nfaces])
        # print("M", np.array2string(M, precision=2, floatmode='fixed'))
        return M
