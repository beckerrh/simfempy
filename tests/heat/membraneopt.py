assert __name__ == '__main__'
from os import sys, path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.applications
import pygmsh
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from simfempy.tools import npext
from simfempy.meshes import pygmshext
import copy
import time

# ----------------------------------------------------------------#
def createMesh2d(h=0.1, hhole=0.03, nholes=2, holesize=0.2):
    x0, x1 = 0, 1
    geometry = pygmsh.built_in.Geometry()
    nholesy = int(np.sqrt(nholes))
    nholesx = int(nholes/nholesy)
    holes, hole_labels = pygmshext.add_holesnew(geometry, h=h, hhole=hhole, x0=0.2, x1=0.8, y0=0.2, y1=0.8, nholesx=nholesx,nholesy=nholesy)
    outer = []
    outer.append([[x0, x0, 0], 1000, h])
    outer.append([[x1, x0, 0], 1001, h])
    outer.append([[x1, x1, 0], 1002, h])
    outer.append([[x0, x1, 0], 1003, h])
    xouter = [out[0] for out in outer]
    labels = [out[1] for out in outer]
    lcars = [out[2] for out in outer]
    p1 = pygmshext.add_polygon(geometry, xouter, lcar=lcars, holes=holes)
    vals, inds = npext.unique_all(labels)
    for val, ind in zip(vals, inds):
        geometry.add_physical_line([p1.line_loop.lines[i] for i in ind], label=int(val))
    geometry.add_physical_surface(p1.surface, label=100)
    # print("code", geometry.get_code())
    data = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(data=data)
    bdrycond = simfempy.applications.problemdata.BoundaryConditions()
    bdrycond.type[1002] = "Dirichlet"
    bdrycond.type[1000] = "Dirichlet"
    bdrycond.type[1001] = "Dirichlet"
    bdrycond.type[1003] = "Dirichlet"
    bdrycond.check(mesh.bdrylabels.keys())
    return mesh, bdrycond, hole_labels


#----------------------------------------------------------------#
class Plotter:
    def __init__(self, heat):
        self.heat = heat
    def plot(self, point_data=None, cell_data=None, info=None):
        if point_data is None:
            point_data, cell_data = self.heat.point_data, self.heat.cell_data
        if info is None:
            addplots = None
        else:
            self.info = info
            addplots = None
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, point_data=point_data, cell_data=cell_data, addplots=addplots)
        plt.show()

#----------------------------------------------------------------#
class Membrane(simfempy.applications.heat.Heat):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        super().__init__(**kwargs)
        self.linearsolver = "pyamg"
        # self.linearsolver = "umf"
        self.plotter = Plotter(self)
        self.measure_labels = kwargs.pop('measure_labels')
        self.measure_labels_inv = {}
        for i in range(len(self.measure_labels)):
            self.measure_labels_inv[int(self.measure_labels[i])] = i
        self.param_labels = kwargs.pop('param_labels')
        self.param_labels_inv = {}
        for i in range(len(self.param_labels)):
            self.param_labels_inv[int(self.param_labels[i])] = i
        # if 'regularize' in kwargs: self.regularize = kwargs.pop('regularize')
        # else: self.regularize = None
        self.problemdata.postproc = {}
        self.problemdata.postproc['measured'] = "meanvalues:{}".format(','.join( [str(l) for l in self.measure_labels]))
        self.nparam = len(self.param_labels)
        self.nmeasures = len(self.measure_labels)
        self.data0 = np.zeros(self.nmeasures)
        self.param = np.arange(self.nparam )
        self.problemdata.rhscell = {}
        for label in self.param_labels:
            self.problemdata.rhscell[label] = simfempy.solvers.optimize.RhsParam(self.param[self.param_labels_inv[label]])


    def getData(self, info):
        # print("infopp", infopp)
        return info['postproc']['measured']

    def computeRes(self, param, u=None):
        self.param = param
        for label in self.param_labels:
            self.problemdata.rhscell[label].param = self.param[self.param_labels_inv[label]]
        A = self.matrix()
        b,u = self.computeRhs(u)
        self.A = A
        u, iter = self.linearSolver(A, b, u, solver=self.linearsolver, verbose=0)
        self.point_data, self.cell_data, self.info = self.postProcess(u)
        data = self.getData(self.info)
        # print("self.data0",self.data0)
        return data - self.data0, u

    def computeDRes(self, param, u, du):
        assert self.data0.shape[0] == self.nmeasures
        jac = np.zeros(shape=(self.nmeasures, self.nparam))
        rhscell = copy.deepcopy(self.problemdata.rhscell)
        if du is None: du = self.nparam*[np.empty(0)]
        for i in range(self.nparam):
            for label in self.param_labels:
                self.problemdata.rhscell[label] = simfempy.solvers.optimize.RhsParam(0)
            self.problemdata.rhscell[self.param_labels[i]] = simfempy.solvers.optimize.RhsParam(1)
            b, du[i] = self.computeRhs()
            du[i], iter = self.linearSolver(self.A, b, du[i], solver=self.linearsolver, verbose=0)
            point_data, cell_data, info = self.postProcess(du[i])
            # self.plot(point_data, cell_data, info)
            jac[:self.nmeasures,i] = self.getData(info)
        self.problemdata.rhscell = rhscell
        return jac, du

    def computeDResAdjW(self, param, u):
        jac = np.zeros(shape=(self.nmeasures, self.nparam))
        pdsplit = self.problemdata.postproc['measured'].split(':')
        assert pdsplit[0] == 'meanvalues'
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if not hasattr(self,'w'): self.w = self.nmeasures*[np.empty(0)]
        self.problemdata.postproc = {}
        self.problemdata.postproc['measured'] = "meanvalues:{}".format(','.join( [str(l) for l in self.param_labels]))
        self.problemdata.rhscell = {}
        for j in range(self.nmeasures):
            for k in range(self.nmeasures):
                if k==j: self.problemdata.rhscell[self.measure_labels[k]] = simfempy.solvers.optimize.RhsParam(1)
                else: self.problemdata.rhscell[self.measure_labels[k]] = None
            if self.w[j].shape[0]==0:
                self.w[j] = np.zeros(self.mesh.nnodes)
            b, self.w[j] = self.computeRhs(self.w[j])
            self.w[j], iter = self.linearSolver(self.A, b, self.w[j], solver=self.linearsolver, verbose=0)
            point_data, cell_data, info = self.postProcess(self.w[j])
            # self.plotter.plot(point_data, cell_data)
            jac[j] = self.getData(info)
        self.problemdata = problemdata_bu
        return jac

    def computeAdj(self, param, r, u, z):
        pdsplit = self.problemdata.postproc['measured'].split(':')
        assert pdsplit[0] == 'meanvalues'
        cellids = [int(l) for l in pdsplit[1].split(',')]
        assert np.all(cellids == self.measure_labels)
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if z is None: z = np.zeros(self.mesh.nnodes)
        self.problemdata.rhscell = {}
        for j in range(self.nmeasures):
            self.problemdata.rhscell[self.measure_labels[j]] = simfempy.solvers.optimize.RhsParam(r[j])
        b, z = self.computeRhs(z)
        z, iter = self.linearSolver(self.A, b, z, solver=self.linearsolver, verbose=0)
        point_data, cell_data, info = self.postProcess(z)
        # self.plotter.plot(point_data, cell_data)
        self.problemdata = problemdata_bu
        return z

    def computeDResAdj(self, param, r, u, z):
        z = self.computeAdj(param, r, u, z)
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        self.problemdata.postproc = {}
        self.problemdata.postproc['measured'] = "meanvalues:{}".format(','.join( [str(l) for l in self.param_labels]))
        point_data, cell_data, info = self.postProcess(z)
        grad = self.getData(info)
        self.problemdata = problemdata_bu
        return grad, z

#----------------------------------------------------------------#
def test():
    nholesperdirection = 4
    holesize = 0.1
    mesh, bdrycond, hole_labels = createMesh2d(nholes=nholesperdirection, holesize=holesize)
    # simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    # plt.show()
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond)
    param_labels = hole_labels[1::2]
    measure_labels = hole_labels[::2]
    regularize = 0.0001
    membrane = Membrane(problemdata=problemdata, measure_labels=measure_labels, param_labels=param_labels)
    membrane.setMesh(mesh)
    nmeasures = len(measure_labels)
    nparams = len(param_labels)

    refparam = 2+2*np.arange(nparams)
    optimizer = simfempy.solvers.optimize.Optimizer(membrane, nparam=nparams, nmeasure=nmeasures, regularize=regularize,
                                                    param0=2*np.ones(nparams))

    percrandom = 0.1
    refdata, perturbeddata = optimizer.create_data(refparam=refparam, percrandom=percrandom)

    initialparam = np.zeros(nparams)
    print("refparam",refparam)
    print("refdata",refdata)
    print("perturbeddata",perturbeddata)
    print("initialparam",initialparam)

    # membrane.data0 = np.zeros(nmeasures)
    # param = np.random.rand(nparams)
    # param = 2+2*np.arange(nparams)
    # # data = membrane.solvestate(param)[:nmeasures]
    # data,u = membrane.computeRes(param)[:nmeasures]
    # print("param", param)
    # # print("data", data)
    # membrane.plotter.plot()
    # perc = 0
    # membrane.data0[:] =  data[:]*(1+0.5*perc* ( 2*np.random.rand(nmeasures)-1))
    # param = np.zeros(nparams)
    # param = np.random.rand(nparams)
    # data = membrane.solvestate(param)[:nmeasures]
    # print("data", data)
    # membrane.plotter.plot()

    # optimizer.gradtest = True
    # for method in optimizer.lsmethods:
    for method in optimizer.minmethods:
        optimizer.minimize(x0=initialparam, method=method)
        membrane.plotter.plot(info=membrane.info)

#================================================================#

test()
