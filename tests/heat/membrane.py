assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

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
    spacesize = (x1-x0-nholes*holesize)/(nholes+1)
    if spacesize < 0.1*holesize:
        maxsize = (x1-x0)/(nholes*1.1 - 0.1)
        raise ValueError("holes too big (max={})".format(maxsize))
    pos = np.empty(2*nholes)
    pos[0] = spacesize
    pos[1] = pos[0] + holesize
    for i in range(1,nholes):
        pos[2*i] = pos[2*i-1] + spacesize
        pos[2*i+1] = pos[2*i] + holesize
    xholes = []
    for i in range(nholes):
        xa, xb = x0+pos[2*i], x0+pos[2*i+1]
        for j in range(nholes):
            ya, yb = x0+pos[2*j], x0+pos[2*j+1]
            xholes.append([[xa, ya, 0], [xb, ya, 0], [xb, yb, 0], [xa, yb, 0]])
    holes = []
    hole_labels = np.arange(200, 200 + len(xholes), dtype=int)
    for xhole, hole_label in zip(xholes, hole_labels):
        xarrm = np.mean(np.array(xhole), axis=0)
        holes.append(geometry.add_polygon(X=xhole, lcar=hhole))
        pygmshext.add_point_in_surface(geometry, holes[-1].surface, xarrm, lcar=h)
        geometry.add_physical_surface(holes[-1].surface, label=int(hole_label))
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
        # self.addplots = [self.plotmeas]
    def plot(self, point_data=None, cell_data=None, info=None):
        if info is None:
            self.point_data, self.cell_data, self.info = self.heat.point_data, self.heat.cell_data, self.heat.info
        else:
            self.point_data, self.cell_data, self.info = point_data, cell_data, info
        # fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, self.point_data, self.cell_data, addplots=self.addplots)
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, self.point_data, self.cell_data)
        plt.show()

#----------------------------------------------------------------#
class RhcCell(object):
    def __init__(self, param):
        self.param = param
    def __call__(self, x, y, z):
        return self.param

class Membrane(simfempy.applications.heat.Heat):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        super().__init__(**kwargs)
        self.linearsolver = "pyamg"
        self.plotter = Plotter(self)
        self.measure_labels = kwargs.pop('measure_labels')
        self.measure_labels_inv = {}
        for i in range(len(self.measure_labels)):
            self.measure_labels_inv[int(self.measure_labels[i])] = i
        self.param_labels = kwargs.pop('param_labels')
        self.param_labels_inv = {}
        for i in range(len(self.param_labels)):
            self.param_labels_inv[int(self.param_labels[i])] = i
        if 'regularize' in kwargs: self.regularize = kwargs.pop('regularize')
        else: self.regularize = None
        # print("self.measure_labels", self.measure_labels)
        # print("self.param_labels", self.param_labels)
        self.problemdata.postproc = {}
        self.problemdata.postproc['measured'] = "meanvalues:{}".format(','.join( [str(l) for l in self.measure_labels]))
        self.nparam = len(self.param_labels)
        self.nmeasures = len(self.measure_labels)
        self.param = np.arange(self.nparam )
        self.problemdata.rhscell = {}
        for label in self.param_labels:
            self.problemdata.rhscell[label] = RhcCell(self.param[self.param_labels_inv[label]])
        # print("self.problemdata", self.problemdata)


    def getData(self, infopp):
        # print("infopp", infopp)
        return infopp['measured']

    def solvestate(self, param):
        # print("#")
        self.param = param
        # print("self.param", self.param)
        for label in self.param_labels:
            self.problemdata.rhscell[label].param = self.param[self.param_labels_inv[label]]
        A = self.matrix()
        if not hasattr(self, 'ustate'):
            self.ustate = np.zeros(self.mesh.nnodes)
        b,self.ustate = self.computeRhs(self.ustate)
        self.A = A
        self.ustate, iter = self.linearSolver(A, b, self.ustate, solver=self.linearsolver, verbose=0)
        self.point_data, self.cell_data, self.info = self.postProcess(self.ustate)
        data = self.getData(self.info['postproc'])
        if self.regularize:
            # diffparam = param-self.diffglobal*np.ones(self.nparam)
            diffparam = param
            return np.append(data - self.data0, self.regularize*(diffparam))
        return data - self.data0

    def solveDstate(self, param):
        assert self.data0.shape[0] == self.nmeasures
        if self.regularize:
            jac = np.zeros(shape=(self.nmeasures+self.nparam, self.nparam))
            jac[self.nmeasures:,:] = self.regularize*np.eye(self.nparam)
        else:
            jac = np.zeros(shape=(self.nmeasures, self.nparam))
        rhscell = copy.deepcopy(self.problemdata.rhscell)
        for i in range(self.nparam):
            for label in self.param_labels:
                self.problemdata.rhscell[label] = RhcCell(0)
            self.problemdata.rhscell[self.param_labels[i]] = RhcCell(1)
            b, du = self.computeRhs()
            du, iter = self.linearSolver(self.A, b, du, solver=self.linearsolver, verbose=0)
            point_data, cell_data, info = self.postProcess(du)
            # print("info['postproc'].shape",self.getData(info['postproc']).shape)
            # print("jac.shape",jac.shape)
            # self.plot(point_data, cell_data, info)
            jac[:self.nmeasures,i] = self.getData(info['postproc'])
        self.problemdata.rhscell = rhscell

        # print("jac", jac.shape)
        return jac

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
    regularize = 0.000001
    membrane = Membrane(problemdata=problemdata, measure_labels=measure_labels, param_labels=param_labels, regularize=regularize)
    membrane.setMesh(mesh)
    nmeasures = len(measure_labels)
    nparams = len(param_labels)
    membrane.data0 = np.zeros(nmeasures)
    param = np.random.rand(nparams)
    param = 2+2*np.arange(nparams)
    data = membrane.solvestate(param)[:nmeasures]
    print("param", param)
    # print("data", data)
    membrane.plotter.plot()
    perc = 0
    membrane.data0[:] =  data[:]*(1+0.5*perc* ( 2*np.random.rand()-1))
    param = np.zeros(nparams)
    # param = np.random.rand(nparams)
    # data = membrane.solvestate(param)[:nmeasures]
    # print("data", data)
    # membrane.plotter.plot()

    methods = ['trf','lm']
    for method in methods:
        t0 = time.time()
        info = scipy.optimize.least_squares(membrane.solvestate, jac=membrane.solveDstate, x0=param, method=method, verbose=0, gtol=1e-12)
        dt = time.time()-t0
        print("{:^10s} x = {} J={:10.2e} nf={:4d} nj={:4d} {:10.2f} s".format(method, info.x, info.cost, info.nfev, info.njev, dt))
        membrane.plotter.plot()

#================================================================#

test()
