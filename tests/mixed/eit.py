# assert __name__ == '__main__'
from os import sys, path
# simfempypath = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),'simfempy')
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

print("sys.path",sys.path)
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
def createMesh2d(**kwargs):
    h = kwargs['h']
    hmeasure = kwargs.pop('hmeasure')
    nmeasures = kwargs.pop('nmeasures')
    measuresize = kwargs.pop('measuresize')
    x0, x1 = -1.5, 1.5
    geometry = pygmsh.built_in.Geometry()
    holes, hole_labels = pygmshext.add_holes(geometry, x0, x1, **kwargs)
    # un point additionnel pas espace entre segments-mesure
    num_sections = 3*nmeasures
    spacing = np.empty(num_sections)
    labels = np.empty(num_sections, dtype=int)
    spacesize = (1-nmeasures*measuresize)/nmeasures
    if spacesize < 0.1*measuresize:
        maxsize = 1/(nmeasures*1.1)
        raise ValueError("measuresize too big (max={})".format(maxsize))
    spacing[0] = 0
    spacing[1] = spacing[0] + measuresize
    spacing[2] = spacing[1] + 0.5*spacesize
    for i in range(1,nmeasures):
        spacing[3*i] = spacing[3*i-1] + 0.5*spacesize
        spacing[3*i+1] = spacing[3*i] + measuresize
        spacing[3*i+2] = spacing[3*i+1] + 0.5*spacesize
    labels[0] = 1000
    labels[1] = labels[0] + 1
    labels[2] = labels[1] + 0
    for i in range(1,nmeasures):
        labels[3*i] = labels[3*i-1] + 1
        labels[3*i+1] = labels[3*i] + 1
        labels[3*i+2] = labels[3*i+1] + 0
    # labels = 1000 + np.arange(num_sections, dtype=int)
    lcars = hmeasure*np.ones(num_sections+1)
    lcars[3::3] = h
    # print("lcars",lcars)

    circ = pygmshext.add_circle(geometry, 3*[0], 2, lcars=lcars, h=h, num_sections=num_sections, holes=holes, spacing=spacing)

    vals, inds = npext.unique_all(labels)
    for val, ind in zip(vals, inds):
        geometry.add_physical_line([circ.line_loop.lines[i] for i in ind], label=int(val))

    geometry.add_physical_surface(circ.plane_surface, label=100)
    # print("circ", dir(circ.line_loop))

    # with open("welcome.geo","w") as file: file.write(geometry.get_code())
    data = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(data=data)
    measure_labels = labels[::3]
    other_labels = set.difference(set(np.unique(labels)),set(np.unique(measure_labels)))
    return mesh, hole_labels, measure_labels, other_labels

#----------------------------------------------------------------#
class Plotter:
    def __init__(self, eit):
        self.eit = eit
        # self.addplots = [self.plotmeas]
    def plot(self, point_data=None, cell_data=None, info=None):
        print("info", info)
        if info is None:
            point_data, cell_data, info = self.eit.point_data, self.eit.cell_data, self.eit.info
        else:
            point_data, cell_data, info = point_data, cell_data, info
        # fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, self.point_data, self.cell_data, addplots=self.addplots)
        # print("cell_data['diff']",cell_data['diff'].shape)
        cell_plot={'p0':cell_data['p'], 'diff':cell_data['diff']}
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.eit.mesh, point_data, cell_data=cell_plot)
        plt.show()

#----------------------------------------------------------------#
class RhcRobin(object):
    def __init__(self, param):
        self.param = param
    def __call__(self, x, y, z):
        # print("RhcRobin", self.param)
        return self.param

#----------------------------------------------------------------#
class EIT(simfempy.applications.laplacemixed.LaplaceMixed):

    def conductivityinv(self, label):
        if label == 100:
            return self.diffglobalinv
        # print("conductivity", label, self.param[self.param_labels_inv[label]], self.param)
        return self.param[self.param_labels_inv[label]]

    def dconductivityinv(self, label):
        if label == self.dlabel:
            return 1
        return 0

    def __init__(self, **kwargs):
        kwargs['plotdiff'] = True
        super().__init__(**kwargs)
        self.plotter = Plotter(self)
        self.linearsolver = "umf"
        self.measure_labels = kwargs.pop('measure_labels')
        self.measure_labels_inv = {}
        for i in range(len(self.measure_labels)):
            self.measure_labels_inv[int(self.measure_labels[i])] = i
        self.param_labels = kwargs.pop('param_labels')
        self.param_labels_inv = {}
        for i in range(len(self.param_labels)):
            self.param_labels_inv[int(self.param_labels[i])] = i
        self.voltage_labels = kwargs.pop('voltage_labels')
        self.voltage_labels_inv = {}
        for i in range(len(self.voltage_labels)):
            self.voltage_labels_inv[int(self.voltage_labels[i])] = i
        self.voltage = kwargs.pop('voltage')

        if 'regularize' in kwargs: self.regularize = kwargs.pop('regularize')
        else: self.regularize = None
        self.nparam = len(self.param_labels)
        self.nmeasures = len(self.measure_labels)
        self.param = np.ones(self.nparam )

        self.diffglobalinv = kwargs.pop('diffglobalinv')
        self.diffinv = np.vectorize(self.conductivityinv)
        self.ddiffinv = np.vectorize(self.dconductivityinv)

        # print("self.problemdata", self.problemdata)
        bdrycond = self.problemdata.bdrycond
        for label in self.voltage_labels:
            bdrycond.fct[label] = RhcRobin(self.voltage[self.voltage_labels_inv[label]])
        # print("self.problemdata", self.problemdata)


    def getData(self, infopp):
        # print("infopp", infopp)
        return infopp['measured']

    def solvestate(self, param):
        # print("#")
        self.param = param
        self.diffcellinv = self.diffinv(self.mesh.cell_labels)
        self.diffcell = 1/self.diffcellinv
        A = self.matrix()
        if hasattr(self, 'ustate'):
            b,self.ustate = self.computeRhs(self.ustate)
        else:
            b, self.ustate = self.computeRhs()
        self.A = A
        self.ustate, iter = self.linearSolver(A, b, self.ustate, verbose=0)
        # print("state iter", iter)
        self.point_data, self.cell_data, self.info = self.postProcess(self.ustate)
        data = self.getData(self.info['postproc'])
        # print("self.param", self.param)
        # print("data- self.data0", data-self.data0)
        if self.regularize:
            diffparam = param-self.diffglobalinv *np.ones(self.nparam)
            return np.append(data - self.data0, self.regularize*(diffparam))
        return data - self.data0

    def solveDstate(self, param):
        nparam = self.param.shape[0]
        assert self.data0.shape[0] == self.nmeasures
        if self.regularize:
            jac = np.zeros(shape=(self.nmeasures+nparam, nparam))
            jac[self.nmeasures:,:] = self.regularize*np.eye(nparam)
        else:
            jac = np.zeros(shape=(self.nmeasures, nparam))
        bdrycond_bu = copy.deepcopy(self.problemdata.bdrycond)
        for color in self.problemdata.bdrycond.fct:
            self.problemdata.bdrycond.fct[color] = None
            self.problemdata.bdrycond.param[color] = 0
        for i in range(nparam):
            self.dlabel = self.param_labels[i]
            self.diffcellinv = self.ddiffinv(self.mesh.cell_labels)
            Ai, B = self.matrix()
            b = np.zeros_like(self.ustate)
            b[:self.mesh.nfaces] = -Ai.dot(self.ustate[:self.mesh.nfaces])
            du = np.zeros_like(b)
            self.diffcellinv = self.diffinv(self.mesh.cell_labels)
            du, iter = self.linearSolver(self.A, b, du, verbose=0)
            # print("dstate iter", iter)
            point_data, cell_data, info = self.postProcess(du)
            # print("info['postproc'].shape",self.getData(info['postproc']).shape)
            # print("jac.shape",jac.shape)
            # self.plot(point_data, cell_data, info)
            jac[:self.nmeasures,i] = self.getData(info['postproc'])
        self.problemdata.bdrycond = bdrycond_bu

        # print("jac", jac.shape)
        return jac

#----------------------------------------------------------------#
def test():
    h = 1
    hhole, hmeasure = 0.2*h, 0.1*h
    nholesperdirection = 1
    nmeasures = 4
    holesize = 2/nholesperdirection
    measuresize = 0.03
    mesh, hole_labels, electrode_labels, other_labels = createMesh2d(h=h, hhole=hhole, hmeasure=hmeasure, nholes=nholesperdirection, nmeasures=nmeasures, holesize=holesize, measuresize=measuresize)
    # simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    # plt.show()
    # print("electrode_labels",electrode_labels)
    # print("other_labels",other_labels)
    bdrycond = simfempy.applications.problemdata.BoundaryConditions()

    for label in other_labels:
        bdrycond.type[label] = "Neumann"
    for label in electrode_labels:
        bdrycond.type[label] = "Robin"
        bdrycond.param[label] = 1

    postproc = {}
    postproc['measured'] = "bdrymean:{}".format(','.join( [str(l) for l in electrode_labels]))

    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)

    param_labels = hole_labels
    nparams = len(param_labels)
    measure_labels = electrode_labels
    nmeasures = len(measure_labels)
    voltage_labels = electrode_labels
    # voltage = 1 + 10*(np.random.rand(nmeasures)-2)
    voltage = 10*np.ones(nmeasures)
    voltage[::2] *= -1
    voltage -= np.mean(voltage)

    regularize = None
    diffglobalinv = 10
    eit = EIT(problemdata=problemdata, measure_labels=measure_labels, param_labels=param_labels, voltage_labels=voltage_labels, voltage=voltage, regularize=regularize, diffglobalinv=diffglobalinv)
    eit.setMesh(mesh)

    eit.data0 = np.zeros(nmeasures)
    param = 1/(1 +np.arange(nparams, dtype=float))
    refdata = eit.solvestate(param)[:nmeasures]
    print("param", param)
    print("refdata", refdata)
    # eit.plotter.plot()

    percrandom = 0.01
    perturbeddata =  refdata[:]*(1+0.5*percrandom*( 2*np.random.rand()-1))
    print("perturbeddata", perturbeddata)
    eit.data0[:] =  perturbeddata
    optimize(eit, param)
    params = np.einsum('i,j->ji', param, np.linspace(0.1,10, 30))
    print("params", params.reshape(-1))
    paramtocost(eit, params)


def paramtocost(eit, params):
    datas=[]
    for param in params:
        print("param", param)
        data = eit.solvestate(param)
        datas.append(0.5*np.linalg.norm(data)**2)
    plt.plot(params, datas)
    plt.show()

def optimize(eit, param):
    methods = ['trf','lm']
    param[:] *= 2
    for method in methods:
        # param[:] = 1/diffglobal
        t0 = time.time()
        # info = scipy.optimize.least_squares(eit.solvestate, x0=param, method=method, gtol=1e-12, verbose=0)
        info = scipy.optimize.least_squares(eit.solvestate, jac=eit.solveDstate, x0=param, method=method, gtol=1e-12, verbose=0)
        dt = time.time()-t0
        # print("status", info.status)
        # print("{:^10s} x = {} J={:10.2e} nf={:4d} nj={:4d} {:10.2f} s".format(method, info.x, info.cost, info.nfev, info.njev, dt))
        print("{:^10s} x = {} J={:10.2e} nf={:4d} {:10.2f} s".format(method, info.x, info.cost, info.nfev,  dt))
        eit.plotter.plot()


#================================================================#

test()
