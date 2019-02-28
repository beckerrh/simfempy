assert __name__ == '__main__'
from os import sys, path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.applications
import pygmsh
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from simfempy.tools import npext
from simfempy.meshes import pygmshext
import copy


# ----------------------------------------------------------------#
def createMesh2d(**kwargs):
    geometry = pygmsh.built_in.Geometry()

    h = kwargs['h']
    hmeasure = kwargs.pop('hmeasure')
    nmeasures = kwargs.pop('nmeasures')
    measuresize = kwargs.pop('measuresize')
    x0, x1 = -1.2, 1.2

    hhole = kwargs.pop('hhole')
    nholes = kwargs.pop('nholes')
    nholesy = int(np.sqrt(nholes))
    nholesx = int(nholes/nholesy)
    print("hhole", hhole, "nholes", nholes, "nholesy", nholesy, "nholesx", nholesx)
    holes, hole_labels = pygmshext.add_holesnew(geometry, h=h, hhole=hhole, x0=x0, x1=x1, y0=x0, y1=x1, nholesx=nholesx,nholesy=nholesy)
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

    with open("welcome.geo","w") as file: file.write(geometry.get_code())
    data = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(data=data)
    measure_labels = labels[::3]
    other_labels = set.difference(set(np.unique(labels)),set(np.unique(measure_labels)))
    return mesh, hole_labels, measure_labels, other_labels

#----------------------------------------------------------------#
class Plotter:
    def __init__(self, heat):
        self.heat = heat
    def plot(self, point_data=None, cell_data=None, info=None, title=""):
        if point_data is None:
            point_data, cell_data = self.heat.point_data, self.heat.cell_data
        if info is None:
            addplots = None
        else:
            self.info = info
            addplots = None
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, point_data=point_data, cell_data=cell_data, addplots=addplots, title=title)
        plt.show()

#----------------------------------------------------------------#
class EIT(simfempy.applications.heat.Heat):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        kwargs['plotk'] = True
        super().__init__(**kwargs)
        self.linearsolver = "pyamg"
        # self.linearsolver = "umf"
        self.kheat = np.vectorize(self.kparam)
        self.dkheat = np.vectorize(self.dkparam)
        self.diffglobal = kwargs.pop('diffglobal')
        self.hole_labels = kwargs.pop('hole_labels')
        self.hole_labels_inv = {}
        for i in range(len(self.hole_labels)):
            self.hole_labels_inv[int(self.hole_labels[i])] = i
        self.nparam = len(self.hole_labels)
        self.param = np.ones(self.nparam)
        self.plotter = Plotter(self)
        pp = self.problemdata.postproc['measured'].split(":")[1]
        self.nmeasures = len(pp.split(","))
        self.data0 = np.zeros(self.nmeasures)

    def kparam(self, label):
        if label==100: return self.diffglobal
        # return self.param[label-200]
        return self.param[self.hole_labels_inv[label]]
    def dkparam(self, label):
        if label==self.dlabel: return 1.0
        return 0.0

    def getData(self, infopp):
        return infopp['measured']
        # return np.concatenate([np.array([infopp[f] for f in self.fluxes]), infopp['measured']], axis=0)

    def computeRes(self, param, u=None):
        # print("#")
        self.param = param
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        A = self.matrix()
        b,u= self.computeRhs(u)
        self.A = A
        u, iter = self.linearSolver(A, b, u, solver=self.linearsolver, verbose=0)
        self.point_data, self.cell_data, self.info = self.postProcess(u)
        data = self.getData(self.info['postproc'])
        # self.plotter.plot()
        # print("self.data0", self.data0, "data", data)
        return data - self.data0, u

    def computeDRes(self, param, u, du):
        self.param = param
        assert self.data0.shape[0] == self.nmeasures
        jac = np.zeros(shape=(self.nmeasures, self.nparam))
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        self.problemdata.postproc = problemdata_bu.postproc
        if du is None: du = self.nparam*[np.empty(0)]
        for i in range(self.nparam):
            self.dlabel = self.hole_labels[i]
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            b = -Bi.dot(u)
            du[i] = np.zeros_like(b)
            # self.kheatcell = self.kheat(self.mesh.cell_labels)
            # b, du[i] = self.boundaryvec(b, du[i])
            du[i], iter = self.linearSolver(self.A, b, du[i], solver=self.linearsolver, verbose=0)
            point_data, cell_data, info = self.postProcess(du[i])
            # point_data["B"] = b
            # self.plotter.plot(point_data, cell_data, title="DU")
            jac[:self.nmeasures,i] = self.getData(info['postproc'])
        self.problemdata = problemdata_bu
        return jac, du

    def computeDResAdjW(self, param, u):
        assert self.data0.shape[0] == self.nmeasures
        jac = np.zeros(shape=(self.nmeasures, self.nparam))
        pdsplit = self.problemdata.postproc['measured'].split(':')
        assert pdsplit[0] == 'pointvalues'
        pointids = [int(l) for l in pdsplit[1].split(',')]
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if not hasattr(self,'w'): self.w = self.nmeasures*[np.empty(0)]
        self.problemdata.rhspoint = {}
        for j in range(self.nmeasures):
            for k in range(self.nmeasures):
                if k==j: self.problemdata.rhspoint[pointids[k]] = simfempy.solvers.optimize.RhsParam(1)
                else: self.problemdata.rhspoint[pointids[k]] = None
            self.kheatcell = self.kheat(self.mesh.cell_labels)
            if self.w[j].shape[0]==0:
                self.w[j] = np.zeros(self.mesh.nnodes)
            b, self.w[j] = self.computeRhs(self.w[j])
            self.w[j], iter = self.linearSolver(self.A, b, self.w[j], solver=self.linearsolver, verbose=0)
            # point_data, cell_data, info = self.postProcess(self.w[j])
            # self.plotter.plot(point_data, cell_data)
            for i in range(self.nparam):
                self.dlabel = self.hole_labels[i]
                self.kheatcell = self.dkheat(self.mesh.cell_labels)
                Bi = self.matrix()
                jac[j, i] = -Bi.dot(u).dot(self.w[j])
        self.problemdata = problemdata_bu
        return jac

    def computeAdj(self, param, r, u, z):
        pdsplit = self.problemdata.postproc['measured'].split(':')
        assert pdsplit[0] == 'pointvalues'
        pointids = [int(l) for l in pdsplit[1].split(',')]
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if z is None:
            z = np.zeros(self.mesh.nnodes)
        self.problemdata.rhspoint = {}
        for j in range(self.nmeasures):
            self.problemdata.rhspoint[pointids[j]] = simfempy.solvers.optimize.RhsParam(r[j])
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        b, z = self.computeRhs(z)
        z, iter = self.linearSolver(self.A, b, z, solver=self.linearsolver, verbose=0)
        # point_data, cell_data, info = self.postProcess(self.z)
        # self.plotter.plot(point_data, cell_data)
        self.problemdata = problemdata_bu
        return z

    def computeDResAdj(self, param, r, u, z):
        z = self.computeAdj(param, r, u, z)
        grad = np.zeros(shape=(self.nparam))
        for i in range(self.nparam):
            self.dlabel = self.hole_labels[i]
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            grad[i] = -Bi.dot(u).dot(z)
        return grad, z

    def computeM(self, param, du, z):
        M = np.zeros(shape=(self.nparam,self.nparam))
        assert z is not None
        for i in range(self.nparam):
            self.dlabel = self.hole_labels[i]
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            for j in range(self.nparam):
                M[i,j] = -Bi.dot(du[j]).dot(z)
        # print("M", np.array2string(M, precision=2, floatmode='fixed'))
        return M

#----------------------------------------------------------------#
def test():
    h = 0.2
    hhole, hmeasure = 0.3*h, 0.2*h
    nmeasures = 4
    measuresize = 0.03
    nholes = 2
    mesh, hole_labels, electrode_labels, other_labels = createMesh2d(h=h, hhole=hhole, hmeasure=hmeasure, nholes=nholes, nmeasures=nmeasures, measuresize=measuresize)
    # simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    # plt.show()
    # print("electrode_labels",electrode_labels)
    # print("other_labels",other_labels)


    param_labels = hole_labels
    nparams = len(param_labels)
    measure_labels = electrode_labels
    nmeasures = len(measure_labels)
    voltage_labels = electrode_labels
    voltage = 2*np.ones(nmeasures)
    voltage[::2] *= -1
    voltage -= np.mean(voltage)

    bdrycond = simfempy.applications.problemdata.BoundaryConditions()
    for label in other_labels:
        bdrycond.type[label] = "Neumann"
    for i,label in enumerate(electrode_labels):
        bdrycond.type[label] = "Robin"
        bdrycond.param[label] = 1000
        bdrycond.fct[label] = simfempy.solvers.optimize.RhsParam(voltage[i])

    postproc = {}
    postproc['measured'] = "bdrydn:{}".format(','.join( [str(l) for l in electrode_labels]))

    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)


    regularize = 0.00000
    diffglobal = 1
    eit = EIT(problemdata=problemdata, measure_labels=measure_labels, hole_labels=param_labels, diffglobal=diffglobal)
    eit.setMesh(mesh)

    optimizer = simfempy.solvers.optimize.Optimizer(eit, nparam=nparams, nmeasure=nmeasures, regularize=regularize, param0=diffglobal*np.ones(nparams))

    refparam = diffglobal*np.ones(nparams, dtype=float)
    refparam[::2] *= 5
    refparam[1::2] *= 10
    # refparam[1::2] *= 100
    print("refparam",refparam)
    percrandom = 0.
    refdata, perturbeddata = optimizer.create_data(refparam=refparam, percrandom=percrandom)
    # perturbeddata[::2] *= 1.2
    # perturbeddata[1::2] *= 0.8
    print("refdata",refdata)
    print("perturbeddata",perturbeddata)

    initialparam = diffglobal*np.ones(nparams)
    print("initialparam",initialparam)

    # optimizer.gradtest = True
    # for method in optimizer.methods:
    for method in optimizer.lsmethods:
        optimizer.minimize(x0=initialparam, method=method)
        eit.plotter.plot(info=eit.info)
#

#================================================================#

test()
