# assert __name__ == '__main__'
from os import sys, path
# simfempypath = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),'simfempy')
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

print("sys.path",sys.path)
import simfempy.applications
import pygmsh
import numpy as np
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
    x0, x1 = -1.4, 1.4

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
    labels = 1000*np.ones(num_sections, dtype=int)
    labels[::3] += np.arange(1,nmeasures+1)
    lcars = hmeasure*np.ones(num_sections+1)
    lcars[3::3] = h
    # print("lcars",lcars)
    # print("labels",labels)
    # labels[3::3] = labels[-1]

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
    def __init__(self, solver):
        self.solver = solver
    def plot(self, point_data=None, cell_data=None, info=None):
        if point_data is None:
            point_data, cell_data = self.solver.point_data, self.solver.cell_data
        if info is None:
            addplots = None
        else:
            self.info = info
            # addplots = [self.plotmeas]
            addplots = None
        quiver_cell_data={'v': (4*cell_data['v0'],4*cell_data['v1'])}
        cell_data={'u':cell_data['p'], 'diff':cell_data['diff']}
        point_data={}
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.solver.mesh, point_data=point_data, cell_data=cell_data, quiver_cell_data=quiver_cell_data, addplots=addplots)

        # print("info", info)
        # if info is None:
        #     point_data, cell_data, info = self.eit.point_data, self.eit.cell_data, self.eit.info
        # else:
        #     point_data, cell_data, info = point_data, cell_data, info
        # # fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, self.point_data, self.cell_data, addplots=self.addplots)
        # # print("cell_data['diff']",cell_data['diff'].shape)
        # cell_plot={'p0':cell_data['p'], 'diff':cell_data['diff']}
        # fig, axs = simfempy.meshes.plotmesh.meshWithData(self.eit.mesh, point_data, cell_data=cell_plot)
        plt.show()

#----------------------------------------------------------------#
class EIT(simfempy.applications.laplacemixed.LaplaceMixed):

    def conductivity(self, label):
        if label == 100:
            return self.diffglobal
        # print("conductivity", label, self.param[self.param_labels_inv[label]], self.param)
        return self.param[self.param_labels_inv[label]]

    def dconductivity(self, label):
        if label == self.dlabel:
            return self.dcoeff
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
        self.nparam = len(self.param_labels)
        self.nmeasures = len(self.measure_labels)
        self.diffglobal = kwargs.pop('diffglobal')
        self.param = self.diffglobal*np.ones(self.nparam )
        self.diff = np.vectorize(self.conductivity)
        self.ddiff = np.vectorize(self.dconductivity)
        self.data0 = np.zeros(self.nmeasures)

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.Ais = [np.empty(shape=(0,0)) for i in range(self.nparam)]
        bdrycond_bu = copy.deepcopy(self.problemdata.bdrycond)
        for color in self.problemdata.bdrycond.fct:
            self.problemdata.bdrycond.fct[color] = None
            self.problemdata.bdrycond.param[color] = 0
        for i in range(self.nparam):
            self.dlabel = self.param_labels[i]
            self.dcoeff = 1
            # self.dcoeff = -1/param[i]**2
            self.diffcellinv = self.ddiff(self.mesh.cell_labels)
            Ai, B = self.matrix()
            self.Ais[i] = Ai
        self.problemdata.bdrycond = bdrycond_bu

    def getData(self, infopp):
        # print("infopp", infopp)
        return infopp['measured']

    def computeRes(self, param, u=None):
        # print("#")
        self.param = 1/param
        if not np.all(param>0):
            print(10*"#", param)
            self.param = np.fmax(param, self.diffglobal)
        self.diffcell = self.diff(self.mesh.cell_labels)
        self.diffcellinv = 1/self.diffcell
        A = self.matrix()
        b, u = self.computeRhs(u)
        self.A = A
        u, iter = self.linearSolver(A, b, u, verbose=0)
        # print("state iter", iter)
        self.point_data, self.cell_data, self.info = self.postProcess(u)
        data = self.getData(self.info['postproc'])
        # print("self.param", self.param)
        # print("data- self.data0", data-self.data0)
        return data - self.data0, u

    def computeDRes(self, param, u, du):
        self.param = 1/param
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
        self.param = 1/param
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if z is None: z = np.zeros(self.mesh.nfaces+self.mesh.ncells)
        assert r.shape[0] == self.nmeasures
        for i, label in enumerate(self.measure_labels):
            self.problemdata.bdrycond.fct[label] = simfempy.solvers.optimize.RhsParam(r[i])
        self.diffcell = self.diff(self.mesh.cell_labels)
        self.diffcellinv = 1/self.diffcell
        b, z = self.computeRhs(z)
        z, iter = self.linearSolver(self.A, b, z, solver=self.linearsolver, verbose=0)
        # point_data, cell_data, info = self.postProcess(z)
        # self.plotter.plot(point_data, cell_data)
        self.problemdata = problemdata_bu
        return z

    def computeM(self, param, du, z):
        self.param = 1/param
        M = np.zeros(shape=(self.nparam,self.nparam))
        assert z is not None
        for i in range(self.nparam):
            for j in range(self.nparam):
                M[j, i] = self.Ais[i].dot(du[j][:self.mesh.nfaces]).dot(z[:self.mesh.nfaces])
        # print("M", np.array2string(M, precision=2, floatmode='fixed'))
        return M

#----------------------------------------------------------------#
def test():
    h = 0.3
    hhole, hmeasure = 0.2*h, 0.1*h
    nmeasures = 8
    measuresize = 0.02
    nholes = 4
    mesh, hole_labels, electrode_labels, other_labels = createMesh2d(h=h, hhole=hhole, hmeasure=hmeasure, nholes=nholes, nmeasures=nmeasures, measuresize=measuresize)
    # print("electrode_labels",electrode_labels)
    # print("other_labels",other_labels)
    simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()


    param_labels = hole_labels
    nparams = len(param_labels)
    measure_labels = electrode_labels
    nmeasures = len(measure_labels)
    voltage_labels = electrode_labels
    # voltage = 1 + 10*(np.random.rand(nmeasures)-2)
    voltage = 2*np.ones(nmeasures)
    voltage[::2] *= -1
    voltage -= np.mean(voltage)
    print("voltage", voltage)


    bdrycond = simfempy.applications.problemdata.BoundaryConditions()

    for label in other_labels:
        bdrycond.type[label] = "Neumann"
    for i,label in enumerate(electrode_labels):
        bdrycond.type[label] = "Robin"
        bdrycond.param[label] = 100
        bdrycond.fct[label] = simfempy.solvers.optimize.RhsParam(voltage[i])

    postproc = {}
    postproc['measured'] = "bdrydn:{}".format(','.join( [str(l) for l in electrode_labels]))

    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)


    regularize = 0.01
    regularize = 0.0
    diffglobal = 10
    eit = EIT(problemdata=problemdata, measure_labels=measure_labels, param_labels=param_labels, diffglobal=diffglobal)
    eit.setMesh(mesh)

    optimizer = simfempy.solvers.optimize.Optimizer(eit, nparam=nparams, nmeasure=nmeasures, regularize=regularize, param0=diffglobal*np.ones(nparams))

    refparam = diffglobal*np.ones(nparams, dtype=float)

    if nholes==25:
        refparam[7] *= 0.1
        refparam[11] *= 0.1
        refparam[17] *= 0.1
    elif nholes == 4:
        refparam[0] = refparam[3] = 0.1
    else:
        # refparam[::2] *= 5
        # refparam[1::2] *= 10
        refparam[::2] *= 0.2
        refparam[1::2] *= 0.1

    print("refparam",refparam)
    percrandom = 0.01
    refdata, perturbeddata = optimizer.create_data(refparam=refparam, percrandom=percrandom)
    eit.plotter.plot(info=eit.info)
    # perturbeddata[::2] *= 1.3
    # perturbeddata[1::2] *= 0.7
    print("refdata",refdata)
    print("perturbeddata",perturbeddata)

    initialparam = diffglobal*np.ones(nparams)
    print("initialparam",initialparam)

    latex = simfempy.tools.latexwriter.LatexWriter(filename="mincompare")
    # optimizer.gradtest = True
    bounds = False
    if bounds:
        bounds = (0.1 * diffglobal, np.inf)
        methods = optimizer.boundmethods
    else:
        bounds = None
        methods = optimizer.methods
    values, valformat = optimizer.testmethods(x0=initialparam, methods=methods, bounds=bounds)
    latex.append(n=methods, nname='method', nformat="20s", values=values, valformat=valformat)
    latex.write()
    latex.compile()

#================================================================#

test()
