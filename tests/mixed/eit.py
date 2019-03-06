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
    def plot(self, **kwargs):
        if not 'point_data' in kwargs: point_data = self.solver.point_data
        else: point_data = kwargs.pop('point_data')
        if not 'cell_data' in kwargs: cell_data = self.solver.cell_data
        else: cell_data = kwargs.pop('cell_data')
        quiver_cell_data={'v': (cell_data['v0'],cell_data['v1'])}
        cell_data={'u':cell_data['p'], 'diff':cell_data['diff']}
        point_data={}
        kwargs['point_data'] = point_data
        kwargs['cell_data'] = cell_data
        kwargs['quiver_cell_data'] = quiver_cell_data
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.solver.mesh, **kwargs)

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
        self.diffglobalinv = kwargs.pop('diffglobalinv')
        self.param = self.diffglobalinv*np.ones(self.nparam )
        self.diffinv = np.vectorize(self.conductivityinv)
        self.ddiffinv = np.vectorize(self.dconductivityinv)
        # self.data0 = np.zeros(self.nmeasures)

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
            self.diffcellinv = self.ddiffinv(self.mesh.cell_labels)
            Ai, B = self.matrix()
            self.Ais[i] = Ai
        self.problemdata.bdrycond = bdrycond_bu

    def plot(self, **kwargs):
        self.plotter.plot(**kwargs)

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
        # print("data- self.data0", data-self.data0)
        return data - self.data0, u

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

#----------------------------------------------------------------#
def problemdef(h, nholes, nmeasures, diffglobalinv = 100, volt=4):
    h = h
    hhole, hmeasure = 0.2*h, 0.1*h
    measuresize = 0.02
    nholes = nholes
    mesh, hole_labels, electrode_labels, other_labels = createMesh2d(h=h, hhole=hhole, hmeasure=hmeasure, nholes=nholes,
                                                                     nmeasures=nmeasures, measuresize=measuresize)
    param_labels = hole_labels
    nparams = len(param_labels)
    measure_labels = electrode_labels
    assert nmeasures == len(measure_labels)
    voltage_labels = electrode_labels
    voltage = volt*np.ones(nmeasures)
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
    diffglobalinv = diffglobalinv
    eit = EIT(problemdata=problemdata, measure_labels=measure_labels, param_labels=param_labels, diffglobalinv=diffglobalinv)
    eit.setMesh(mesh)
    return eit



#----------------------------------------------------------------#
def test():
    h = 0.5
    nmeasures = 32
    nholes = 25
    diffglobalinv = 100
    eit = problemdef(h, nholes, nmeasures, diffglobalinv)

    regularize = 0.000
    param0 = diffglobalinv*np.ones(nholes)
    optimizer = simfempy.solvers.optimize.Optimizer(eit, nparam=nholes, nmeasure=nmeasures, regularize=regularize,
                                                    param0=param0)

    refparam = diffglobalinv*np.ones(nholes, dtype=float)

    if nholes==25:
        refparam[7] /= 10
        refparam[11] /= 10
        refparam[17] /= 10
    elif nholes == 4:
        refparam[0] /= 10
        refparam[3] /= 5
    else:
        # refparam[::2] /= 5
        # refparam[1::2] /= 10
        refparam[::2] /= 5
        refparam[1::2] /= 10

    print("refparam",refparam)
    percrandom = 0.
    refdata, perturbeddata = optimizer.create_data(refparam=refparam, percrandom=percrandom)
    eit.plotter.plot(info=eit.info)
    perturbeddata[::2] *= 1.3
    perturbeddata[1::2] *= 0.7
    # print("refdata",refdata)
    # print("perturbeddata",perturbeddata)

    initialparam = diffglobalinv*np.ones(nholes)
    print("initialparam",initialparam)

    bounds = False
    if bounds:
        bounds = (0.01*diffglobalinv, diffglobalinv)
        methods = optimizer.boundmethods
        methods = ['trf','dogbox']
    else:
        bounds = None
        methods = optimizer.methods

    # optimizer.hestest = True
    methods = optimizer.methods
    values, valformat = optimizer.testmethods(x0=initialparam, methods=methods, bounds=bounds, plot=True)
    # eit.plotter.plot(info=eit.info)

    latex = simfempy.tools.latexwriter.LatexWriter(filename="mincompare")
    latex.append(n=methods, nname='method', nformat="20s", values=values, valformat=valformat)
    latex.write()
    latex.compile()


#----------------------------------------------------------------#
def plotJhat():
    h = 0.4
    nmeasures = 32
    nholes = 2
    diffglobalinv = 100
    eit = problemdef(h, nholes, nmeasures, diffglobalinv)

    optimizer = simfempy.solvers.optimize.Optimizer(eit, nparam=nholes, nmeasure=nmeasures)
    refparam = diffglobalinv*np.ones(nholes, dtype=float)
    refparam[::2] /= 5
    refparam[1::2] /= 10
    percrandom = 0.1
    refdata, perturbeddata = optimizer.create_data(refparam=refparam, percrandom=percrandom)
    eit.plotter.plot(info=eit.info)

    n = 25
    c = np.empty(shape=(n,n,nmeasures))
    px = np.linspace(0.2*refparam[0], 10*refparam[0], n)
    py = np.linspace(0.2*refparam[1], 10*refparam[1], n)
    param = np.empty(2, dtype=float)
    for i in range(n):
        for j in range(n):
            param[0] = px[i]
            param[1] = py[j]
            data, u = eit.computeRes(param)
            # print("data", data)
            # print("param", param, "data",data)
            c[i,j] = data
    xx, yy = np.meshgrid(px, py)
    ncols = min(nmeasures,3)
    nrows = nmeasures//3 + bool(nmeasures%3)
    ncols = 1
    nrows = 3
    # print("nrows, ncols", nrows, ncols)
    fig, axs = plt.subplots(ncols, nrows, figsize=(nrows*4.5,ncols*4), squeeze=False)
    fig.suptitle("pert = {}%".format(percrandom))
    # aspect = (np.max(x)-np.mean(x))/(np.max(y)-np.mean(y))
    ind = [0,7]
    for i in range(nrows-1):
        # ax = axs[i // ncols, i % ncols]
        ax = axs[0,i]
        cnt = ax.contourf(xx, yy, c[:,:,ind[i]], 16, cmap='jet')
        ax.set_aspect(1)
        # clb = plt.colorbar(cnt, ax=ax)
        ax.set_title(r"$c_{}(u)$".format(ind[i]))
    Jhat = np.sum(c*c, axis=(2))
    Jhat /= np.max(Jhat)
    # print("Jhat", Jhat)
    ax = axs[0,-1]
    CS = ax.contour(px, py, Jhat, levels=np.linspace(0.,1,20))
    ax.clabel(CS, inline=1, fontsize=8)
    ax.set_title(r'$\hat J$')
    plt.show()


#================================================================#

test()
# plotJhat()
