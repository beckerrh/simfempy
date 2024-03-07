import pathlib, sys
simfempypath = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0,simfempypath)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from simfempy.models.elliptic import Elliptic
from simfempy.models.application import Application
from simfempy.meshes import plotmesh
from simfempy.tools.comparemethods import CompareMethods

class InterfaceAnalyticalSolution(Application):
    def __init__(self, h=0.1):
        self.k1, self.k2, self.r02 = 1, 100, 0.75**2
        self.u = self.ExactSolution(self)
        super().__init__(h=h, exactsolution=self.u, generatePDforES=False)
        # fill problem data
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003])
        self.problemdata.bdrycond.fct[1000] = self.u
        self.problemdata.bdrycond.fct[1001] = self.u
        self.problemdata.bdrycond.fct[1002] = self.u
        self.problemdata.bdrycond.fct[1003] = self.u
        self.problemdata.params.fct_glob['kheat'] = self.k
        self.problemdata.params.fct_glob['rhs'] = np.vectorize(lambda x,y,z: -4)
    class ExactSolution():
        def __init__(self, sup):
            self.k1, self.k2, self.r02 = sup.k1, sup.k2, sup.r02
        def __call__(self, x, y, z):
            x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
            r2 = x ** 2 + y ** 2 + z ** 2
            return np.where(r2 <= self.r02, r2 / self.k1, r2 / self.k2 + self.r02 * (1 / self.k1 - 1 / self.k2))
        def d(self, i, x, y, z):
            x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
            r2 = x ** 2 + y ** 2 + z ** 2
            if i==0: return np.where(r2 <= self.r02, 2*x / self.k1, 2*x / self.k2)
            elif i==1: return np.where(r2 <= self.r02, 2*y / self.k1, 2*y / self.k2)
            else: return np.where(r2 <= self.r02, 2*z / self.k1, 2*z / self.k2)
    def k(self, color, x, y, z):
        x,y,z = np.asarray(x),np.asarray(y),np.asarray(z)
        r2 = x**2+y**2+z**2
        return np.where(r2 <= self.r02, self.k1, self.k2)
    def defineGeometry(self, geom, h):
        # raise ValueError(f"{h=}")
        rectangle = geom.add_rectangle(xmin=-1, xmax=1, ymin=-1, ymax=1, z=0, mesh_size=h)
        geom.add_physical(rectangle.surface, label="100")
        for i, line in enumerate(rectangle.lines): geom.add_physical(line, label=f"{1000 + i}")

def solve(h=0.1, modelargs={}):
    heat = Elliptic(application=InterfaceAnalyticalSolution(h=h), **modelargs)
    result, u = heat.static(mode="newton")
    data = heat.sol_to_data(u)
    for p, v in result.data['scalar'].items(): print(f"{p}: {v}")
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"{heat.application.__class__.__name__} (static)", fontsize=16)
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    ax = fig.add_subplot(outer[0], projection='3d')
    x, y, tris = heat.mesh.points[:, 0], heat.mesh.points[:, 1], heat.mesh.simplices
    z = data['point']['U']
    # print(f"{data['point'].keys()=} {data['cell'].keys()=} {result.data=}")
    ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
    data.update({'cell': {'k': heat.kheatcell}})
    data['cell'].update({'err': np.abs(result.data['cell']['err'])})
    plotmesh.meshWithData(heat.mesh, data=data, alpha=0.5, fig=fig, outer=outer[1])
    plt.show()

def compare(niter=6, modelargs={}):
    paramsdict={}
    app = InterfaceAnalyticalSolution()
    comp = CompareMethods(application=app, paramsdict=paramsdict, model=Elliptic, modelargs=modelargs, niter=niter)
    res = comp.compare()

if __name__ == "__main__":
    linearsolver = {'method':'pyamg', 'disp':0, 'symmetric':True}
    modelargs = {'fem': 'cr1', 'linearsolver': linearsolver, 'disc_params': {'dirichletmethod': 'strong'}, 'scale_ls': True,
                 'newton_rtol': 1e-10}
    # solve(h=0.02, modelargs=modelargs)
    compare(niter=7, modelargs=modelargs)