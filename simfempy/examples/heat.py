assert __name__ == '__main__'
# use own pygmsh
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'pygmsh'))
sys.path.insert(0,simfempypath)

import matplotlib.pyplot as plt
import pygmsh
from simfempy.applications.heat import Heat
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes.hole import hole
from simfempy.meshes.plotmesh import meshWithBoundaries, meshWithData
from simfempy.meshes.animdata import AnimData

# ---------------------------------------------------------------- #
def main():
    problemdata = createDataDyn()
    # problemdata = createData()
    # problemdata = createDataConvection()
    mesh = createMesh(h=0.1)
    print(f"{mesh=}")
    heat = Heat(mesh=mesh, problemdata=problemdata)
    meshWithBoundaries(heat.mesh)
    mode = 'dynamic'
    if mode == 'static':
        result = heat.static()
        print(f"{result.info['timer']}")
        print(f"postproc: {result.data['global']['postproc']}")
        meshWithData(heat.mesh, data=result.data, title="Heat static", alpha=1)
        plt.show()
    elif mode == 'dynamic':
        u0 = heat.initialCondition("200")
        result = heat.dynamic(u0, t_span=(0,2000), nframes=40, dt=10)
        anim = AnimData(mesh, result.data['point']['U'])
        plt.show()
        # class HeatAnim:
        #     def __init__(self, ax, mesh, u):
        #         x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
        #         ax.triplot(x, y, tris, color='gray', lw=1, alpha=1)
        #         self.norm = matplotlib.colors.Normalize(vmin=100, vmax=200)
        #         self.argscf = {'levels':32, 'norm':self.norm, 'cmap':'jet'}
        #         self.argsc = {'colors':'k', 'levels':np.linspace(100,200,20)}
        #         ax.tricontourf(x, y, tris, u[0], **self.argscf)
        #         # ax.tricontour(x, y, tris, u0, **self.argsc)
        #         cmap = matplotlib.cm.jet
        #         plt.colorbar(matplotlib.cm.ScalarMappable(norm=self.norm, cmap=cmap), ax=ax)
        #         self.u, self.ax  = u, ax
        #         self.x, self.y, self.tris = x, y, tris
        #     def __call__(self, i):
        #         u, ax = self.u, self.ax
        #         x, y, tris = self.x, self.y, self.tris
        #         ax.cla()
        #         ax.set_title(f"Iter {i}")
        #         print(f"{i=} {np.linalg.norm(self.u[i])}")
        #         ax.tricontourf(x, y, tris, self.u[i], **self.argscf)
        #         ax.tricontour(x, y, tris, self.u[i], **self.argsc)
        #         return ax
        # fig = plt.figure()
        # sol = result.data['point']['U']
        # ha = HeatAnim(fig.gca(), mesh, sol)
        # anim = animation.FuncAnimation(fig, ha, frames=len(sol), repeat=False)
        # plt.show()
        # class HeatAnim:
        #     def __init__(self, ax, mesh, u0, heat):
        #         x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
        #         ax.triplot(x, y, tris, color='gray', lw=1, alpha=1)
        #         self.norm = matplotlib.colors.Normalize(vmin=100, vmax=200)
        #         self.argscf = {'levels':32, 'norm':self.norm, 'cmap':'jet'}
        #         self.argsc = {'colors':'k', 'levels':np.linspace(100,200,20)}
        #         ax.tricontourf(x, y, tris, u0, **self.argscf)
        #         # ax.tricontour(x, y, tris, u0, **self.argsc)
        #         cmap = matplotlib.cm.jet
        #         plt.colorbar(matplotlib.cm.ScalarMappable(norm=self.norm, cmap=cmap), ax=ax)
        #         self.heat, self.u0, self.ax  = heat, u0, ax
        #         self.x, self.y, self.tris = x, y, tris
        #         self.u = u0
        #
        #     def __call__(self, i):
        #         heat, u0, ax = self.heat, self.u0, self.ax
        #         x, y, tris = self.x, self.y, self.tris
        #         self.u = heat.dynamic(self.u, niter=100, dt=1.)
        #         print(f"{i=} {np.linalg.norm(self.u)}")
        #         ax.cla()
        #         ax.set_title(f"Iter {i}")
        #         ax.tricontourf(x, y, tris, self.u, **self.argscf)
        #         ax.tricontour(x, y, tris, self.u, **self.argsc)
        # # simfempy.meshes.plotmesh.meshWithData(heat.mesh, point_data={'u':u}, title="Heat dynamic", alpha=1)
        # fig = plt.figure()
        # ha = HeatAnim(fig.gca(), mesh, u0, heat)
        # anim = animation.FuncAnimation(fig, ha, frames=25, repeat=False)
        # plt.show()
    else: raise ValueError(f"unknown{ mode=}")

# ---------------------------------------------------------------- #
def createMesh(h=0.2):
    rect = [-2, 2, -2, 2]
    with pygmsh.geo.Geometry() as geom:
        holes = []
        holes.append(hole(geom, xc=0, yc=0, r=0.4, mesh_size=h, label="3000", circle=True))
        holes.append(hole(geom, xc=-1, yc=-1, r=0.5, mesh_size=h, label="200", make_surface=True))
        p = geom.add_rectangle(*rect, z=0, mesh_size=h, holes=holes)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    # from simfempy.meshes.plotmesh import plotmeshWithNumbering
    # plotmeshWithNumbering(mesh)
    return SimplexMesh(mesh=mesh)
# ---------------------------------------------------------------- #
def createDataDyn():
    data = ProblemData()
    bdrycond =  data.bdrycond
    bdrycond.set("Robin", [1000])
    bdrycond.param[1000] = 10
    bdrycond.set("Neumann", [1002, 1001, 1003, 3000, 3001, 3002])
    bdrycond.fct[1000] = lambda x,y,z: 120
    postproc = data.postproc
    postproc.type['bdrymean_low'] = "bdry_mean"
    postproc.color['bdrymean_low'] = [1000]
    postproc.type['bdrymean_up'] = "bdry_mean"
    postproc.color['bdrymean_up'] = [1002]
    postproc.type['fluxn'] = "bdry_nflux"
    postproc.color['fluxn'] = [1001, 1003]
    params = data.params
    params.set_scal_cells("kheat", [100], 0.001)
    params.set_scal_cells("kheat", [200], 10.0)
    # alternative:
    # def kheat(label, x, y, z):
    #     if label==100: return 0.0001
    #     return 0.1*label
    # params.fct_glob["kheat"] = kheat
    # params.fct_glob["convection"] = ["0", "0.01"]
    return data
# ---------------------------------------------------------------- #
def createData():
    data = ProblemData()
    bdrycond =  data.bdrycond
    # bdrycond.set("Robin", [1000])
    # bdrycond.set("Dirichlet", [1001, 1003])
    bdrycond.set("Dirichlet", [1000, 1001, 1003])
    bdrycond.set("Neumann", [1002, 3000, 3001, 3002])
    bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz: 0.0
    bdrycond.fct[1000] = bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    # bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    # bdrycond.fct[1000] = lambda x, y, z, nx, ny, nz: 100
    # bdrycond.param[1000] = 1
    postproc = data.postproc
    postproc.type['bdrymean_low'] = "bdry_mean"
    postproc.color['bdrymean_low'] = [1000]
    postproc.type['bdrymean_up'] = "bdry_mean"
    postproc.color['bdrymean_up'] = [1002]
    postproc.type['fluxn'] = "bdry_nflux"
    postproc.color['fluxn'] = [1001, 1003]
    params = data.params
    params.set_scal_cells("kheat", [100], 0.001)
    params.set_scal_cells("kheat", [200], 10.0)
    # alternative:
    # def kheat(label, x, y, z):
    #     if label==100: return 0.0001
    #     return 0.1*label
    # params.fct_glob["kheat"] = kheat
    # params.fct_glob["convection"] = ["0", "0.01"]
    return data


# ---------------------------------------------------------------- #
def createDataConvection():
    data = ProblemData()
    bdrycond =  data.bdrycond
    bdrycond.set("Robin", [1000])
    bdrycond.set("Dirichlet", [1000, 3000, 3001, 3002])
    bdrycond.set("Neumann", [1001, 1002, 1003])
    bdrycond.fct[1001] = bdrycond.fct[1003] = bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz: 0.0
    bdrycond.fct[3000] = bdrycond.fct[3001] = bdrycond.fct[3002] = lambda x,y,z: 320
    bdrycond.fct[1000] = lambda x, y, z: 200
    postproc = data.postproc
    postproc.type['bdrymean_low'] = "bdry_mean"
    postproc.color['bdrymean_low'] = [1000]
    postproc.type['bdrymean_up'] = "bdry_mean"
    postproc.color['bdrymean_up'] = [1002]
    postproc.type['fluxn'] = "bdry_nflux"
    postproc.color['fluxn'] = [1001, 1003]
    params = data.params
    params.set_scal_cells("kheat", [100], 0.001)
    params.set_scal_cells("kheat", [200], 10.0)
    params.fct_glob["convection"] = ["0", "0.001"]
    return data

# ================================================================c#


main()