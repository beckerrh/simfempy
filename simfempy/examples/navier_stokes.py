import numpy as np
import simfempy.models.application

# ================================================================ #
class Application(simfempy.models.application.Application):
    def __init__(self, ncomp=2, h=None, mu=0.1):
        super().__init__(ncomp=ncomp, h=h, scal_glob={'mu':mu})
    def plot(self, mesh, data, **kwargs):
        import matplotlib.pyplot as plt
        fig = kwargs.pop('fig', None)
        gs = kwargs.pop('gs', None)
        if fig is None:
            if gs is not None:
                raise ValueError(f"got gs but no fig")
            appname = kwargs.pop('title', self.__class__.__name__)
            fig = plt.figure(constrained_layout=True)
            plt.title(appname)
        if gs is None:
            gs = fig.add_gridspec(1, 1)[0, 0]
        if mesh.dimension == 2:
            self.plot2d(mesh, data, fig, gs, **kwargs)
        elif mesh.dimension == 3:
            self.plot3d(mesh, data, fig, gs, **kwargs)
        else:
            raise ValueError(f"not written {mesh=}")
    def plot2d(self, mesh, data, fig, gs, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.gridspec as gridspec
        nplots = 4
        inner = gridspec.GridSpecFromSubplotSpec(nrows=nplots, ncols=1, subplot_spec=gs, wspace=0.3, hspace=0.3)
        x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
        iplot = 0
        p = data['cell']['P']
        ax = fig.add_subplot(inner[iplot])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.1)
        cnt = ax.tripcolor(x, y, tris, facecolors=p, edgecolors='k', cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.4)
        clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
        clb.ax.set_title('P')
        iplot += 1
        for name, values in data['point'].items():
            ax = fig.add_subplot(inner[iplot])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.3)
            cnt = ax.tricontourf(x, y, tris, values, levels=16, cmap='jet', alpha=1.)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.4)
            clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
            clb.ax.set_title(name)
            iplot += 1
        v0,v1 = data['point']['V_0'], data['point']['V_1']
        ax = fig.add_subplot(inner[iplot])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.1)
        qv = ax.quiver(x, y, v0, v1, units='xy')
    def plot3d(self, mesh, data, fig, gs, **kwargs):
        try:
            import pyvista
            tets = mesh.simplices
            ntets = tets.shape[0]
            celltypes = pyvista.CellType.TETRA * np.ones(ntets, dtype=int)
            cells = np.insert(tets, 0, 4, axis=1).ravel()
            pyvistamesh = pyvista.UnstructuredGrid(cells, celltypes, mesh.points)
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            inner = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs, wspace=0.3, hspace=0.3)
            alpha = kwargs.pop('alpha', 0.6)
            p = data['cell']['P']
            v = np.zeros(shape=(mesh.nnodes, 3))
            v[:,0] = data['point']['V_0']
            v[:,1] = data['point']['V_1']
            v[:,2] = data['point']['V_2']
            vnorm = np.linalg.norm(v, axis=1)
            pyvistamesh["V"] = v
            pyvistamesh["vn"] = vnorm
            pyvistamesh.cell_data['P'] = p
            plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
            plotter.renderer.SetBackground(255, 255, 255)
            plotter.add_mesh(pyvistamesh, opacity=alpha, color='gray', show_edges=True)
            plotter.show(title=kwargs.pop('title', self.__class__.__name__))
            ax = fig.add_subplot(inner[0])
            ax.imshow(plotter.image)
            ax.set_xticks([])
            ax.set_yticks([])
            scalar_bar_args = {'title': 'p', 'color':'black'}
            plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
            plotter.renderer.SetBackground(255, 255, 255)
            plotter.add_mesh(pyvistamesh, opacity=alpha, color='gray', scalars='P', scalar_bar_args=scalar_bar_args)
            plotter.show(title=kwargs.pop('title', self.__class__.__name__))
            ax = fig.add_subplot(inner[1])
            ax.imshow(plotter.image)
            ax.set_xticks([])
            ax.set_yticks([])
            plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
            plotter.renderer.SetBackground(255, 255, 255)
            glyphs = pyvistamesh.glyph(orient="V", scale="vn", factor=10)
            # pyvistamesh.set_active_vectors("vectors")
            plotter.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='coolwarm')
            # plotter.show(title=kwargs.pop('title', self.__class__.__name__))
            # pyvistamesh.arrows.plot(off_screen=kwargs.pop('off_screen',True))
            plotter.show(title=kwargs.pop('title', self.__class__.__name__))
            ax = fig.add_subplot(inner[2])
            ax.imshow(plotter.image)
            ax.set_xticks([])
            ax.set_yticks([])
        except:
            print("pyvista is not installed")


# ================================================================ #
class Poiseuille2d(Application):
    def __init__(self, mu=0.01, h=0.5):
        super().__init__(mu=mu, h=h)
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1002, 1000, 1003])
        self.problemdata.bdrycond.set("Neumann", [1001])
        self.problemdata.bdrycond.set("Navier", [])
        self.problemdata.bdrycond.set("Pressure", [])
        self.problemdata.bdrycond.fct[1003] = [lambda x, y, z: 4 * y * (1 - y), lambda x, y, z: 0]
        # parameters
        self.problemdata.params.scal_glob["navier"] = 1.01
        # TODO pass ncomp with mesh ?!
    def defineGeometry(self, geom, h):
        p = geom.add_rectangle(xmin=0, xmax=4, ymin=0, ymax=1, z=0, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
# ================================================================ #
class Poiseuille3d(Application):
    def __init__(self, mu=0.01, h=0.5):
        super().__init__(mu=mu, h=h, ncomp=3)
        data = self.problemdata
        # boundary conditions
        data.bdrycond.set("Dirichlet", [100, 103])
        data.bdrycond.set("Neumann", [101])
        data.bdrycond.fct[103] = [lambda x, y, z: 16 * y * (1 - y) * z * (1 - z), lambda x, y, z: 0, lambda x, y, z: 0]
    def defineGeometry(self, geom, h):
        p = geom.add_rectangle(xmin=0, xmax=4, ymin=0, ymax=1, z=0, mesh_size=h)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical([top, p.surface, lat[0], lat[2]], label="100")
        geom.add_physical(lat[1], label="101")
        geom.add_physical(lat[3], label="103")
        geom.add_physical(vol, label="10")
# ================================================================ #
class DrivenCavity2d(Application):
    def __init__(self, h=0.1, mu=0.003):
        super().__init__(mu=mu, h=h, ncomp=2)
        data = self.problemdata
        # boundary conditions
        data.bdrycond.set("Dirichlet", [1000, 1002])
        data.bdrycond.fct[1002] = [lambda x, y, z: 1, lambda x, y, z: 0]
        # parameters
        data.params.scal_glob["mu"] = mu
    #
    def defineGeometry(self, geom, h):
        ms = [h*v for v in [1.,1.,0.2,0.2]]
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=ms)
        geom.add_physical(p.surface, label="100")
        geom.add_physical(p.lines[2], label="1002")
        geom.add_physical([p.lines[0], p.lines[1], p.lines[3]], label="1000")
# ================================================================ #
class DrivenCavity3d(Application):
    def __init__(self, h=0.1, mu=0.003):
        super().__init__(mu=mu, h=h, ncomp=3)
        data = self.problemdata
        data.bdrycond.set("Dirichlet", [100, 102])
        data.bdrycond.fct[102] = [lambda x, y, z: 1, lambda x, y, z: 0, lambda x, y, z: 0]
        # parameters
        data.params.scal_glob["mu"] = mu
    def defineGeometry(self, geom, h):
        ms = [h*v for v in [1.,1.,0.1,0.1]]
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=ms)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical(lat[2], label="102")
        geom.add_physical([top, p.surface, lat[0], lat[1], lat[3]], label="100")
        geom.add_physical(vol, label="10")
# ================================================================ #
class BackwardFacingStep2d(Application):
    def __init__(self, mu=0.02, h=0.2):
        super().__init__(mu=mu, h=h)
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1000, 1002])
        # self.problemdata.bdrycond.set("Pressure", [1004])
        self.problemdata.bdrycond.set("Neumann", [1004])
        self.problemdata.bdrycond.fct[1000] = [lambda x, y, z: y * (1 - y), lambda x, y, z: 0]
    def defineGeometry(self, geom, h):
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        hs = 6*[h]
        hs[2] *= 0.2
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=hs)
        #  np.insert(np.array(X), 2, 0, axis=1): fills zeros for z-coord
        geom.add_physical(p.surface, label="100")
        dirlines = [p for i,p in enumerate(p.lines) if i != 0 and i != 4]
        geom.add_physical(dirlines, "1002")
        geom.add_physical(p.lines[0], "1000")
        geom.add_physical(p.lines[4], "1004")
# ================================================================ #
class BackwardFacingStep3d(Application):
    def __init__(self, h=0.2, mu=0.02):
        super().__init__(mu=mu, h=h, ncomp=3)
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [100, 102])
        self.problemdata.bdrycond.set("Neumann", [104])
        self.problemdata.bdrycond.fct[102] = [lambda x, y, z: y*(1-y)*z*(1-z), lambda x, y, z: 0, lambda x, y, z: 0]
    def defineGeometry(self, geom, h):
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        hs = 6*[h]
        hs[2] *= 0.1
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=hs)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        dirf = [lat[i] for i in range(1,6) if i!=4 ]
        dirf.extend([p.surface, top])
        geom.add_physical(dirf, label="100")
        geom.add_physical(lat[0], label="102")
        geom.add_physical(lat[4], label="104")
        # for i in range(len(lat)):
        #     geom.add_physical(lat[i], label=f"{101+i}")
        geom.add_physical(vol, label="10")
        # geom.add_physical(p.surface, label="100")
        # dirlines = [p for i,p in enumerate(p.lines) if i != 0 and i != 4]
        # geom.add_physical(dirlines, "1002")
        # geom.add_physical(p.lines[0], "1000")
        # geom.add_physical(p.lines[4], "1004")
# ================================================================ #
class SchaeferTurek2d(Application):
    def __init__(self, hcircle=None, mu=0.01, h=0.5, errordrag=True):
        super().__init__(mu=mu, h=h)
        self.hcircle, self.errordrag = hcircle, errordrag
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1002, 1000, 1003, 3000])
        self.problemdata.bdrycond.set("Neumann", [1001])
        self.problemdata.bdrycond.fct[1003] = [lambda x, y, z: 0.3 * y * (4.1 - y) / 2.05 ** 2, lambda x, y, z: 0]
        self.problemdata.params.scal_glob["mu"] = mu
        self.problemdata.postproc.set(name='bdrynflux', type='bdry_nflux', colors=3000)
        self.problemdata.postproc.plot = ['drag', 'lift1', 'lift2']
    def changepostproc(self, info):
        bdrynflux = info.pop('bdrynflux_3000')
        # print(f"changepostproc: {bdrynflux=}")
        info['drag'] = -50 * bdrynflux[0]
        info['lift'] = -50 * bdrynflux[1]
        if self.errordrag:
            info['err_drag'] = 5.57953523384 + 50 * bdrynflux[0]
            info['err_lift'] = 0.010618937712 + 50 * bdrynflux[1]
    def defineGeometry(self, geom, h):
        if self.hcircle is None: hcircle = 0.2 * h
        circle = geom.add_circle(x0=[2,2], radius=0.5, mesh_size=hcircle, num_sections=10, make_surface=False)
        geom.add_physical(circle.curve_loop.curves, label="3000")
        p = geom.add_rectangle(xmin=0, xmax=11, ymin=0, ymax=4.1, z=0, mesh_size=h, holes=[circle])
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
# ================================================================ #
class SchaeferTurek3d(Application):
    def __init__(self, hcircle=None, mu=0.01, h=0.5, errordrag=True):
        super().__init__(mu=mu, h=h, ncomp=3)
        self.hcircle, self.errordrag = hcircle, errordrag
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [100, 103, 300])
        self.problemdata.bdrycond.set("Neumann", [101])
        self.problemdata.bdrycond.fct[103] = [lambda x, y, z: 0.45 * y * (4.1 - y) * z * (4.1 - z) / 2.05 ** 4, lambda x, y, z: 0,
                                  lambda x, y, z: 0]
        self.problemdata.params.scal_glob["mu"] = mu
        self.problemdata.postproc.set(name='bdrynflux', type='bdry_nflux', colors=300)
    def changepostproc(self, info):
        bdrynflux = info.pop('bdrynflux_300')
        scale = 50/4.1
        info['drag'] = -scale * bdrynflux[0]
        info['lift1'] = -scale * bdrynflux[1]
        info['lift2'] = -scale * bdrynflux[2]
        if self.errordrag:
            info['err_drag'] = 6 + scale * bdrynflux[0]
            info['err_lift1'] = 0.01 + scale * bdrynflux[1]
            info['err_lift2'] = 0.0175 + scale * bdrynflux[2]
    def defineGeometry(self, geom, h):
        if self.hcircle is None: hcircle = 0.3 * h
        circle = geom.add_circle(x0=[5,2], radius=0.5, mesh_size=hcircle, num_sections=8, make_surface=False)
        p = geom.add_rectangle(xmin=0, xmax=15, ymin=0, ymax=4.1, z=0, mesh_size=h, holes=[circle])
        axis = [0, 0, 4.1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical([top,p.surface, lat[0], lat[2]], label="100")
        geom.add_physical(lat[1], label="101")
        geom.add_physical(lat[3], label="103")
        geom.add_physical(lat[4:], label="300")
        geom.add_physical(vol, label="10")
