import pygmsh
from simfempy.models.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh

# ================================================================ #
class Application:
    def __init__(self, ncomp=1, h=None, has_exact_solution=False, random_exactsolution=False, scal_glob={}):
        self.h = h
        self.has_exact_solution = has_exact_solution
        self.random_exactsolution = random_exactsolution
        self.problemdata = ProblemData()
        self.problemdata.ncomp = ncomp
        for k,v in scal_glob.items():
            self.problemdata.params.scal_glob[k] = v
    def createMesh(self, h=None):
        if h is None: h = self.h
        with pygmsh.geo.Geometry() as geom:
            self.defineGeometry(geom, h)
            mesh = geom.generate_mesh()
        return SimplexMesh(mesh)
    def plot(self, mesh, data, **kwargs):
        if mesh.dimension == 2:
            import matplotlib.pyplot as plt
            from matplotlib.figure import figaspect
            import matplotlib.gridspec as gridspec
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            fig = kwargs.pop('fig', None)
            gs = kwargs.pop('gs', None)
            nplots = len(data['cell'].keys()) + len(data['point'].keys())
            if fig is None:
                if gs is not None:
                    raise ValueError(f"got gs but no fig")
                fig = plt.figure(constrained_layout=True, figsize=figaspect(nplots))
                # appname = kwargs.pop('title', self.__class__.__name__)
                # fig.set_title(f"{appname}")
            if gs is None:
                gs = fig.add_gridspec(1, 1)[0,0]
            inner = gridspec.GridSpecFromSubplotSpec(nrows=nplots, ncols=1, subplot_spec=gs, wspace=0.3, hspace=0.3)
            x, y, tris = mesh.points[:,0], mesh.points[:,1], mesh.simplices
            iplot = 0
            for name,values in data['cell'].items():
                ax = fig.add_subplot(inner[iplot])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.1)
                cnt = ax.tripcolor(x, y, tris, facecolors=values, edgecolors='k', cmap='jet')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.4)
                clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
                clb.ax.set_title(name)
                iplot += 1
            for name,values in data['point'].items():
                # print(f"{name=} {values.min()=}  {values.max()=}")
                ax = fig.add_subplot(inner[iplot])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.1)
                cnt = ax.tricontourf(x, y, tris, values, levels=16, cmap='jet', alpha=1.)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.4)
                clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
                clb.ax.set_title(name)
                iplot += 1
        else:
            raise ValueError("not written")
