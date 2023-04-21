import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib


#----------------------------------------------------------------#
class AnimData:
    def __init__(self, mesh, u, plotfct=None, initfct=None):
        fig = plt.figure()
        ax = fig.gca()
        self.mesh, self.u, self.ax = mesh, u, ax
        self.nframes = len(u)
        if plotfct == None:
            self.plotfct = self.__plotfct__
        else:
            self.plotfct = plotfct
        if initfct == None:
            self.__initfct__(ax, u)
        else:
            initfct(ax, u)
        self.anim = animation.FuncAnimation(fig, self, frames=self.nframes, repeat=False)

    def __initfct__(self, ax, u):
        mesh = self.mesh
        ax.set_aspect(aspect='equal')
        x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
        ax.triplot(x, y, tris, color='gray', lw=1, alpha=1)
        smax, smin  = -np.inf, np.inf
        for s in u:
            smin = min(smin,np.min(s))
            smax = max(smax, np.max(s))
        self.norm = matplotlib.colors.Normalize(vmin=smin, vmax=smax)
        self.argscf = {'levels': 32, 'norm': self.norm, 'cmap': 'jet'}
        self.argsc = {'colors': 'k', 'levels': np.linspace(smin, smax, 32)}
        cmap = matplotlib.cm.jet
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=self.norm, cmap=cmap), ax=ax)

    def __plotfct__(self, ax, u):
        x, y, tris = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.simplices
        ax.tricontourf(x, y, tris, u, **self.argscf)
        ax.tricontour(x, y, tris, u, **self.argsc)

    def __call__(self, i):
        ax = self.ax
        ax.cla()
        ax.set_title(f"Iter {i+1}/{self.nframes}")
        self.plotfct(ax, self.u[i])
        return ax
