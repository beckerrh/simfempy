import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib


#----------------------------------------------------------------#
class AnimData:
    def __init__(self, mesh, u):
        fig = plt.figure()
        ax = fig.gca()
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
        ax.tricontourf(x, y, tris, u[0], **self.argscf)
        cmap = matplotlib.cm.jet
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=self.norm, cmap=cmap), ax=ax)
        self.u, self.ax = u, ax
        self.x, self.y, self.tris = x, y, tris
        self.nframes = len(u)
        self.anim = animation.FuncAnimation(fig, self, frames=self.nframes, repeat=False)

    def __call__(self, i):
        u, ax = self.u, self.ax
        x, y, tris = self.x, self.y, self.tris
        ax.cla()
        ax.set_title(f"Iter {i+1}/{self.nframes}")
        # print(f"{i=} {np.linalg.norm(self.u[i])}")
        ax.tricontourf(x, y, tris, self.u[i], **self.argscf)
        ax.tricontour(x, y, tris, self.u[i], **self.argsc)
        return ax
