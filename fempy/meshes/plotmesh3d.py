import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import colors as mcolors

#----------------------------------------------------------------#
def _settitle(ax, text):
    try:
        ax.set_title(text)
    except:
        ax.title(text)

#----------------------------------------------------------------#
def _plotNodeLabels(x, y, z, ax=plt):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], r'%d' % (i), fontweight='bold', bbox=props)


#=================================================================#
def _plotCells(x, y, z, tets, ax=plt):
    for i in range(len(tets)):
        ax.plot(x[tets[i]], y[tets[i]], z[tets[i]], color=(0.5,0.5,0.5))

#=================================================================#
def _plotCellLabels(tets, xc, yc, zc, ax=plt):
    for i in range(len(tets)):
        ax.text(xc[i], yc[i], zc[i], r'%d' % (i), color='r', fontweight='bold', fontsize=10)


#=================================================================#
def meshWithBoundaries(x, y, z, tets, faces, bdrylabels, nodelabels=False, ax=plt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # if nodelabels: _plotNodeLabels(x, y, z, ax=ax)
    _plotCells(x, y, z, tets, ax=ax)
    cmap = plt.get_cmap("tab10")
    patches=[]
    i=0
    for color, bdryfaces in bdrylabels.items():
        patches.append(mpatches.Patch(color=cmap(i), label=color))
        for ie in bdryfaces:
            poly3d = [ [x[f], y[f], z[f]] for f in faces[ie]]
            ax.add_collection3d(Poly3DCollection([poly3d], facecolors=cmap(i), linewidths=1))
        i += 1
    ax.legend(handles=patches)
    _settitle(ax, "Mesh and Boundary Labels")

#=================================================================#
def meshWithData(x, y, z, tets, xc, yc, zc, point_data, cell_data, ax=plt, numbering=False, title=None, suptitle=None):
    return

#=================================================================#
def meshWithData3(x, y, z, tets, xc, yc, zc, point_data, cell_data, ax=plt, numbering=False, title=None, suptitle=None):
    from mayavi import mlab
    from tvtk.api import tvtk
    v = mlab.figure()
    ug = tvtk.UnstructuredGrid()
    ug.points = np.stack((x,y,z)).T
    tet_type = tvtk.Tetra().cell_type
    ug.set_cells(tet_type, tets)
    m = tvtk.PolyDataMapper()
    m.input = ug.output
    a = tvtk.Actor()
    a.mapper = m
    mlab.show()

# =================================================================#
def meshWithData2(x, y, z, tets, xc, yc, zc, point_data, cell_data, ax=plt, numbering=False, title=None, uptitle=None):
    nplots = len(point_data)+len(cell_data)
    if nplots==0:
        print("meshWithData() no point_data")
        return
    fig = plt.figure(figsize=plt.figaspect(1/nplots))
    if suptitle: fig.suptitle(suptitle)
    axs = []
    for i in range(nplots):
        axs.append(fig.add_subplot("1{:1d}{:1d}".format(nplots, i+1), projection='3d'))
    count=0
    for pdn, pd in point_data.items():
        ax = axs[count]
        xyz = np.stack((x,y,z)).T
        print("xyz", xyz.shape)
        vts = xyz[tets, :]
        tri = mplot3d.art3d.Poly3DCollection(vts)
        tri.set_alpha(0.2)
        tri.set_color('grey')
        ax.add_collection3d(tri)
        ax.plot(x,y,z, 'k.')
        ax.set_axis_off()
        # for i in range(len(tets)):
        #     ax.plot(x[tets[i]], y[tets[i]], z[tets[i]])
            # ax.plot_trisurf(x[tets[i]], y[tets[i]], z[tets[i]])
        _settitle(ax, pdn)
        count += 1
    if title: fig.canvas.set_window_title(title)
    plt.show()
