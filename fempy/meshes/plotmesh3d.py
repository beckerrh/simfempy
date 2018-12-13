import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpl_toolkits import mplot3d

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
        ax.plot(x[tets[i]], y[tets[i]], z[tets[i]], color='k')

#=================================================================#
def _plotCellLabels(tets, xc, yc, zc, ax=plt):
    for i in range(len(tets)):
        ax.text(xc[i], yc[i], zc[i], r'%d' % (i), color='r', fontweight='bold', fontsize=10)


#=================================================================#
def meshWithBoundaries(x, y, z, tets, faces, bdrylabels, ax=plt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.unique(bdrylabels)
    _plotNodeLabels(x, y, z, ax=ax)
    _plotCells(x, y, z, tets, ax=ax)
    pltcolors = 'bgrcmyk'
    patches=[]
    i=0
    for color, bdryfaces in bdrylabels.items():
        patches.append(mpatches.Patch(color=pltcolors[i], label=color))
        for ie in bdryfaces:
            poly3d = [ [x[f], y[f], z[f]] for f in faces[ie]]
            ax.add_collection3d(Poly3DCollection([poly3d], facecolors=pltcolors[i], linewidths=1))
        i += 1
    ax.legend(handles=patches)
    _settitle(ax, "Mesh and Boundary Labels")

#=================================================================#
def meshWithData(x, y, z, tets, xc, yc, zc, point_data, cell_data, ax=plt, numbering=False):
    nplots = len(point_data)+len(cell_data)
    if nplots==0:
        print("meshWithData() no point_data")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _plotNodeLabels(x, y, z, ax=ax)
    _plotCellLabels(x, y, z, tets, xc, yc, zc, ax=ax)
