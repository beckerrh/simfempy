import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#----------------------------------------------------------------#
def _settitle(ax, text):
    try:
        ax.set_title(text)
    except:
        ax.title(text)

#----------------------------------------------------------------#
def _plotVertices(x, y, ax=plt):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for iv in range(len(x)):
        ax.text(x[iv], y[iv], r'%d' % (iv), fontweight='bold', bbox=props)

#----------------------------------------------------------------#
def _plotCellsLabels(x, y, triangles, xc, yc, ax=plt, plotlocalNumbering=False):
    for i in range(len(triangles)):
        ax.text(xc[i], yc[i], r'%d' % (i), color='r', fontweight='bold', fontsize=10)
        if plotlocalNumbering:
            for ii in range(3):
                iv = triangles[i, ii]
                ax.text(0.75 * x[iv] + 0.25 * xc[i], 0.75 * y[iv] + 0.25 * yc[i],
                        r'%d' % (ii), color='g', fontweight='bold')

#----------------------------------------------------------------#
def _plotFaces(xf, yf, faces, ax=plt):
    for i in range(len(faces)):
        ax.text(xf[i], yf[i], f'{i}', color='b', fontweight='bold', fontsize=10)

#=================================================================#
def meshWithBoundaries(x, y, tris, lines, bdrylabels, ax=plt):
    colors = np.unique(bdrylabels)
    # print("colors", colors)
    ax.triplot(x, y, tris, color='k')
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    pltcolors = 'bgrcmyk'
    patches=[]
    i=0
    for color, edges in bdrylabels.items():
        patches.append(mpatches.Patch(color=pltcolors[i], label=color))
        for ie in edges:
            ax.plot(x[lines[ie]], y[lines[ie]], color=pltcolors[i], lw=4)
        i += 1
    ax.legend(handles=patches)
    _settitle(ax, "Mesh and Boundary Labels")

#=================================================================#
def meshWithNodesAndTriangles(x, y, tris, xc, yc, ax=plt):
    ax.triplot(x, y, tris, color='k', lw=1)
    _plotVertices(x, y, ax=ax)
    _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
    _settitle(ax, "Nodes and Triangles")

#=================================================================#
def meshWithNodesAndFaces(x, y, tris, xf, yf, faces, ax=plt):
    ax.triplot(x, y, tris, color='k', lw=1)
    _plotFaces(xf, yf, faces, ax=ax)
    _settitle(ax, "Nodes and Faces")

#=================================================================#
def meshWithData(x, y, tris, xc, yc, point_data, cell_data=None, ax=plt, numbering=False, title=None, suptitle=None):
    nplots = len(point_data)
    if cell_data: nplots += len(cell_data)
    if nplots==0:
        print("meshWithData() no point_data")
        return
    fig, axs = plt.subplots(1, nplots,figsize=(nplots*4.5,4), squeeze=False)
    print("suptitle",suptitle)
    if suptitle: fig.suptitle(suptitle)
    count=0
    for pdn, pd in point_data.items():
        assert x.shape == pd.shape
        ax = axs[0,count]
        ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.4)
        cnt = ax.tricontourf(x, y, tris, pd, 16, cmap='jet')
        if numbering:
            _plotVertices(x, y, tris, xc, yc, ax=ax)
            _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
        plt.colorbar(cnt, ax=ax)
        _settitle(ax, pdn)
        count += 1
    if cell_data:
        for cdn, cd in cell_data.items():
            assert tris.shape[0] == cd.shape[0]
            ax = axs[0,count]
            cnt = ax.tripcolor(x, y, tris, facecolors=cd, edgecolors='k', cmap='jet')
            if numbering:
                _plotVertices(x, y, tris, xc, yc, ax=ax)
                _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
            plt.colorbar(cnt, ax=ax)
            _settitle(ax, cdn)
            count += 1
    if title: fig.canvas.set_window_title(title)
    plt.tight_layout()
    plt.show()
