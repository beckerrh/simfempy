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
def _plotFaces(x, y, xf, yf, faces, ax=plt, plotlocalNumbering=False):
    for i in range(len(faces)):
        ax.text(xf[i], yf[i], f'{i}', color='b', fontweight='bold', fontsize=10)
        if plotlocalNumbering:
            for ii in range(2):
                iv = faces[i, ii]
                ax.text(0.75 * x[iv] + 0.25 * xf[i], 0.75 * y[iv] + 0.25 * yf[i],
                        r'%d' % (ii), color='g', fontweight='bold')

#----------------------------------------------------------------#
def _plotCellSideLocal(xc, yc, xf, yf, triangles, sidesofcells, ax=plt):
    for ic in range(len(triangles)):
        for ii in range(3):
            ie = sidesofcells[ic, ii]
            ax.text(0.7 * xf[ie] + 0.3 * xc[ic], 0.7 * yf[ie] + 0.3 * yc[ic],
                        r'%d' % (ii), color='g', fontweight='bold')

#----------------------------------------------------------------#
def _plotSideCellLocal(xc, yc, xf, yf, sides, cellsofsides, ax=plt):
    for ie in range(len(sides)):
        for ii in range(2):
            ic = cellsofsides[ie, ii]
            if ic < 0: continue
            ax.text(0.7 * xf[ie] + 0.3 * xc[ic], 0.7 * yf[ie] + 0.3 * yc[ic],
                        r'%d' % (ii), color='m', fontweight='bold')

#----------------------------------------------------------------#
def _plotNormalsAndSigma(xc, yc, xf, yf, normals, sidesofcells, sigma, ax=plt):
    ax.quiver(xf, yf, normals[:, 0], normals[:, 1])
    for ic in range(len(xc)):
        for ii in range(3):
            ie = sidesofcells[ic, ii]
            s = sigma[ic, ii]
            ax.text(0.5 * xf[ie] + 0.5 * xc[ic], 0.5 * yf[ie] + 0.5 * yc[ic],
                        r'%d' % (s), color='y', fontweight='bold')

#=================================================================#
def meshWithBoundaries(x, y, tris, lines, bdrylabels, ax=plt):
    colors = np.unique(bdrylabels)
    # print("colors", colors)
    ax.triplot(x, y, tris, color='k')
    if ax ==plt:
        ax.xlabel(r'x')
        ax.ylabel(r'y')
    else:
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
def mesh(x, y, tris, **kwargs):
    ax = plt
    if 'ax' in kwargs: ax = kwargs.pop('ax')
    ax.triplot(x, y, tris, color='k', lw=1)
    title = "Mesh"
    nodes = True
    if 'nodes' in kwargs: nodes = kwargs.pop('nodes')
    cells = True
    if 'cells' in kwargs: cells = kwargs.pop('cells')
    sides = True
    if 'sides' in kwargs: sides = kwargs.pop('sides')
    cellsidelocal = False
    if 'cellsidelocal' in kwargs: cellsidelocal = kwargs.pop('cellsidelocal')
    sidecelllocal = False
    if 'sidecelllocal' in kwargs: sidecelllocal = kwargs.pop('sidecelllocal')
    normals = False
    if 'normals' in kwargs: normals = kwargs.pop('normals')

    if cells or cellsidelocal or sidecelllocal or normals:
        xc, yc = x[tris].mean(axis=1), y[tris].mean(axis=1)
    if sides or cellsidelocal or sidecelllocal or normals:
        if 'meshsides' not in kwargs: raise KeyError("need meshsides")
        meshsides = kwargs.pop('meshsides')
        xf, yf = x[meshsides].mean(axis=1), y[meshsides].mean(axis=1)
    if cellsidelocal or normals:
        sidesofcells = kwargs.pop('sidesofcells')
    if sidecelllocal:
        cellsofsides = kwargs.pop('cellsofsides')
    if normals:
        meshnormals = kwargs.pop('meshnormals')
        meshsigma = kwargs.pop('meshsigma')

    if nodes:
        title += " and Nodes"
        _plotVertices(x, y, ax=ax)
    if cells:
        title += " and Cells"
        cellslocal = False
        if 'cellslocal' in kwargs: cellslocal = kwargs.pop('cellslocal')
        _plotCellsLabels(x, y, tris, xc, yc, ax=ax, plotlocalNumbering=cellslocal)
    if sides:
        title += " and Sides"
        sideslocal = False
        if 'sideslocal' in kwargs: sideslocal = kwargs.pop('sideslocal')
        _plotFaces(x, y, xf, yf, meshsides, ax=ax, plotlocalNumbering=sideslocal)
    if cellsidelocal:
        title += " and Cells-Sides"
        _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
        _plotFaces(x, y, xf, yf, meshsides, ax=ax)
        _plotCellSideLocal(xc, yc, xf, yf, tris, sidesofcells, ax=ax)
    if sidecelllocal:
        title += " and Cells-Sides"
        _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
        _plotFaces(x, y, xf, yf, meshsides, ax=ax)
        _plotSideCellLocal(xc, yc, xf, yf, meshsides, cellsofsides, ax=ax)
    if normals:
        title += " and Normals"
        _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
        _plotFaces(x, y, xf, yf, meshsides, ax=ax)
        _plotNormalsAndSigma(xc, yc, xf, yf, meshnormals, sidesofcells, meshsigma, ax=ax)

    _settitle(ax, title)

#=================================================================#
def meshWithData(x, y, tris, xc, yc, point_data=None, cell_data=None, numbering=False, title=None, suptitle=None):
    nplots=0
    if point_data: nplots += len(point_data)
    if cell_data: nplots += len(cell_data)
    if nplots==0:
        print("meshWithData() no point_data")
        return
    ncols = min(nplots,3)
    nrows = nplots//3 + bool(nplots%3)
    # print("nrows, ncols", nrows, ncols)
    fig, axs = plt.subplots(nrows, ncols,figsize=(ncols*4.5,nrows*4), squeeze=False)
    if suptitle: fig.suptitle(suptitle)
    count=0
    if point_data:
        for pdn, pd in point_data.items():
            assert x.shape == pd.shape
            ax = axs[count//3,count%3]
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
            ax = axs[count//3,count%3]
            cnt = ax.tripcolor(x, y, tris, facecolors=cd, edgecolors='k', cmap='jet')
            if numbering:
                _plotVertices(x, y, tris, xc, yc, ax=ax)
                _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
            plt.colorbar(cnt, ax=ax)
            _settitle(ax, cdn)
            count += 1
    if title: fig.canvas.set_window_title(title)
    return fig, axs
    # plt.tight_layout()
