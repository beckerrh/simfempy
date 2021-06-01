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
def plotmesh(**kwargs):
    ax = kwargs.pop('ax', plt)
    title = kwargs.pop('title', 'Mesh')
    alpha = kwargs.pop('alpha', 1)
    if 'mesh' in kwargs:
        mesh = kwargs.pop('mesh')
        x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
    else:
        x, y, tris = kwargs.pop('x'), kwargs.pop('y'), kwargs.pop('tris')
    ax.triplot(x, y, tris, color='k', alpha=0.5)
    if ax == plt:
        plt.gca().set_aspect(aspect='equal')
        ax.xlabel(r'x')
        ax.ylabel(r'y')
    else:
        ax.set_aspect(aspect='equal')
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'y')
    # celllabels = mesh.celllabels
    # cnt = ax.tripcolor(x, y, tris, facecolors=celllabels, edgecolors='k', cmap='jet', alpha=0.4)
    # clb = plt.colorbar(cnt)
    # clb.set_label("cellcolors")
    # if len(mesh.verticesoflabel):
    #     pltcolors = 'bgrcmykbgrcmyk'
    #     patches = []
    #     for i, (color, vertices) in enumerate(mesh.verticesoflabel.items()):
    #         patches.append(mpatches.Patch(color=pltcolors[i], label=color))
    #         for vertex in vertices:
    #             ax.plot(x[vertex], y[vertex],'X', color=pltcolors[i])
    #     ax.legend(handles=patches)
    _settitle(ax, title)
#=================================================================#
def mesh(x, y, tris, **kwargs):
    ax = kwargs.pop('ax', plt)
    ax.triplot(x, y, tris, color='k', lw=1)
    title = "Mesh"
    nodes = kwargs.pop('nodes', True)
    cells = kwargs.pop('cells', True)
    sides = kwargs.pop('sides', False)
    cellsidelocal = kwargs.pop('cellsidelocal', False)
    sidecelllocal = kwargs.pop('sidecelllocal', False)
    normals = kwargs.pop('normals', False)
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
    return ax
#=================================================================#
def plotMeshWithPointData(ax, pdn, pd, x, y, tris, alpha):
    if not isinstance(pd, np.ndarray):
        raise ValueError(f"Problem in data {type(pd)=}")
    if x.shape != pd.shape:
        raise ValueError(f"Problem in data {x.shape=} {pd.shape=}")
    ax.triplot(x, y, tris, color='gray', lw=1, alpha=alpha)
    cnt = ax.tricontourf(x, y, tris, pd, levels=16, cmap='jet')
    clb = plt.colorbar(cnt, ax=ax, shrink=0.6)
    clb.ax.set_title(pdn)
    _settitle(ax, pdn)
#=================================================================#
def plotMeshWithCellData(ax, cdn, cd, x, y, tris, alpha):
    if tris.shape[0] != cd.shape[0]:
        raise ValueError("wrong length in '{}' {}!={}".format(cdn, tris.shape[0], cd.shape[0]))
    ax.triplot(x, y, tris, color='gray', lw=1, alpha=alpha)
    cnt = ax.tripcolor(x, y, tris, facecolors=cd, edgecolors='k', cmap='jet')
    clb = plt.colorbar(cnt, ax=ax, shrink=0.6)
    clb.ax.set_title(cdn)
    _settitle(ax, cdn)
#=================================================================#
def meshWithBoundaries(x, y, tris, **kwargs):
    fig = None
    if 'outer' in kwargs:
        import matplotlib.gridspec as gridspec
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=kwargs['outer'], wspace=0.1, hspace=0.1)
        if not 'fig' in kwargs: raise KeyError(f"needs argument 'fig")
        fig = kwargs['fig']
        ax = plt.Subplot(fig, inner[0])
    elif 'ax' in kwargs: ax = kwargs.pop('ax')
    else: ax = plt
    lines = kwargs.pop('lines')
    bdrylabels = kwargs.pop('bdrylabels')
    ax.triplot(x, y, tris, color='k')
    if ax ==plt:
        plt.gca().set_aspect(aspect='equal')
        ax.xlabel(r'x')
        ax.ylabel(r'y')
    else:
        ax.set_aspect(aspect='equal')
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'y')
    pltcolors = 'bgrcmykbgrcmyk'
    patches=[]
    i=0
    for color, edges in bdrylabels.items():
        patches.append(mpatches.Patch(color=pltcolors[i], label=color))
        for ie in edges:
            ax.plot(x[lines[ie]], y[lines[ie]], color=pltcolors[i], lw=4)
        i += 1
    if 'celllabels' in kwargs:
        celllabels = kwargs.pop('celllabels')
        cnt = ax.tripcolor(x, y, tris, facecolors=celllabels, edgecolors='k', cmap='jet', alpha=0.4)
        clb = plt.colorbar(cnt)
        # clb = plt.colorbar(cnt, ax=ax)
        # clb.ax.set_title(cdn)
        clb.set_label("cellcolors")
    if 'cellsoflabel' in kwargs:
        cellsoflabel = kwargs.pop('cellsoflabel')
        # print(f"{tris.shape=}")
        celllabels = np.empty(tris.shape[0])
        for color, cells in cellsoflabel.items(): celllabels[cells] = color
        cnt = ax.tripcolor(x, y, tris, facecolors=celllabels, edgecolors='k', cmap='jet', alpha=0.4)
        # clb = plt.colorbar(cnt)
        # clb.set_label("cellcolors")
    # clb = plt.colorbar(cnt, ax=ax)
    # clb.ax.set_title(cdn)
    ax.legend(handles=patches)
    _settitle(ax, "Mesh and Boundary Labels")
    if fig: fig.add_subplot(ax)
#=================================================================#
