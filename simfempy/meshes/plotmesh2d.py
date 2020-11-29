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
    if 'ax' in kwargs: ax = kwargs.pop('ax')
    else: ax = plt
    title = 'Mesh'
    if 'title' in kwargs: title = kwargs.pop('title')
    alpha = 1
    if 'alpha' in kwargs: alpha = kwargs.pop('alpha')
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
def meshWithBoundaries(x, y, tris, **kwargs):
    if 'ax' in kwargs: ax = kwargs.pop('ax')
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
        for color, cells in cellsoflabel.items():
            celllabels[cells] = color
        cnt = ax.tripcolor(x, y, tris, facecolors=celllabels, edgecolors='k', cmap='jet', alpha=0.4)
        clb = plt.colorbar(cnt)
        # clb = plt.colorbar(cnt, ax=ax)
        # clb.ax.set_title(cdn)
        clb.set_label("cellcolors")

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
    sides = False
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
def meshWithData(**kwargs):
    """
    point_data  : dictionary name->data
    cell_data  : dictionary name->data
    addplots  : additional plot functions (in new axes)
    """
    x, y, tris, xc, yc = kwargs['x'], kwargs['y'], kwargs['tris'], kwargs['xc'], kwargs['yc']
    addplots = []
    if 'addplots' in kwargs: addplots = kwargs['addplots']
    if addplots is None: addplots=[]
    point_data, cell_data, quiver_cell_data = None, None, None
    numbering = False
    title, suptitle = None, None
    # if 'point_data' in kwargs: point_data = kwargs['point_data']
    # if 'cell_data' in kwargs: cell_data = kwargs['cell_data']
    if 'data' in kwargs:
        point_data = kwargs['data']['point']
        cell_data = kwargs['data']['cell']
    if 'point_data' in kwargs: point_data = kwargs['point_data']
    if 'cell_data' in kwargs: cell_data = kwargs['cell_data']
    if 'quiver_cell_data' in kwargs: quiver_cell_data = kwargs['quiver_cell_data']
    if 'numbering' in kwargs: numbering = kwargs['numbering']
    if 'title' in kwargs: title = kwargs['title']
    if 'suptitle' in kwargs: suptitle = kwargs['suptitle']

    nplots=0
    if point_data: nplots += len(point_data)
    if cell_data: nplots += len(cell_data)
    if quiver_cell_data: nplots += len(quiver_cell_data)
    nplots += len(addplots)
    if nplots==0:
        raise ValueError("meshWithData(): no data")
    ncols = min(nplots,3)
    nrows = nplots//3 + bool(nplots%3)
    # print("nrows, ncols", nrows, ncols)
    fig, axs = plt.subplots(nrows, ncols,figsize=(ncols*4.5,nrows*4), squeeze=False)
    if suptitle: fig.suptitle(suptitle)
    # aspect = (np.max(x)-np.mean(x))/(np.max(y)-np.mean(y))
    count=0
    if point_data:
        for pdn, pd in point_data.items():
            assert x.shape == pd.shape
            ax = axs[count//ncols,count%ncols]
            ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.4)
            cnt = ax.tricontourf(x, y, tris, pd, 16, cmap='jet')
            ax.set_aspect(aspect='equal')
            if numbering:
                _plotVertices(x, y, tris, xc, yc, ax=ax)
                _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
            clb = plt.colorbar(cnt, ax=ax)
            # clb.set_label(pdn)
            _settitle(ax, pdn)
            count += 1
    if cell_data:
        for cdn, cd in cell_data.items():
            if tris.shape[0] != cd.shape[0]:
                raise ValueError("wrong length in '{}' {}!={}".format(cdn,tris.shape[0],cd.shape[0]))
            # ax = axs[count//3,count%3]
            ax = axs[count//ncols,count%ncols]
            cnt = ax.tripcolor(x, y, tris, facecolors=cd, edgecolors='k', cmap='jet')
            ax.set_aspect(aspect='equal')
            if numbering:
                _plotVertices(x, y, tris, xc, yc, ax=ax)
                _plotCellsLabels(x, y, tris, xc, yc, ax=ax)
            clb = plt.colorbar(cnt, ax=ax)
            # clb.ax.set_title(cdn)
            clb.set_label(cdn)
            _settitle(ax, cdn)
            count += 1
    if quiver_cell_data:
        for cdn, cd in quiver_cell_data.items():
            ax = axs[count//ncols,count%ncols]
            ax.set_aspect(aspect='equal')
            if 'plotmesh' in kwargs and kwargs['plotmesh']: plotmesh(x=x, y=y, tris=tris, ax=ax, alpha=0.3)
            ax.quiver(xc, yc, cd[0], cd[1], units='xy')
            count += 1

    for addplot in addplots:
        ax = axs[count // ncols, count % ncols]
        addplot(ax)
    if title: fig.canvas.set_window_title(title)
    return fig, axs
    # plt.tight_layout()
