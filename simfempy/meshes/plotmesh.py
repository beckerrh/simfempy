import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#----------------------------------------------------------------#
def plotmesh(mesh, **kwargs):
    assert mesh.dimension == 2
    ax = kwargs.pop('ax', plt)
    title = kwargs.pop('title', 'Mesh')
    alpha = kwargs.pop('alpha', 1)
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
    try:
        ax.set_title(title)
    except:
        ax.title(title)

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
    try:
        ax.set_title(pdn)
    except:
        ax.title(pdn)
#=================================================================#
def plotMeshWithCellData(ax, cdn, cd, x, y, tris, alpha):
    if tris.shape[0] != cd.shape[0]:
        raise ValueError("wrong length in '{}' {}!={}".format(cdn, tris.shape[0], cd.shape[0]))
    ax.triplot(x, y, tris, color='gray', lw=1, alpha=alpha)
    cnt = ax.tripcolor(x, y, tris, facecolors=cd, edgecolors='k', cmap='jet')
    clb = plt.colorbar(cnt, ax=ax, shrink=0.6)
    clb.ax.set_title(cdn)
    try:
        ax.set_title(cdn)
    except:
        ax.title(cdn)
#----------------------------------------------------------------#
def meshWithData(mesh, **kwargs):
    simp = mesh.simplices
    x, y, xc, yc = mesh.points[:, 0], mesh.points[:, 1], mesh.pointsc[:, 0], mesh.pointsc[:, 1]
    addplots = kwargs.pop('addplots',[])
    numbering = kwargs.pop('numbering',False)
    alpha = kwargs.pop('alpha', 0.6)
    plotmesh = kwargs.pop('plotmesh', None)
    if 'data' in kwargs:
        point_data = kwargs['data'].get('point', {})
        cell_data = kwargs['data'].get('cell', {})
    else:
        point_data = {}
        cell_data = {}
    if 'point_data' in kwargs:
        assert isinstance(kwargs['point_data'], dict)
        point_data.update(kwargs['point_data'])
    if 'cell_data' in kwargs:
        assert isinstance(kwargs['cell_data'], dict)
        cell_data.update(kwargs['cell_data'])
    quiver_data = kwargs.get('quiver_data', {})
    nplots = len(point_data) + len(cell_data) + len(quiver_data) + len(addplots)
    if nplots==0: raise ValueError("meshWithData(): no data")
    if 'outer' in kwargs:
        import matplotlib.gridspec as gridspec
        inner = gridspec.GridSpecFromSubplotSpec(nplots, 1, subplot_spec=kwargs['outer'], wspace=0.1, hspace=0.1)
        if not 'fig' in kwargs: raise KeyError(f"needs argument 'fig")
        fig = kwargs['fig']
    else:
        ncols = min(nplots,3)
        nrows = nplots//3 + bool(nplots%3)
        fig, axs = plt.subplots(nrows, ncols,figsize=(ncols*4.5,nrows*4), squeeze=False)
    count=0
    for pdn, pd in point_data.items():
        if 'outer' in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count//ncols,count%ncols]
        plotMeshWithPointData(ax, pdn, pd, x, y, simp, alpha)
        ax.set_aspect(aspect='equal')
        if 'title' in kwargs: ax.set_title(kwargs['title'])
        fig.add_subplot(ax)
        count += 1
    for cdn, cd in cell_data.items():
        if 'outer' in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count//ncols,count%ncols]
        plotMeshWithCellData(ax, cdn, cd, x, y, simp, alpha)
        ax.set_aspect(aspect='equal')
        fig.add_subplot(ax)
        count += 1
    for qdn, qd in quiver_data.items():
        if 'outer' in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count//ncols,count%ncols]
        ax.set_aspect(aspect='equal')
        if len(qd)!=2: raise ValueError(f"{len(qd)=} {quiver_data=}")
        if qd[0].shape[0] == x.shape[0]:
            ax.quiver(x, y, qd[0], qd[1], units='xy')
        else:
            ax.quiver(xc, yc, qd[0], qd[1], units='xy')
        ax.set_aspect(aspect='equal')
        fig.add_subplot(ax)
        count += 1
    for addplot in addplots:
        if 'outer' in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count//ncols,count%ncols]
        addplot(ax)
        count += 1
    return fig
#----------------------------------------------------------------#


#----------------------------------------------------------------#
def meshWithBoundaries(mesh, **kwargs):
    fig = None
    if 'outer' in kwargs:
        import matplotlib.gridspec as gridspec
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=kwargs['outer'], wspace=0.1, hspace=0.1)
        if not 'fig' in kwargs: raise KeyError(f"needs argument 'fig")
        fig = kwargs['fig']
        ax = plt.Subplot(fig, inner[0])
    elif 'ax' in kwargs: ax = kwargs.pop('ax')
    else: ax = plt
    lines = mesh.faces
    bdrylabels = mesh.bdrylabels
    x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
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
        col = pltcolors[i%len(pltcolors)]
        patches.append(mpatches.Patch(color=col, label=f"{color}"))
        for ie in edges:
            ax.plot(x[lines[ie]], y[lines[ie]], color=col, lw=4)
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
    title = "Mesh and Boundary Labels"
    try:
        ax.set_title(title)
    except:
        ax.title(title)
    if fig: fig.add_subplot(ax)
