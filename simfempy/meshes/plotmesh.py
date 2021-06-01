import matplotlib.pyplot as plt
from simfempy.meshes import plotmesh1d, plotmesh2d, plotmesh3d
import meshio

#----------------------------------------------------------------#
def _getDim(meshdata):
    try:
        dim = meshdata.dimension
        meshdataismesh = True
    except:
        dim = len(meshdata)-3
        meshdataismesh = False
    return dim, meshdataismesh
#----------------------------------------------------------------#
def plotmesh(mesh, **kwargs):
    dim, meshdataismesh = _getDim(mesh)
    if dim == 1:
        plotmesh1d.plotmesh(mesh=mesh, **kwargs)
    elif dim == 2:
        plotmesh2d.plotmesh(mesh=mesh, **kwargs)
    else:
        plotmesh3d.plotmesh(mesh=mesh, **kwargs)
    # if not 'ax' in kwargs or kwargs['ax']==plt: plt.show()
#----------------------------------------------------------------#
def plotmeshWithNumbering(meshdata, **kwargs):
    if isinstance(meshdata,meshio._mesh.Mesh):
        types = [c.type for c in meshdata.cells]
        # print(f"{types=}")
        if 'tetra' in types: raise ValueError(f"so far only 2D")
        x, y  = meshdata.points[:,0], meshdata.points[:,1]
        for c, cb in meshdata.cells:
            if c=='triangle': tris = cb
            # elif c=='lines': faces = cb
    else:
        dim, meshdataismesh = _getDim(meshdata)
        if dim==3:
            raise NotImplementedError("3d not yet implemented")
        if meshdataismesh:
            x, y, tris, faces = meshdata.points[:,0], meshdata.points[:,1], meshdata.simplices, meshdata.faces
            kwargs['meshsides'] = faces
        else:
            x, y, tris = meshdata[0], meshdata[1], meshdata[2]
    if 'localnumbering' in kwargs and kwargs.pop('localnumbering'):
        fig, axs = plt.subplots(2, 3, figsize=(13.5, 8), squeeze=False)

        newkwargs = {}
        newkwargs['meshsides'] = faces
        newkwargs['cellsofsides'] = meshdata.cellsOfFaces
        newkwargs['sidesofcells'] = meshdata.facesOfCells
        newkwargs['meshnormals'] = meshdata.normals
        newkwargs['meshsigma'] = meshdata.sigma

        newkwargs['ax']= axs[0,0]
        plotmesh2d.mesh(x, y, tris, **newkwargs)

        newkwargs['ax']= axs[0,1]
        newkwargs['cellslocal']= True
        newkwargs['sides']= False
        plotmesh2d.mesh(x, y, tris, **newkwargs)

        newkwargs['ax']= axs[0,2]
        newkwargs['sideslocal']= True
        newkwargs['sides']= True
        newkwargs['cells']= False
        plotmesh2d.mesh(x, y, tris, **newkwargs)

        newkwargs['ax']= axs[1,0]
        newkwargs['nodes']= False
        newkwargs['sides']= False
        newkwargs['cells']= False
        newkwargs['cellsidelocal']= True
        plotmesh2d.mesh(x, y, tris, **newkwargs)

        newkwargs['ax']= axs[1,1]
        newkwargs['cellsidelocal']= False
        newkwargs['sidecelllocal']= True
        plotmesh2d.mesh(x, y, tris, **newkwargs)

        newkwargs['ax']= axs[1,2]
        newkwargs['normals']= True
        newkwargs['sidecelllocal']= False
        plotmesh2d.mesh(x, y, tris, **newkwargs)
    else:
        kwargs['ax']= plt
        plotmesh2d.mesh(x, y, tris, **kwargs)
#----------------------------------------------------------------#
def meshWithBoundaries(meshdata, **kwargs):
    dim, meshdataismesh = _getDim(meshdata)
    if dim==1:
        x, lines = meshdata.points[:, 0], meshdata.simplices
        plotmesh1d.meshWithBoundaries(x, lines, **kwargs)
    elif dim==2:
        if meshdataismesh:
            x, y, tris = meshdata.points[:,0], meshdata.points[:,1], meshdata.simplices
            kwargs['lines'] = meshdata.faces
            kwargs['bdrylabels'] = meshdata.bdrylabels
            if hasattr(meshdata, 'cell_labels'):
                kwargs['celllabels'] = meshdata.cell_labels
            if hasattr(meshdata, 'cellsoflabel'):
                kwargs['cellsoflabel'] = meshdata.cellsoflabel
        else:
            x, y, tris = meshdata[0], meshdata[1], meshdata[2]
            kwargs['lines'] = meshdata[3]
            kwargs['bdrylabels'] = meshdata[4]
        plotmesh2d.meshWithBoundaries(x, y, tris, **kwargs)
    else:
        if meshdataismesh:
            x, y, z, tets = meshdata.points[:,0], meshdata.points[:,1], meshdata.points[:,2], meshdata.simplices
            faces, bdrylabels = meshdata.faces, meshdata.bdrylabels
            plotmesh3d.meshWithBoundaries(x, y, z, tets, faces, bdrylabels, **kwargs)
        else:
            plotmesh3d.meshWithBoundaries(meshdata, **kwargs)
#----------------------------------------------------------------#
def meshWithData(meshdata, **kwargs):
    """
    meshdata    : either mesh or coordinates and connectivity
    point_data  : dictionary name->data
    cell_data  : dictionary name->data
    """
    dim, meshdataismesh = _getDim(meshdata)
    if meshdataismesh:
        simp = meshdata.simplices
        if dim == 2:
            x, y, xc, yc = meshdata.points[:,0], meshdata.points[:,1], meshdata.pointsc[:,0], meshdata.pointsc[:,1]
        else:
            x, y, z = meshdata.points[:, 0], meshdata.points[:, 1], meshdata.points[:, 2]
            xc, yc, zc = meshdata.pointsc[:, 0], meshdata.pointsc[:, 1], meshdata.pointsc[:, 2]
    else:
        if dim == 2:
            x, y, simp, xc, yc = meshdata
        else:
            x, y, z, simp, xc, yc, zc = meshdata
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
        if dim==2:
            fig, axs = plt.subplots(nrows, ncols,figsize=(ncols*4.5,nrows*4), squeeze=False)
        else:
            fig = plt.figure(figsize=(ncols*4.5,nrows*4))
            axl=[]
            for ir in range(nrows):
                for ic in range(ncols):
                    pos = 100*nplots + 10*(ir+1) + (ic+1)
                    axl.append(fig.add_subplot(pos, projection='3d'))
            axs = np.array(axl).reshape(nrows,ncols)
    count=0
    for pdn, pd in point_data.items():
        if 'outer' in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count//ncols,count%ncols]
        if dim==1:
            plotmesh1d.plotMeshWithPointData(ax, pdn, pd, x, alpha)
        elif dim==2:
            plotmesh2d.plotMeshWithPointData(ax, pdn, pd, x, y, simp, alpha)
        else:
            plotmesh3d.plotMeshWithPointData(ax, pdn, pd, x, y, z, simp, alpha)
        if numbering:
            if dim == 2:
                plotmesh2d._plotVertices(x, y, simp, xc, yc, ax=ax)
                plotmesh2d._plotCellsLabels(x, y, simp, xc, yc, ax=ax)
            else: raise NotImplementedError("3d...")
        ax.set_aspect(aspect='equal')
        fig.add_subplot(ax)
        count += 1
    for cdn, cd in cell_data.items():
        if 'outer' in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count//ncols,count%ncols]
        if dim==1:
            plotmesh1d.plotMeshWithCellData(ax, cdn, cd, x, alpha)
        elif dim==2:
            plotmesh2d.plotMeshWithCellData(ax, cdn, cd, x, y, simp, alpha)
        else:
            plotmesh3d.plotMeshWithCellData(ax, cdn, cd, x, y, z, simp, alpha)
        if numbering:
            if dim == 2:
                plotmesh2d._plotVertices(x, y, simp, xc, yc, ax=ax)
                plotmesh2d._plotCellsLabels(x, y, simp, xc, yc, ax=ax)
            else:
                raise NotImplementedError("3d...")
        ax.set_aspect(aspect='equal')
        fig.add_subplot(ax)
        count += 1
    for qdn, qd in quiver_data.items():
        if 'outer' in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count//ncols,count%ncols]
        ax.set_aspect(aspect='equal')
        if dim == 2:
            if plotmesh: plotmesh2d.plotmesh(x=x, y=y, tris=simp, ax=ax, alpha=0.3)
            if len(qd)!=2: raise ValueError(f"{len(qd)=} {quiver_data=}")
            if qd[0].shape[0] == x.shape[0]:
                ax.quiver(x, y, qd[0], qd[1], units='xy')
            else:
                ax.quiver(xc, yc, qd[0], qd[1], units='xy')
        else:
            raise NotImplementedError("3d...")
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

# ----------------------------------------------------------------#
def meshTriSurf(mesh, data, fig, outer, **kwargs):
    """
    """
    ax = fig.add_subplot(outer, projection='3d')
    x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
    ax.plot_trisurf(x, y, tris, Z=data, cmap='jet')

# ----------------------------------------------------------------#
def meshTriShading(mesh, data, fig, outer, **kwargs):
    """
    """
    ax = fig.add_subplot(outer)
    x, y, tris = mesh.points[:, 0], mesh.points[:, 1], mesh.simplices
    cm = plt.get_cmap('Greys')
    cnt = ax.tripcolor(x, y, tris, data, cmap=cm)
    clb = plt.colorbar(cnt, ax=ax, shrink=0.6)
    ax.set_aspect(aspect='equal')

#=================================================================#
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from simfempy.meshes import testmeshes

    # mesh = testmeshes.backwardfacingstep(h=1)
    mesh = testmeshes.backwardfacingstep3d(h=1)
    xc, yc, zc = mesh.pointsc.T
    u = xc**2 + yc**2 + zc**2
    meshWithData(mesh, cell_data={'U':xc**2 + yc**2 + zc**2}, plotmesh=True)
