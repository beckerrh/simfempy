import matplotlib.pyplot as plt
# from . import plotmesh1d, plotmesh2d, plotmesh3d
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


#=================================================================#
def plotmesh(mesh, **kwargs):
    dim, meshdataismesh = _getDim(mesh)
    if dim == 1:
        plotmesh1d.plotmesh(mesh=mesh, **kwargs)
    elif dim == 2:
        plotmesh2d.plotmesh(mesh=mesh, **kwargs)
    else:
        plotmesh3d.plotmesh(mesh=mesh, **kwargs)
    # if not 'ax' in kwargs or kwargs['ax']==plt: plt.show()

#=================================================================#
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

#=================================================================#
# def meshWithData(meshdata, point_data=None, cell_data=None, numbering=False, title=None, suptitle=None, addplots=[]):
def meshWithData(meshdata, **kwargs):

    dim, meshdataismesh = _getDim(meshdata)
    """
    meshdata    : either mesh or coordinates and connectivity
    point_data  : dictionary name->data
    cell_data  : dictionary name->data
    """
    newkwargs = kwargs.copy()

    if dim==1:
        if meshdataismesh:
            newkwargs['x'] = meshdata.points[:,0]
        else:
            newkwargs['x'] = meshdata[0]
        plotmesh1d.meshWithData(**newkwargs)
    elif dim==2:
        if meshdataismesh:
            newkwargs['x'] = meshdata.points[:,0]
            newkwargs['y'] = meshdata.points[:,1]
            newkwargs['tris'] = meshdata.simplices
            newkwargs['xc'] = meshdata.pointsc[:,0]
            newkwargs['yc'] = meshdata.pointsc[:,1]
        else:
            newkwargs['x'] = meshdata[0]
            newkwargs['y'] = meshdata[1]
            newkwargs['tris'] = meshdata[2]
            newkwargs['xc'] = meshdata.pointsc[3]
            newkwargs['yc'] = meshdata.pointsc[4]
        return plotmesh2d.meshWithData(**newkwargs)
    elif dim==3:
        if meshdataismesh:
            newkwargs['x'] = meshdata.points[:,0]
            newkwargs['y'] = meshdata.points[:,1]
            newkwargs['z'] = meshdata.points[:,2]
            newkwargs['tets'] = meshdata.simplices
            newkwargs['xc'] = meshdata.pointsc[:,0]
            newkwargs['yc'] = meshdata.pointsc[:,1]
            newkwargs['zc'] = meshdata.pointsc[:,2]
        else:
            newkwargs['x'] = meshdata[0]
            newkwargs['y'] = meshdata[1]
            newkwargs['z'] = meshdata[2]
            newkwargs['tets'] = meshdata[3]
            newkwargs['xc'] = meshdata.pointsc[4]
            newkwargs['yc'] = meshdata.pointsc[5]
            newkwargs['zc'] = meshdata.pointsc[6]
        return plotmesh3d.meshWithData(**newkwargs)
    else:
        raise ValueError(f"wrong dimension {dim=}")

#=================================================================#
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
    plt.show()

#=================================================================#
#=================================================================#
if __name__ == '__main__':
    import simplexmesh
    import geomdefs
    geometry = geomdefs.unitsquare.Unitsquare()
    mesh = simplexmesh.SimplexMesh(geometry=geometry, hmean=1)
    # plotmeshWithNumbering(mesh, localnumbering=True)
    plotmesh(mesh)
