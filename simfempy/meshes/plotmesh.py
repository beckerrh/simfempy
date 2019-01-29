import matplotlib.pyplot as plt
try:
    import plotmesh2d
    import plotmesh3d
except ModuleNotFoundError:
    from . import plotmesh2d
    from . import plotmesh3d


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
def meshWithBoundaries(meshdata, ax=plt):
    dim, meshdataismesh = _getDim(meshdata)
    if dim==2:
        if meshdataismesh:
            x, y, tris, lines, labels = meshdata.points[:,0], meshdata.points[:,1], meshdata.simplices, meshdata.faces, meshdata.bdrylabels
            plotmesh2d.meshWithBoundaries(x, y, tris, lines, labels, ax)
        else:
            plotmesh2d.meshWithBoundaries(meshdata, ax)
    else:
        if meshdataismesh:
            x, y, z, tets = meshdata.points[:,0], meshdata.points[:,1], meshdata.points[:,2], meshdata.simplices
            faces, bdrylabels = meshdata.faces, meshdata.bdrylabels
            plotmesh3d.meshWithBoundaries(x, y, z, tets, faces, bdrylabels, ax)
        else:
            plotmesh3d.meshWithBoundaries(meshdata, ax)
    if ax==plt: plt.show()

#=================================================================#
def meshWithData(meshdata, point_data=None, cell_data=None, numbering=False, title=None, suptitle=None, addplots=[]):
    dim, meshdataismesh = _getDim(meshdata)
    """
    meshdata    : either mesh or coordinates and connectivity
    point_data  : dictionary name->data
    cell_data  : dictionary name->data
    """
    if dim==2:
        if meshdataismesh:
            x, y, tris, xc, yc = meshdata.points[:,0], meshdata.points[:,1], meshdata.simplices, meshdata.pointsc[:,0], meshdata.pointsc[:,1]
            return plotmesh2d.meshWithData(x, y, tris, xc, yc, point_data, cell_data, title=title, suptitle=suptitle,addplots=addplots)
        else:
            return plotmesh2d.meshWithData(meshdata, point_data, cell_data, title=title, suptitle=suptitle,addplots=addplots)
    else:
        if meshdataismesh:
            x, y, z, tets = meshdata.points[:,0], meshdata.points[:,1], meshdata.points[:,2], meshdata.simplices
            xc, yc, zc = meshdata.pointsc[:,0], meshdata.pointsc[:,1], meshdata.pointsc[:,2]
            return plotmesh3d.meshWithData(x, y, z, tets, xc, yc, zc, point_data, cell_data, title=title, suptitle=suptitle,addplots=addplots)
        else:
            return plotmesh3d.meshWithData(meshdata, point_data, cell_data, title=title, suptitle=suptitle,addplots=addplots)


#=================================================================#
def plotmesh(meshdata, **kwargs):
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
    mesh = simplexmesh.SimplexMesh(geomname="unitsquare", hmean=1)
    plotmesh(mesh, localnumbering=True)
