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

#=================================================================#
def meshWithNodesAndTriangles(meshdata, ax=plt):
    dim, meshdataismesh = _getDim(meshdata)
    if dim==2:
        if meshdataismesh:
            x, y, tris, xc, yc = meshdata.points[:,0], meshdata.points[:,1], meshdata.simplices, meshdata.pointsc[:,0], meshdata.pointsc[:,1]
            plotmesh2d.meshWithNodesAndTriangles(x, y, tris, xc, yc, ax)
        else:
            plotmesh2d.meshWithNodesAndTriangles(meshdata, ax)
    else:
        msg = "Dimension is {} but plot is not written".format(dim)
        raise ValueError(msg)

#=================================================================#
def meshWithNodesAndFaces(meshdata, ax=plt):
    dim, meshdataismesh = _getDim(meshdata)
    if dim==2:
        if meshdataismesh:
            x, y, tris, faces = meshdata.points[:,0], meshdata.points[:,1], meshdata.simplices, meshdata.faces
            try:
                xf, yf = meshdata.pointsf[:,0], meshdata.pointsf[:,1]
            except:
                pointsf = meshdata.points[faces].mean(axis=1)
            plotmesh2d.meshWithNodesAndFaces(x, y, tris, pointsf[:,0], pointsf[:,1], faces, ax)
        else:
            plotmesh2d.meshWithNodesAndFaces(meshdata, ax)
    else:
        msg = "Dimension is {} but plot is not written".format(dim)
        raise ValueError(msg)

#=================================================================#
def meshWithData(meshdata, point_data, cell_data, ax=plt, numbering=False):
    dim, meshdataismesh = _getDim(meshdata)
    if dim==2:
        if meshdataismesh:
            x, y, tris, xc, yc = meshdata.points[:,0], meshdata.points[:,1], meshdata.simplices, meshdata.pointsc[:,0], meshdata.pointsc[:,1]
            plotmesh2d.meshWithData(x, y, tris, xc, yc, point_data, cell_data, ax)
        else:
            plotmesh2d.meshWithData(meshdata, point_data, cell_data, ax)
    else:
        if meshdataismesh:
            x, y, z, tets = meshdata.points[:,0], meshdata.points[:,1], meshdata.points[:,2], meshdata.simplices
            xc, yc, zc = meshdata.pointsc[:,0], meshdata.pointsc[:,1], meshdata.pointsc[:,2]
            plotmesh3d.meshWithData(x, y, z, tets, xc, yc, zc, point_data, cell_data, ax)
        else:
            plotmesh3d.meshWithData(meshdata, point_data, cell_data, ax)
