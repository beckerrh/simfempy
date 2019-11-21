# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import os, sys, importlib
import meshio
import numpy as np
from scipy import sparse
from simfempy.tools import npext


# try:
#     import geomdefs
# except ModuleNotFoundError:
#     from . import geomdefs


#=================================================================#
class SimplexMesh(object):
    """
    simplicial mesh, can be initialized from the output of pygmsh.
    Needs physical labels geometry objects of highest dimension and co-dimension one

    dimension, nnodes, ncells, nfaces: dimension, number of nodes, simplices, faces
    points: coordinates of the vertices of shape (nnodes,3)
    pointsc: coordinates of the barycenters of cells (ncells,3)
    pointsf: coordinates of the barycenters of faces (nfaces,3)

    simplices: node ids of simplices of shape (ncells, dimension+1)
    faces: node ids of faces of shape (nfaces, dimension)

    facesOfCells: shape (ncells, dimension+1): contains simplices[i,:]-setminus simplices[i,ii], sorted
    cellsOfFaces: shape (nfaces, dimension): cellsOfFaces[i,1]=-1 if boundary

    normals: normal per face of length dS, oriented from  ids of faces of shape (nfaces, dimension)
             normals on boundary are external
    sigma: orientation of normal per cell and face (ncells, dimension+1)

    dV: shape (ncells), volumes of simplices
    bdrylabels: dictionary(keys: colors, values: id's of boundary faces)
    """

    def __repr__(self):
        return "SimplexMesh({}): dim/nnodes/ncells/nfaces: {}/{}/{}/{} bdrylabels={}".format(self.geometry, self.dimension, self.nnodes, self.ncells, self.nfaces, list(self.bdrylabels.keys()))
    def __init__(self, **kwargs):
        if 'mesh' in kwargs:
            self.geometry = 'own'
            mesh = kwargs.pop('mesh')
        else:
            import pygmsh
            self.geometry = kwargs.pop('geometry')
            if 'hmean' in kwargs: hmean = kwargs.pop('hmean')
            else: hmean = 1
            self.geometry.define(hmean)
            # code = self.geometry.get_code()
            # with open("toto.geo",'w') as file:
            #     file.write(code)
            mesh = pygmsh.generate_mesh(self.geometry, verbose=False)
        self._initMeshPyGmsh(mesh.points, mesh.cells, mesh.cell_data)

    def _checkPhyscalLabels(self, points, cells, celldata):
        if 'vertex' in cells.keys():
            if not 'vertex' in celldata.keys() or not 'gmsh:physical' in celldata['vertex'].keys():
                raise KeyError(f"vertices given without physical_label")
        if self.dimension == 2:
            if not 'line' in celldata.keys() or not 'gmsh:physical' in celldata['line'].keys():
                raise KeyError(f"lines given without physical_label")
            if not 'triangle' in celldata.keys() or not 'gmsh:physical' in celldata['triangle'].keys():
                raise KeyError(f"triangles given without physical_label")
        else:
            if not 'triangle' in celldata.keys() or not 'gmsh:physical' in celldata['triangle'].keys():
                raise KeyError(f"triangles given without physical_label")
            if not 'tetra' in celldata.keys() or not 'gmsh:physical' in celldata['tetra'].keys():
                raise KeyError(f"tetras given without physical_label")

    def _initMeshPyGmsh(self, points, cells, celldata):
        if 'tetra' in cells.keys():
            self.dimension = 3
        elif 'triangle' in cells.keys():
            self.dimension = 2
        else:
            self.dimension = 1
        assert points.shape[1] ==3
        self._checkPhyscalLabels(points, cells, celldata)
        self.points = points
        self.nnodes = self.points.shape[0]
        if 'vertex' in cells.keys():
            self.vertices = cells['vertex'].reshape(-1)
            self.vertex_labels = celldata['vertex']['gmsh:physical']
            verticesoflabel = npext.unique_all(self.vertex_labels)
            self.verticesoflabel={}
            for color, ind in zip(verticesoflabel[0], verticesoflabel[1]):
                self.verticesoflabel[color] = self.vertices[ind]
        if self.dimension==2:
            self.simplices = cells['triangle']
            self._facedata = (cells['line'], celldata['line']['gmsh:physical'])
            self._constructFacesFromSimplices(cells['line'], celldata['line']['gmsh:physical'])
            self.cell_labels = celldata['triangle']['gmsh:physical']
        else:
            self.simplices = cells['tetra']
            self._facedata = (cells['triangle'], celldata['triangle']['gmsh:physical'])
            self._constructFacesFromSimplices(cells['triangle'], celldata['triangle']['gmsh:physical'])
            self.cell_labels = celldata['tetra']['gmsh:physical']
        cellloflabel = npext.unique_all(self.cell_labels)
        self.cellsoflabel = {}
        for color, ind in zip(cellloflabel[0], cellloflabel[1]):
            self.cellsoflabel[color] = ind
        # for color, ind in zip(cellloflabel[0], cellloflabel[1]):
        #     print("color", color)
        #     print("split", cell_labels[ind])

        assert self.dimension+1 == self.simplices.shape[1]
        self.ncells = self.simplices.shape[0]
        self.pointsc = self.points[self.simplices].mean(axis=1)
        self.pointsf = self.points[self.faces].mean(axis=1)
        self._constructNormalsAndAreas()
        # print(self)

    def _constructFacesFromSimplices(self, bdryfacesgmsh, bdrylabelsgmsh):
        simplices = self.simplices
        ncells = simplices.shape[0]
        nnpc = simplices.shape[1]
        allfaces = np.empty(shape=(nnpc*ncells,nnpc-1), dtype=int)
        for i in range(ncells):
            for ii in range(nnpc):
                mask = np.array( [jj !=ii for jj in range(nnpc)] )
                allfaces[i*nnpc+ii] = np.sort(simplices[i,mask])
        s = "{0}" + (nnpc-2)*", {0}"
        s = s.format(allfaces.dtype)
        order = ["f0"]+["f{:1d}".format(i) for i in range(1,nnpc-1)]
        perm = np.argsort(allfaces.view(s), order=order, axis=0).flatten()
        allfacescorted = allfaces[perm]
        self.faces, indices = np.unique(allfacescorted, return_inverse=True, axis=0)
        locindex = np.tile(np.arange(0,nnpc), ncells)
        cellindex = np.repeat(np.arange(0,ncells), nnpc)
        self.nfaces = self.faces.shape[0]
        self.cellsOfFaces = -1 * np.ones(shape=(self.nfaces, 2), dtype=int)
        self.facesOfCells = np.zeros(shape=(ncells, nnpc), dtype=int)
        for ii in range(indices.shape[0]):
            f = indices[ii]
            loc = locindex[perm[ii]]
            cell = cellindex[perm[ii]]
            self.facesOfCells[cell, loc] = f
            if self.cellsOfFaces[f,0] == -1: self.cellsOfFaces[f,0] = cell
            else: self.cellsOfFaces[f,1] = cell
        self._constructBoundaryLabels(bdryfacesgmsh, bdrylabelsgmsh)
    # def _constructCellLabels(self, simpgmsh, labelsgmsh):
    #     if self.simplices.shape != simpgmsh.shape:
    #         msg ="wrong shapes self.simplices={} simpgmsh={}".format(self.simplices.shape,simpgmsh.shape)
    #         raise ValueError(msg)
    #     simpsorted = np.sort(self.simplices, axis=1)
    #     labelssorted = np.sort(simpgmsh, axis=1)
    #     assert simpsorted.shape == labelssorted.shape
    #     if self.dimension==2:
    #         dts = "{0}, {0}, {0}".format(simpsorted.dtype)
    #         dtl = "{0}, {0}, {0}".format(labelssorted.dtype)
    #         sp = np.argsort(simpsorted.view(dts), order=('f0','f1','f2'), axis=0).flatten()
    #         lp = np.argsort(labelssorted.view(dtl), order=('f0','f1','f2'), axis=0).flatten()
    #     else:
    #         dts = "{0}, {0}, {0}, {0}".format(simpsorted.dtype)
    #         dtl = "{0}, {0}, {0}, {0}".format(labelssorted.dtype)
    #         sp = np.argsort(simpsorted.view(dts), order=('f0','f1','f2','f3'), axis=0).flatten()
    #         lp = np.argsort(labelssorted.view(dtl), order=('f0','f1','f2','f3'), axis=0).flatten()
    #     spi = np.empty(sp.size, sp.dtype)
    #     spi[sp] = np.arange(sp.size)
    #     perm = lp[spi]
    #     self.cell_labels = labelsgmsh[perm]

    def _constructFaces(self, bdryfacesgmsh, bdrylabelsgmsh):
        simps, neighbrs = self.delaunay.simplices, self.delaunay.neighbors
        count=0
        for i in range(len(simps)):
            for idim in range(self.dimension+1):
                if i > neighbrs[i, idim]: count +=1
        self.nfaces = count
        self.faces = np.empty(shape=(self.nfaces,self.dimension), dtype=int)
        self.cellsOfFaces = -1 * np.ones(shape=(self.nfaces, 2), dtype=int)
        self.facesOfCells = np.zeros(shape=(self.ncells, self.dimension+1), dtype=int)
        count=0
        for i in range(len(simps)):
            for idim in range(self.dimension+1):
                j = neighbrs[i, idim]
                if i<j: continue
                mask = np.array( [ii !=idim for ii in range(self.dimension+1)] )
                self.faces[count] = np.sort(simps[i,mask])
                self.facesOfCells[i, idim] = count
                self.cellsOfFaces[count, 0] = i
                if j > -1:
                    for jdim in range(self.dimension+1):
                        if neighbrs[j, jdim] == i:
                            self.facesOfCells[j, jdim] = count
                            self.cellsOfFaces[count, 1] = j
                            break
                count +=1
        # for i in range(len(simps)):
        #     print("self.facesOfCells {} {}".format(i,self.facesOfCells[i]))
        # for i in range(self.nfaces):
        #     print("self.cellsOfFaces {} {}".format(i,self.cellsOfFaces[i]))
        # bdries
        self._constructBoundaryLabels(bdryfacesgmsh, bdrylabelsgmsh)

    def _constructBoundaryLabels(self, bdryfacesgmsh, bdrylabelsgmsh):
        # bdries
        bdryids = np.flatnonzero(self.cellsOfFaces[:,1] == -1)
        assert np.all(bdryids == np.flatnonzero(np.any(self.cellsOfFaces == -1, axis=1)))
        bdryfaces = np.sort(self.faces[bdryids],axis=1)
        nbdryfaces = len(bdryids)
        if len(bdrylabelsgmsh) != nbdryfaces:
            raise ValueError("wrong number of boundary labels {} != {}".format(len(bdrylabelsgmsh),nbdryfaces))
        if len(bdryfacesgmsh) != nbdryfaces:
            raise ValueError("wrong number of bdryfaces {} != {}".format(len(bdryfacesgmsh), nbdryfaces))
        self.bdrylabels = {}
        colors, counts = np.unique(bdrylabelsgmsh, return_counts=True)
        # print ("colors, counts", colors, counts)
        for i in range(len(colors)):
            self.bdrylabels[colors[i]] = -np.ones( (counts[i]), dtype=np.int32)
        bdryfacesgmsh = np.sort(bdryfacesgmsh)
        if self.dimension==2:
            dtb = "{0}, {0}".format(bdryfacesgmsh.dtype)
            dtf = "{0}, {0}".format(bdryfaces.dtype)
            bp = np.argsort(bdryfacesgmsh.view(dtb), order=('f0','f1'), axis=0).flatten()
            fp = np.argsort(bdryfaces.view(dtf), order=('f0','f1'), axis=0).flatten()
        else:
            dtb = "{0}, {0}, {0}".format(bdryfacesgmsh.dtype)
            dtf = "{0}, {0}, {0}".format(bdryfaces.dtype)
            bp = np.argsort(bdryfacesgmsh.view(dtb), order=('f0','f1','f2'), axis=0).flatten()
            fp = np.argsort(bdryfaces.view(dtf), order=('f0','f1','f2'), axis=0).flatten()
        bpi = np.empty(bp.size, bp.dtype)
        bpi[bp] = np.arange(bp.size)
        perm = bdryids[fp[bpi]]
        counts = {}
        for key in list(self.bdrylabels.keys()): counts[key]=0
        for i in range(len(perm)):
            if np.any(bdryfacesgmsh[i] != self.faces[perm[i]]):
                raise ValueError("Did not find boundary indices")
            color = bdrylabelsgmsh[i]
            self.bdrylabels[color][counts[color]] = perm[i]
            counts[color] += 1
        # print ("self.bdrylabels", self.bdrylabels)

    def _constructNormalsAndAreas(self):
        if self.dimension==2:
            x,y = self.points[:,0], self.points[:,1]
            sidesx = x[self.faces[:, 1]] - x[self.faces[:, 0]]
            sidesy = y[self.faces[:, 1]] - y[self.faces[:, 0]]
            self.normals = np.stack((-sidesy, sidesx, np.zeros(self.nfaces)), axis=-1)
            elem = self.simplices
            dx1 = x[elem[:, 1]] - x[elem[:, 0]]
            dx2 = x[elem[:, 2]] - x[elem[:, 0]]
            dy1 = y[elem[:, 1]] - y[elem[:, 0]]
            dy2 = y[elem[:, 2]] - y[elem[:, 0]]
            self.dV = 0.5 * np.abs(dx1*dy2-dx2*dy1)
        else:
            x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
            x1 = x[self.faces[:, 1]] - x[self.faces[:, 0]]
            y1 = y[self.faces[:, 1]] - y[self.faces[:, 0]]
            z1 = z[self.faces[:, 1]] - z[self.faces[:, 0]]
            x2 = x[self.faces[:, 2]] - x[self.faces[:, 0]]
            y2 = y[self.faces[:, 2]] - y[self.faces[:, 0]]
            z2 = z[self.faces[:, 2]] - z[self.faces[:, 0]]
            sidesx = y1*z2 - y2*z1
            sidesy = x2*z1 - x1*z2
            sidesz = x1*y2 - x2*y1
            self.normals = 0.5*np.stack((sidesx, sidesy, sidesz), axis=-1)
            elem = self.simplices
            dx1 = x[elem[:, 1]] - x[elem[:, 0]]
            dx2 = x[elem[:, 2]] - x[elem[:, 0]]
            dx3 = x[elem[:, 3]] - x[elem[:, 0]]
            dy1 = y[elem[:, 1]] - y[elem[:, 0]]
            dy2 = y[elem[:, 2]] - y[elem[:, 0]]
            dy3 = y[elem[:, 3]] - y[elem[:, 0]]
            dz1 = z[elem[:, 1]] - z[elem[:, 0]]
            dz2 = z[elem[:, 2]] - z[elem[:, 0]]
            dz3 = z[elem[:, 3]] - z[elem[:, 0]]
            self.dV = (1/6) * np.abs(dx1*(dy2*dz3-dy3*dz2) - dx2*(dy1*dz3-dy3*dz1) + dx3*(dy1*dz2-dy2*dz1))
        for i in range(self.nfaces):
            i0, i1 = self.cellsOfFaces[i, 0], self.cellsOfFaces[i, 1]
            if i1 == -1:
                xt = np.mean(self.points[self.faces[i]], axis=0) - np.mean(self.points[self.simplices[i0]], axis=0)
                if np.dot(self.normals[i], xt)<0:  self.normals[i] *= -1
            else:
                xt = np.mean(self.points[self.simplices[i1]], axis=0) - np.mean(self.points[self.simplices[i0]], axis=0)
                if np.dot(self.normals[i], xt) < 0:  self.normals[i] *= -1
        # self.sigma = np.array([1.0 - 2.0 * (self.cellsOfFaces[self.facesOfCells[ic, :], 0] == ic) for ic in range(self.ncells)])
        self.sigma = np.array([2 * (self.cellsOfFaces[self.facesOfCells[ic, :], 0] == ic)-1 for ic in range(self.ncells)])

    def write(self, filename, dirname = None, point_data=None):
        cell_data_meshio = {}

        if hasattr(self,'vertex_labels'):
            cell_data_meshio['vertex'] = {}
            cell_data_meshio['vertex']['gmsh:physical'] = self.vertex_labels


        if self.dimension ==2:
            cells = {'triangle': self.simplices}
            cells['line'] = self._facedata[0]
            cell_data_meshio['line']={}
            cell_data_meshio['line']['gmsh:physical'] = self._facedata[1]
            cell_data_meshio['triangle']={}
            cell_data_meshio['triangle']['gmsh:physical'] = self.cell_labels
        else:
            cells = {'tetra': self.simplices}
            cells['triangle'] = self._facedata[0]
            cell_data_meshio['triangle']={}
            cell_data_meshio['triangle']['gmsh:physical'] = self._facedata[1]
            cell_data_meshio['tetra']={}
            cell_data_meshio['tetra']['gmsh:physical'] = self.cell_labels
        if dirname is not None:
            dirname = dirname + os.sep + "mesh"
            if not os.path.isdir(dirname) :
                os.makedirs(dirname)
            filename = os.path.join(dirname, filename)
        print("cell_data_meshio['line']['gmsh:physical']", cell_data_meshio['line']['gmsh:physical'])
        meshio.write_points_cells(filename=filename, points=self.points, cells=cells, point_data=point_data, cell_data=cell_data_meshio, file_format='gmsh2-ascii')

    def computeSimpOfVert(self, test=False):
        S = sparse.dok_matrix((self.nnodes, self.ncells), dtype=int)
        for ic in range(self.ncells):
            S[self.simplices[ic,:], ic] = ic+1
        S = S.tocsr()
        S.data -= 1
        self.simpOfVert = S
        if test:
            # print("S=",S)
            from . import plotmesh
            import matplotlib.pyplot as plt
            simps, xc, yc = self.simplices, self.pointsc[:,0], self.pointsc[:,1]
            meshdata =  self.x, self.y, simps, xc, yc
            plotmesh.meshWithNodesAndTriangles(meshdata)
            plt.show()

    def plot(self, **kwargs):
        from simfempy.meshes import plotmesh
        plotmesh.plotmesh(self, **kwargs)
    def plotWithBoundaries(self):
        # from . import plotmesh
        from simfempy.meshes import plotmesh
        plotmesh.meshWithBoundaries(self)
    def plotWithNumbering(self, **kwargs):
        # from . import plotmesh
        from simfempy.meshes import plotmesh
        plotmesh.plotmeshWithNumbering(self, **kwargs)
    def plotWithData(self, **kwargs):
        # from . import plotmesh
        from simfempy.meshes import plotmesh
        plotmesh.meshWithData(self, **kwargs)


#=================================================================#
if __name__ == '__main__':
    import geomdefs
    geometry = geomdefs.unitsquare.Unitsquare()
    mesh = SimplexMesh(geometry=geometry, hmean=2)
    import plotmesh
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, 1, sharex='col')
    plotmesh.meshWithBoundaries(mesh, ax=axarr[0])
    plotmesh.plotmeshWithNumbering(mesh, ax=axarr[1])
    plotmesh.plotmeshWithNumbering(mesh, localnumbering=True)
