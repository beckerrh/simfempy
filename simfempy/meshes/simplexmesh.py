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

from .testmeshes import __pygmsh6__

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
            raise KeyError("Needs a mesh (no longer geometry)")
        self._initMeshPyGmsh(mesh)
        from . import plotmesh
        plotmesh.meshWithBoundaries(self)
        plotmesh.plotmeshWithNumbering(self)
    def _initMeshPyGmsh(self, mesh):
        # only 3d-coordinates
        assert mesh.points.shape[1] ==3
        self.points = mesh.points
        self.nnodes = self.points.shape[0]
        self.celltypes = [key for key, cellblock in mesh.cells]
        # for key, cellblock in cells: keys.append(key)
        print("self.celltypes", self.celltypes)
        if 'tetra' in self.celltypes:
            self.dimension = 3
            self.simplicesname, self.facesname = 'tetra', 'triangle'
        elif 'triangle' in self.celltypes:
            self.dimension = 2
            self.simplicesname, self.facesname = 'triangle', 'line'
        else:
            self.dimension = 1
            self.simplicesname, self.facesname = 'line', 'vertex'
        bdryfacesgmsh = np.empty(shape=(0,self.dimension), dtype=int)
        for key, cellblock in mesh.cells:
            print(f"{key=} {cellblock=}")
            if key == self.simplicesname: self.simplices = cellblock
            if key == 'vertex': self.vertices = cellblock
            if key == self.facesname: bdryfacesgmsh = np.append(bdryfacesgmsh, cellblock, axis=0)
        self._constructFacesFromSimplices()
        assert self.dimension+1 == self.simplices.shape[1]
        self.ncells = self.simplices.shape[0]
        self.pointsc = self.points[self.simplices].mean(axis=1)
        self.pointsf = self.points[self.faces].mean(axis=1)
        self._constructNormalsAndAreas()
        if __pygmsh6__:
            self._initMeshPyGmsh6(mesh.cells, mesh.cell_data['gmsh:physical'], bdryfacesgmsh)
        else:
            self._initMeshPyGmsh7(mesh.cells, mesh.cell_sets, bdryfacesgmsh)

    def _initMeshPyGmsh7(self, cells, cell_sets, bdryfacesgmsh):
        # cell_sets: dict label --> list of None or np.array for each cell_type
        # the indices of the np.array are not the cellids !
        # ???
        print(f"{cell_sets=}")
        typesoflabel = {}
        sizes = {key:0 for key in self.celltypes}
        cellsoflabel = {key:{} for key in self.celltypes}
        ctorderd = []
        for label, cb in cell_sets.items():
            if len(cb) != len(self.celltypes): raise KeyError(f"mismatch {label=}")
            for celltype, info in zip(self.celltypes, cb):
                # only one is supposed to be not None
                if info is not None:
                    cellsoflabel[celltype][label] = info
                    sizes[celltype] += info.shape[0]
                    typesoflabel[label] = celltype
                    ctorderd.append(celltype)
        #correcting the numbering in cell_sets
        n = 0
        for ct in list(dict.fromkeys(ctorderd)):
            #eliminates duplicates
            for l, cb in cellsoflabel[ct].items(): cb -= n
            n += sizes[ct]

        # print(f"{label=} {celltype=} {info=}")
        print(f"{cellsoflabel=}")
        self.cellsoflabel = cellsoflabel[self.simplicesname]
        self.verticesoflabel = {}
        if self.dimension > 1:
            self.verticesoflabel = cellsoflabel['vertex']
        # bdry faces
        # for key, cellblock in cells:
        #     if key == self.facesnames[self.dimension - 1]: bdryfacesgmsh = cellblock
        bdrylabelsgmsh = cellsoflabel[self.facesnames[self.dimension-1]]
        self._constructBoundaryFaces7(bdryfacesgmsh, bdrylabelsgmsh)
        # self.bdrylabels = {}
        # for color, cells in bdrylabelsgmsh.items():
        #     self.bdrylabels[int(color)] = cells
        print(f"{self.bdrylabels=}")
        # bdryids = np.flatnonzero(self.cellsOfFaces[:,1] == -1)
        # bdryfaces = np.sort(self.faces[bdryids],axis=1)
        # nbdryfaces = len(bdryids)
        # raise NotImplementedError("no idea")

    def _initMeshPyGmsh6(self, cells, cdphys, bdryfacesgmsh):
        if len(cdphys) != len(self.celltypes):
            raise KeyError(f"not enough physical labels:\n {self.celltypes=}\n {len(cdphys)=}")
        # first attempt, 'np.append' copies data...
        _cells = {}
        _labels = {}
        for (key, cellblock), cd in zip(cells,cdphys):
            if len(cellblock) != len(cd):
                raise ValueError(f"mismatch in {key} {len(cellblock)} {len(cd)}")
            if not key in _cells.keys():
                _cells[key] = cellblock
                _labels[key] = cd
            else:
                _cells[key] = np.append(_cells[key], cellblock, axis=0)
                _labels[key] = np.append(_labels[key], cd, axis=0)
        # print("_labels", _labels)
        # if self.dimension==1:
        #     # self._facedata = (_cells['vertex'], _labels['vertex'])
        #     bdrylabelsgmsh = _labels['vertex']
        #     # self.cell_labels = _labels['line']
        # elif self.dimension==2:
        #     # self._facedata = (_cells['line'], _labels['line'])
        #     bdrylabelsgmsh = _labels['line']
        #     # self.cell_labels = _labels['triangle']
        # else:
        #     # self._facedata = (_cells['triangle'], _labels['triangle'])
        #     bdrylabelsgmsh = _labels['triangle']
        #     # self.cell_labels = _labels['tetra']
        # bdryfacesgmsh, bdrylabelsgmsh = self._facedata
        # print(f"{self.facesname=}")
        # print(f"{bdryfacesgmsh=}")
        # print(f"{_cells['line']=}")
        # assert np.all(bdryfacesgmsh==_cells['line'])
        self._constructBoundaryFaces6(bdryfacesgmsh, _labels[self.facesname])
        # self.cellsoflabel = npext.creatdict_unique_all(self.cell_labels)
        self.cellsoflabel = npext.creatdict_unique_all(_labels[self.simplicesname])
        self.verticesoflabel = {}
        if self.dimension > 1 and 'vertex' in _cells.keys():
            self.vertices = _cells['vertex'].reshape(-1)
            self.verticesoflabel = npext.creatdict_unique_all(_labels['vertex'])

    def _constructFacesFromSimplices(self):
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
        if self.dimension==1:
            perm = np.argsort(allfaces, axis=0).ravel()
        else:
            perm = np.argsort(allfaces.view(s), order=order, axis=0).ravel()
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
    def _constructBoundaryFaces7(self, bdryfacesgmsh, bdrylabelsgmsh):
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
        nnpc = self.simplices.shape[1]
        s = "{0}" + (nnpc-2)*", {0}"
        dtb = s.format(bdryfacesgmsh.dtype)
        dtf = s.format(bdryfaces.dtype)
        order = ["f0"]+["f{:1d}".format(i) for i in range(1,nnpc-1)]
        if self.dimension==1:
            bp = np.argsort(bdryfacesgmsh.view(dtb), axis=0).ravel()
            fp = np.argsort(bdryfaces.view(dtf), axis=0).ravel()
        else:
            bp = np.argsort(bdryfacesgmsh.view(dtb), order=order, axis=0).ravel()
            fp = np.argsort(bdryfaces.view(dtf), order=order, axis=0).ravel()
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
    def _constructBoundaryFaces6(self, bdryfacesgmsh, bdrylabelsgmsh):
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
        nnpc = self.simplices.shape[1]
        s = "{0}" + (nnpc-2)*", {0}"
        dtb = s.format(bdryfacesgmsh.dtype)
        dtf = s.format(bdryfaces.dtype)
        order = ["f0"]+["f{:1d}".format(i) for i in range(1,nnpc-1)]
        if self.dimension==1:
            bp = np.argsort(bdryfacesgmsh.view(dtb), axis=0).ravel()
            fp = np.argsort(bdryfaces.view(dtf), axis=0).ravel()
        else:
            bp = np.argsort(bdryfacesgmsh.view(dtb), order=order, axis=0).ravel()
            fp = np.argsort(bdryfaces.view(dtf), order=order, axis=0).ravel()
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
        elem = self.simplices
        self.sigma = np.array([2 * (self.cellsOfFaces[self.facesOfCells[ic, :], 0] == ic)-1 for ic in range(self.ncells)])
        if self.dimension==1:
            x = self.points[:,0]
            self.normals = np.stack((np.ones(self.nfaces), np.zeros(self.nfaces), np.zeros(self.nfaces)), axis=-1)
            dx1 = x[elem[:, 1]] - x[elem[:, 0]]
            self.dV = np.abs(dx1)
        elif self.dimension==2:
            x,y = self.points[:,0], self.points[:,1]
            sidesx = x[self.faces[:, 1]] - x[self.faces[:, 0]]
            sidesy = y[self.faces[:, 1]] - y[self.faces[:, 0]]
            self.normals = np.stack((-sidesy, sidesx, np.zeros(self.nfaces)), axis=-1)
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
