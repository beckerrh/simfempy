# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import os
import meshio
import numpy as np
from scipy import sparse

#=================================================================#
class SimplexMesh(object):
    """
    simplicial mesh, can be initialized from the output of pygmsh.
    Needs physical labels geometry objects of highest dimension and co-dimension one

    dimension, nnodes, ncells, nfaces: dimension, number of nodes, simplices, faces
    points: coordinates of the vertices of shape (nnodes,3)
    pointsc: coordinates of the barycenters of cells (ncells,3)
    pointsf: coordinates of the barycenters of faces (nfaces,3)

    cells: dictionary of all given cells
    cells_phys: dictionary of all given cells with physical labels
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
        msg = f"SimplexMesh({self.name}): dim({self.dimension})/nnodes({self.nnodes})/ncells({self.ncells})/nfaces({self.nfaces})"
        msg += f" Physical labels:\n"
        for k,v in self.cells_phys.items():
            msg += f"\t{k} --> {v.keys()}\n"
        return msg
    def __init__(self, mesh=None, name='no_name'):
        if mesh is None:
            raise KeyError("Can only work with mesh (at the moment)")
        self.mesh = mesh
        self.name = name
        # points
        self.points = mesh.points
        assert self.points.shape[1] ==3
        self.nnodes = self.points.shape[0]
        # Get all relevant data : 1) cells
        cellkeys = [m[0] for m in mesh.cells]
        self.cells = {}
        for key, cellblock in mesh.cells:
            if key in self.cells.keys():
                raise ValueError(f"Multiple key '{key}' given in cells")
            self.cells[key] = cellblock
        # Get all relevant data : 2) physical
        def errmsg(k):
            msg = "Physical label must be of the form\n"
            msg += "\t'X:id' with X=N|L|S|V meaning(node|line|surface|volume\n"
            msg += f"\tgiven label: '{k}'"
            return msg
        def errmsg2(k):
            return f"Multiple key '{k}' given in cells"
        ct = {'N':'vertex','L':'line','S':'triangle','V':'tetra'}
        self.cells_phys = {'vertex':{}, 'line':{}, 'triangle':{}, 'tetra':{}}
        for k,v in mesh.point_sets.items():
            ks = k.split(":")
            if len(ks) != 2 or ks[0] != ct.keys()[0]: raise ValueError(errmsg(k))
            if ks[1] in self.cells_phys['vertex'].keys(): raise ValueError(errmsg2(ks[1]))
            self.cells_phys['vertex'][ks[1]] = v[0]
        for k,v in mesh.cell_sets.items():
            ks = k.split(":")
            if len(ks) != 2 or ks[0] not in ct.keys(): raise ValueError(errmsg(k))
            if ks[1] in self.cells_phys[ct[ks[0]]].keys(): raise ValueError(errmsg2(ks[1]))
            # print(f"{ks=} {v=}", len(v))
            for i,vi in enumerate(v)    :
                if isinstance(vi, np.ndarray): break
            self.cells_phys[ct[ks[0]]][ks[1]] = v[i]
        # numbering is with respect to all cells, so we hav eto correct it
        n, cklast = 0, cellkeys[0]
        for ck in cellkeys[1:]:
            # print(f"{cklast=} {self.cells[cklast].shape=}")
            n += self.cells[cklast].shape[0]
            cklast = ck
            # print(f"{ck=}")
            for k,v in self.cells_phys[ck].items():
                v -= n
        # for k,v in self.cells.items():
        #     print(f"Cells {k=} {v=}")
        # for k,v in self.cells_phys.items():
        #     print(f"Cells Phys {k=} {v=}")
        # set dimension
        self.dimension = 1
        if 'tetra' in cellkeys: self.dimension = 3
        elif 'triangle' in cellkeys: self.dimension = 2
        # set simplices
        if self.dimension==1: simpkey, facekey = 'line', 'vertex'
        elif self.dimension==2: simpkey, facekey = 'triangle', 'line'
        else: simpkey, facekey = 'tetra', 'triangle'
        self.simplices = self.cells[simpkey]
        self.ncells = self.simplices.shape[0]
        self.pointsc = self.points[self.simplices].mean(axis=1)
        # set cell_labels
        self.cell_labels = np.zeros(self.ncells)
        for k,v in self.cells_phys[simpkey].items():
            self.cell_labels[v] = k
        # compute faces
        self._constructFacesFromSimplices()
        self.pointsf = self.points[self.faces].mean(axis=1)
        # compute boundary faces
        # set faces_labels
        bdryids = np.flatnonzero(self.cellsOfFaces[:,1] == -1)
        faces_labels = np.zeros(len(bdryids))
        for k,v in self.cells_phys[facekey].items():
            faces_labels[v] = k
        self._constructBoundaryLabels(bdryids, self.cells[facekey], faces_labels)
        # compute normals
        self._constructNormalsAndAreas()

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

    def _constructBoundaryLabels(self, bdryids, bdryfacesgmsh, bdrylabelsgmsh):
        print(f"{self.mesh.cell_sets=}")
        # print(f"{bdryfacesgmsh=}")
        # print(f"{bdrylabelsgmsh=}")
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
        from simfempy_pygmsh7.meshes import plotmesh
        plotmesh.plotmesh(self, **kwargs)
    def plotWithBoundaries(self):
        # from . import plotmesh
        from simfempy_pygmsh7.meshes import plotmesh
        plotmesh.meshWithBoundaries(self)
    def plotWithNumbering(self, **kwargs):
        # from . import plotmesh
        from simfempy_pygmsh7.meshes import plotmesh
        plotmesh.plotmeshWithNumbering(self, **kwargs)
    def plotWithData(self, **kwargs):
        # from . import plotmesh
        from simfempy_pygmsh7.meshes import plotmesh
        plotmesh.meshWithData(self, **kwargs)


#=================================================================#
if __name__ == '__main__':
    import testmesh
    import plotmesh
    import matplotlib.pyplot as plt
    m = testmesh.mesh2d(mesh_size=1.5)
    mesh = SimplexMesh(mesh=m)
    # plotmesh.plotmesh(mesh)
    fig, axarr = plt.subplots(2, 1, sharex='col')
    plotmesh.meshWithBoundaries(mesh, ax=axarr[0])
    plotmesh.plotmeshWithNumbering(mesh, ax=axarr[1])
    plt.show()
    # plotmesh.plotmeshWithNumbering(mesh, localnumbering=True)
