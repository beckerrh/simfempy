# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import os, sys, importlib
import meshio
import numpy as np
from numpy.lib.shape_base import take_along_axis
from scipy import sparse
from simfempy.tools import npext, timer

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
    cellsOfFaces: shape (nfaces, 2): cellsOfFaces[i,1]=-1 if boundary

    normals: normal per face of length dS, oriented from  ids of faces of shape (nfaces, dimension)
             normals on boundary are external
    sigma: orientation of normal per cell and face (ncells, dimension+1)

    innerfaces: mask for interior faces
    cellsOfInteriorFaces: cellsOfFaces[innerfaces]

    dV: shape (ncells), volumes of simplices
    bdrylabels: dictionary(keys: colors, values: id's of boundary faces)
    cellsoflabel: dictionary(keys: colors, values: id's of cells)
    """

    def __repr__(self):
        s = f"SimplexMesh({self.geometry}): "
        s += f"dim/nnodes/nfaces/ncells: {self.dimension}/{self.nnodes}/{self.nfaces}/{self.ncells}"
        s += f"\nbdrylabels={list(self.bdrylabels.keys())}"
        s += f"\ncellsoflabel={list(self.cellsoflabel.keys())}"
        return s
    def __init__(self, **kwargs):
        if 'mesh' in kwargs:
            self.geometry = 'own'
            mesh = kwargs.pop('mesh')
        else:
            raise KeyError("Needs a mesh (no longer geometry)")
        self.timer = timer.Timer(name="SimplexMesh")
        self._initMeshPyGmsh(mesh)
        self.check()
        # print(self.timer)
    def check(self):
        if len(np.unique(self.simplices)) != self.nnodes:
            raise ValueError(f"{len(np.unique(self.simplices))=} BUT {self.nnodes=}")
    def bdryFaces(self, colors=None):
        if colors is None: colors = self.bdrylabels.keys()
        pos = [0]
        for color in colors: pos.append(pos[-1]+len(self.bdrylabels[color]))
        faces = np.empty(pos[-1], dtype=np.uint32)
        for i,color in enumerate(colors): faces[pos[i]:pos[i+1]] = self.bdrylabels[color]
        return faces

    def _initMeshPyGmsh(self, mesh):
        self.pygmsh = mesh
        self.celltypes = [key for key, cellblock in mesh.cells]
        assert self.celltypes==list(mesh.cells_dict.keys())
        # for key, cellblock in cells: keys.append(key)
        # print("self.celltypes", self.celltypes)
        if 'tetra' in self.celltypes:
            self.dimension = 3
            self.simplicesname, self.facesname = 'tetra', 'triangle'
        elif 'triangle' in self.celltypes:
            self.dimension = 2
            self.simplicesname, self.facesname = 'triangle', 'line'
        elif 'line' in self.celltypes:
            self.dimension = 1
            self.simplicesname, self.facesname = 'line', 'vertex'
        else:
            raise ValueError(f"something wrong {self.celltypes=} {mesh=}")
        # bdryfacesgmshlist = []
        for key, cellblock in mesh.cells:
            # print(f"{key=} {cellblock=}")
            if key == self.simplicesname: self.simplices = cellblock
            elif key == self.facesname:
                self.facesdata = cellblock
                # print(f"{key=} {cellblock=}")
                # bdryfacesgmshlist.extend(cellblock)
            # elif key == 'vertex': self.vertices = cellblock
            # elif key == 'line': self.lines = cellblock
            else:
                continue
                # raise ValueError(f"unknown cellblock {cellblock=}")
        if not hasattr(self,"simplices"):
            raise ValueError(f"something wrong {self.dimension=}")
        assert np.all(self.facesdata==mesh.cells_dict[self.facesname])
        assert np.all(self.simplices==mesh.cells_dict[self.simplicesname])
        # only 3d-coordinates
        assert mesh.points.shape[1] ==3
        # eliminate drangling points
        nnp = len(np.unique(self.simplices))
        if not np.all(np.unique(self.simplices)==np.arange(nnp)):
            msg = f"points in simplices {nnp} but {self.points.shape=}"
            msg += f"\n{self.celltypes=}\n{mesh.cell_sets=}"
            msg += f"\n{np.unique(self.simplices)=}"
            msg += f"\n{mesh.cells_dict=}"
            raise ValueError(msg)
        self.points = mesh.points[:nnp]
        self.nnodes = self.points.shape[0]
        self._constructFacesFromSimplices()
        self.timer.add("_constructFacesFromSimplices")
        assert self.dimension+1 == self.simplices.shape[1]
        self.ncells = self.simplices.shape[0]
        self.pointsc = self.points[self.simplices].mean(axis=1)
        self.pointsf = self.points[self.faces].mean(axis=1)
        self._constructNormalsAndAreas()
        self.timer.add("_constructNormalsAndAreas")
        # self.cell_sets = mesh.cell_sets
        bdrylabelsgmsh = self._initMeshPyGmsh7(mesh.cell_sets, mesh.cells_dict)
        self.timer.add("_initMeshPyGmsh7")
        # boundaries
        # self._constructBoundaryFaces7(bdryfacesgmshlist, bdrylabelsgmsh)
        self._constructBoundaryFaces7(self.facesdata, bdrylabelsgmsh)
        self.timer.add("_constructBoundaryFaces7")
        # print(f"{self.bdrylabels.keys()=}")
        #TODO : remplacer -1 par nan dans les indices
    def constructInnerFaces(self):
        self.innerfaces = self.cellsOfFaces[:,1]>=0
        self.cellsOfInteriorFaces= self.cellsOfFaces[self.innerfaces]
    def _initMeshPyGmsh7(self, cell_sets, cells_dict):
        # cell_sets: dict label --> list of None or np.array for each cell_type
        # the indices of the np.array are not the cellids !
        # ???
        # print(f"{cell_sets=}")
        typesoflabel = {}
        sizes = {key:0 for key in self.celltypes}
        cellsoflabel = {key:{} for key in self.celltypes}
        ctorderd = []
        for label, cb in cell_sets.items():
            if label=='gmsh:bounding_entities': continue
            # print(f"{label=} {cb=}")
            if len(cb) != len(self.celltypes): raise KeyError(f"mismatch {label=}")
            for celltype, info in zip(self.celltypes, cb):
                # only one is supposed to be not None
                if info is not None:
                    try: ilabel=int(label)
                    except: raise ValueError(f"cannot convert to int {label=} {cell_sets=}")
                    cellsoflabel[celltype][ilabel] = info
                    # print(f"{label=} {type(label)=} {info=}")
                    sizes[celltype] += info.shape[0]
                    typesoflabel[ilabel] = celltype
                    ctorderd.append(celltype)
        # print(f"{self.celltypes=}\n{cellsoflabel=}")
        #correcting the numbering in cell_sets
        n = 0
        for ct in list(dict.fromkeys(ctorderd)):
            #eliminates duplicates
            for l, cb in cellsoflabel[ct].items(): cb -= n
            n += sizes[ct]
        self.cellsoflabel = cellsoflabel[self.simplicesname]
        # print(f"{cellsoflabel['vertex']=}\n{cells_dict['vertex']}")
        # self.verticesoflabel = cellsoflabel['vertex']
        self.verticesoflabel = {k:cells_dict['vertex'][v] for k,v in cellsoflabel['vertex'].items()}
        # self.linesoflabel = cellsoflabel['line']
        self.linesoflabel = {k:cells_dict['line'][v] for k,v in cellsoflabel['line'].items()}
        # print(f"{self.verticesoflabel=}")
        # print(f"{cellsoflabel=}\n{cellsoflabel.keys()}")
        # if self.dimension > 1: self.verticesoflabel = cellsoflabel['vertex']
        # print(f"{self.verticesoflabel=}")
        # bdry faces
        # for key, cellblock in cells:
        #     if key == self.facesnames[self.dimension - 1]: bdryfacesgmsh = cellblock
        if self.facesname not in cellsoflabel:
            raise ValueError(f"{self.facesname=} not in {cellsoflabel=}")
        return cellsoflabel[self.facesname]
    def _constructFacesFromSimplices(self):
        simplices = self.simplices
        ncells = simplices.shape[0]
        nnpc = simplices.shape[1]
        # allfaces = np.empty(shape=(nnpc*ncells,nnpc-1), dtype=np.uint64)
        # for i in range(ncells):
        #     for ii in range(nnpc):
        #         # face ii is opposite to node ii
        #         mask = np.array( [jj !=ii for jj in range(nnpc)] )
        #         allfaces[i*nnpc+ii] = np.sort(simplices[i,mask])
        nd = np.logical_not(np.eye(nnpc,dtype=bool)).ravel()
        allfaces = np.sort(np.tile(simplices, nnpc)[:,nd].reshape(ncells, nnpc, nnpc-1), axis=2).reshape(nnpc*ncells,nnpc-1)
        # if np.any(allfaces!=allfaces2):
        #     raise ValueError(f"{allfaces=}\n{allfaces2=}")
        # s = "{0}" + (nnpc-2)*", {0}"
        s = (nnpc-1)*"{0},"
        s = s[:-1].format(allfaces.dtype)
        # order = ["f0"]+["f{:1d}".format(i) for i in range(1,nnpc-1)]
        order = ["f{:1d}".format(i) for i in range(nnpc-1)]
        # print(f"{s=} {order=}")
        if self.dimension==1:
            perm = np.argsort(allfaces, axis=0).ravel()
        else:
            perm = np.argsort(allfaces.view(s), order=order, axis=0).ravel()
        # print(f"{allfaces=}")
        # print(f"{perm=}")
        allfacesorted = allfaces[perm]
        # print(f"{allfacesorted=}")
        self.faces, indices = np.unique(allfacesorted, return_inverse=True, axis=0)
        # print(f"{self.faces=}")
        self.nfaces = self.faces.shape[0]
        self.facesOfCells = np.zeros(shape=(ncells, nnpc), dtype=int)
        locindex = np.tile(np.arange(0,nnpc), ncells).ravel()
        cellindex = np.repeat(np.arange(0,ncells), nnpc)
        # for ii in range(indices.shape[0]):
        #     f = indices[ii]
        #     loc = locindex[perm[ii]]
        #     cell = cellindex[perm[ii]]
        #     self.facesOfCells[cell, loc] = f
        #     if self.cellsOfFaces[f,0] == -1: self.cellsOfFaces[f,0] = cell
        #     else: self.cellsOfFaces[f,1] = cell
        # foc = np.zeros(shape=(ncells, nnpc), dtype=int)
        self.facesOfCells[cellindex[perm],locindex[perm]] = indices

        # cellsOfFacesOld = -1 * np.ones(shape=(self.nfaces, 2), dtype=int)
        # for i in range(ncells):
        #     for ii in range(nnpc):
        #         f = self.facesOfCells[i,ii]
        #         if cellsOfFacesOld[f,0] == -1: cellsOfFacesOld[f,0] = i
        #         else: cellsOfFacesOld[f,1] = i
        unique, indices = np.unique(self.facesOfCells,return_index=True)
        assert np.all(unique == np.arange(self.nfaces))
        i0, i1 = np.unravel_index(indices, shape=self.facesOfCells.shape)
        foc = self.facesOfCells.copy()
        foc[i0,i1] = -1   
        unique, indices = np.unique(foc,return_index=True)
        i2, i3 = np.unravel_index(indices[1:], shape=foc.shape)
        i = -1*np.ones(self.nfaces, dtype=self.facesOfCells.dtype)
        i[unique[1:]] = i2
        self.cellsOfFaces =np.vstack([i0,i]).T
        # print(f"{self.cellsOfFaces.shape=} {cellsOfFacesOld.shape=}")
        # assert np.all(i0 == cellsOfFacesOld[:,0])
        # assert np.all(i == cellsOfFacesOld[:,1])
 
    def _constructBoundaryFaces7(self, bdryfacesgmsh, bdrylabelsgmsh):
        # bdries
        # bdryfacesgmsh may contains interior edges for len(celllabels)>1
        bdryfacesgmsh = np.sort(bdryfacesgmsh)
        bdryids = np.flatnonzero(self.cellsOfFaces[:,1] == -1)
        # print(f"{bdryids=}")
        # assert np.all(bdryids == np.flatnonzero(np.any(self.cellsOfFaces == -1, axis=1)))
        bdryfaces = np.sort(self.faces[bdryids],axis=1)
        # print(f"{bdryfacesgmsh=}\n{bdryfaces=}")
        # ind = np.isin(bdryfacesgmsh, bdryfaces)
        # print(f"{ind=} {bdryfacesgmsh[ind]=}")
        # print(f"{bdryfaces=}")
        # nbdryfaces = len(bdryids)
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
        # print(f"{bp=}")
        # print(f"{fp=}")
#https://stackoverflow.com/questions/51352527/check-for-identical-rows-in-different-numpy-arrays
        indices = (bdryfacesgmsh[bp, None] == bdryfaces[fp]).all(-1).any(-1)
        if not np.all(bdryfaces[fp]==bdryfacesgmsh[bp[indices]]):
            raise ValueError(f"{bdryfaces.T=}\n{bdryfacesgmsh.T=}\n{indices=}\n{bdryfaces[fp].T=}\n{bdryfacesgmsh[bp[indices]].T=}")
        bp2 = bp[indices]
        for i in range(len(fp)):
            if not np.all(bdryfacesgmsh[bp2[i]] == bdryfaces[fp[i]]):
                raise ValueError(f"{i=} {bdryfacesgmsh[bp2[i]]=} {bdryfaces[fp[i]]=}")
        bpi = np.argsort(bp)
        # bp2i = {bp2[i]:i for i in range(len(bp2))}
        # print(f"{bp=} \n{bp2=} \n{bpi=} \n{bp2i=} \n{indices=}")
        binv = -1*np.ones_like(bp)
        binv[bp2] = np.arange(len(bp2))
        self.bdrylabels = {}
        for col, cb in bdrylabelsgmsh.items():
            # if cb[0] in bp2i.keys():
            if indices[bpi[cb[0]]]:
                # for i in range(len(cb)):
                #     if not bp2i[cb[i]] == binv[cb[i]]:
                #         raise ValueError(f"{bp2i[cb[i]]} {binv[cb[i]]}")
                # print(f"{col=}")
                # self.bdrylabels[int(col)] = np.empty_like(cb)
                # for i in range(len(cb)): self.bdrylabels[int(col)][i] = bdryids[fp[binv[cb[i]]]]
                self.bdrylabels[int(col)] = bdryids[fp[binv[cb]]]
            # else:
            #     assert not indices[bpi[cb[0]]]
    def _constructNormalsAndAreas(self):
        # t = timer.Timer("_constructNormalsAndAreas")
        elem = self.simplices
        #TODO improve computation of sigma
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
        # t.add("ndV")
        # self.sigma = np.array([2 * (self.cellsOfFaces[self.facesOfCells[ic, :], 0] == ic)-1 for ic in range(self.ncells)])
        # for ic in range(self.ncells):
        #     print(self.cellsOfFaces[self.facesOfCells[ic, :], 0], ic, self.cellsOfFaces[self.facesOfCells[ic, :], 0] == ic)
        ind = np.arange(self.ncells)
        # print(f"{self.cellsOfFaces[self.facesOfCells[ind, :], 0].shape=} {ind.shape=}")
        self.sigma = 2* np.equal(self.cellsOfFaces[self.facesOfCells[ind, :], 0], ind[:,np.newaxis]) -1
        # print(f"{ind2.shape=}")
        # sigma2 = 2*ind2 - 1
        # if not np.all(self.sigma==sigma2): raise ValueError(f"{self.sigma=}\n{sigma2=}")
        # t.add("sigma")
        ib = np.arange(self.nfaces)[self.cellsOfFaces[:, 1] == -1]
        xt = np.mean(self.points[self.faces[ib]], axis=1) - np.mean(self.points[self.simplices[self.cellsOfFaces[ib, 0]]], axis=1)
        m = np.einsum('nk,nk->n', self.normals[ib], xt)<0
        self.normals[ib[m]] *= -1

        ib = np.arange(self.nfaces)[self.cellsOfFaces[:, 1] != -1]
        xt = np.mean(self.points[self.simplices[self.cellsOfFaces[ib, 1]]], axis=1) - np.mean(self.points[self.simplices[self.cellsOfFaces[ib, 0]]], axis=1)
        m = np.einsum('nk,nk->n', self.normals[ib], xt)<0
        self.normals[ib[m]] *= -1

        # for i in range(self.nfaces):
        #     i0, i1 = self.cellsOfFaces[i, 0], self.cellsOfFaces[i, 1]
        #     if i1 == -1:
        #         xt = np.mean(self.points[self.faces[i]], axis=0) - np.mean(self.points[self.simplices[i0]], axis=0)
        #         if np.dot(self.normals[i], xt)<0: self.normals[i] *= -1
        #     else:
        #         xt = np.mean(self.points[self.simplices[i1]], axis=0) - np.mean(self.points[self.simplices[i0]], axis=0)
        #         if np.dot(self.normals[i], xt) < 0: self.normals[i] *= -1
        # t.add("switch")
        # print(t)
        # self.sigma = np.array([1.0 - 2.0 * (self.cellsOfFaces[self.facesOfCells[ic, :], 0] == ic) for ic in range(self.ncells)])
    # ----------------------------------------------------------------#
    def write(self, filename, dirname = None, data=None):
        type = filename.split('.')[1]
        if dirname is not None:
            dirname = dirname + os.sep + "mesh"
            if not os.path.isdir(dirname) :
                os.makedirs(dirname)
            filename = os.path.join(dirname, filename)
        if self.dimension == 1:
            cells = {'lines': self.simplices}
            cells['vertex'] = self.facesdata
        elif self.dimension ==2:
            # cells = {'triangle': self.simplices}
            # cells['line'] = self.facesdata
            cells = [('triangle', self.simplices), ('line', self.facesdata)]
            # cells = [('triangle', self.simplices)]
        else:
            cells = [('tetra', self.simplices)]
            # cells = [('tetra', self.simplices), ('triangle', self.facesdata)]
        # meshio.write_points_cells(filename, self.points, cells, file_format="gmsh")
        args = {'points': self.points, 'cells':cells}
        if data is not None:
            if 'point' in data:
                args['point_data'] = data['point']
            if 'cell' in data:
                # print(f"{data['cell']=}")
                args['cell_data'] = {k: [data['cell'][k]] for k in data['cell'].keys()}
        mesh = meshio.Mesh(**args)
        # print(f"{mesh=}")
        meshio.write(filename, mesh)
    # ----------------------------------------------------------------#
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

#=================================================================#
if __name__ == '__main__':
    import pygmsh
    rect = [-2, 2, -2, 2]
    with pygmsh.geo.Geometry() as geom:
        z=0
        xc, yc, r = 0.5, 0.5, 0.5
        mesh_size = 0.1
        hole = geom.add_circle(x0=[xc,yc], radius=r, mesh_size=mesh_size, num_sections=6, make_surface=False)
        lines = hole.curve_loop.curves
        geom.add_physical(lines, label="3000")
        holes = [hole]
        p = geom.add_rectangle(*rect, z=0, mesh_size=1, holes=holes)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    print(f"{mesh=}")
    mesh = SimplexMesh(mesh=mesh)
    import plotmesh
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, 1, sharex='col')
    plotmesh.meshWithBoundaries(mesh, ax=axarr[0])
    plotmesh.plotmeshWithNumbering(mesh, ax=axarr[1])
    # plotmesh.plotmeshWithNumbering(mesh, localnumbering=True)
