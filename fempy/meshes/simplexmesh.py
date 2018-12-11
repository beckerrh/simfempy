# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import os
import meshio
import numpy as np
from scipy import sparse
from scipy import spatial

try:
    import geometry
except ModuleNotFoundError:
    from . import geometry


#=================================================================#
class SimplexMesh(object):
    """
    simplicial mesh based on scipy.delaunay
    """
    def __str__(self):
        return "TriangleMesh({}): dim/nvert/ncells/nedges: {}/{}/{}/{} bdrylabels={}".format(self.geomname, self.dimension, self.nnodes, self.ncells, self.nfaces, list(self.bdrylabels.keys()))
    def __init__(self, **kwargs):
        if 'data' in kwargs:
            self.geomname = 'own'
            data = kwargs.pop('data')
            self._initMeshPyGmsh(data[0], data[1], data[3])
            return
        self.geomname = kwargs.pop('geomname')
        hmean = None
        if 'hmean' in kwargs: hmean = kwargs.pop('hmean')
        filenamemsh = self.geomname + '.msh'
        if hmean is not None or not os.path.isfile(filenamemsh):
            geom = geometry.Geometry(geomname=self.geomname, h=hmean)
            geom.runGmsh(newgeometry=(hmean is not None))
        mesh = meshio.read(filename=filenamemsh)
        self._initMeshPyGmsh(mesh.points, mesh.cells, mesh.cell_data)
    def _initMeshPyGmsh(self, points, cells, celldata):
        if 'tetra' in cells.keys():
            self.dimension = 3
        elif 'triangle' in cells.keys():
            self.dimension = 2
        else:
            self.dimension = 1
        self.delaunay = spatial.Delaunay(points[:,:self.dimension])
        assert points.shape[1] ==3
        self.points = points
        self.simplices = self.delaunay.simplices
        assert self.dimension+1 == self.simplices.shape[1]
        self.nnodes = self.points.shape[0]
        assert self.nnodes == points.shape[0]
        self.ncells = self.simplices.shape[0]
        self.pointsc = self.points[self.simplices].mean(axis=1)
        if self.dimension==2:
            self._constructCellLabels(cells['triangle'], celldata['triangle']['gmsh:physical'])
            self._constructFaces(cells['line'], celldata['line']['gmsh:physical'])
        else:
            self._constructCellLabels(cells['tetra'], celldata['tetra']['gmsh:physical'])
            self._constructFaces(cells['triangle'], celldata['triangle']['gmsh:physical'])
        self._constructNormalsAndAreas()
        print(self)
    def _findIndices(self, A,B):
        #http://numpy-discussion.10968.n7.nabble.com/How-to-find-indices-of-values-in-an-array-indirect-in1d-td41972.html
        B_sorter = np.argsort(B)
        B_sorted = B[B_sorter]
        B_sorted_index = np.searchsorted(B_sorted, A)
        # Go back into the original index:
        B_index = B_sorter[B_sorted_index]
        valid = B.take(B_index, mode='clip') == A
        if not np.all(valid):
            print("A", A)
            print("B", B)
            raise ValueError("Did not find indices", valid)
        return B_index[valid]
    def _constructCellLabels(self, simp2, labels):
        simpsorted = np.sort(self.simplices, axis=1)
        labelssorted = np.sort(simp2, axis=1)
        if self.dimension==2:
            dts = "{0}, {0}, {0}".format(simpsorted.dtype)
            dtl = "{0}, {0}, {0}".format(labelssorted.dtype)
            sp = np.argsort(simpsorted.view(dts), order=('f0','f1','f2'), axis=0).flatten()
            lp = np.argsort(labelssorted.view(dtl), order=('f0','f1','f2'), axis=0).flatten()
        else:
            dts = "{0}, {0}, {0}, {0}".format(simpsorted.dtype)
            dtl = "{0}, {0}, {0}, {0}".format(labelssorted.dtype)
            sp = np.argsort(simpsorted.view(dts), order=('f0','f1','f2','f3'), axis=0).flatten()
            lp = np.argsort(labelssorted.view(dtl), order=('f0','f1','f2','f3'), axis=0).flatten()
        spi = np.empty(sp.size, sp.dtype)
        spi[sp] = np.arange(sp.size)
        perm = lp[spi]
        self.cell_labels = labels[perm]
    def _constructFaces(self, bdryfaces, bdrylabels):
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
        self.bdryfaces = np.flatnonzero(np.any(self.cellsOfFaces == -1, axis=1))
        self.nbdryfaces = len(self.bdryfaces)
        if len(bdrylabels) != self.nbdryfaces:
            raise ValueError("wrong number of boundary labels {} != {}".format(len(bdrylabelsmsh),self.nbdryfaces))
        if len(bdryfaces) != self.nbdryfaces:
            raise ValueError("wrong number of bdryfaces {} != {}".format(len(bdryfaces), self.nbdryfaces))
        self.bdrylabels = {}
        colors, counts = np.unique(bdrylabels, return_counts=True)
        # print ("colors, counts", colors, counts)
        for i in range(len(colors)):
            self.bdrylabels[colors[i]] = -np.ones( (counts[i]), dtype=np.int32)
        bdryfaces = np.sort(bdryfaces)
        n = self.nfaces
        if self.dimension==2:
            A = n*bdryfaces[:,0] + bdryfaces[:,1]
            B = n * self.faces[:, 0] + self.faces[:, 1]
        else:
            A = n*n*bdryfaces[:,0] + n*bdryfaces[:,1] + bdryfaces[:,2]
            B = n*n * self.faces[:, 0] + n*self.faces[:, 1] + bdryfaces[:,2]
        toto = self._findIndices(A,B)

        #nouvelle version
        # if self.dimension==2:
        #     dtb = "{0}, {0}".format(bdryfaces.dtype)
        #     dtf = "{0}, {0}".format(self.faces.dtype)
        #     bp = np.argsort(bdryfaces.view(dtb), order=('f0','f1'), axis=0).flatten()
        #     fp = np.argsort(self.faces.view(dtf), order=('f0','f1'), axis=0).flatten()
        # else:
        #     dtb = "{0}, {0}, {0}".format(bdryfaces.dtype)
        #     dtf = "{0}, {0}, {0}".format(self.faces.dtype)
        #     bp = np.argsort(bdryfaces.view(dtb), order=('f0','f1','f2'), axis=0).flatten()
        #     fp = np.argsort(self.faces.view(dtf), order=('f0','f1','f2'), axis=0).flatten()
        #
        # print ("bp", bp)
        # print ("fp", fp)
        # bpinfp = np.searchsorted(fp, bp)
        # print ("bpinfp", bpinfp)
        # bpi = np.empty(bp.size, bp.dtype)
        # bpi[bp] = np.arange(bp.size)
        # perm = fp[bpi]
        # print ("toto", toto)
        # print ("perm", perm)





        counts = {}
        for key in list(self.bdrylabels.keys()): counts[key]=0
        for i in range(len(toto)):
            if np.any(bdryfaces[i] != self.faces[toto[i]]):
                raise ValueError("Did not find boundary indices")
            color = bdrylabels[i]
            self.bdrylabels[color][counts[color]] = toto[i]
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
            self.dx = 0.5 * np.abs(dx1*dy2-dx2*dy1)
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
            self.normals = np.stack((sidesx, sidesy, sidesz), axis=-1)
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
            self.dx = (1/6) * np.abs(dx1*(dy2*dz3-dy3*dz2) - dx2*(dy1*dz3-dy3*dz1) + dx3*(dy1*dz2-dy2*dz1))
        for i in range(self.nfaces):
            i0, i1 = self.cellsOfFaces[i, 0], self.cellsOfFaces[i, 1]
            if i1 == -1:
                xt = np.mean(self.points[self.faces[i]], axis=0) - np.mean(self.points[self.simplices[i0]], axis=0)
                if np.dot(self.normals[i], xt)<0:  self.normals[i] *= -1
            else:
                xt = np.mean(self.points[self.simplices[i1]], axis=0) - np.mean(self.points[self.simplices[i0]], axis=0)
                if np.dot(self.normals[i], xt) < 0:  self.normals[i] *= -1
    def write(self, filename, dirname = "out", point_data=None, cell_data=None):
        cell_data_meshio = {}
        if self.dimension ==2:
            cells = {'triangle': self.simplices}
            if cell_data is not None:
                cell_data_meshio = {'triangle': cell_data}
        else:
            cells = {'tetra': self.simplices}
            if cell_data is not None:
                cell_data_meshio = {'tetra': cell_data}
        dirname = dirname + os.sep + "mesh"
        if not os.path.isdir(dirname) :
            os.makedirs(dirname)
        filename = os.path.join(dirname, filename)
        meshio.write_points_cells(filename=filename, points=self.points, cells=cells, point_data=point_data, cell_data=cell_data_meshio)
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


# ------------------------------------- #
if __name__ == '__main__':
    # tmesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.7)
    tmesh = SimplexMesh(geomname="unitsquare", hmean=0.7)
    import plotmesh
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(3, 1, sharex='col')
    # plotdata = tmesh.x, tmesh.y, tmesh.triangles, tmesh.lines, tmesh.labels_lines
    # plotmesh.meshWithBoundaries(plotdata, ax=axarr[0])
    plotmesh.meshWithBoundaries(tmesh, ax=axarr[0])
    # plotdata = tmesh.x, tmesh.y, tmesh.triangles, tmesh.centersx, tmesh.centersy
    # plotmesh.meshWithNodesAndTriangles(plotdata, ax=axarr[1])
    plotmesh.meshWithNodesAndTriangles(tmesh, ax=axarr[1])
    plotmesh.meshWithNodesAndFaces(tmesh, ax=axarr[2])
    plt.show()
