# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import os
import meshio
import matplotlib.tri
import numpy as np
import scipy.sparse as sparse

try:
    import geometry
except ModuleNotFoundError:
    from . import geometry


#=================================================================#
class TriangleMesh(matplotlib.tri.Triangulation):
    """
    triangular mesh based on matplotlib.tri.Triangulation
    """
    def __init__(self, **kwargs):
        if 'data' in kwargs:
            self.geomname = 'own'
            data = kwargs.pop('data')
            self._initMesh(data[0], data[1], data[3])
            return
        self.geomname = kwargs.pop('geomname')
        hmean = None
        if 'hmean' in kwargs: hmean = kwargs.pop('hmean')
        filenamemsh = self.geomname + '.msh'
        if hmean is not None or not os.path.isfile(filenamemsh):
            geom = geometry.Geometry(geomname=self.geomname, h=hmean)
            geom.runGmsh(newgeometry=(hmean is not None))
        mesh = meshio.read(filename=filenamemsh)
        self._initMesh(mesh.points, mesh.cells, mesh.cell_data)
    def _initMesh(self, points, cells, celldata):
        self.bdrylabelsmsh = celldata['line']['gmsh:physical']
        matplotlib.tri.Triangulation.__init__(self, x=points[:, 0], y=points[:, 1], triangles=cells['triangle'])
        self.nedges = len(self.edges)
        self.ncells = len(self.triangles)
        self.nnodes = len(self.x)
        self.bdryvert = self.triangles.flat[np.flatnonzero(self.neighbors == -1)]
        self.intvert = np.setxor1d(np.arange(self.nnodes), self.bdryvert)
        self.nbdryvert = len(self.bdryvert)
        self.normals = None
        self.area = None
        self.cellsOfEdge = None
        self.edgesOfCell = None
        self.centersx = self.x[self.triangles].mean(axis=1)
        self.centersy = self.y[self.triangles].mean(axis=1)
        self.edgesx = self.x[self.edges].mean(axis=1)
        self.edgesy = self.y[self.edges].mean(axis=1)
        self.construcCellEdgeConnectivity()
        self.construcNormalsAndAreas()
        self.bdryedges = np.flatnonzero(np.any(self.cellsOfEdge == -1, axis=1))
        self.intedges = np.setxor1d(np.arange(self.nedges), self.bdryedges)
        self.nbdryedges = len(self.bdryedges)
        self.nintedges = len(self.intedges)
        self.bdrylabels = None
        self.lines = cells['line']
        self.constructBoundaryEdges(self.bdrylabelsmsh, cells['line'])
        print(self)

    def __str__(self):
        return "TriangleMesh({}): nvert/ncells/nedges: {}/{}/{} bdrylabels={}".format(self.geomname, len(self.x), len(self.triangles), len(self.edges), list(self.bdrylabels.keys()))
    def write(self, filename, dirname = "out", point_data=None, cell_data=None):
        points = np.zeros((self.nnodes, 3))
        points[:, 0:2] = np.stack((self.x, self.y), axis=-1)
        assert np.all(points[:, 0] == self.x)
        assert np.all(points[:, 1] == self.y)
        cells = {'triangle': self.triangles}
        if cell_data is not None:
            cell_data_meshio = {'triangle': cell_data}
        else:
            cell_data_meshio=None
        dirname = dirname + os.sep + "mesh"
        if not os.path.isdir(dirname) :
            os.makedirs(dirname)
        filename = os.path.join(dirname, filename)
        meshio.write_points_cells(filename=filename, points=points, cells=cells, point_data=point_data, cell_data=cell_data_meshio)
    def construcNormalsAndAreas(self):
        sidesx = self.y[self.edges[:, 0]] - self.y[self.edges[:, 1]]
        sidesy = self.x[self.edges[:, 1]] - self.x[self.edges[:, 0]]
        self.normals = np.stack((sidesx, sidesy), axis=-1)
        elem = self.triangles
        sidesx0 = self.x[elem[:, 2]] - self.x[elem[:, 1]]
        sidesx1 = self.x[elem[:, 0]] - self.x[elem[:, 2]]
        sidesx2 = self.x[elem[:, 1]] - self.x[elem[:, 0]]
        sidesx = np.stack((sidesx0, sidesx1, sidesx2), axis=-1)
        sidesy0 = self.y[elem[:, 2]] - self.y[elem[:, 1]]
        sidesy1 = self.y[elem[:, 0]] - self.y[elem[:, 2]]
        sidesy2 = self.y[elem[:, 1]] - self.y[elem[:, 0]]
        sidesy = np.stack((sidesy0, sidesy1, sidesy2), axis=-1)
        self.area = 0.5 * np.abs(-sidesx[:, 2] * sidesy[:, 1] + sidesy[:, 2] * sidesx[:, 1])
    def construcCellEdgeConnectivity(self):
        self.cellsOfEdge = -1 * np.ones(shape=(self.nedges, 2), dtype=int)
        self.edgesOfCell = np.zeros(shape=(self.ncells, 3), dtype=int)
        eovs = {}
        for (ie, iv) in enumerate(self.edges):
            iv.sort()
            eovs[tuple(iv)] = ie
        for (it, iv) in enumerate(self.triangles):
            for ii in range(3):
                ivs = [iv[(ii + 1) % 3], iv[(ii + 2) % 3]]
                ivs.sort()
                ie = eovs[tuple(ivs)]
                self.edgesOfCell[it, ii] = ie
                if iv[(ii + 1) % 3] == self.edges[ie, 0] and iv[(ii + 2) % 3] == self.edges[ie, 1]:
                    self.cellsOfEdge[ie, 1] = it
                else:
                    self.cellsOfEdge[ie, 0] = it
                    # print 'self.cellsOfEdge', self.cellsOfEdge
                    # print 'self.edgesOfCell', self.edgesOfCell
    def constructBoundaryEdges(self, bdrylabelsmsh, lines):
        """
        we suppose, self.edges is sorted !!!
        :param bdrylabelsmsh:
        :param lines:
        :return:
        """
        if len(bdrylabelsmsh) != self.nbdryedges:
            raise ValueError("wrong number of boundary labels %d != %d (self.nbdryedges)" %(len(bdrylabelsmsh),self.nbdryedges))
        if len(lines) != self.nbdryedges:
            raise ValueError("wrong number of lines %d != %d (self.nedges)" % (len(lines), self.nbdryedges))
        self.bdrylabels = {}
        colors, counts = np.unique(bdrylabelsmsh, return_counts=True)
        # print ("colors, counts", colors, counts)
        for i in range(len(colors)):
            self.bdrylabels[colors[i]] = -np.ones( (counts[i]), dtype=np.int32)
        lines = np.sort(lines)
        # print("lines", lines)
        # print("self.edges", self.edges)

        n = self.nedges
        A = np.zeros( lines.shape[0], dtype=np.int32)
        B = np.zeros( n, dtype=np.int32)
        for i in range(len(A)):
            A[i] = n*lines[i,0] + lines[i,1]
        for i in range(len(B)):
            B[i] = n*self.edges[i,0] + self.edges[i,1]

        #http://numpy-discussion.10968.n7.nabble.com/How-to-find-indices-of-values-in-an-array-indirect-in1d-td41972.html
        B_sorter = np.argsort(B)
        B_sorted = B[B_sorter]
        B_sorted_index = np.searchsorted(B_sorted, A)

        # Go back into the original index:
        B_index = B_sorter[B_sorted_index]
        valid = B.take(B_index, mode='clip') == A
        if not np.all(valid):
            raise ValueError("Did not find indices", valid)
        toto = B_index[valid]
        counts = {}
        for key in list(self.bdrylabels.keys()): counts[key]=0
        for i in range(len(toto)):
            if np.any(lines[i] != self.edges[toto[i]]):
                raise ValueError("Did not find boundary indices")
            color = bdrylabelsmsh[i]
            self.bdrylabels[color][counts[color]] = toto[i]
            counts[color] += 1
        # print ("self.bdrylabels", self.bdrylabels)
    def computeSimpOfVert(self, test=False):
        S = sparse.dok_matrix((self.nnodes, self.ncells), dtype=int)
        for ic in range(self.ncells):
            S[self.triangles[ic,:], ic] = ic+1
        S = S.tocsr()
        S.data -= 1
        self.simpOfVert = S
        if test:
            # print("S=",S)
            from . import plotmesh
            import matplotlib.pyplot as plt
            simps, xc, yc = self.triangles, self.centersx, self.centersy
            meshdata =  self.x, self.y, simps, xc, yc
            plotmesh.meshWithNodesAndTriangles(meshdata)
            plt.show()


# ------------------------------------- #
if __name__ == '__main__':
    tmesh = TriangleMesh(geomname="backwardfacingstep", hmean=0.7)
    import plotmesh
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, 1, sharex='col')
    plotdata = tmesh.x, tmesh.y, tmesh.triangles, tmesh.lines, tmesh.bdrylabelsmsh
    plotmesh.meshWithBoundaries(plotdata, ax=axarr[0])
    plotdata = tmesh.x, tmesh.y, tmesh.triangles, tmesh.centersx, tmesh.centersy
    plotmesh.meshWithNodesAndTriangles(plotdata, ax=axarr[1])
    plt.show()
