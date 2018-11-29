# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from trimesh import TriMesh
# import trimesh

class TriMeshWithNodePatches(TriMesh):
    def __init__(self, tmesh=None, geomname="unitsquare", hmean=None):
        TriMesh.__init__(self, trimesh=tmesh, geomname=geomname, hmean=hmean)
        # self.constructPatches()
        # self.constructNodePatchIndices()
        # self.computeNodePatchInfo()
        self.patches_tris = None
        self.patches_bnodes = None
        self.patchinfo = None
        self.patches_opind = None
        self.patches_opcoeff = None
        self.computeNodePatches()
        print(self)

    def __str__(self):
        return 'TriMeshWithNodePatches: nvert/ncells/nedges: %d %d %d' % (len(self.x), len(self.triangles), len(self.edges))

    def computeNodePatches(self):
        if self.patches_tris is not None: return
        self.constructNodePatchIndices()
        self.computeNodePatchInfo()
        self.computeNodeSideInfo()

    def testNodePatchesOld(self):
        problem = False
        for iv in range(self.nnodes):
            boundary = -1 in self.patches_bnodes[iv][:, 5]
            # if boundary: continue
            for i, patch in enumerate(self.patches_bnodes[iv]):
                iv2 = patch[0]
            # for i, patch in self.patchinfo[iv].iteritems():
            #     iv2 = patch[0][0]
                gradS = np.array([self.x[iv2] - self.x[iv], self.y[iv2] - self.y[iv]])
                i0 = self.patches_opind[iv][i, 0]
                i1 = self.patches_opind[iv][i, 1]
                a0 = self.patches_opcoeff[iv][i, 0]
                a1 = self.patches_opcoeff[iv][i, 1]
                if not boundary:
                    if a0 > 1e-8 or a1 > 1e-8:
                        print('problem in iv', iv, 'iv2', iv2, 'a0 a1', a0, a1, 'i0, i1', i0, i1)
                        problem = True
                gradR = np.array([a0 * (self.x[i0] - self.x[iv]) + a1 * (self.x[i1] - self.x[iv]),a0 * (self.y[i0] - self.y[iv]) + a1 * (self.y[i1] - self.y[iv])])
                if linalg.norm(gradR-gradS)>1e-14:
                    print('iv', iv, 'iv2', iv2)
                    print('gradS', gradS, 'gradR', gradR)
                    self.plotPatches(plt)
                    raise ValueError('wrong')
        if problem:
            self.plotPatches(plt)
            raise ValueError("wrong")
                # else:
                #     print 'no difference for iv=', iv, 'iv2=', iv2
    def testNodePatches(self):
        problem = False
        x, y = self.x, self.y
        for iv in range(self.nnodes):
            boundary = -1 in self.patches_bnodes[iv][:, 5]
            for ii,patch in self.patchinfo[iv].items():
                iv2 = patch[0][0]
                ie = patch[0][1]
                gradS = np.array([x[iv2] - x[iv], y[iv2] - y[iv]])
                for ii in range(patch[1].shape[0]):
                    i0 = patch[1][ii][0]
                    i1 = patch[1][ii][1]
                    a0 = patch[2][ii][0]
                    a1 = patch[2][ii][1]
                    gradR = np.array([a0 * (x[i0] - x[iv]) + a1 * (x[i1] - x[iv]),a0 * (y[i0] - y[iv]) + a1 * (y[i1] - y[iv])])
                    if linalg.norm(gradR-gradS)>1e-14:
                        print('gradR', gradR)
                        print('gradS', gradS)
                        raise ValueError('wrong')
                        problem = True
        if problem:
            self.plotPatches(plt)
            raise ValueError("wrong")
                # else:
                #     print 'no difference for iv=', iv, 'iv2=', iv2
    def constructNodePatchIndices(self):
        self.patches_tris = dict((i,[]) for i in range(self.nnodes))
        for it in range(self.ncells):
            tri = self.triangles[it]
            for ii in range(3):
                inode = tri[ii]
                self.patches_tris[inode].append( (it,ii) )
        patches_bsides = {}
        self.patches_bnodes = {}
        # chercher les edges autour du patch
        for iv,patch in self.patches_tris.items():
            edges_unique = {}
            edges_twice = []
            for it,iit in patch:
                for ii in range(3):
                    ie = self.edgesOfCell[it,ii]
                    if ie in list(edges_unique.keys()):
                        del edges_unique[ie]
                        edges_twice.append(ie)
                    else:
                        edges_unique[ie] = it
            npatch = len(edges_unique)
            patches_bsides[iv] = -np.ones((npatch,2), dtype=int)
            self.patches_bnodes[iv] = -np.ones((npatch,6), dtype=int)
            for ii, (ie, it) in enumerate(edges_unique.items()):
                patches_bsides[iv][ii,0] = ie
                patches_bsides[iv][ii,1] = it
            nodes = []
            for ii in range(npatch):
                ie = patches_bsides[iv][ii,0]
                for iii in range(2):
                    nodes.append(self.edges[ie,iii])
            nodes = np.unique(nodes)
            nodesi = {}
            for ii in range(npatch):
                nodesi[nodes[ii]]=ii
            for ii in range(npatch):
                self.patches_bnodes[iv][ii,0] = nodes[ii]
            for ii in range(npatch):
                ie = patches_bsides[iv][ii, 0]
                it = patches_bsides[iv][ii, 1]
                sign = 2*(it == self.cellsOfEdge[ie,0])-1
                for iii in range(2):
                    inode = nodesi[self.edges[ie, iii]]
                    if self.patches_bnodes[iv][inode,1] == -1:
                        self.patches_bnodes[iv][inode, 1] = ie
                        self.patches_bnodes[iv][inode, 3] = sign
                    else:
                        self.patches_bnodes[iv][inode, 2] = ie
                        self.patches_bnodes[iv][inode, 4] = sign
                for iii in range(3):
                    ie2 = self.edgesOfCell[it,iii]
                    if self.edges[ie2,0] == iv:
                        if self.edges[ie2,1] in nodesi:
                            self.patches_bnodes[iv][nodesi[self.edges[ie2,1]],5] = ie2
                    elif self.edges[ie2,1] == iv:
                        if self.edges[ie2, 0] in nodesi:
                            self.patches_bnodes[iv][nodesi[self.edges[ie2, 0]],5] = ie2
    def computeNodePatchInfo(self):
        self.patchinfo = {}
        for iv in range(self.nnodes):
            # indices des deux aretes par tri et des tris qui ne contiennent PAS cette arrete
            boundary = -1 in self.patches_bnodes[iv][:, 5]
            npatch = self.patches_bnodes[iv].shape[0]
            tris = self.patches_tris[iv]
            ntris = len(tris)
            edgeoftri = -np.ones((ntris, 2), dtype=int)
            trisofedge = {}
            for i in range(npatch):
                edge = self.patches_bnodes[iv][i,5]
                trisofedge[i] = {i for i in range(ntris)}
                for itri in range(ntris):
                    tri = tris[itri][0]
                    for ii in range(3):
                        if edge == self.edgesOfCell[tri,ii]:
                            if edgeoftri[itri,0]==-1:
                                edgeoftri[itri, 0] = i
                                trisofedge[i].remove(itri)
                            else:
                                edgeoftri[itri, 1] = i
                                trisofedge[i].remove(itri)
                                break
            self.patchinfo[iv] = {}
            for i in range(npatch):
                ntriinpatch = len(trisofedge[i])
                iv2 = self.patches_bnodes[iv][i,0]
                ie = self.patches_bnodes[iv][i,5]
                self.patchinfo[iv][i] = [ [iv2, ie, -1], -np.ones((ntriinpatch, 2), dtype=int), np.zeros((ntriinpatch, 2), dtype=np.float64)]
            # print 'trisofedge', trisofedge
            # print 'edgeoftri', edgeoftri
            assert not -1 in edgeoftri
            # compute all edges
            S = np.zeros(shape=(npatch, 2), dtype=np.float64)
            # invind = {}
            for i,patch in enumerate(self.patches_bnodes[iv]):
                # invind[patch[0]] = i
                S[i,0] = self.x[patch[0]]-self.x[iv]
                S[i,1] = self.y[patch[0]]-self.y[iv]
            for itri in range(ntris):
                tri = tris[itri][0]
                SP = np.vstack((S[edgeoftri[itri, 0]], S[edgeoftri[itri, 1]]))
                try:
                    SP = linalg.inv(SP)
                except:
                    print('matrix singular, iv', iv, patch[0], self.patches_bnodes[iv][i2,0])
                    print(self.patches_bnodes[iv])
                    print('Si0 Si1', S[i], S[i2])
                    print('SP', SP)
                    self.plotPatches(plt)
                    raise ValueError("matrix singular")
                for j in range(npatch):
                    if itri in trisofedge[j]:
                        index = list(trisofedge[j]).index(itri)
                        sa = np.dot(SP.T, S[j])
                        self.patchinfo[iv][j][1][index][0] = self.patches_bnodes[iv][edgeoftri[itri, 0],0]
                        self.patchinfo[iv][j][1][index][1] = self.patches_bnodes[iv][edgeoftri[itri, 1],0]
                        self.patchinfo[iv][j][2][index] = sa
                        if np.all(sa<=0):
                            self.patchinfo[iv][j][0][2] = index
            for j in range(npatch):
                if self.patchinfo[iv][j][0][2] ==-1:
                    if not boundary:
                        print('not found', self.patchinfo[iv][j][2])
                        print('iv', iv)
                        print('j', j, self.patchinfo[iv][j])
                        print('patch', self.patchinfo[iv])
                        self.plotPatches(plt)
                        raise ValueError("opposite edge not found")
            # print 'self.patchinfo[iv]', self.patchinfo[iv]
    def computeNodeSideInfo(self):
        self.patches_opind = {}
        self.patches_opcoeff = {}
        for iv in range(self.nnodes):
            boundary = -1 in self.patches_bnodes[iv][:, 5]
            # if boundary: continue
            npatch = self.patches_bnodes[iv].shape[0]
            self.patches_opind[iv] = -np.ones((npatch,2), dtype=int)
            self.patches_opcoeff[iv] = np.zeros((npatch,2), dtype=np.float64)
            assert npatch==len(self.patchinfo[iv])
            for iii,patch in self.patchinfo[iv].items():
                if patch[0][1] == -1: continue
                found = False
                for ii in range(patch[1].shape[0]):
                    a0 = patch[2][ii][0]
                    a1 = patch[2][ii][1]
                    if a0<=0.0 and a1<=0.0:
                        if found:
                            print(self.patches_opcoeff[iv][0], self.patches_opcoeff[iv][1])
                            print(a0, a1)
                            print('iv', iv)
                            print(patch)
                            self.plotPatches(plt)
                            stop
                        found = True
                        self.patches_opind[iv][iii, 0] = patch[1][ii, 0]
                        self.patches_opind[iv][iii, 1] = patch[1][ii, 1]
                        self.patches_opcoeff[iv][iii, 0] = a0
                        self.patches_opcoeff[iv][iii, 1] = a1
                if not found:
                    # print 'iv', iv, 'patch', patch
                    # print patch[2][:,0]+patch[2][:,1]
                    if patch[1].shape[0]==0:
                        self.patches_opind[iv][iii, 0] = patch[0][0]
                        self.patches_opind[iv][iii, 1] = iv
                        self.patches_opcoeff[iv][iii, 0] = 1.0
                        self.patches_opcoeff[iv][iii, 1] = 0.0
                    else:
                        ii = np.argmin( patch[2][:,0]+patch[2][:,1] )
                    # print iii, npatch, ii
                    # print self.patches_opind[iv]
                    # print patch[2]
                    # a0 = patch[2][ii][0]
                        self.patches_opind[iv][iii, 0] = patch[1][ii, 0]
                        self.patches_opind[iv][iii, 1] = patch[1][ii, 1]
                        self.patches_opcoeff[iv][iii, 0] = patch[2][ii, 0]
                        self.patches_opcoeff[iv][iii, 1] = patch[2][ii, 1]
                    # else:
                    #     print 'iv', iv
                    #     print 'iv2', patch[0][0], '(ie)', patch[0][1]
                    #     print patch
                    #     self.plotPatches(plt)
                    #     stop
    # def constructNodePatches(self):
    #     self.patches_tris = dict((i,[]) for i in range(self.nnodes))
    #     for it in range(self.ncells):
    #         tri = self.triangles[it]
    #         for ii in range(3):
    #             inode = tri[ii]
    #             self.patches_tris[inode].append( (it,ii) )
    #     patches_bsides = {}
    #     self.patches_bnodes = {}
    #     for iv,patch in self.patches_tris.iteritems():
    #         edges_unique = {}
    #         edges_twice = []
    #         for it,iit in patch:
    #             for ii in range(3):
    #                 ie = self.edgesOfCell[it,ii]
    #                 if ie in edges_unique.keys():
    #                     del edges_unique[ie]
    #                     edges_twice.append(ie)
    #                 else:
    #                     edges_unique[ie] = it
    #         npatch = len(edges_unique)
    #         patches_bsides[iv] = -np.ones((npatch,2), dtype=int)
    #         self.patches_bnodes[iv] = -np.ones((npatch,6), dtype=int)
    #         for ii, (ie, it) in enumerate(edges_unique.iteritems()):
    #             patches_bsides[iv][ii,0] = ie
    #             patches_bsides[iv][ii,1] = it
    #         nodes = []
    #         for ii in range(npatch):
    #             ie = patches_bsides[iv][ii,0]
    #             for iii in range(2):
    #                 nodes.append(self.edges[ie,iii])
    #         nodes = np.unique(nodes)
    #         nodesi = {}
    #         for ii in range(npatch):
    #             nodesi[nodes[ii]]=ii
    #         for ii in range(npatch):
    #             self.patches_bnodes[iv][ii,0] = nodes[ii]
    #         for ii in range(npatch):
    #             ie = patches_bsides[iv][ii, 0]
    #             it = patches_bsides[iv][ii, 1]
    #             sign = 2*(it == self.cellsOfEdge[ie,0])-1
    #             for iii in range(2):
    #                 inode = nodesi[self.edges[ie, iii]]
    #                 if self.patches_bnodes[iv][inode,1] == -1:
    #                     self.patches_bnodes[iv][inode, 1] = ie
    #                     self.patches_bnodes[iv][inode, 3] = sign
    #                 else:
    #                     self.patches_bnodes[iv][inode, 2] = ie
    #                     self.patches_bnodes[iv][inode, 4] = sign
    #             for iii in range(3):
    #                 ie2 = self.edgesOfCell[it,iii]
    #                 if self.edges[ie2,0] == iv:
    #                     if self.edges[ie2,1] in nodesi:
    #                         self.patches_bnodes[iv][nodesi[self.edges[ie2,1]],5] = ie2
    #                 elif self.edges[ie2,1] == iv:
    #                     if self.edges[ie2, 0] in nodesi:
    #                         self.patches_bnodes[iv][nodesi[self.edges[ie2, 0]],5] = ie2
    #
    #     self.patches_opind = {}
    #     self.patches_opcoeff = {}
    #     for iv in range(self.nnodes):
    #         boundary = -1 in self.patches_bnodes[iv][:, 5]
    #         npatch = self.patches_bnodes[iv].shape[0]
    #         self.patches_opind[iv] = -np.ones((npatch,2), dtype=int)
    #         self.patches_opcoeff[iv] = np.zeros((npatch,2), dtype=np.float64)
    #         S = np.zeros(shape=(npatch, 2), dtype=np.float64)
    #         # Snorm = np.zeros(shape=(npatch, 2), dtype=np.float64)
    #         invind = {}
    #         for i,patch in enumerate(self.patches_bnodes[iv]):
    #             invind[patch[0]] = i
    #             S[i,0] = self.x[patch[0]]-self.x[iv]
    #             S[i,1] = self.y[patch[0]]-self.y[iv]
    #             sl = np.linalg.norm(S[i])
    #             # if sl: Snorm[i] = S[i]/sl
    #         iiv = -1
    #         for i,patch in enumerate(self.patches_bnodes[iv]):
    #             if iv == patch[0]:
    #                 iiv = i
    #                 continue
    #             sp = np.dot(S, S[i,:])
    #             if np.all(sp >=0.0):
    #                 continue
    #             spsort = np.argsort(sp)
    #             i0 = spsort[0]
    #             e1 = self.edges[self.patches_bnodes[iv][i0,1]]
    #             e2 = self.edges[self.patches_bnodes[iv][i0,2]]
    #             possibleids = np.setxor1d(e1, e2)
    #             i1found=False
    #             for pp in possibleids:
    #                 i1 = invind[pp]
    #                 assert i1 != i0
    #                 if self.patches_bnodes[iv][i1,0] == iv:
    #                     continue
    #                 if sp[i1]>0.5:
    #                     continue
    #                 #---------------
    #                 SP =  np.outer(S[i0], S[i0])
    #                 SP +=  np.outer(S[i1], S[i1])
    #                 try:
    #                     SP = linalg.inv(SP)
    #                 except:
    #                     print 'i0, i1, iiv', i0, i1, iiv, 'iv', iv, 'iv2', patch[0]
    #                     print 'S', S
    #                     print 'sp', sp, 'spsort', spsort
    #                     print 'SP', SP
    #                     raise ValueError("matrix singular")
    #                 a0 = np.inner(np.dot(SP, S[i0]),S[i])
    #                 a1 = np.inner(np.dot(SP, S[i1]),S[i])
    #                 print 'a0 a1', a0, a1, 'i1', i1, 'pp', pp
    #                 #---------------
    #                 SP2 = np.vstack( (S[i0],S[i1]))
    #                 try:
    #                     SP2 = linalg.inv(SP2)
    #                 except:
    #                     print 'Si0 Si1', S[i0],S[i1]
    #                     print 'SP2', SP2
    #                     raise ValueError("matrix singular")
    #                 sa2 = np.dot(SP2.T, S[i])
    #                 if np.abs(a0-sa2[0])>1e-14 or np.abs(a1-sa2[1])>1e-14:
    #                     print 'a0 a1', a0, a1, 'sa2', sa2
    #                     raise ValueError("wrong projection")
    #                 #---------------
    #                 # #---------------
    #                 # SP2 = np.zeros((2,2))
    #                 # SP2[0,0] = np.inner(S[i0],S[i0])
    #                 # SP2[0,1] = np.inner(S[i0],S[i1])
    #                 # SP2[1,0] = SP2[0,1]
    #                 # SP2[1,1] = np.inner(S[i1],S[i1])
    #                 # sb2 = np.zeros(2)
    #                 # sb2[0] = np.inner(S[i0], S[i])
    #                 # sb2[1] = np.inner(S[i1], S[i])
    #                 # sa2 = linalg.solve(SP2, sb2)
    #                 # if np.abs(a0-sa2[0])>1e-14 or np.abs(a1-sa2[1])>1e-14:
    #                 #     print 'a0 a1', a0, a1, 'sa2', sa2
    #                 #     raise ValueError("wrong projection")
    #                 # iv2 = patch[0]
    #                 # iv0 = self.patches_bnodes[iv][i0,0]
    #                 # iv1 = self.patches_bnodes[iv][i1,0]
    #                 # pp = np.array([self.x[iv],  self.y[iv]])
    #                 # pi = np.array([self.x[iv2], self.y[iv2]])
    #                 # p0 = np.array([self.x[iv0], self.y[iv0]])
    #                 # p1 = np.array([self.x[iv1], self.y[iv1]])
    #                 # k0 = linalg.norm(p0-pp)/linalg.norm(pi-pp)
    #                 # k1 = linalg.norm(p1-pp)/linalg.norm(pi-pp)
    #                 # print 'a0+a1', k0*a0 + k1*a1
    #                 # #---------------
    #                 a0 = np.around(a0, 16)+0
    #                 a1 = np.around(a1, 16)+0
    #                 if a0 <= 0 and a1 <= 0:
    #                     i1found = True
    #                     break
    #             if i1found:
    #                 self.patches_opcoeff[iv][i][0] = a0
    #                 self.patches_opcoeff[iv][i][1] = a1
    #                 iv0 = self.patches_bnodes[iv][i0,0]
    #                 iv1 = self.patches_bnodes[iv][i1,0]
    #                 self.patches_opind[iv][i,0] = iv0
    #                 self.patches_opind[iv][i,1] = iv1
    #                 if a0 > 1e-8 or a1 > 1e-8:
    #                     print 'a0, a1', a0, a1, 'iv', iv, 'iv2', patch[0]
    #                     print '\tsp', sp, 'spsort', spsort, 'i0, i1', i0, i1, 'iv0, iv1', iv0, iv1
    #                     raise ValueError("so nicht")
    #             else:
    #                 if not boundary:
    #                     print 'iv', iv, 'patch[0]', patch[0 ]
    #                     for ii in range(len(sp)):
    #                         print '\t ', self.patches_bnodes[iv][ii,0], sp[ii]
    #                     self.plotPatches(plt)
    #                     raise ValueError("y en a marre")
    #     self.patch_indexofedge = -np.ones(shape=(self.nedges,2), dtype= int)
    #     for iv in range(self.nnodes):
    #         for i,patch in enumerate(self.patches_bnodes[iv]):
    #             ie = patch[5]
    #             if ie == -1: continue
    #             if iv == self.edges[ie,0]:
    #                 assert self.patch_indexofedge[ie,0]==-1
    #                 self.patch_indexofedge[ie,0] = i
    #             else:
    #                 assert iv == self.edges[ie,1]
    #                 assert self.patch_indexofedge[ie,1]==-1
    #                 self.patch_indexofedge[ie,1] = i
    class PatchPlotterEdges(object):
        def __init__(self, ax, tmesh):
            self.ax = ax
            assert isinstance(tmesh, TriMesh)
            self.trimesh = tmesh
            self.canvas = ax.figure.canvas
            self.cid = self.canvas.mpl_connect('button_press_event', self)
        def __call__(self, event):
            # Ignore clicks outside axes
            if event.inaxes != self.ax.axes:
                return
            xdif = self.trimesh.x- event.xdata
            ydif = self.trimesh.y- event.ydata
            dist = xdif**2 + ydif**2
            iv = np.argmin(dist)
            tris =  [x[0] for x in self.trimesh.patches_tris[iv]]
            ntri = self.trimesh.ncells
            self.ax.clear()
            props = dict(boxstyle='round', facecolor='wheat')
            self.ax.text(self.trimesh.x[iv], self.trimesh.y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
            for it in tris:
                self.ax.text(self.trimesh.centersx[it], self.trimesh.centersy[it], r'%d' % (it), color='r', fontweight='bold')
            col = ['b', 'r']
            x=[]; y=[]; sx=[]; sy=[]
            print('iv=', iv, 'bnodes', self.trimesh.patches_bnodes[iv])
            for patch in self.trimesh.patches_bnodes[iv]:
                inode = patch[0]
                self.ax.text(self.trimesh.x[inode], self.trimesh.y[inode], '%d - %d' %(patch[1], patch[2]))
                if patch[5] != -1:
                    iv0 = self.trimesh.edges[patch[5], 0]
                    iv1 = self.trimesh.edges[patch[5], 1]
                    xm = 0.5*(self.trimesh.x[iv0] + self.trimesh.x[iv1])
                    ym = 0.5*(self.trimesh.y[iv0] + self.trimesh.y[iv1])
                    self.ax.text(xm, ym, '%d' %(patch[5]), color='g')
                5 * (self.trimesh.edges[patch[5], 0] + self.trimesh.edges[patch[5], 1])
                x.append(self.trimesh.x[inode])
                y.append(self.trimesh.y[inode])
                normal = 0.5*( patch[3]*self.trimesh.normals[patch[1]] + patch[4]*self.trimesh.normals[patch[2]])
                sx.append(normal[0])
                sy.append(normal[1])
            self.ax.quiver(x, y, sx, sy, headwidth=5, scale=1., units='xy', color='y')
            self.ax.triplot(self.trimesh, 'k--')
            self.trimesh._preparePlot(self.ax, 'Patches', setlimits=True)
            self.canvas.draw()
    class PatchPlotterNodes(object):
        def __init__(self, ax, tmesh):
            self.ax = ax
            assert isinstance(tmesh, TriMesh)
            self.trimesh = tmesh
            self.canvas = ax.figure.canvas
            self.cid = self.canvas.mpl_connect('button_press_event', self)
        def __call__(self, event):
            # Ignore clicks outside axes
            if event.inaxes != self.ax.axes:
                return
            xdif = self.trimesh.x- event.xdata
            ydif = self.trimesh.y- event.ydata
            dist = xdif**2 + ydif**2
            iv = np.argmin(dist)
            print('iv', iv, 'patch', self.trimesh.patchinfo[iv])
            tris =  [x[0] for x in self.trimesh.patches_tris[iv]]
            ntri = self.trimesh.ncells
            self.ax.clear()
            props = dict(boxstyle='round', facecolor='wheat')
            self.ax.text(self.trimesh.x[iv], self.trimesh.y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
            for patch in self.trimesh.patches_bnodes[iv]:
                inode = patch[0]
                self.ax.text(self.trimesh.x[inode], self.trimesh.y[inode], '%d' %(inode), fontweight='bold', bbox=props, color='b')
            self.ax.triplot(self.trimesh, 'k--')
            self.trimesh._preparePlot(self.ax, 'Patches', setlimits=True)
            self.canvas.draw()
    def plotNodePatches(self, plt):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
        self.plotVerticesAndCells(ax1, False)
        self.plotEdges(ax2, True)
        self._preparePlot(ax3, 'Patches Edges', setlimits=False)
        self._preparePlot(ax4, 'Patches Nodes', setlimits=False)
        patchplotteredges = self.PatchPlotterEdges(ax3, self)
        patchplotternodes = self.PatchPlotterNodes(ax4, self)
        plt.show()
    def plotDownwindEdgeInfo(self, downwindinfo, downwindinfoedge):
        nplots = 4
        nrows = (nplots+1)/2
        ncols = min(2,nplots)
        fig = plt.figure(figsize=(ncols*3,nrows*3))
        iplot = 1
        ax = fig.add_subplot(nrows, ncols, iplot)
        self.plotVerticesAndCells(ax, plotlocalNumbering = True)
        iplot += 1
        ax = fig.add_subplot(nrows, ncols, iplot)
        self._preparePlot(ax, 'Edges')
        self.plotVerticesAndCells(ax, False)
        self.plotEdges(ax, True)
        iplot += 1
        ax = fig.add_subplot(nrows, ncols, iplot)
        self._preparePlot(ax, 'Downwind points')
        self.plotVertices(ax)
        for it in range(len(self.triangles)):
            ax.text(self.centersx[it], self.centersy[it], 'X', color='r', fontweight='bold')
            i0 = downwindinfo[0][it, 0]
            i1 = downwindinfo[0][it, 1]
            s = downwindinfo[1][it, 0]
            p = downwindinfo[1][it, 1]
            xdown = p * self.x[i0] + (1 - p) * self.x[i1]
            ydown = p * self.y[i0] + (1 - p) * self.y[i1]
            l = np.sqrt((self.centersx[it]-xdown)**2 + (self.centersy[it]-ydown)**2)
            # print 'l, s', l, s
            ax.text(xdown, ydown, 'X', color='b', fontweight='bold')
        iplot += 1
        ax = fig.add_subplot(nrows, ncols, iplot)
        self._preparePlot(ax, 'Downwind edges')
        self.plotVertices(ax)
        for ie in range(self.nedges):
            iv0 = self.edges[ie, 0]
            iv1 = self.edges[ie, 1]
            pindices, pcoefs = downwindinfoedge[0][ie], downwindinfoedge[1][ie]
            p = pcoefs[0][0]
            w = pcoefs[0][1]
            # if np.all( pindices==-1): continue
            xe = [ self.x[iv0], self.x[iv1] ]
            ye = [ self.y[iv0], self.y[iv1] ]
            plt.plot( xe, ye, color='r')
            plt.text(np.mean(xe), np.mean(ye), r"p=%g" %p)
            # print 'ie', ie, 'iv0, iv1', iv0, iv1, 'p', p
        plt.show()

# ------------------------------------- #

if __name__ == '__main__':
    tmesh1 = TriMesh(hmean=0.4)
    tmesh = TriMeshWithNodePatches(tmesh1)
    tmesh.testNodePatches()
    tmesh.testNodePatchesOld()
    tmesh.plotNodePatches(plt)