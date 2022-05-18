'''
Classes for NVB-like refinement routines. Currently implemented:

    * NVB1: NVB, where every marked element gets bisected once, thus bisecting
    the refinement edge.

    * NVB3: NVB, where every marked element is bisected twice, thus bisecting
    all edges. This is the standard algorithm; hence, it is also aliased as
    NVB.

    * NVB5: NVB, where every marked element gets bisected so much that an
    interior node appears.

    * NVBEdge: NVB, where not elements but edges are marked for refinement.

    * RGB: Red-Green-Blue refinement.

Apart from RGB (where the refinement edge is the longest edge in every
triangle), the refinement edge is taken to be the edge between the nodes with
indices elements[i,0] and elements[i,1].
'''

import numpy as np
from mesh import *
from topology import computeTopology


################################################################################


def getRefinementRoutine(mesh, method):
       methodsDict = { \
               'NVB': NVB,
               'NVB1': NVB1,
               'NVB3': NVB3,
               'NVB5': NVB5,
               'NVBEdge': NVBEdge,
               'RGB': RGB}
       if method in methodsDict.keys():
           return methodsDict[method](mesh)
       else:
           raise RuntimeError(f'No refinement method named {method}.')


################################################################################

class NVB():
    def __init__(self, mesh):
        self._mesh = mesh
        self._edge2nodes, self._element2edges = computeTopology(mesh)

    def prepareRefinementData(self, markedElements):
        refineData = self._meshClosure(markedElements)
        refineData = self._groupElements(refineData)
        return refineData
    
    def refineMesh(self, refineData):
        newCoordinates = self._createNewCoordinates(refineData)
        newElements = self._createNewElements(refineData)
        return (newCoordinates, newElements)

    def _meshClosure(self, markedElements):
        # initial marking
        markedEdges = np.zeros(self._edge2nodes.shape[0], dtype=bool)
        markedEdges[self._element2edges[markedElements,:]] = True
        # actual closure
        hasHangingNodes = np.array([True])
        while np.count_nonzero(hasHangingNodes) > 0:
            edge = markedEdges[self._element2edges]
            hasHangingNodes = np.logical_and(np.logical_not(edge[:,0]), np.logical_or(edge[:,1], edge[:,2]))
            markedEdges[self._element2edges[hasHangingNodes,0]] = True
        return {'markedEdges': markedEdges}
    
    def _groupElements(self, refineData):
        # group elements according to their refinement pattern
        markedEdges = refineData['markedEdges']
        edge = markedEdges[self._element2edges]
        refinementGroup = []
        refinementGroup.append(np.where(np.all( edge == [True,False,False], axis=1 ))[0]) # bisec(1): newest vertex bisection of 1st edge
        refinementGroup.append(np.where(np.all( edge == [True,True,False], axis=1 ))[0])  # bisec(2): newest vertex bisection of 1st and 2nd edge
        refinementGroup.append(np.where(np.all( edge == [True,False,True], axis=1 ))[0])  # bisec(2): newest vertex bisection of 1st and 3rd edge
        refinementGroup.append(np.where(np.all( edge == [True,True,True], axis=1 ))[0])   # bisec(3): newest vertex bisection of all edges
        refineData['markedEdges'], = np.where(markedEdges)
        
        # obtain new indices for parent elements
        nDescendants = [2,3,3,4]
        idx = np.ones(self._mesh.nElements)
        for k in range(len(refinementGroup)):
            idx[refinementGroup[k]] = nDescendants[k]
        parent2newElement = np.cumsum(np.hstack([0, idx])).astype(int)
        refineData.update({'refinementGroup': refinementGroup, 'nDescendants': nDescendants, 'parent2newElement': parent2newElement})
        return refineData
    
    def _createNewCoordinates(self, refineData):
        markedEdges = refineData['markedEdges']
        return np.vstack([self._mesh.nodes, \
                edgeCoordinatesFromBarycentric(self._mesh.nodes, self._edge2nodes, np.array([1,1])/2, markedEdges)])
    
    def _createNewElements(self, refineData):
        # get numbering of new nodes per element
        markedEdges = refineData['markedEdges']
        edge2newNodes = np.zeros(self._edge2nodes.shape[0])
        edge2newNodes[markedEdges] = self._mesh.nNodes + np.arange(len(markedEdges))
        newNodes = edge2newNodes[self._element2edges]
        
        # generate new elements
        refinementGroup = refineData['refinementGroup']
        parent2newElement = refineData['parent2newElement']
        nDescendants = refineData['nDescendants']
        newElements = np.zeros((parent2newElement[-1], 3))
        newElements[parent2newElement[:-1],:] = self._mesh.elements
        for k in range(len(refinementGroup)):
            parents = parent2newElement[refinementGroup[k]]
            if not parents.size == 0:
                children = np.add.outer(np.arange(nDescendants[k]), parents).flatten()
                newElements[children,:] = getattr(self, f'_refine{k}')(self._mesh.elements, newNodes, refinementGroup[k])
        return newElements

    ## static refinement functions
    # bisec(1): newest vertex bisection of 1st edge
    @staticmethod
    def _refine0(elements, newNodes, idx):
        return np.vstack([np.column_stack([elements[idx,2],elements[idx,0],newNodes[idx,0]]), \
                          np.column_stack([elements[idx,1],elements[idx,2],newNodes[idx,0]])])

    # bisec(2): newest vertex bisection of 1st and 2nd edge
    @staticmethod
    def _refine1(elements, newNodes, idx):
        return np.vstack([np.column_stack([elements[idx,2],elements[idx,0],newNodes[idx,0]]), \
                          np.column_stack([newNodes[idx,0],elements[idx,1],newNodes[idx,1]]), \
                          np.column_stack([elements[idx,2],newNodes[idx,0],newNodes[idx,1]])])

    # bisec(2): newest vertex bisection of 1st and 3rd edge
    @staticmethod
    def _refine2(elements, newNodes, idx):
        return np.vstack([np.column_stack([newNodes[idx,0],elements[idx,2],newNodes[idx,2]]), \
                          np.column_stack([elements[idx,0],newNodes[idx,0],newNodes[idx,2]]), \
                          np.column_stack([elements[idx,1],elements[idx,2],newNodes[idx,0]])])

    # bisec(3): newest vertex bisection of all edges
    @staticmethod
    def _refine3(elements, newNodes, idx):
        return np.vstack([np.column_stack([newNodes[idx,0],elements[idx,2],newNodes[idx,2]]), \
                          np.column_stack([elements[idx,0],newNodes[idx,0],newNodes[idx,2]]), \
                          np.column_stack([newNodes[idx,0],elements[idx,1],newNodes[idx,1]]), \
                          np.column_stack([elements[idx,2],newNodes[idx,0],newNodes[idx,1]])])


################################################################################


class NVB1(NVB):
    # Class realizing the NVB1 refinement rule. Only the closure step differs
    # from generic NVB.

    def __init__(self, mesh):
        super().__init__(mesh)

    def _meshClosure(self, markedElements):
        # initial marking
        markedEdges = np.zeros(self._edge2nodes.shape[0], dtype=bool)
        markedEdges[self._element2edges[markedElements,0]] = True
        # actual closure
        hasHangingNodes = np.array([True])
        while np.count_nonzero(hasHangingNodes) > 0:
            edge = markedEdges[self._element2edges]
            hasHangingNodes = np.logical_and(np.logical_not(edge[:,0]), np.logical_or(edge[:,1], edge[:,2]))
            markedEdges[self._element2edges[hasHangingNodes,0]] = True
        return {'markedEdges': markedEdges}


################################################################################


class NVB3(NVB):
    # Class realizing the NVB3 refinement rule (basically just an alias for NVB).

    def __init__(self, mesh):
        super().__init__(mesh)

################################################################################


class NVB5(NVB):
    # Class realizing the NVB5 refinement rule. This differs from generic NVB
    # by the refinement of the marked elements, where an interior node is
    # introduced.

    def __init__(self, mesh):
        super().__init__(mesh)

    def _meshClosure(self, markedElements):
        refineData = super()._meshClosure(markedElements)
        refineData.update({'markedElements': markedElements})
        return refineData
    
    def _groupElements(self, refineData):
        # group elements according to their refinement pattern
        markedEdges = refineData['markedEdges']
        markedElements = refineData['markedElements']
        edge = markedEdges[self._element2edges]
        refinementGroup = []
        refinementGroup.append(np.where(np.all( edge == [True,False,False], axis=1 ))[0])
        refinementGroup.append(np.where(np.all( edge == [True,True,False], axis=1 ))[0])
        refinementGroup.append(np.where(np.all( edge == [True,False,True], axis=1 ))[0])
        refinementGroup.append(np.all( edge == [True,True,True], axis=1 ))
        refinementGroup[3][markedElements] = False
        refinementGroup[3], = np.where(refinementGroup[3])

        # obtain new indices for parent elements
        nDescendants = [2,3,3,4]
        idx = np.ones(self._mesh.nElements)
        for k in range(len(refinementGroup)):
            idx[refinementGroup[k]] = nDescendants[k]
        idx[markedElements] = 6 # treat marked elements separately
        parent2newElement = np.cumsum(np.hstack([0, idx])).astype(int)
        refineData['markedEdges'], = np.where(markedEdges)
        refineData.update({'refinementGroup': refinementGroup, 'nDescendants': nDescendants, 'parent2newElement': parent2newElement})
        return refineData
    
    def _createNewCoordinates(self, refineData):
        markedElements = refineData['markedElements']
        return np.vstack([super()._createNewCoordinates(refineData), \
                          coordinatesFromBarycentric(self._mesh.nodes, self._mesh.elements, np.array([1,1,2])/4, markedElements)])
    
    def _createNewElements(self, refineData):
        # get numbering of new nodes per element
        markedEdges = refineData['markedEdges']
        markedElements = refineData['markedElements']
        edge2newNodes = np.zeros(self._edge2nodes.shape[0])
        edge2newNodes[markedEdges] = self._mesh.nNodes + np.arange(len(markedEdges))
        edgeNodes = edge2newNodes[self._element2edges]
        nNewCoordinates = self._mesh.nNodes + len(markedEdges)
        interiorNodes = nNewCoordinates + np.arange(len(markedElements))
        
        newElements = super()._createNewElements(refineData)
        parent2newElement = refineData['parent2newElement']
        children = np.add.outer(np.arange(6), parent2newElement[markedElements]).flatten()
        idx = markedElements
        newElements[children,:] = \
            np.vstack([np.column_stack([edgeNodes[idx,2],edgeNodes[idx,0],interiorNodes]), \
                       np.column_stack([self._mesh.elements[idx,2],edgeNodes[idx,2],interiorNodes]), \
                       np.column_stack([self._mesh.elements[idx,0],edgeNodes[idx,0],edgeNodes[idx,2]]), \
                       np.column_stack([edgeNodes[idx,0],self._mesh.elements[idx,1],edgeNodes[idx,1]]), \
                       np.column_stack([edgeNodes[idx,1],self._mesh.elements[idx,2],interiorNodes]), \
                       np.column_stack([edgeNodes[idx,0],edgeNodes[idx,1],interiorNodes])])
        return newElements


################################################################################


class NVBEdge(NVB):
    # Class realizing an edge driven version of the NVB1 refinement rule. This
    # rule only differs from generic NVB in the closure step.

    def __init__(self, mesh):
        super().__init__(mesh)

    def _meshClosure(self, marked):
        # initial marking
        markedEdges = np.zeros(self._edge2nodes.shape[0], dtype=bool)
        markedEdges[marked] = True
        # actual closure
        hasHangingNodes = np.array([True])
        while np.count_nonzero(hasHangingNodes) > 0:
            edge = markedEdges[self._element2edges]
            hasHangingNodes = np.logical_and(np.logical_not(edge[:,0]), np.logical_or(edge[:,1], edge[:,2]))
            markedEdges[self._element2edges[hasHangingNodes,0]] = True
        return {'markedEdges': markedEdges}


################################################################################


class RGB(NVB):
    # Class realizing the RGB refinement rule. This differs from generic NVB
    # only by the refinement of elements with three marked edges and the choice
    # of the refinement edge (longest edge).

    def __init__(self, mesh):
        super().__init__(mesh)

    def _meshClosure(self, markedElements):
        # sort edges by length (while still retaining the orientation of the element)
        dx = self._mesh.nodes[self._mesh.elements[:,[1,2,0]],0]-self._mesh.nodes[self._mesh.elements,0]
        dy = self._mesh.nodes[self._mesh.elements[:,[1,2,0]],1]-self._mesh.nodes[self._mesh.elements,1]
        maxIdx = np.argmax(dx**2 + dy**2, axis=1)
        element2edgesMax = np.copy(self._element2edges)
        idx, = np.where(maxIdx == 1)
        if not len(idx) == 0:
            element2edgesMax[idx,:] = element2edgesMax[idx,[1,2,0]]
        idx, = np.where(maxIdx == 2)
        if not len(idx) == 0:
            element2edgesMax[idx,:] = element2edgesMax[idx,[2,0,1]]
        # initial marking
        markedEdges = np.zeros(self._edge2nodes.shape[0], dtype=bool)
        markedEdges[element2edgesMax[markedElements,:]] = True
        # actual closure
        hasHangingNodes = np.array([True])
        while np.count_nonzero(hasHangingNodes) > 0:
            edge = markedEdges[element2edgesMax]
            hasHangingNodes = np.logical_and(np.logical_not(edge[:,0]), np.logical_or(edge[:,1], edge[:,2]))
            markedEdges[element2edgesMax[hasHangingNodes,0]] = True
        return {'markedEdges': markedEdges}

    # green = newest vertex bisection of 1st edge
    # blue = newest vertex bisection of 1st and 2nd/3rd edge
    # red refinement: connect all edge midpoints of the triangle (replaces bisec(3))
    @staticmethod
    def _refine3(elements, newNodes, idx):
        return np.vstack([np.column_stack([elements[idx,0],newNodes[idx,0],newNodes[idx,2]]), \
                          np.column_stack([newNodes[idx,0],elements[idx,1],newNodes[idx,1]]), \
                          np.column_stack([newNodes[idx,2],newNodes[idx,1],elements[idx,2]]), \
                          np.column_stack([newNodes[idx,1],newNodes[idx,2],newNodes[idx,0]])])
