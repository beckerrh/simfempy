'''
Define a mesh class with read-only properties suitable for plotting with matplotlib.
This file also contains functions for converting barycentric coordinates into
cartesian ones on edges & elements.
'''

import numpy as np


class Mesh:
    def __init__(self, nodes, elements):
        self._nodes = np.array(nodes)
        self._elements = np.array(elements)

    @property
    def nodes(self):
        return self._nodes

    @property
    def elements(self):
        return self._elements
    
    @property
    def nNodes(self):
        return self._nodes.shape[0]

    @property
    def nElements(self):
        return self._elements.shape[0]

    @classmethod
    def loadFromFolder(cls, name):
        with open(f'./geometries/{name}/nodes.dat', 'r') as f:
            nodes = np.genfromtxt(f, delimiter=',')
        with open(f'./geometries/{name}/elements.dat', 'r') as f:
            elements = np.genfromtxt(f, delimiter=',').astype(int)
        return cls(nodes, elements)


###


# Given barycentric coordinates [b1,b2,b3] and elements with nodes [z1,z2,z3],
# compute the cartesian coordinate z1*b1+z2*b2+z3*b3 on certain elements.
def coordinatesFromBarycentric(coordinates, elements, bary, idx):
    return np.vstack( \
        [np.dot(coordinates[elements[idx,:],0].reshape((-1,3)), bary.T), \
         np.dot(coordinates[elements[idx,:],1].reshape((-1,3)), bary.T)] \
        ).T


# Given barycentric coordinates [b1,b2] and edges with nodes [z1,z2], compute
# the cartesian coordinate z1*b1+z2*b2 on certain edges.
def edgeCoordinatesFromBarycentric(coordinates, edges, bary, idx):
    return np.vstack( \
        [np.dot(coordinates[edges[idx,:],0].reshape((-1,2)), bary.T), \
         np.dot(coordinates[edges[idx,:],1].reshape((-1,2)), bary.T)] \
        ).T
