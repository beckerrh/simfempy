'''
Function that provides the topology of a given Mesh:
    
    * edge2nodes: nEdges x 2
        edge2nodes[i,:] provides the starting and ending coordinate of the i-th
        edge, where the numbering is arbitrary. Edges allways point from the
        node with lower to the one with higher index.

    * element2edges: nElements x 3
        element2edges[i,j] provides the j-th edge of the i-th element, which
        lies between the nodes with indices element[i,j] and element[i,j+1]
        (with 2+1=0).
'''

import numpy as np


def computeTopology(mesh):
    edges = mesh.elements[:,[0,1,2,1,2,0]].reshape((-1,2));           
    edge2nodes, idx = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True);
    element2edges = idx.reshape((-1,3))
    return (edge2nodes, element2edges)
