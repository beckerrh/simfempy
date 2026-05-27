# -*- coding: utf-8 -*-
import numpy as np
try:
    from .mesh_edges import sorted_edge
except ImportError:
    from mesh_edges import sorted_edge

def bisec123(cell, mids):
    """
    Uniform red refinement of one triangle.

    cell = (a,b,c)
    mids must contain the three edge midpoints with keys:
        sorted_edge(a,b), sorted_edge(b,c), sorted_edge(c,a)
    """
    a, b, c = map(int, cell)

    mab = mids[tuple(sorted((a, b)))]
    mbc = mids[tuple(sorted((b, c)))]
    mca = mids[tuple(sorted((c, a)))]

    return [
        [a,   mab, mca],
        [mab, b,   mbc],
        [mca, mbc, c  ],
        [mab, mbc, mca],
    ]
def bisec1(cell, mids):
    return bisec123(cell, mids)

def bisec12(cell, mids):
    return bisec123(cell, mids)

def bisec13(cell, mids):
    return bisec123(cell, mids)

# def bisec1(tri, edge_to_mid):
#     v0, v1, v2 = map(int, tri)
#     m12 = edge_to_mid[sorted_edge(v1, v2)]
#     return bisec1(v0, v1, v2, m12)
# def bisec12(tri, edge_to_mid):
#     v0, v1, v2 = map(int, tri)
#     m12 = edge_to_mid[sorted_edge(v1, v2)]
#     m20 = edge_to_mid[sorted_edge(v2, v0)]
#     return bisec12(v0, v1, v2, m12, m20)
# def bisec13(tri, edge_to_mid):
#     v0, v1, v2 = map(int, tri)
#     m12 = edge_to_mid[sorted_edge(v1, v2)]
#     m01 = edge_to_mid[sorted_edge(v0, v1)]
#     return bisec13(v0, v1, v2, m12, m01)
# def bisec123(tri, edge_to_mid):
#     v0, v1, v2 = map(int, tri)
#     m12 = edge_to_mid[sorted_edge(v1, v2)]
#     m20 = edge_to_mid[sorted_edge(v2, v0)]
#     m01 = edge_to_mid[sorted_edge(v0, v1)]
#     return bisec123(v0, v1, v2, m12, m20, m01)


# def bisec1(a, b, c, m12):
#     return np.array([
#         [c, a, m12],
#         [b, c, m12],
#     ], dtype=int)
#
#
# def bisec12(a, b, c, m12, m23):
#     return np.array([
#         [c,   a,   m12],
#         [m12, b,   m23],
#         [c,   m12, m23],
#     ], dtype=int)
#
#
# def bisec13(a, b, c, m12, m31):
#     return np.array([
#         [m12, c,   m31],
#         [a,   m12, m31],
#         [b,   c,   m12],
#     ], dtype=int)
#
#
# def bisec123(a, b, c, m12, m23, m31):
#     return np.array([
#         [m12, c,   m31],
#         [a,   m12, m31],
#         [m12, b,   m23],
#         [c,   m12, m23],
#     ], dtype=int)