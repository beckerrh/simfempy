import numpy as np
import matplotlib.pyplot as plt

def plotmesh(mesh, **kwargs):
    ax = plt
    if 'ax' in kwargs: ax = kwargs.pop('ax')
    x, lines = mesh.points[:, 0], mesh.simplices
    ax.plot(x, np.zeros_like(x), "x-")

def meshWithBoundaries(x, lines, **kwargs):
    ax = plt
    if 'ax' in kwargs: ax = kwargs.pop('ax')
    ax.plot(x, np.zeros_like(x), "x-")

