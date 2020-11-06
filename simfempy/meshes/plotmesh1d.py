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

def meshWithData(**kwargs):
    ax = plt
    if 'ax' in kwargs: ax = kwargs.pop('ax')
    x = kwargs.pop('x')
    title, suptitle = None, None
    if 'title' in kwargs: title = kwargs['title']
    if 'suptitle' in kwargs: suptitle = kwargs['suptitle']
    point_data, cell_data = None, None
    if 'data' in kwargs:
        point_data = kwargs['data']['point']
        cell_data = kwargs['data']['cell']
    nplots=0
    if point_data: nplots += len(point_data)
    if cell_data: nplots += len(cell_data)
    ax.clf()
    ax.subplot(1, 2, 1)
    if suptitle: plt.gcf().suptitle(suptitle)
    # if title: plt.gcf().set_title(title)
    # aspect = (np.max(x)-np.mean(x))/(np.max(y)-np.mean(y))
    count=0
    if point_data:
        for pdn, pd in point_data.items():
            # print("print", pdn)
            assert x.shape == pd.shape
            ax.plot(x, pd, label=pdn)
        ax.legend()
    ax.subplot(1, 2, 2)
    if cell_data:
        xc =  0.5*(x[:-1] + x[1:])
        for cdn, cd in cell_data.items():
            assert xc.shape == cd.shape
            ax.plot(xc, cd, label=cdn)
        ax.legend()


