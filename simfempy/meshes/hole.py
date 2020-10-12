import numpy as np

def square(geom, x, y, r, lcar, label, make_surface):
    """
    :param x,y,r: position and size of hole
    :param label:
    :param make_surface:
    :param lcar:
    :return: hole
    """
    # add z-component
    z=0
    hcoord = [[x-r, y-r], [x-r, y+r], [x+r, y+r], [x+r, y-r]]
    xhole = np.insert(np.array(hcoord), 2, z, axis=1)
    if make_surface:
        hole = geom.add_polygon(X=xhole, lcar=lcar, make_surface=True)
        geom.add_physical(hole.surface, label=label)
    else:
        hole = geom.add_polygon(X=xhole, lcar=lcar,make_surface=False)
        for j in range(len(hole.lines)): geom.add_physical(hole.lines[j], label=label+j)
    return hole
