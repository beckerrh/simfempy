import numpy as np

def square(geom, xc, yc, r, mesh_size, label, make_surface=False):
    """
    :param xc,yc,r: position and size of hole
    :param label:
    :param make_surface:
    :param lcar:
    :return: hole
    """
    # add z-component
    z=0
    hcoord = [[xc-r, yc-r], [xc-r, yc+r], [xc+r, yc+r], [xc+r, yc-r]]
    xhole = np.insert(np.array(hcoord), 2, z, axis=1)
    if make_surface:
        # assert isinstance(label, int)
        hole = geom.add_polygon(points=xhole, mesh_size=mesh_size, make_surface=True)
        geom.add_physical(hole.surface, label=str(label))
        for j in range(len(hole.lines)): geom.add_physical(hole.lines[j], label=f"{10*int(label)+j}")
    else:
        hole = geom.add_polygon(points=xhole, mesh_size=mesh_size,make_surface=False)
        for j in range(len(hole.lines)): geom.add_physical(hole.lines[j], label=f"{int(label)+j}")
    return hole
