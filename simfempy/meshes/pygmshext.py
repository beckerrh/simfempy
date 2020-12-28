import numpy as np
import meshio
import subprocess
from . import simplexmesh

#----------------------------------------------------------------#
def gmshRefine(mesh):
    # TODO: gmsh refine
    filenamemsh = "coarse.msh"
    filenamemshref = "fine.msh"
    mesh.write(filename=filenamemsh)
    cmd = ['gmsh']
    cmd += ["-refine"]
    cmd += ["{}".format(filenamemsh)]
    cmd += ["-o"]
    cmd += ["{}".format(filenamemshref)]
    # cmd += ["-format"]
    # cmd += ["{}".format("msh2")]
    p = subprocess.Popen(cmd, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    # if stderr != "":
    #     print("cmd=", cmd)
    #     print("cmd=", ' '.join(cmd))
    #     print("stderr=", stderr)
    #     raise RuntimeError('Gmsh exited with error (return code %d).' % p.returncode)
    mesh = meshio.read(filenamemshref)
    # print("mesh.cells", mesh.cells.keys())
    # print("mesh.cell_data", mesh.cell_data.keys())
    # data =  mesh.points, mesh.cells, mesh.point_data, mesh.cell_data, mesh.field_data
    return simplexmesh.SimplexMesh(mesh=mesh)


#----------------------------------------------------------------#
def add_circle(
        geom,
        x0,
        radius,
        spacing=None,
        lcars=None,
        h=None,
        R=None,
        compound=False,
        num_sections=3,
        holes=None,
        make_surface=True,
):
    """Add circle in the :math:`x`-:math:`y`-plane.
    """
    if holes is None:
        holes = []
    else:
        assert make_surface

    # Define points that make the circle (midpoint and the four cardinal
    # directions).
    if spacing is None:
        spacing = np.arange(num_sections)/ num_sections
    # print("spacing",spacing)
    X = np.zeros((num_sections + 1, len(x0)))
    if num_sections == 4:
        # For accuracy, the points are provided explicitly.
        X[1:, [0, 1]] = np.array(
            [[radius, 0.0], [0.0, radius], [-radius, 0.0], [0.0, -radius]]
        )
    else:
        X[1:, [0, 1]] = np.array(
            [
                [
                    # radius * np.cos(2 * np.pi * k / num_sections),
                    # radius * np.sin(2 * np.pi * k / num_sections),
                    radius * np.cos(2 * np.pi * spacing[k]),
                    radius * np.sin(2 * np.pi * spacing[k]),
                ]
                for k in range(num_sections)
            ]
        )

    if R is not None:
        assert np.allclose(
            abs(np.linalg.eigvals(R)), np.ones(X.shape[1])
        ), "The transformation matrix doesn't preserve circles; at least one eigenvalue lies off the unit circle."
        X = np.dot(X, R.T)

    X += x0

    # Add Gmsh Points.
    p = [geom.add_point(x, lcar=lcars[i]) for i,x in enumerate(X)]

    # Define the circle arcs.
    arcs = [geom.add_circle_arc(p[k], p[0], p[k + 1]) for k in range(1, len(p) - 1)]
    arcs.append(geom.add_circle_arc(p[-1], p[0], p[1]))

    if compound:
        arcs = [geom.add_compound_line(arcs)]

    line_loop = geom.add_line_loop(arcs)

    if make_surface:
        plane_surface = geom.add_plane_surface(line_loop, holes)
    else:
        plane_surface = None

    class Circle(object):
        def __init__(
                self,
                x0,
                radius,
                R,
                compound,
                num_sections,
                holes,
                line_loop,
                plane_surface,
                lcar=None,
        ):
            self.x0 = x0
            self.radius = radius
            self.lcar = lcar
            self.R = R
            self.compound = compound
            self.num_sections = num_sections
            self.holes = holes
            self.line_loop = line_loop
            self.plane_surface = plane_surface
            return

    return Circle(
        x0,
        radius,
        R,
        compound,
        num_sections,
        holes,
        line_loop,
        plane_surface,
    )


#----------------------------------------------------------------#
# def add_holes(geom, x0, x1, **kwargs):
#     h = kwargs.pop('h')
#     hhole = kwargs.pop('hhole')
#     nholes = kwargs.pop('nholes')
#     holesize = kwargs.pop('holesize')
#     if 'hole_labels' in kwargs: hole_labels = kwargs.pop('hole_labels')
#     else: hole_labels = None
#     if 'make_surface' in kwargs: make_surface = kwargs.pop('make_surface')
#     else: make_surface = True
#     spacesize = (x1-x0-nholes*holesize)/(nholes+1)
#     if spacesize < 0.1*holesize:
#         maxsize = (x1-x0)/(nholes*1.1 - 0.1)
#         raise ValueError("holes too big (max={})".format(maxsize))
#     pos = np.empty(2*nholes)
#     pos[0] = spacesize
#     pos[1] = pos[0] + holesize
#     for i in range(1,nholes):
#         pos[2*i] = pos[2*i-1] + spacesize
#         pos[2*i+1] = pos[2*i] + holesize
#     xholes = []
#     for i in range(nholes):
#         xa, xb = x0+pos[2*i], x0+pos[2*i+1]
#         for j in range(nholes):
#             ya, yb = x0+pos[2*j], x0+pos[2*j+1]
#             xholes.append([[xa, ya, 0], [xb, ya, 0], [xb, yb, 0], [xa, yb, 0]])
#
#     holes = []
#     hole_labels = np.arange(200, 200 + len(xholes), dtype=int)
#     for xhole, hole_label in zip(xholes, hole_labels):
#         holes.append(geom.add_polygon(X=xhole, lcar=hhole))
#         xarrm = np.mean(np.array(xhole), axis=0)
#         add_point_in_surface(geom, holes[-1].surface, xarrm, lcar=h)
#         geom.add_physical(holes[-1].surface, label=int(hole_label))
#     return holes, hole_labels
#----------------------------------------------------------------#
def add_holesnew(geom, **kwargs):
    h = kwargs.pop('h')
    hhole = kwargs.pop('hhole')
    x0 = kwargs.pop('x0')
    x1 = kwargs.pop('x1')
    y0 = kwargs.pop('y0')
    y1 = kwargs.pop('y1')
    nholesx = kwargs.pop('nholesx')
    nholesy = kwargs.pop('nholesy')
    if 'holesizex' in kwargs: holesizex = kwargs.pop('holesizex')
    else: holesizex=None
    if 'holesizey' in kwargs: holesizey = kwargs.pop('holesizey')
    else: holesizey=None
    if 'hole_labels' in kwargs: hole_labels = kwargs.pop('hole_labels')
    else: hole_labels = None
    if 'make_surface' in kwargs: make_surface = kwargs.pop('make_surface')
    else: make_surface = True
    holesizexmax = (x1 - x0) / (nholesx * 1.1 - 0.1)
    if holesizex is None: holesizex = holesizexmax*0.99
    if nholesx>1:
        spacesizex = (x1-x0-nholesx*holesizex)/(nholesx-1)
        if spacesizex < 0.1*holesizex:
            raise ValueError("holesizex({}) too big (max={})".format(holesizex,holesizexmax))
    holesizeymax = (y1 - y0) / (nholesy * 1.1 - 0.1)
    if holesizey is None: holesizey = holesizeymax*0.99
    if nholesy>1:
        spacesizey = (y1-y0-nholesy*holesizey)/(nholesy-1)
        if spacesizey < 0.1*holesizey:
            raise ValueError("holesizey({}) too big (max={})".format(holesizey,holesizeymax))
    posx = np.empty(2*nholesx)
    posx[0] = x0
    posx[1] = posx[0] + holesizex
    for i in range(1,nholesx):
        posx[2*i] = posx[2*i-1] + spacesizex
        posx[2*i+1] = posx[2*i] + holesizex
    posy = np.empty(2*nholesy)
    posy[0] = y0
    posy[1] = posy[0] + holesizey
    for i in range(1,nholesy):
        posy[2*i] = posy[2*i-1] + spacesizey
        posy[2*i+1] = posy[2*i] + holesizey

    xholes = []
    for i in range(nholesx):
        xa, xb = posx[2*i], posx[2*i+1]
        for j in range(nholesy):
            ya, yb = posy[2*j], posy[2*j+1]
            xholes.append([[xa, ya, 0], [xb, ya, 0], [xb, yb, 0], [xa, yb, 0]])

    holes = []
    hole_labels = np.arange(200, 200 + len(xholes), dtype=int)
    for xhole, hole_label in zip(xholes, hole_labels):
        holes.append(geom.add_polygon(X=xhole, lcar=hhole))
        xarrm = np.mean(np.array(xhole), axis=0)
        add_point_in_surface(geom, holes[-1].surface, xarrm, lcar=h)
        geom.add_physical(holes[-1].surface, label=int(hole_label))
    return holes, hole_labels

#----------------------------------------------------------------#
def add_polygon(geom, X, lcar=None, holes=None, make_surface=True):
    assert len(X) == len(lcar)
    if holes is None:
        holes = []
    else:
        assert make_surface

    # Create points.
    p = [geom.add_point(x, lcar=l) for x,l in zip(X,lcar)]
    # Create lines
    lines = [geom.add_line(p[k], p[k + 1]) for k in range(len(p) - 1)]
    lines.append(geom.add_line(p[-1], p[0]))
    ll = geom.add_line_loop((lines))
    surface = geom.add_plane_surface(ll, holes) if make_surface else None

    print("ll", ll)
    print("surface", surface)
    return geom.Polygon(ll, surface)
    return geom.Polygon(ll, surface, lcar=lcar)

#----------------------------------------------------------------#
def add_point_in_surface(geom, surf, X, lcar, label=None):
    p = geom.add_point(X, lcar=lcar)
    geom.add_raw_code("Point {{{}}} In Surface {{{}}};".format(p.id, surf.id))
    if label:
        geom.add_physical_point(p, label=label)
