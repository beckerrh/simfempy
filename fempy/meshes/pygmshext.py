
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

    return geom.Polygon(ll, surface)
    return geom.Polygon(ll, surface, lcar=lcar)

#----------------------------------------------------------------#
def add_point_in_surface(geom, surf, X, lcar, label=None):
    p = geom.add_point(X, lcar=lcar)
    geom.add_raw_code("Point {{{}}} In Surface {{{}}};".format(p.id, surf.id))
    if label:
        geom.add_physical_point(p, label=label)
