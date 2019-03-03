class BoundaryConditions(object):
    """
    Information on boundary conditions
    type: dictionary int->srting
    fct: dictionary int->callable
    """
    # def __init__(self, colors=None):
    #     if colors is None:
    #         self.type = {}
    #         self.fct = {}
    #         self.param = {}
    #     else:
    #         self.type = {color: None for color in colors}
    #         self.fct = {color: None for color in colors}
    #         self.param = {color: None for color in colors}
    def __init__(self):
        self.type = {}
        self.fct = {}
        self.param = {}

    def __repr__(self):
        return "types={} fct={} param={}".format(self.type, self.fct, self.param)

    def hasExactSolution(self):
        return hasattr(self, 'fctexact')

    def colors(self):
        return self.type.keys()

    def types(self):
        return self.type.values()

    def colorsOfType(self, type):
        colors = []
        for color, typeofcolor in self.type.items():
            if typeofcolor == type: colors.append(color)
        return colors

    def check(self, colors):
        colors = set(colors)
        typecolors = set(self.type.keys())
        withouttype = colors.difference(typecolors)
        if withouttype:
            raise ValueError("without types: {}".format(withouttype))
        typenotfound = typecolors.difference(colors)
        if typenotfound:
            raise ValueError("unused types: {}".format(typenotfound))


class ProblemData(object):
    """
    Contains all (?) data: boundary conditions and right-hand sides
    """
    def __init__(self, bdrycond=None, rhs=None, rhscell=None, rhspoint = None, postproc=None, ncomp=-1):
        self.ncomp=ncomp
        if bdrycond: self.bdrycond = bdrycond
        else: self.bdrycond = BoundaryConditions()
        self.rhs = None
        self.rhscell = rhscell
        self.rhspoint = rhspoint
        self.solexact = None
        self.postproc = postproc

    def __repr__(self):
        # sf = ""
        # sc = []
        # for a in [a for a in dir(self) if not a.startswith('__')]:
        #     sf += "{}={{}}\\n".format(a)
        #     sc.append("self."+a)
        # s = sf.format(*sc)
        # print("sf",sf)
        # print("sc",sc)
        # print("s",s)
        # s = eval(s)
        # return s
        return "ncomp={:2d}\nbdrycond={}\nrhs={}\nrhspoint={}\nrhscell={}\npostproc={}\nsolexact={}".format(self.ncomp, self.bdrycond, self.rhs, self.rhspoint, self.rhscell, self.postproc, self.solexact)

    def clear(self):
        """
        keeps the boundary condition types and parameters !
        """
        self.rhs = None
        self.rhscell = None
        self.rhspoint = None
        self.solexact = None
        self.postproc = None
        for color in self.bdrycond.fct:
            self.bdrycond.fct[color] = None
