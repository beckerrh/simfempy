def _check1setinother_(set1, set2, name1="set1", name2="set2"):
    notin2 = set1.difference(set2)
    if notin2:
        raise KeyError(f"in {name1} but not in {name2}: {notin2}")
def _check2setsequal_(set1, set2, name1="set1", name2="set2"):
    _check1setinother_(set1, set2, name1, name2)
    _check1setinother_(set2, set1, name2, name1)

# ---------------------------------------------------------------- #
class BoundaryConditions(object):
    """
    Information on boundary conditions
    type: dictionary int->string
    fct: dictionary int->callable
    param: dictionary int->float
    """
    def __init__(self, colors=None):
        if colors is None:
            self.type = {}
            self.fct = {}
            self.param = {}
        else:
            self.type = {color: None for color in colors}
            self.fct = {color: None for color in colors}
            self.param = {color: None for color in colors}

    def __repr__(self):
        return f"types={self.type}\nfct={self.fct}\nparam={self.param}"

    def hasExactSolution(self):
        return hasattr(self, 'fctexact')
    def colors(self):
        return self.type.keys()
    def types(self):
        return self.type.values()
    def set(self, type, colors):
        for color in colors:
            self.type[color] = type
    def colorsOfType(self, type):
        colors = []
        for color, typeofcolor in self.type.items():
            if typeofcolor == type: colors.append(color)
        return colors
    def check(self, colors):
        colors = set(colors)
        typecolors = set(self.type.keys())
        _check2setsequal_(colors, typecolors, "mesh colors", "types")


# ---------------------------------------------------------------- #
class PostProcess(object):
    """
    Information on postprocess
    type: dictionary string(name)->string
    color: dictionary string(name)->list(int)
    """
    def __init__(self):
        self.type = {}
        self.color = {}
    def __repr__(self):
        return f"types={self.type}\ncolor={self.color}"
    def colors(self, name):
        return self.color[name]
    def check(self, colors):
        if self.type.keys() != self.color.keys():
            raise KeyError(f"postprocess keys differ: type={self.type.keys()}, color={self.color.keys()}")
        colors = set(colors)
        usedcolors = set().union(*self.color.values())
        _check1setinother_(usedcolors, colors, "used", "mesh colors")

# ---------------------------------------------------------------- #
class ProblemData(object):
    """
    Contains all (?) data:
    - boundary conditions
    - right-hand sides
    - exact solution (if ever)
    - postprocess
    - datafct: dictionary string(name)->fct
    - dataparam: dictionary string(name)->float
    """
    def __init__(self, bdrycond=None, rhs=None, rhscell=None, rhspoint = None, postproc=None, ncomp=-1):
        self.ncomp=ncomp
        if bdrycond is None: self.bdrycond = BoundaryConditions()
        else: self.bdrycond = bdrycond
        if postproc is None: self.postproc = PostProcess()
        else: self.postproc = postproc
        self.rhs = None
        self.rhscell = rhscell
        self.rhspoint = rhspoint
        self.solexact = None
        self.datafct = {}
        self.dataparam = {}

    def _split2string(self, string):
        return '\n\t\t'+'\n\t\t'.join(str(string).split('\n'))

    def __repr__(self):
        repr = f"\n{self.__class__}:"
        repr += f"\n\tncomp = {self.ncomp:2d}"
        repr += f"\n\tbdrycond:{self._split2string(self.bdrycond)}"
        if self.postproc: repr += f"\n\tpostproc:{self._split2string(self.postproc)}"
        if self.rhs: repr += f"\n\trhs={self.rhs}"
        if self.rhscell: repr += f"\n\trhscell={self.rhscell}"
        if self.rhspoint: repr += f"\n\trhspoint={self.rhspoint}"
        if self.solexact: repr += f"\n\tsolexact={self.solexact}"
        if self.datafct: repr += f"\n\tdatafct={self.datafct}"
        if self.dataparam: repr += f"\n\tdataparam={self.dataparam}"
        return repr

    def check(self, mesh):
        colors = mesh.bdrylabels.keys()
        self.bdrycond.check(colors)
        if self.postproc: self.postproc.check(colors)

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
