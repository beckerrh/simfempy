class BoundaryConditions(object):
    """
    Information on boundary conditions
    type: dictionary int->srting
    fct: dictionary int->callable
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
        return "types={} fct={} param={}".format(self.type, self.fct, self.param)
    def colors(self):
        return self.type.keys()


# class RightHandSides(object):
#     """
#     Information on right-hand sides
#     type: dictionary int->srting
#     fct: dictionary int->callable
#     """
#     def __init__(self, colors=None):
#         if colors is None:
#             self.type = {}
#             self.fct = {}
#             self.param = {}
#         else:
#             self.type = {color: None for color in colors}
#             self.fct = {color: None for color in colors}
#             self.param = {color: None for color in colors}
#     def __repr__(self):
#         return "types={} fct={} param={}".format(self.type, self.fct, self.param)
#     def colors(self):
#         return self.type.keys()

class ProblemData(object):
    """
    Contains all (?) data: boundary conditions and right-hand sides
    """
    def __init__(self, bdrycond=None, rhs=None, postproc=None, ncomp=-1):
        self.ncomp=ncomp
        if bdrycond: self.bdrycond = bdrycond
        else: self.bdrycond = BoundaryConditions()
        if rhs: self.rhs = rhs
        else: self.rhs = None
        self.solexact = None
        self.postproc = postproc

    def __repr__(self):
        return "ncomp={:2d}\nbdrycond={}\nrhs={}\npostproc={}\nsolexact={}".format(self.ncomp, self.bdrycond, self.rhs, self.postproc, self.solexact)
