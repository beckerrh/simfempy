import numpy as np


class BoundaryConditions():
    """
    Information on boundary condotions
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
