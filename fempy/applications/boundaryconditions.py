import numpy as np


class BoundaryConditions():
    """
    Information on boundary condotions
    type: dictionary int->srting
    fct: dictionary int->callable
    """

    def __init__(self, bdrylabels=None):
        if bdrylabels is None:
            self.type = {}
            self.fct = {}
        else:
            colors = np.unique(bdrylabels)
            self.type = {color: None for color in colors}
            self.fct = {color: None for color in colors}
    def __str__(self):
        return "types={} fct={}".format(self.type, self.fct)
    def colors(self):
        return self.type.keys()
