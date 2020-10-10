class BdryData(object):
    """
    Information for boundary conditions
    """
    def __init__(self):
        self.bsaved = {}
        self.Asaved = {}

    def __repr__(self):
        return ", ".join("'{}': {}".format(attr, value) for attr, value in self.__dict__.items())