def dictproduct(paramsdicts):
    import itertools
    paramslist = [[(name,param) for param in params] for name,params in paramsdicts.items()]
    return [{p[0]: p[1] for p in params} for params in itertools.product(*paramslist)]
