import numpy as np

# ------------------------------------- #
def unique_all(a):
    """
    https://stackoverflow.com/questions/30003068/get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    """
    a = np.asarray(a)
    ind_s = np.argsort(a)
    a_s = a[ind_s]
    vals, ind_start = np.unique(a_s, return_index=True)
    return vals, np.split(ind_s, ind_start[1:])

def creatdict_unique_all(cl):
    clunique = unique_all(cl)
    clinv = {}
    for color, ind in zip(clunique[0], clunique[1]):
        clinv[color] = ind
    return clinv


# ------------------------------------- #
if __name__ == '__main__':
    a = np.array([1, 7, 3, 1, 6, 7, 1, 6, 1, 7])
    vals, inds = unique_all(a)
    print("a", a)
    print("vals", vals)
    print("inds", inds)
    for ind, val in zip(inds,vals):
        print("val", val, "ind", ind, "val[ind]", a[ind])

