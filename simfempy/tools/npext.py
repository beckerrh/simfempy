import numpy as np

# ------------------------------------- #
def positionin(x,y):
    # assert len(x.shape) ==2
    assert len(y.shape) ==2
    assert x.shape[0]==y.shape[0]
    pos = np.empty_like(x)
    if len(x.shape) == 1:
        # ne respecte pas l'ordre !!
        for i in range(x.shape[0]):
            pos[i]= np.nonzero(y[i]==x[i])[0]
            # print(f"{x[i]=} {y[i]=} {pos[i]=}")
        return pos
    for i in range(x.shape[0]):
        index = np.argsort(y[i])
        ysorted = y[i][index]
        sorted_index = np.searchsorted(ysorted, x[i])
        pos[i] = np.take(index, sorted_index, mode="clip")
        # print(f"{x[i]=}")
        # print(f"{y[i]=}")
        # print(f"{pos[i]=}")
    return pos

# ------------------------------------- #
def positionnotin(x,y):
    assert len(y.shape) ==2
    assert len(x.shape) == 1
    assert x.shape[0]==y.shape[0]
    pos = np.empty(shape=(x.shape[0],y.shape[1]-1), dtype=y.dtype)
    for i in range(x.shape[0]):
        ind = np.where(y[i]!=x[i])
        pos[i] = y[i,ind]
        # pos[i] = np.nonzero(y[i]!=x[i])[0]
        # print(f"{x[i]=} {y[i]=} {pos[i]=}")
    return pos

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

