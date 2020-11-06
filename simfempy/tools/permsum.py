#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:49:13 2020

@author: becker
"""

import numpy as np
import scipy

k = 3
d = 1

def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation

for i in sums(d+1,k):
    print(f"{i=} {scipy.special.factorial(np.array(i, dtype=int))}")

L = list(sums(d+1,k))
print('total permutations:',len(L))
print(f"{L=}")
