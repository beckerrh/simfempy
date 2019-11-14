#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:04:34 2019

@author: becker
"""

import matplotlib.pyplot as plt
import numpy as np

def xon (ton, t):
    if ton <= t:
        return (t-ton)/5.
    else:
        return 0

vxon = np.vectorize(xon)
t = np.linspace(0, 49, 50)    
xontest = vxon(2, t)
plt.plot(t, xontest, '-')
plt.show()