import numpy as np

#--------------------------------------------
def phione(x, y):
    return 1.0
def dphione(x, y):
    return (np.zeros_like(x), np.zeros_like(x))

#--------------------------------------------
def spline(s):
    if s>=1: return 1
    return 3*s**2 - 2*s**3
def splinec(s):
    c=0.5
    if s<=c: return 0
    t = (s-c)/(1-c)
    return 3*t**2 - 2*t**3
def phicon(x, y):
    n = x.shape[0]
    num = 0.0
    for i in range(n):
        if x[i]*y[i] < 0.0:
            num += 1
    return num/float(n)
def dphicon(x, y):
    n = x.shape[0]
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    num = 0.0
    den = 0.0
    for i in range(n):
        num += (x[i]-y[i])**2
        den += x[i]**2 + y[i]**2 + 2*abs(x[i]*y[i])
    if den:
        for i in range(n):
            dnumx = 2*(x[i]-y[i])
            dnumy = 2*(x[i]-y[i])
            ddenx = 2*x[i]
            ddeny = 2*y[i]
            if x[i]*y[i] > 0:
                ddenx += 2 * y[i]
                ddeny += 2 * x[i]
            elif x[i]*y[i] < 0:
                ddenx -= 2 * y[i]
                ddeny -= 2 * x[i]
            dx[i] = dnumx/num - ddenx/den
            dy[i] = dnumy/num - ddeny/den
    return dx, dy

#--------------------------------------------
def phiabs(x, y):
    n = x.shape[0]
    num = 0.0
    den = 0.0
    for i in range(n):
        num += abs(x[i]-y[i])
        den += abs(x[i])+abs(y[i])
    if den: sc = num/den
    else: sc=0.0
    return sc
def dphiabs(x, y):
    n = x.shape[0]
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    num = 0.0
    den = 0.0
    for i in range(n):
        num += abs(x[i]-y[i])
        den += abs(x[i])+abs(y[i])
    if den:
        for i in range(n):
            dnumx = 0.0
            dnumy = 0.0
            if x[i] > y[i]:
                dnumx = 1.0
                dnumy = -1.0
            elif x[i] < y[i]:
                dnumx = -1.0
                dnumy = 1.0
            ddenx = 0.0
            ddeny = 0.0
            if x[i] > 0.0:
                ddenx = 1.0
            elif x[i] < 0.0:
                ddenx = -1.0
            if y[i] > 0.0:
                ddeny = 1.0
            elif y[i] < 0.0:
                ddeny = -1.0
            dx[i] = dnumx/num - ddenx/den
            dy[i] = dnumy/num - ddeny/den
    return dx, dy
#--------------------------------------------
def phisum(x, y):
    n = x.shape[0]
    num = 0.0
    den = 0.0
    for i in range(n):
        num += x[i]
        den += abs(x[i])
    if den: sc = abs(num)/den
    else: sc=0.0
    return sc
def dphisum(x, y):
    stop
