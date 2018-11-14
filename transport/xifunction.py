import numpy as np

def spline(a,b,x):
    return ( (3*b-a)*(x**2-a**2)+6*a*b*x-2*x**3 )/(b-a)**3
def sspline(a,b,x):
    return ( 2*(3*b-a)*x+6*a*b-6*x**2 )/(b-a)**3
#--------------------------------------------
def xilin(x, y):
    return x - y
    if x*y>=0.0:
        return x-y
    elif x>0.0 and y <= -2*x:
        return -x
    elif x < 0.0 and y >= -2 * x:
        return -x
    else:
        return x - y
def dxilindx(x, y):
    return 1.0
    if x*y>=0.0:
        return 1.0
    elif x>0.0 and y <= -2*x:
        return -1.0
    elif x < 0.0 and y >= -2 * x:
        return -1.0
    else:
        return 1.0
def dxilindy(x, y):
    return -1.0
    if x*y>=0.0:
        return -1.0
    elif x>0.0 and y <= -2*x:
        return 0.0
    elif x < 0.0 and y >= -2 * x:
        return 0.0
    else:
        return -1.0

#--------------------------------------------
def xiconst(x, y):
    return x
def dxiconstdx(x, y):
    return 1
def dxiconstdy(x, y):
    return 0

#--------------------------------------------
def xinew(x, y):
    if x * y <= 0.0:
        return x
    if y/x >= 1.0:
        return 0.0
    s = (x-y)/(x+y)
    f = 0.5*s**2+3*s**3 -2.5*s**4
    return x*f**2
def dxinewdx(x, y):
    if x * y <= 0.0:
        return 1.0
    if y/x >= 1.0:
        return 0.0
    s = (x-y)/(x+y)
    f = 0.5*s**2+3*s**3 -2.5*s**4
    df = s + 9*s**2 - 10*s**3
    return f**2 + 2*x*y/(x+y)**2* ( 2*df*f)
def dxinewdy(x, y):
    if x * y <= 0.0:
        return 0.0
    if y/x >= 1.0:
        return 0.0
    s = (x-y)/(x+y)
    f = 0.5 * s ** 2 + 3 * s ** 3 - 2.5 * s ** 4
    df = s + 9 * s ** 2 - 10 * s ** 3
    return -2 * x ** 2 / (x + y) ** 2 * ( 2*df*f)


# --------------------------------------------
def xinew2(x, y):
    if x * y <= 0.0:
        return x
    s = (x-y)/(x+y)
    c= 0.6
    if s < -c:
        return -(s+c)/(1-c)*x
    elif s < c:
        return 0.0
    else:
        return (s-c)/(1-c)*x
    # f = (2*s**2 - s**4)**4
    f = s**2
    return x * f
def dxinew2dx(x, y):
    if x * y <= 0.0:
        return 1.0
    s = (x-y)/(x+y)
    c= 0.6
    if s < -c:
        return -(s+c)/(1-c)  - 2*x*y/(x+y)**2/(1-c)
    elif s < c:
        return 0.0
    else:
        return (s-c)/(1-c) + 2*x*y/(x+y)**2/(1-c)

    f = s**2
    df = 2*s
    ds = 2*y/(x+y)**2
    return f + x* ds * df
def dxinew2dy(x, y):
    if x * y <= 0.0:
        return 0.0
    s = (x-y)/(x+y)
    c= 0.6
    if s < -c:
        return 2*x**2/(x+y)**2/(1-c)
    elif s <= c:
        return 0.0
    else:
        return -2*x**2/(x+y)**2/(1-c)
    f = s**2
    df = 2*s
    ds = -2*x/(x+y)**2
    return x * df *ds

# --------------------------------------------
def xispline(x, y):
    if x * y <= 0.0:
        return x
    elif x/y < 1:
        f = spline(-1,1,x/y)
        return x * f
    else:
        return 0.0
def dxisplinedx(x, y):
    c = 0.5
    if x * y <= 0.0:
        return 1.0
    elif y / x < c:
        r = y/x/c
        return (1 - 1.5*r**2 + 0.5*r**3) -r*(-3.0*r + 1.5*r**2)
    else:
        return 0.0
def dxisplinedy(x, y):
    c = 0.5
    if x * y <= 0.0:
        return 0.0
    elif y / x < c:
        r = y/x/c
        return (-3.0*r + 1.5*r**2)/c
    else:
        return 0.0


# --------------------------------------------
def xi2(x, y):
    if x * y <= 0.0:
        return x
    elif x / y > 1:
        s = (x-y)/(x+y)
        return (x-y)*s
    else:
        return 0.0
def dxi2dx(x, y):
    if x * y <= 0.0:
        return 1.0
    elif x / y >= 1:
        s = (x-y)/(x+y)
        dsx = 2*y/(x+y)**2
        return (x-y)*dsx + s
    else:
        return 0.0
def dxi2dy(x, y):
    if x * y <= 0.0:
        return 0.0
    elif x / y > 1:
        s = (x - y) / (x + y)
        dsy = -2 * x / (x + y) ** 2
        return (x-y)*dsy - s
    else:
        return 0.0

# --------------------------------------------
def xi3old(x, y):
    if x * y <= 0.0:
        return x
    elif x / y > 1:
        s = (x - y) / (x + y)
        f = 2.5 * s - 1.5 * s ** 2
        return (x - y) * f
    else:
        return 0.0

def dxi3dxold(x, y):
    if x * y <= 0.0:
        return 1.0
    elif x / y >= 1:
        s = (x - y) / (x + y)
        dsx = 2 * y / (x + y) ** 2
        f = 2.5 * s - 1.5 * s ** 2
        df = 2.5 - 3.0 * s
        return (x - y) * dsx * df + f
    else:
        return 0.0

def dxi3dyold(x, y):
    if x * y <= 0.0:
        return 0.0
    elif x / y > 1:
        s = (x - y) / (x + y)
        dsy = -2 * x / (x + y) ** 2
        f = 2.5 * s - 1.5 * s ** 2
        df = 2.5 - 3.0 * s
        return (x - y) * dsy * df - f
    else:
        return 0.0


# --------------------------------------------
def xi3(x, y):
    if x * y >= 0.0:
        return 0
    elif x>0:
        return x-max(y,-2*x)
    else:
        return x-min(y,-2*x)


def dxi3dx(x, y):
    if x * y >= 0.0:
        return 0
    elif abs(x)<=abs(-y):
        return 1
    else:
        return 0

def dxi3dy(x, y):
    if x * y >= 0.0:
        return 0
    elif abs(x) <= abs(-y):
        return 0
    else:
        return -1


# --------------------------------------------
def xi(x, y):
    c = 1
    if x * y <= 0.0:
        return x
    elif x / y > c:
        return x-c*y
    else:
        return 0.0
def dxidx(x, y):
    c = 1
    if x * y <= 0.0:
        return 1.0
    elif x / y >= c:
        return 1.0
    else:
        return 0.0
def dxidy(x, y):
    c = 1
    if y==0.0 or x==c*y:
        return -0.5*c
    elif x * y >= 0.0 and x / y > c:
        return -c
    else:
        return 0.0

# --------------------------------------------
def xisignmin(x, y):
    signy = 1.0
    if y < 0.0: signy = -1.0
    return x - signy * min(abs(x), abs(y))
def dxisignmindx(x, y):
    signy = 0.0
    if y > 0.0:
        signy = 1.0
    elif y < 0.0:
        signy = -1.0
    signx = 0.0
    if x > 0.0:
        signx = 1.0
    elif x < 0.0:
        signx = -1.0
    if abs(x) == abs(y):
        return 1.0 - 0.5 * signy * signx
    elif abs(x) < abs(y):
        return 1.0 - signy * signx
    else:
        return 1.0
def dxisignmindy(x, y):
    signy = 0.0
    if y > 0.0:
        signy = 1.0
    elif y < 0.0:
        signy = -1.0
    signx = 0.0
    if x > 0.0:
        signx = 1.0
    elif x < 0.0:
        signx = -1.0
    if abs(x) == abs(y):
        return -0.5
    elif abs(x) < abs(y):
        return 0.0
    else:
        return -1.0


# --------------------------------------------
def xibis(x, y):
    if x * y <= 0.0:
        return x
    elif x / y > 1.0:
        return x*(x - y)/(x+y)
    else:
        return 0.0
def dxibisdx(x, y):
    if x * y < 0.0:
        return 1.0
    elif x == 0.0:
        return 1.0
    elif y == 0.0:
        return 1.0
    elif x == y:
        return 0.5
    elif x / y > 1.0:
        return (x**2-y**2+2*x*y)/(x+y)**2
    else:
        return 0.0
def dxibisdy(x, y):
    if x * y < 0.0:
        return 0.0
    elif x == 0.0:
        return 0.0
    elif y == 0.0:
        return 0.0
    elif x == y:
        return 0.0
    elif x / y > 1.0:
        return -2*x**2/(x+y)**2
    else:
        return 0.0

# --------------------------------------------
def xiter(x, y):
    if x * y <= 0.0:
        return x
    elif x / y > 1.0:
        s = (x-y)/(x+y)
        f = s*s*s
        return x * f
    else:
        return 0.0
def dxiterdx(x, y):
    if x * y < 0.0:
        return 1.0
    elif x == 0.0:
        return 1.0
    elif y == 0.0:
        return 1.0
    elif x == y:
        return 0.0
    elif x / y > 1.0:
        s = (x-y)/(x+y)
        dsx = 2*y/(x+y)**2
        f = s*s*s
        df = 3*s**2
        return f + x*df*dsx
    else:
        return 0.0
def dxiterdy(x, y):
    if x * y < 0.0:
        return 0.0
    elif x == 0.0:
        return 0.0
    elif y == 0.0:
        return 0.0
    elif x == y:
        return 0.0
    elif x / y > 1.0:
        s = (x-y)/(x+y)
        dsy = -2*x/(x+y)**2
        f = s*s*s
        df = 3*s**2
        return x*df*dsy
    else:
        return 0.0


# --------------------------------------------
def xiquater(x, y):
    if x * y <= 0.0:
        return x
    elif x / y > 1.0:
        s = (x-y)/(x+y)
        f = 3*s**2 - 2*s**3
        return x * f
    else:
        return 0.0
def dxiquaterdx(x, y):
    if x * y < 0.0:
        return 1.0
    elif x == 0.0:
        return 1.0
    elif y == 0.0:
        return 1.0
    elif x == y:
        return 0.0
    elif x / y > 1.0:
        s = (x-y)/(x+y)
        dsx = 2*y/(x+y)**2
        f = 3*s**2 - 2*s**3
        df = 6*s - 6*s**2
        return f + x*df*dsx
    else:
        return 0.0
def dxiquaterdy(x, y):
    if x * y < 0.0:
        return 0.0
    elif x == 0.0:
        return 0.0
    elif y == 0.0:
        return 0.0
    elif x == y:
        return 0.0
    elif x / y > 1.0:
        s = (x-y)/(x+y)
        dsy = -2*x/(x+y)**2
        f = 3*s**2 - 2*s**3
        df = 6*s - 6*s**2
        return x*df*dsy
    else:
        return 0.0

