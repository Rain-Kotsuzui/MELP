from .lib import *

@wp.func
def Poly6(r:wp.vec3f, h:wp.float32)->wp.float32:
    d=wp.length_sq(r)
    coe=315.0 / (64.0 * wp.pi * wp.pow(h, 9.0))
    tmp = 0.0
    if d <= h*h:
        tmp = coe * wp.pow((h * h - d), 3.0)
    return tmp

@wp.func
def Wendland(r:wp.vec3f, h:wp.float32)->wp.float32:
    q=wp.length_sq(r)/(2.0*h)
    coe= 21.0/(16.0*wp.pi*h*h*h)
    tmp = 0.0
    if q <= 1.0:
        tmp = coe * wp.pow((1.0 - q), 4.0)*(1.0 + 4.0*q)
    return tmp

@wp.func
def Poly6_2D(r:wp.vec3f,n:wp.vec3f, h:wp.float32)->wp.float32:
    tang_proj=wp.diag(wp.vec3f(1.0))-wp.outer(n,n)
    r=tang_proj * r
    d=wp.length_sq(r)
    coe = 4.0 / (wp.pi * wp.pow(h, 8.0))
    tmp=0.0
    if d <= h*h:
        tmp = coe * wp.pow((h * h - d), 3.0)
    return tmp

@wp.func
def Poly6_2D_Grad(r:wp.vec3f,n:wp.vec3f, h:wp.float32)->wp.vec3f:
    tang_proj=wp.diag(wp.vec3f(1.0))-wp.outer(n,n)
    r=tang_proj * r
    d=wp.length_sq(r)
    coe = -24.0 / (wp.pi * wp.pow(h, 8.0))
    tmp=wp.vec3f(0.0)
    if d <= h*h:
        tmp = coe * wp.pow((h * h - d), 2.0) * r
    return tmp

@wp.func
def Poly6_2D_Lap(r:wp.vec3f,n:wp.vec3f, h:wp.float32)->wp.float32:
    tang_proj=wp.diag(wp.vec3f(1.0))-wp.outer(n,n)
    r=tang_proj * r
    d=wp.length_sq(r)
    coe = 48.0 / (wp.pi * wp.pow(h, 8.0))
    tmp=0.0
    if d <= h*h:
        tmp = coe * (h * h - d) * (-h * h + 3.0 * d)
    return tmp
