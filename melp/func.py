from .lib import *

@wp.func
def Poly6(r:wp.vec3f, h:wp.float32)->wp.float32:
    d=wp.length_sq(r)
    coe=315.0 / (64.0 * wp.pi * wp.pow(h, 9.0))
    tmp = 0.0
    if d <= h*h:
        tmp = coe * wp.pow((h * h - d), 3.0)
    return tmp
