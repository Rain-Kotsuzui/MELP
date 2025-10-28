from .lib import *


@wp.struct
class EParticle:
    pos: wp.vec3f
    vel: wp.vec3f
    nvel: wp.vec3f
    tvel: wp.vec3f

    normal: wp.vec3f

    thickness: wp.float32
    mass: wp.float32
    c: wp.float32
    volume: wp.float32
    momentum: wp.vec3f

    affine_momentum : wp.vec3f
    num_density: wp.float32
    area: wp.float32
    h:wp.float32
    g:wp.mat22f    

@wp.struct
class LParticle:
    pos: wp.vec3f
    vel: wp.vec3f
    normal: wp.vec3f

    thickness: wp.float32
    mass: wp.float32
    c: wp.float32
    volume: wp.float32
    momentum: wp.vec3f

    b: wp.mat33f
    d: wp.mat33f

    alpha_poly6: wp.float32
    alpha_wendland: wp.float32


