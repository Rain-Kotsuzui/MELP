from .lib import *


@wp.struct
class EParticle:
    pos: wp.vec3f
    vel: wp.vec3f
    normal: wp.vec3f

    mass: wp.float32
    c: wp.float32
    volume: wp.float32
    momentum: wp.vec3f

    affine_momentum : wp.vec3f

@wp.struct
class LParticle:
    pos: wp.vec3f
    vel: wp.vec3f
    normal: wp.vec3f

    mass: wp.float32
    c: wp.float32
    volume: wp.float32
    momentum: wp.vec3f

    alpha: wp.float32


