from .lib import *
from .func import *
from .data import *
# ========L2E=========
# alpha compute
@wp.kernel
def getAlpha(Egrid:wp.uint64,Eparticles: wp.array(dtype=EParticle),Lparticles: wp.array(dtype=LParticle),h:float) -> None: # type: ignore
    tid = wp.tid()
    p = Lparticles[tid].pos
    query = wp.hash_grid_query(Egrid, p, h)
    index = int(0)
    alpha = wp.float32(0.0)
    while(wp.hash_grid_query_next(query, index)):
        neighbor = Eparticles[index]
        alpha += Poly6(p-neighbor.pos,h)
    Lparticles[tid].alpha = alpha
    pass
# m,c,V,p compute
@wp.kernel
def getMCVP(Lgrid:wp.uint64,Eparticles: wp.array(dtype=EParticle),Lparticles: wp.array(dtype=LParticle),h:float) -> None: # type: ignore
    tid = wp.tid()
    p = Eparticles[tid].pos
    query = wp.hash_grid_query(Lgrid, p, h)
    index = int(0)

    mass = float(0.0)
    c = wp.vec3f(0.0,0.0,0.0)
    V = float(0.0)
    p = wp.vec3f(0.0,0.0,0.0)
    while(wp.hash_grid_query_next(query, index)):
        neighbor = Lparticles[index]
        coe = Poly6(p-neighbor.pos,h)/neighbor.alpha
        mass += coe*neighbor.mass
        c += coe*neighbor.surfactant
        V += coe*neighbor.volume
        p += coe*neighbor.momentum
    pass

# ========辅助kernel=========
@wp.kernel
def updateELposs(particles: wp.array(dtype=Any), pos: wp.array(dtype=Any)) -> None: # type: ignore
    tid = wp.tid()
    p = particles[tid]
    pos[tid] = p.pos
