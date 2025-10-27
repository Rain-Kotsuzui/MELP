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

# m,c,V,p,affine compute
@wp.kernel
def getMCVPA(Lgrid:wp.uint64,Eparticles: wp.array(dtype=EParticle),Lparticles: wp.array(dtype=LParticle),h:float) -> None: # type: ignore
    tid = wp.tid()

    p = Eparticles[tid].pos
    query = wp.hash_grid_query(Lgrid, p, h)
    index = int(0)

    mass = float(0.0)
    c = float(0.0)
    volume = float(0.0)
    moment = wp.vec3f(0.0,0.0,0.0)
    while(wp.hash_grid_query_next(query, index)):
        neighbor = Lparticles[index]
        coe = Poly6(p-neighbor.pos,h)/neighbor.alpha
        mass += coe*neighbor.mass
        c += coe*neighbor.c
        volume += coe*neighbor.volume
        moment += coe*neighbor.momentum
    Eparticles[tid].mass = mass
    Eparticles[tid].c = c
    Eparticles[tid].volume = volume
    Eparticles[tid].momentum = moment

    # TODO affine momentum
    pass

# ========辅助kernel=========
@wp.kernel
def updateELposs(particles: wp.array(dtype=Any), pos: wp.array(dtype=Any)) -> None: # type: ignore
    tid = wp.tid()
    p = particles[tid]
    pos[tid] = p.pos

@wp.kernel
def centerCompute(n:wp.int32,particles: wp.array(dtype=Any), center:wp.array(dtype=Any)) -> None: # type: ignore
    tid = wp.tid()
    p = particles[tid]
    center[0] +=p.pos/wp.float32(n)

@wp.kernel
def PCAnormalBuild(grid: wp.uint64,particles: wp.array(dtype=Any),view_point: wp.array(dtype=Any),h:float,norm:wp.array(dtype=Any)) -> None: # type: ignore
    tid = wp.tid()
    p = particles[tid].pos
    query = wp.hash_grid_query(grid, p, h)
    index = int(0)
    count = int(0)
    center = wp.vec3f(0.0,0.0,0.0)
    while(wp.hash_grid_query_next(query, index)):
        count +=1
        center += particles[index].pos
    if(count<3):
        particles[tid].normal = p - view_point[0]
        particles[tid].normal = particles[tid].normal/wp.length(particles[tid].normal)
        
        norm[tid]=p+particles[tid].normal*0.3
        return
    
    center /= wp.float32(count)
    query = wp.hash_grid_query(grid, p, h)
    index = int(0)
    cov = wp.mat33f(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0)
    while wp.hash_grid_query_next(query, index):
        r = particles[index].pos - center
        cov += wp.outer(r, r)


    cov_inv = wp.inverse(cov)
    v = wp.vec3f(1.0, 1.0, 1.0) 
    for i in range(10):
        x = cov_inv*v
        v = wp.normalize(x)

    normal = v

    view_vec =p- view_point[0]
    if wp.dot(normal, view_vec) < 0.0:
        normal = -normal

    norm[tid]=p+normal*0.3
    particles[tid].normal = normal



