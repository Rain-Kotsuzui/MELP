from .lib import *
from .func import *
from .data import *

# ========L2E=========
# alpha compute


@wp.kernel
def getAlpha(Egrid: wp.uint64, Eparticles: wp.array(dtype=EParticle), Lparticles: wp.array(dtype=LParticle), h: float) -> None:  # type: ignore
    tid = wp.tid()
    p = Lparticles[tid].pos
    norm = Lparticles[tid].normal

    query = wp.hash_grid_query(Egrid, p, h)
    index = int(0)
    alpha_poly6 = wp.float32(0.0)
    alpha_wendland = wp.float32(0.0)
    while (wp.hash_grid_query_next(query, index)):
        Eneighbor = Eparticles[index]
        alpha_poly6 += Poly6(p-Eneighbor.pos, h)
        alpha_wendland += Wendland(p-Eneighbor.pos, h)

    Lparticles[tid].alpha_poly6 = alpha_poly6
    Lparticles[tid].alpha_wendland = alpha_wendland
    pass

# m,c,V,p,affine compute


@wp.kernel
def getMCVPA(Lgrid: wp.uint64, Eparticles: wp.array(dtype=EParticle), Lparticles: wp.array(dtype=LParticle), h: float) -> None:  # type: ignore
    tid = wp.tid()

    p = Eparticles[tid].pos
    query = wp.hash_grid_query(Lgrid, p, h)
    index = int(0)

    mass = float(0.0)
    c = float(0.0)
    volume = float(0.0)
    moment = wp.vec3f(0.0)
    affine_moment = wp.vec3f(0.0)

    count = int(0)
    while (wp.hash_grid_query_next(query, index)):
        count += 1
        Lneighbor = Lparticles[index]
        coe = Poly6(p-Lneighbor.pos, h)/Lneighbor.alpha_poly6

        mass += coe*Lneighbor.mass
        c += coe*Lneighbor.c
        volume += coe*Lneighbor.volume

        coe = Wendland(p-Lneighbor.pos, h)/Lneighbor.alpha_wendland
        tang_proj = wp.diag(
            wp.vec3f(1.0))-wp.outer(Lparticles[tid].normal, Lparticles[tid].normal)
        proj_r = tang_proj*(p-Lneighbor.pos)
        inv_d = wp.inverse(Lneighbor.d)

        moment += coe*Lneighbor.momentum
        affine_moment += coe*Lneighbor.b*inv_d*proj_r

    Eparticles[tid].mass = mass
    Eparticles[tid].c = c
    Eparticles[tid].volume = volume

    Eparticles[tid].momentum = moment
    Eparticles[tid].affine_momentum = affine_moment

    if count == 0:
        # TODO 泡泡炸裂
        pass
    norm_proj = wp.outer(Eparticles[tid].normal, Eparticles[tid].normal)
    Eparticles[tid].vel = (Eparticles[tid].momentum +
                           Eparticles[tid].affine_momentum)/Eparticles[tid].mass
    Eparticles[tid].nvel = norm_proj*Eparticles[tid].vel
    Eparticles[tid].tvel = Eparticles[tid].vel-Eparticles[tid].nvel

    pass

# ========Geometry=========


@wp.kernel
def getLthickness(Lparticles: wp.array(dtype=LParticle), dt: float) -> None:  # type: ignore
    # TODO
    pass


@wp.kernel
def getEgeometry(Egrid: wp.uint64, Eparticles: wp.array(dtype=EParticle), radius: float) -> None:  # type: ignore
    tid = wp.tid()
    pEi = Eparticles[tid].pos

    norm = Eparticles[tid].normal
    vec_up = wp.vec3f(0.0, 1.0, 0.0)
    if wp.dot(norm, vec_up) > 0.99:
        vec_up = wp.vec3f(0.0, 0.0, 1.0)
    e1 = wp.normalize(wp.cross(vec_up, norm))
    e2 = wp.normalize(wp.cross(norm, e1))
    grad_z = wp.vec2f(0.0)  # (dz/du, dz/dv)
    lap_z = wp.float32(0.0)  # laplacian z

    query = wp.hash_grid_query(Egrid, pEi, radius)
    j = int(0)

    num_density = wp.float32(0.0)
    area = wp.float32(0.0)
    while (wp.hash_grid_query_next(query, j)):
        vec_ij = pEi-Eparticles[j].pos

        num_density += Poly6_2D(vec_ij, norm, radius)

        r = wp.length(vec_ij)
        if r > 1e-6:
            z = wp.dot(vec_ij, norm)
            grad_W = Poly6_2D_Grad(vec_ij, norm, radius)

            grad_W_u = wp.dot(grad_W, e1)
            grad_W_v = wp.dot(grad_W, e2)

            grad_z += wp.vec2(z * grad_W_u, z * grad_W_v)
            lap_z += z * Poly6_2D_Lap(vec_ij, norm, radius)

    Eparticles[tid].h = -0.5*lap_z/num_density
    dz_du = grad_z[0]/num_density
    dz_dv = grad_z[1]/num_density
    Eparticles[tid].g = wp.mat22f(1.0+dz_du*dz_du, dz_du*dz_dv,
                                  dz_du*dz_dv, 1.0+dz_dv*dz_dv)

    area = 1.0/(num_density+1e-9)
    Eparticles[tid].num_density = num_density
    Eparticles[tid].thickness = Eparticles[tid].volume*num_density
    Eparticles[tid].area = area
    pass


# =========Euler Dynamics============
@wp.kernel
def bubbleVolume(center: wp.array(dtype=wp.vec3f), Eparticles: wp.array(dtype=EParticle), bubble_volume: wp.array(dtype=float), surface_area: wp.array(dtype=float)) -> None:  # type: ignore
    tid = wp.tid()
    p = Eparticles[tid].pos-center[0]
    d = wp.length(p)
    n = Eparticles[tid].normal
    sign = -1.0
    if wp.dot(n, p) >= 0.0:
        sign = 1.0
    bubble_volume[0] += sign*Eparticles[tid].area*d/3.0
    surface_area[0] += Eparticles[tid].area
    pass


# ========E2L=========


@wp.kernel
def E2L(Egrid: wp.uint64, Eparticles: wp.array(dtype=EParticle), Lparticles: wp.array(dtype=LParticle), h: float) -> None:  # type: ignore
    tid = wp.tid()
    p = Lparticles[tid].pos
    query = wp.hash_grid_query(Egrid, p, h)
    index = int(0)
    vel = wp.vec3f(0.0)
    b = wp.mat33f(0.0)
    d = wp.mat33f(0.0)
    norm_proj = wp.outer(Lparticles[tid].normal, Lparticles[tid].normal)
    tang_proj = wp.diag(wp.vec3f(1.0))-norm_proj

    while (wp.hash_grid_query_next(query, index)):
        neighbor = Eparticles[index]
        proj_Ve = tang_proj*neighbor.vel
        proj_r = tang_proj*(neighbor.pos-p)
        coe = Wendland(p-neighbor.pos, h)/Lparticles[tid].alpha_wendland

        vel += coe*neighbor.vel
        b += coe*wp.outer(proj_Ve, proj_r)
        d += coe*Wendland(p-neighbor.pos, h)*wp.outer(proj_r, proj_r)

    Lparticles[tid].vel = vel
    Lparticles[tid].b = b
    Lparticles[tid].d = d


# ========辅助kernel=========
@wp.kernel
def updateELposs(particles: wp.array(dtype=Any), pos: wp.array(dtype=Any)) -> None:  # type: ignore
    tid = wp.tid()
    p = particles[tid]
    pos[tid] = p.pos


@wp.kernel
def centerCompute(n: wp.int32, particles: wp.array(dtype=Any), center: wp.array(dtype=Any)) -> None:  # type: ignore
    tid = wp.tid()
    p = particles[tid]
    center[0] += p.pos/wp.float32(n)


@wp.kernel
def PCAnormalBuild(grid: wp.uint64, particles: wp.array(dtype=Any), view_point: wp.array(dtype=Any), h: float, norm: wp.array(dtype=Any)) -> None:  # type: ignore
    tid = wp.tid()
    p = particles[tid].pos
    query = wp.hash_grid_query(grid, p, h)
    index = int(0)
    count = int(0)
    center = wp.vec3f(0.0, 0.0, 0.0)
    while (wp.hash_grid_query_next(query, index)):
        count += 1
        center += particles[index].pos
    if (count < 3):
        particles[tid].normal = p - view_point[0]
        particles[tid].normal = particles[tid].normal / \
            wp.length(particles[tid].normal)

        norm[tid] = p+particles[tid].normal*0.3
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

    view_vec = p - view_point[0]
    if wp.dot(normal, view_vec) < 0.0:
        normal = -normal

    norm[tid] = p+normal*0.3
    particles[tid].normal = normal


@wp.kernel
def deTest(debug: wp.array(dtype=wp.vec3f), Eparticles: wp.array(dtype=EParticle), time: float) -> None:  # type: ignore
    tid = wp.tid()
    Eparticles[tid].pos = debug[tid]*(1.0+0.5*wp.sin(2.0*time))
    pass
