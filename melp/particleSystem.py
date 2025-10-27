from .lib import *
from .data import *
from .func import *
from .kernel import *
import warp.math as wm

@wp.kernel
def init_EParticles(n: int, particles: wp.array(dtype=EParticle), Eposs: wp.array(dtype=wp.vec3f)):  # type: ignore
    tid = wp.tid()
    if tid < n:
        p = particles[tid]

        y = 1.0 - (float(tid) / float(n - 1)) * 2.0
        radius = wp.sqrt(1.0 - y*y)
        theta = PHI * float(tid)
        x = wp.cos(theta) * radius
        z = wp.sin(theta) * radius

        p.pos = wp.vec3f(x, y, z) * 2.0+ wp.vec3f(0.0, 5.0, 0.0)
        p.vel = wp.vec3f(0.0, 0.0, 0.0)

        Eposs[tid] = p.pos
        particles[tid] = p
    pass


@wp.kernel
def init_LParticles(n: int, particles: wp.array(dtype=LParticle), Lposs: wp.array(dtype=wp.vec3f)):  # type: ignore
    tid = wp.tid()
    if tid < n:
        p = particles[tid]

        y = 1.0 - (float(tid) / float(n - 1)) * 2.0
        radius = wp.sqrt(1.0 - y*y)
        theta = PHI * float(tid)
        x = wp.cos(theta) * radius
        z = wp.sin(theta) * radius

        p.pos = wp.vec3f(x, y, z) * (3.0+wp.sin(2.0*wp.pi* float(tid) / float(n - 1)))+ wp.vec3f(0.0, 5.0, 0.0)
        p.vel = wp.vec3f(0.0, 0.0, 0.0)
        p.mass = PARTICLE_MASS
        p.surfactant = PARTICLE_SURFACTANT
        p.volume = PARTICLE_VOLUME
        p.momentum = p.mass*p.vel
        p.alpha = 1.0

        Lposs[tid] = p.pos
        particles[tid] = p
    pass


class ParticleSystem:
    EParticles: EParticle
    LParticles: LParticle
    Eposs: wp.array
    Lposs: wp.array
    n: int
    
    Egrid: wp.HashGrid
    Lgrid: wp.HashGrid

    kernel_r:wp.float32
    def __init__(self, n: int,kernel_r:wp.float32 =KERNEL_RAIDUS) -> None:
        self.n = int(n)
        self.kernel_r = kernel_r
        self.EParticles = wp.empty(self.n, dtype=EParticle, device="cuda")
        self.LParticles = wp.empty(self.n, dtype=LParticle, device="cuda")
        self.Eposs = wp.empty(self.n, dtype=wp.vec3f, device="cuda")
        self.Lposs = wp.empty(self.n, dtype=wp.vec3f, device="cuda")

        self.Egrid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device="cuda")
        self.Lgrid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device="cuda")

        self.init_particles()
        pass

    def init_particles(self) -> None:  # type: ignore
        wp.launch(init_EParticles, dim=self.EParticles.shape[0], inputs=[self.n,
                  self.EParticles, self.Eposs], device="cuda")
        wp.launch(init_LParticles, dim=self.LParticles.shape[0], inputs=[self.n,
                  self.LParticles, self.Lposs], device="cuda")
        pass

    def substep(self,dt: wp.float32) -> None:  # type: ignore
        # TODO

        # hashgrid build
        self.hashBuild()
        # L2E ransfer
        self.L2E()
        # Geometry(Eparticles, LParticles, Eposs, Lposs, dt)
        # DynamicsWithEuler(Eparticles, LParticles, Eposs, Lposs, dt)
        # E2L(Eparticles, LParticles, Eposs, Lposs, dt)
        # ERedistribute(Eparticles, Eposs, dt)
        # Eadvance(Eparticles, Eposs, dt)
        # Ladvance(LParticles, Lposs, dt) and Lparticle nature update (momentum)
        
        self.updateELposs()
        pass
    def L2E(self) -> None:
        # alpha compute
        wp.launch(getAlpha, dim=self.n, inputs=[self.Egrid.id,self.EParticles,self.LParticles,self.kernel_r], device="cuda")
        # m,c,V,p compute
        wp.launch(getMCVP, dim=self.n, inputs=[self.Lgrid.id,self.EParticles,self.LParticles,self.kernel_r], device="cuda")
        pass

    def updateELposs(self) -> None:
        wp.launch(updateELposs, dim=self.n, inputs=[self.EParticles, self.Eposs], device="cuda")
        wp.launch(updateELposs, dim=self.n, inputs=[self.LParticles, self.Lposs], device="cuda")
        pass

    def hashBuild(self) -> None:
        self.Egrid.build(points=self.Eposs, radius=self.kernel_r)
        self.Lgrid.build(points=self.Lposs, radius=self.kernel_r)
        pass
