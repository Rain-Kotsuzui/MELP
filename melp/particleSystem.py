from .lib import *
from .data import *
from .func import *
from .kernel import *
import warp.math as wm


@wp.kernel
def init_EParticles(n: int, particles: wp.array(dtype=EParticle), Eposs: wp.array(dtype=wp.vec3f),debug: wp.array(dtype=wp.vec3f)):  # type: ignore
    tid = wp.tid()
    if tid < n:
        p = particles[tid]

        y = 1.0 - (float(tid) / float(n - 1)) * 2.0
        radius = wp.sqrt(1.0 - y*y)
        theta = PHI * float(tid)
        x = wp.cos(theta) * radius
        z = wp.sin(theta) * radius

        p.pos = wp.vec3f(x, y, z) * 2.0 + wp.vec3f(0.0, 5.0, 0.0)
        p.vel = wp.vec3f(0.0)
        p.nvel = wp.vec3f(0.0)
        p.tvel = wp.vec3f(0.0)
        p.mass = wp.float32(0.0)
        p.c = wp.float32(0.0)
        p.volume = wp.float32(0.0)
        p.momentum = p.mass*p.vel
        p.thickness = wp.float32(0.0)

        p.affine_momentum = wp.vec3f(0.0)
        p.num_density = wp.float32(0.5)
        p.area = wp.float32(0.0)
        p.h = wp.float32(0.0)
        p.g = wp.diag(wp.vec2f(1.0))
        #TODO
        p.external_force = GRIVATY

        Eposs[tid] = p.pos
        debug[tid] = p.pos
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

        # p.pos = wp.vec3f(x, y, z) * (3.0+wp.sin(2.0*wp.pi *
        #                                         float(tid) / float(n - 1))) + wp.vec3f(0.0, 5.0, 0.0)
        
        p.pos = wp.vec3f(x, y, z) * 2.0 + wp.vec3f(0.0, 5.0, 0.0)
        p.vel = wp.vec3f(0.0)
        p.mass = PARTICLE_MASS
        p.c = PARTICLE_SURFACTANT
        p.volume = PARTICLE_VOLUME
        p.momentum = p.mass*p.vel

        p.b = wp.mat33f(0.0)
        p.d = wp.mat33f(0.0)

        p.normal= wp.vec3f(0.0)
        Lposs[tid] = p.pos
        particles[tid] = p
    pass


class ParticleSystem:
    n: int
    t: wp.float32
    dt: wp.float32
    bubble_volume: wp.array
    surface_area: wp.array
    n0: wp.float32
    T: wp.float32
    p_in : wp.float32

    EParticles: EParticle
    LParticles: LParticle
    Eposs: wp.array
    Lposs: wp.array

    Enorm: wp.array
    Lnorm: wp.array

    center: wp.array

    Egrid: wp.HashGrid
    Lgrid: wp.HashGrid

    kernel_r: wp.float32

    debug: wp.array

    def __init__(self,dt:float, n: int, kernel_r: wp.float32 = KERNEL_RAIDUS) -> None:
        self.n = int(n)
        self.t= wp.float32(0.0)
        self.dt = wp.float32(dt)
        self.kernel_r = kernel_r
        self.EParticles = wp.empty(self.n, dtype=EParticle, device="cuda")
        self.LParticles = wp.empty(self.n, dtype=LParticle, device="cuda")
        self.Eposs = wp.empty(self.n, dtype=wp.vec3f, device="cuda")
        self.debug = wp.empty(self.n, dtype=wp.vec3f, device="cuda")
        self.Lposs = wp.empty(self.n, dtype=wp.vec3f, device="cuda")

        self.Egrid = wp.HashGrid(dim_x=128, dim_y=128,
                                 dim_z=128, device="cuda")
        self.Lgrid = wp.HashGrid(dim_x=128, dim_y=128,
                                 dim_z=128, device="cuda")

        self.center = wp.zeros(1, dtype=wp.vec3f, device="cuda")

        self.Enorm = wp.zeros(self.n, dtype=wp.vec3f, device="cuda")
        self.Lnorm = wp.zeros(self.n, dtype=wp.vec3f, device="cuda")

        self.bubble_volume = wp.zeros(1, dtype=wp.float32, device="cuda")
        self.surface_area = wp.zeros(1, dtype=wp.float32, device="cuda")
        
        self.T=ENV_TEMPERATURE
        self.p_in = ENV_PRESSURE
        self.init_particles()

        self.Geometry()
        self.bubble_volume.fill_(wp.float32(0.0))
        self.surface_area.fill_(wp.float32(0.0))
        wp.launch(bubbleVolume, dim=self.n, inputs=[self.EParticles, self.bubble_volume,self.surface_area], device="cuda")
        self.n0=self.bubble_volume.list()[0]*self.p_in/(self.T*IDEAL_GAS_CONSTANT)

        pass

    def init_particles(self) -> None:  # type: ignore
        wp.launch(init_EParticles, dim=self.EParticles.shape[0], inputs=[self.n,
                  self.EParticles, self.Eposs,self.debug], device="cuda")
        

        wp.launch(init_LParticles, dim=self.LParticles.shape[0], inputs=[self.n,
                  self.LParticles, self.Lposs], device="cuda")
        
        self.hashBuild()
        self.normalBuild()
        # init alpha
        wp.launch(getAlpha, dim=self.n, inputs=[
                  self.Egrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        # init B,D
        wp.launch(E2L, dim=self.n, inputs=[
                  self.Egrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        pass

    def substep(self, dt: wp.float32) -> None:  # type: ignore
        # TODO

        # hashgrid build
        self.hashBuild()
        
        # L2E ransfer
        self.L2E()
        # Geometry
        self.Geometry()
        
        # DynamicsWithEuler 
        # TODO bug!!!
        #self.EulerDynamics()

        # E2L(Eparticles, LParticles, Eposs, Lposs, dt)
        self.E2L()

        # Eadvance(Eparticles, Eposs, dt)
        #self.Eadv()
        # TODO
        ## ERedistribute(Eparticles, Eposs, dt)

        # Ladvance(LParticles, Lposs, dt) and Lparticle nature update (momentum)
        self.Ladv()

        #self.de()

        self.normalBuild()
        self.updateELposs()
        self.t += dt
        pass
    
    def Ladv(self) -> None:
        wp.launch(Ladvance, dim=self.n, inputs=[self.LParticles, self.dt,self.Egrid.id,self.EParticles, self.kernel_r,self.Lnorm], device="cuda")
        pass

    def Eadv(self) -> None:
        wp.launch(Eadvance, dim=self.n, inputs=[self.EParticles, self.dt], device="cuda")
        wp.launch(getEgeometry, dim=self.n, inputs=[self.Egrid.id,self.EParticles, self.kernel_r], device="cuda")
        pass

    def E2L(self) -> None:
        wp.launch(E2L, dim=self.n, inputs=[self.Egrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        pass

    def EulerDynamics(self) -> None:
        # enclosed volume
        self.bubble_volume.fill_(wp.float32(0.0))
        self.surface_area.fill_(wp.float32(0.0))
        wp.launch(bubbleVolume, dim=self.n, inputs=[self.EParticles, self.bubble_volume,self.surface_area], device="cuda")
        # p_in
        self.p_in = self.n0*self.T*IDEAL_GAS_CONSTANT/self.bubble_volume.list()[0]
        # solve gamma
        ## get c1,c2,c3,b
        c1 = wp.empty(self.n, dtype=wp.float32, device="cuda")
        c2 = wp.empty(self.n, dtype=wp.vec3f, device="cuda")
        c3 = wp.empty(self.n, dtype=wp.float32, device="cuda")
        b = wp.empty(self.n, dtype=wp.float32, device="cuda")
        wp.launch(getC1C2C3B, dim=self.n, inputs=[c1,c2,c3,b,self.EParticles,self.Egrid.id,self.kernel_r,self.dt])
        ## relaxedJacobi
        for _ in range(MAX_ITERATIONS):
            wp.launch(RelaxedJacobi, dim=self.n, inputs=[self.Egrid.id,self.EParticles,c1,c2,c3,b,OMEGA,self.kernel_r], device="cuda")
        
        # updateVelocity
        wp.launch(updateEVelocity, dim=self.n, inputs=[self.Egrid.id,self.EParticles,self.kernel_r,ENV_PRESSURE,self.p_in,self.dt], device="cuda")
        pass

    def Geometry(self) -> None: 
        # TODO Lthickness
        wp.launch(getEgeometry, dim=self.n, inputs=[self.Egrid.id,self.EParticles, self.kernel_r], device="cuda")
        pass

    def L2E(self) -> None:
        # alpha compute
        wp.launch(getAlpha, dim=self.n, inputs=[
                  self.Egrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        # m,c,V,p compute
        wp.launch(getMCVPA, dim=self.n, inputs=[
                  self.Lgrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        pass

    def updateELposs(self) -> None:
        wp.launch(updateELposs, dim=self.n, inputs=[
                  self.EParticles, self.Eposs], device="cuda")
        wp.launch(updateELposs, dim=self.n, inputs=[
                  self.LParticles, self.Lposs], device="cuda")
        pass

    def normalBuild(self) -> None:
        self.center.fill_(wp.vec3f(0.0))
        wp.launch(centerCompute, dim=self.n, inputs=[
                  self.n, self.EParticles, self.center], device="cuda")
        #wp.launch(PCAnormalBuild, dim=self.n, inputs=[wp.uint64(
        #    self.Lgrid.id), self.LParticles, self.center, self.kernel_r, self.Lnorm], device="cuda")
        wp.launch(PCAnormalBuild, dim=self.n, inputs=[wp.uint64(
            self.Egrid.id), self.EParticles, self.center, self.kernel_r, self.Enorm], device="cuda")
        pass

    def hashBuild(self) -> None:
        self.Egrid.build(points=self.Eposs, radius=self.kernel_r)
        self.Lgrid.build(points=self.Lposs, radius=self.kernel_r)
        pass



    def de(self) -> None:
        # TODO
        wp.launch(deTest, dim=self.n, inputs=[self.debug,self.EParticles, self.t], device="cuda")
        pass
