from .lib import *
from .data import *
from .func import *
from .kernel import *
import warp.math as wm


@wp.kernel
def init_EParticles(n: int, particles: wp.array(dtype=EParticle), Eposs: wp.array(dtype=wp.vec3f), debug: wp.array(dtype=wp.vec3f)):  # type: ignore
    tid = wp.tid()
    if tid < n:
        p = particles[tid]

        y = 1.0 - (float(tid) / float(n - 1)) * 2.0
        radius = wp.sqrt(1.0 - y*y)
        theta = PHI * float(tid)
        x = wp.cos(theta) * radius
        z = wp.sin(theta) * radius

        p.pos = wp.vec3f(x*2.0, y*1.0, z*2.0)+ wp.vec3f(0.0, 5.0, 0.0)
        p.vel = wp.vec3f(0.0)
        p.nvel = wp.vec3f(0.0)
        p.tvel = wp.vec3f(0.0)
        p.Evel = wp.vec3f(0.0)
        p.pseudo_pressure = wp.float32(0.0)

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
        # TODO
        p.external_force = wp.vec3f(0.0)

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

        p.pos = wp.vec3f(x*2.0, y*1.0, z*2.0) + wp.vec3f(0.0, 5.0, 0.0)

        p.vel = wp.vec3f(0.0)
        p.mass = PARTICLE_MASS
        p.c = PARTICLE_SURFACTANT
        p.volume = PARTICLE_VOLUME
        p.momentum = p.mass*p.vel

        p.b = wp.mat33f(0.0)
        p.d = wp.mat33f(0.0)

        p.normal = wp.vec3f(0.0)
        Lposs[tid] = p.pos
        particles[tid] = p
    pass


class ParticleSystem:
    n: int
    m: int
    t: wp.float32
    dt: wp.float32
    dt_redistribute: wp.float32
    bubble_volume: wp.array
    surface_area: wp.array
    n0: wp.float32
    T: wp.float32
    p_in: wp.array

    EParticles: wp.array
    LParticles: wp.array
    Eposs: wp.array
    Lposs: wp.array

    num_density_max: wp.array
    num_density_min: wp.array
    num_density_averge: wp.array

    Enorm: wp.array
    Lnorm: wp.array

    center: wp.array

    Egrid: wp.HashGrid
    Lgrid: wp.HashGrid

    kernel_r: wp.float32

    debug: wp.array

    def __init__(self, dt: float, n: int, m: int, kernel_r: wp.float32 = KERNEL_RAIDUS) -> None:
        self.n = int(n)
        self.m = int(m)
        self.t = wp.float32(0.0)
        self.dt = wp.float32(dt)
        self.dt_redistribute = wp.float32(dt/5.0)

        self.kernel_r = kernel_r
        self.EParticles = wp.empty(self.m, dtype=EParticle, device="cuda")
        self.LParticles = wp.empty(self.n, dtype=LParticle, device="cuda")
        self.Eposs = wp.empty(self.m, dtype=wp.vec3f, device="cuda")
        self.debug = wp.empty(self.m, dtype=wp.vec3f, device="cuda")
        self.Lposs = wp.empty(self.n, dtype=wp.vec3f, device="cuda")

        self.Egrid = wp.HashGrid(dim_x=128, dim_y=128,
                                 dim_z=128, device="cuda")
        self.Lgrid = wp.HashGrid(dim_x=128, dim_y=128,
                                 dim_z=128, device="cuda")

        self.center = wp.zeros(1, dtype=wp.vec3f, device="cuda")

        self.Enorm = wp.zeros(self.m, dtype=wp.vec3f, device="cuda")
        self.Lnorm = wp.zeros(self.n, dtype=wp.vec3f, device="cuda")

        self.bubble_volume = wp.zeros(1, dtype=wp.float32, device="cuda")
        self.surface_area = wp.zeros(1, dtype=wp.float32, device="cuda")

        self.T = ENV_TEMPERATURE
        self.p_in = wp.ones(1, dtype=wp.float32, device="cuda")
        self.init_particles()

        self.num_density_max = wp.zeros(1, dtype=wp.float32, device="cuda")
        self.num_density_min = wp.zeros(1, dtype=wp.float32, device="cuda")
        self.num_density_averge = wp.zeros(1, dtype=wp.float32, device="cuda")
        self.Geometry()
        self.bubble_volume.fill_(wp.float32(0.0))
        self.surface_area.fill_(wp.float32(0.0))
        wp.launch(bubbleVolume_, dim=self.m, inputs=[
                  self.EParticles, self.bubble_volume, self.surface_area, self.p_in], device="cuda")
        self.n0 = self.bubble_volume.list()[0]*self.p_in.list()[0]/(self.T*IDEAL_GAS_CONSTANT)

        pass

    def init_particles(self) -> None:  # type: ignore
        wp.launch(init_EParticles, dim=self.EParticles.shape[0], inputs=[self.m,
                  self.EParticles, self.Eposs, self.debug], device="cuda")

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
        # TODO affine bug
        self.L2E()

        # Geometry
        self.Geometry()

        #print(f"{(self.num_density_max-self.num_density_min)/self.num_density_averge}")
        # DynamicsWithEuler
        self.EulerDynamics()
        # E2L(Eparticles, LParticles, Eposs, Lposs, dt)
        self.E2L()

        # Eadvance(Eparticles, Eposs, dt)
        self.Eadv()

        # ERedistribute(Eparticles, Eposs, dt)
        self.ERedistribute()

        # Ladvance(LParticles, Lposs, dt) and Lparticle nature update (momentum)
        self.Ladv()

        # self.de()

        # self.normalBuild()
        self.updateELposs()
        self.t += dt
        print(f"t: {self.t:.2f}")
        pass

    def Ladv(self) -> None:
        wp.launch(Ladvance, dim=self.n, inputs=[
                  self.LParticles, self.dt, self.Egrid.id, self.EParticles, self.kernel_r, self.Lnorm], device="cuda")
        pass

    def ERedistribute(self) -> None:
        wp.launch(init_redistribute, dim=self.m, inputs=[
                  self.EParticles], device="cuda")
        i = 0
        while (((self.num_density_max-self.num_density_min)/self.num_density_averge).numpy()[0] > 1.5) and i<10:
            i += 1
            print(f"REDISTRIBUTE!")
            # TODO solve pseudo_pressure
            # get c,b
            c = wp.empty(self.m, dtype=wp.float32, device="cuda")
            b = wp.empty(self.m, dtype=wp.float32, device="cuda")
            beta = wp.zeros(1, dtype=wp.float32, device="cuda")
            self.Geometry()
            wp.launch(getCB, dim=self.m, inputs=[self.EParticles, self.num_density_averge, c,
                      b, beta, self.EParticles, self.Egrid.id, self.kernel_r, self.dt_redistribute])
            for _ in range(MAX_ITERATIONS):
                wp.launch(redistribute_RelaxedJacobi, dim=self.m, inputs=[
                          self.Egrid.id, self.EParticles, c, b, OMEGA, self.kernel_r], device="cuda")
                
            wp.launch(getBeta, dim=1, inputs=[beta, self.Egrid.id,self.EParticles,self.kernel_r], device="cuda")
            #print(f"beta: {beta.list()[0]}")

            wp.launch(redistribute, dim=self.m, inputs=[
                      self.Egrid.id, self.EParticles, beta, self.dt_redistribute, self.kernel_r], device="cuda")
            

            #print(f"{self.num_density_max} {self.num_density_min} {self.num_density_averge}")
           # print(f"{self.EParticles.list()[0]}")
            #print(f"{1.0/beta.list()[0]}")
            pass
        wp.launch(apply_reEvel, dim=self.m, inputs=[
                  self.EParticles], device="cuda")
        pass

    def Eadv(self) -> None:
        wp.launch(Eadvance, dim=self.m, inputs=[
                  self.EParticles, self.dt], device="cuda")

        self.num_density_max.fill_(wp.float32(0.0))
        self.num_density_min.fill_(wp.float32(10.0*self.m))
        self.num_density_averge.fill_(wp.float32(0.0))
        wp.launch(getEgeometry, dim=self.m, inputs=[self.Egrid.id, self.EParticles, self.kernel_r,
                  self.num_density_max, self.num_density_min, self.num_density_averge], device="cuda")
        pass

    def E2L(self) -> None:
        wp.launch(E2L, dim=self.n, inputs=[
                  self.Egrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        pass

    def EulerDynamics(self) -> None:
        # enclosed volume
        self.bubble_volume.fill_(wp.float32(0.0))
        self.surface_area.fill_(wp.float32(0.0))
        wp.launch(bubbleVolume, dim=self.m, inputs=[
                  self.EParticles, self.bubble_volume, self.surface_area], device="cuda")
        wp.launch(pressure,dim=1, inputs=[self.p_in, self.bubble_volume, self.n0, self.T], device="cuda")
        # solve gamma
        # get c1,c2,c3,b
        c1 = wp.empty(self.m, dtype=wp.float32, device="cuda")
        c2 = wp.empty(self.m, dtype=wp.vec3f, device="cuda")
        c3 = wp.empty(self.m, dtype=wp.float32, device="cuda")
        b = wp.empty(self.m, dtype=wp.float32, device="cuda")
        wp.launch(getC1C2C3B, dim=self.m, inputs=[
                  c1, c2, c3, b, self.EParticles, self.Egrid.id, self.kernel_r, self.dt])
        # relaxedJacobi
        for _ in range(MAX_ITERATIONS):
            wp.launch(RelaxedJacobi, dim=self.m, inputs=[
                      self.Egrid.id, self.EParticles, c1, c2, c3, b, OMEGA, self.kernel_r], device="cuda")
        
        # debugGammamax = wp.zeros(1, dtype=wp.float32, device="cuda")
        # debugGammamin = wp.zeros(1, dtype=wp.float32, device="cuda")
        # debugGammamin.fill_(wp.float32(1000.0))
        # wp.launch(debugGamma, dim=self.m, inputs=[debugGammamin,debugGammamax,self.Egrid.id,self.EParticles,self.kernel_r], device="cuda")
        # print(f"gamma min: {debugGammamin.list()[0]}, max: {debugGammamax.list()[0]}")

        # apply external force
        wp.launch(applyExternalForce, dim=self.m, inputs=[
                  self.EParticles], device="cuda")
        # updateVelocity
        wp.launch(updateEVelocity, dim=self.m, inputs=[
                  self.Egrid.id, self.EParticles, self.kernel_r, ENV_PRESSURE, self.p_in, self.dt], device="cuda")
        
        #print(f"p: {self.p_in.list()[0]-ENV_PRESSURE}, V: {self.bubble_volume.list()[0]}")
        # i=1000
        # Ep=self.EParticles.list()[i]
        # gammai =Ep.c
        # h = Ep.h
        # thick = Ep.thickness
        # rho = Ep.mass/Ep.volume
        # n = Ep.normal
        # na=(self.p_in.list()[0]-ENV_PRESSURE+100.0*2.0*(PURE_WATER_SURFACE_TENSION-IDEAL_GAS_CONSTANT*ENV_TEMPERATURE*gammai)*h)/(rho*thick)
        # print(f"{na}")
        
        pass

    def Geometry(self) -> None:
        # TODO Lthickness
        self.num_density_max.fill_(wp.float32(0.0))
        self.num_density_min.fill_(wp.float32(10.0*self.m))
        self.num_density_averge.fill_(wp.float32(0.0))
        self.normalBuild()
        self.hashBuild()

        wp.launch(getEgeometry, dim=self.m, inputs=[self.Egrid.id, self.EParticles, self.kernel_r,
                  self.num_density_max, self.num_density_min, self.num_density_averge], device="cuda")
        #print(f"{(self.num_density_max-self.num_density_min)/self.num_density_averge}")
        
        #print(f"{self.num_density_max} {self.num_density_min} {self.num_density_averge}")
        pass

    def L2E(self) -> None:
        # alpha compute
        wp.launch(getAlpha, dim=self.n, inputs=[
                  self.Egrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        # m,c,V,p compute
        wp.launch(getMCVPA, dim=self.m, inputs=[
                  self.Lgrid.id, self.EParticles, self.LParticles, self.kernel_r], device="cuda")
        pass

    def updateELposs(self) -> None:
        wp.launch(updateELposs, dim=self.m, inputs=[
                  self.EParticles, self.Eposs], device="cuda")
        wp.launch(updateELposs, dim=self.n, inputs=[
                  self.LParticles, self.Lposs], device="cuda")
        pass

    def normalBuild(self) -> None:
        self.center.fill_(wp.vec3f(0.0))
        wp.launch(centerCompute, dim=self.m, inputs=[
                  self.m, self.EParticles, self.center], device="cuda")
        # wp.launch(PCAnormalBuild, dim=self.n, inputs=[wp.uint64(
        #    self.Lgrid.id), self.LParticles, self.center, self.kernel_r, self.Lnorm], device="cuda")
        wp.launch(PCAnormalBuild, dim=self.m, inputs=[wp.uint64(
            self.Egrid.id), self.EParticles, self.center, self.kernel_r, self.Enorm], device="cuda")
        pass

    def hashBuild(self) -> None:
        self.updateELposs()
        self.Egrid.build(points=self.Eposs, radius=self.kernel_r)
        self.Lgrid.build(points=self.Lposs, radius=self.kernel_r)
        pass

    def de(self) -> None:
        # TODO
        wp.launch(deTest, dim=self.m, inputs=[
                  self.debug, self.EParticles, self.t], device="cuda")
        pass
