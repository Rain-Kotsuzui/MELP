import warp as wp
from typing import Any

BUBBLE_COUNT = wp.constant(wp.int32(1))
PARTICLE_COUNT = wp.constant(wp.int32(10000))
PARTICLE_RADIUS = wp.constant(wp.float32(0.05))
PARTICLE_VOLUME = wp.constant(wp.float32(
    4.0/3.0) * wp.pi * wp.pow(wp.float32(PARTICLE_RADIUS), wp.float32(3.0)))
PARTICLE_SURFACTANT = wp.constant(wp.float32(0.3))
PARTICLE_THICKNESS = wp.constant(wp.float32(1e-6))
PARTICLE_DENSITY = wp.constant(wp.float32(1000.0))
PARTICLE_MASS = PARTICLE_DENSITY * PARTICLE_VOLUME

INF_SMALL = wp.constant(wp.float32(1e-6))
RADIUS = wp.constant(wp.float32(0.1))
MASS = wp.constant(wp.float32(1.0))

KERNEL_RAIDUS = 10.0*PARTICLE_RADIUS
FRAME = wp.constant(wp.int32(1000))
SUBSTEPS = wp.constant(wp.int32(5))

PHI = wp.constant(wp.float32(2.399963229728653))
IDEAL_GAS_CONSTANT = wp.constant(wp.float32(8.31446261815324))

# ENVIRONMENT
ENV_TEMPERATURE = wp.constant(wp.float32(298.0))
ENV_PRESSURE = wp.constant(wp.float32(101325.0))
GRIVATY = wp.constant(wp.vec3f(0, -9.81, 0))

PURE_WATER_SURFACE_TENSION = wp.constant(wp.float32(0.072))

# JACOBI
MAX_ITERATIONS = wp.constant(wp.int32(10))
OMEGA = wp.constant(wp.float32(0.3))

# REDISTRIBUTION
THRESHOLD = wp.constant(0.4)
