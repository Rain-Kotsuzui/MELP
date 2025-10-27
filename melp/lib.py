import warp as wp
from typing import Any

BUBBLE_COUNT = wp.constant(wp.int32(1))
PARTICLE_COUNT = wp.constant(wp.int32(10000))
PARTICLE_RADIUS = wp.constant(wp.float32(0.05))
PARTICLE_MASS = wp.constant(wp.float32(1.0))
PARTICLE_VOLUME = wp.constant(wp.float32(4.0/3.0) * wp.pi * wp.pow(wp.float32(PARTICLE_RADIUS), wp.float32(3.0)))
PARTICLE_SURFACTANT = wp.constant(wp.float32(1.0))

INF_SMALL = wp.constant(wp.float32(1e-6))
RADIUS = wp.constant(wp.float32(0.1))
MASS = wp.constant(wp.float32(1.0))

KERNEL_RAIDUS = wp.constant(wp.float32(0.2))
FRAME = wp.constant(wp.int32(1000))
SUBSTEPS = wp.constant(wp.int32(5))

PHI = wp.constant(wp.float32(2.399963229728653))
