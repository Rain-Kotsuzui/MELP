import warp as wp
import warp.render as wr
import numpy as np
from .data import *
from .lib import *
from .particleSystem import *
from .render import *

wp.init()

def MELP(dt:float,N:int =BUBBLE_COUNT,n:int=PARTICLE_COUNT, frame:int=FRAME) -> None:

    Bubble = [ParticleSystem(dt,n) for _ in range(N)]
    dt = 1.0 / float(frame)
    preview = render()

    while preview.running():
        for _ in range(SUBSTEPS):
            for i in range(N):
                Bubble[i].substep(dt/float(SUBSTEPS))
        preview.rend(Bubble)