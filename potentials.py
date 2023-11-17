from typing import Any
import taichi as ti

@ti.data_oriented
class HarmonicPotential:
    def __init__(self, k):
        self.k = k
    
    @ti.func
    def __call__(self, x):
        return self.k * x
