"""
@author Donglai Ma
@email donglaima96@gmail.com
@create date 2022-12-06 22:14:44
@modify date 2022-12-06 22:14:44
@desc test particle simulation using taichi
"""

import taichi as ti
import constants as cst
ti.init(arch = ti.gpu)


""" 
Full relativistic Lorentz equation

https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2011GL046787
"""



@ti.dataclass
class Particle:
    r: ti.types.vector(3,ti.f32)
    p: ti.types.vector(3,ti.f32) #momentum
    gamma: ti.f32
    mass: ti.f32
    charge: ti.f32
    gamma: ti.f32

    @ti.func
    def initParticles(self, m, q):
        self.mass = m
        self.charge = q
    @ti.func
    def initPos(self, x, y, z):
        self.r = ti.Vector([x, y, z])
    @ti.func
    def initMomentum(self, px, py, pz):
        self.p = ti.Vector([px, py, pz])

    @ti.func
    def boris_push(self, dt, E, B):
        """Push the particles using Boris' method
        Update the velocity of particles
        An example for non-relativistic:
         https://www.particleincell.com/2011/vxb-rotation/

        p_n_minus = p_minus - qE*dt/2
        p_n_plus = p_plus + qE*dt/2
        (p_plus - p_minus) / dt = q/2 * (p_plus + p_minus)/(gamma * m0)
        ...
        Args:
            dt (_type_): _description_
            E (_type_): _description_
            B (_type_): _description_
        """

        p_minus = self.p + self.q * E * dt / 2.0
        t = self.q * dt * B / (2 * self.gamma * cst.C * self.m) # Gaussian unit
        p_p = p_minus + p_minus.cross(t)

        s = 2 * t /(1 + t.dot(t))

        p_plus = p_minus + p_p.cross(s)

        self.p = p_plus + self.q * E * dt / 2.0

    



