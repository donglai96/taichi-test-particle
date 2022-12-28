"""
@author Donglai Ma
@email donglaima96@gmail.com
@create date 2022-12-10 16:51:17
@modify date 2022-12-10 16:51:17
@desc Simulate the particles, the Simulation class lauch the simulation 
and prepare for the plot
"""


from particle import Particle
import taichi as ti
import constants as cst
import numpy as np
from dipolefield import *
ti.init(debug = True)

N = 5
charge = cst.Charge
mass = cst.Mp
print(mass)
E0 = ti.Vector([0,25,0])
B0 = cst.B0
w0 = cst.Charge * 1 /mass
dt = 1
L = 5
Np = 100             # Number of cyclotronic periods

#Tf = Np * 2 * np.pi / w0
#Nt = int(Tf // dt) 
Nt = 10000
print(Nt)
r_taichi = ti.Vector.field(n = 3,dtype = ti.f32,shape = (N,Nt))
p_taichi = ti.Vector.field(n = 3,dtype = ti.f32,shape = (N,Nt))


particles = Particle.field(shape = (N,))

@ti.kernel
def init():
    
    for n in range(N):
        #particles[n].initParticles(cst.Mp, cst.Charge)
        #v = particles[1].q
        particles[n].m = 1.6726e-27
        particles[n].q = cst.Charge
        particles[n].initPos(0.0,0.0,0.1)
        print('test')
        print(particles[n].m)
        particles[n].initMomentum(1e-5,0.02,1e-5)
        B = dipole_field(L, particles[n].r,B0)
        particles[n].boris_push(-dt/2.0, E0, B)
        # x = particles[n].r
        # print('hello', x)
@ti.kernel
def update():
    for i in range(Nt):
        for n in range(N):
            B = dipole_field(L, particles[n].r,B0)
            x = particles[n].r
            print('hello', x)

            rr = B
            print('hellor ' , rr)

            particles[n].leap_frog(dt,E0,B,L)
            r_taichi[n,i] = particles[n].r
            p_taichi[n,i] = particles[n].p

print('start')
init()
print('init finished')
#update()
print('finished')
print(r_taichi[3,3])
    