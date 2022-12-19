"""
@author Donglai Ma
@email donglaima96@gmail.com
@create date 2022-12-06 22:14:44
@modify date 2022-12-06 22:14:44
@desc test particle simulation using taichi
"""

import taichi as ti
import constants as cst
#ti.init(arch = ti.gpu)


""" 
Full relativistic Lorentz equation

https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2011GL046787
"""
ti.init(arch= ti.gpu)



@ti.dataclass
class Particle:
    r: ti.types.vector(3,ti.f32)
    p: ti.types.vector(3,ti.f32) #momentum
    gamma: ti.f32
    m: ti.f32
    q: ti.f32

    @ti.func
    def initParticles(self, mass, charge):
        self.m = mass
        self.q = charge
    @ti.func
    def initPos(self, x, y, z):
        self.r = ti.Vector([x, y, z]) #z would be latitude
    @ti.func
    def initMomentum(self, px, py, pz):
        self.p = ti.Vector([px, py, pz])
    @ti.func
    def getgamma(self):
        self.gamma = ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
        return self.gamma


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
        gamma = self.getgamma()
        print(self.q)
        t = self.q * dt * B / (2 * gamma* cst.C * self.m) # Gaussian unit
        
        p_p = p_minus + p_minus.cross(t)

        s = 2 * t /(1 + t.dot(t))

        p_plus = p_minus + p_p.cross(s)

        self.p = p_plus + self.q * E * dt / 2.0
        

    @ti.func
    def leap_frog(self, dt, E, B, L ):
        self.boris_push(dt, E, B)
        # boris update the momentum

        gamma = self.getgamma()
        gammam = gamma + self.m

        v_xyz = self.p / gammam
        lat = self.r[2]

        ratio = ti.Vector([1,1, (L * cst.Planet_Radius * \
            ti.sqrt((1 + 3 * ti.sin(lat) * ti.sin(lat))) * ti.cos(lat))])


        self.r = self.r + dt * v_xyz * ratio
 

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
        particles[n].initParticles(cst.Mp, cst.Charge)
        #v = particles[1].q
        
        print(particles[n].m)
        particles[n].initPos(0.0,0.0,0.1)

        particles[n].initMomentum(1e-5,0.02,1e-5)
        # B = dipole_field(L, particles[n].r,B0)
        # particles[n].boris_push(-dt/2.0, E0, B)
init()