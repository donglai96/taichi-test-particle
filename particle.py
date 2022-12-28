"""
@author Donglai Ma
@email donglaima96@gmail.com
@create date 2022-12-06 22:14:44
@modify date 2022-12-06 22:14:44
@desc test particle simulation using taichi
"""

from curses import tigetnum
import taichi as ti
import constants as cst
from dipolefield import dipole_field
#ti.init(arch = ti.gpu)


""" 
Full relativistic Lorentz equation

https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2011GL046787
"""
#ti.init(arch= ti.gpu)




@ti.dataclass
class Particle:
    r: ti.types.vector(3,ti.f64)
    p: ti.types.vector(3,ti.f64) #momentum
    #gamma: ti.f64
    m: ti.f64
    q: ti.f64

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
    # @ti.func
    # def getgamma(self):
    #     self.gamma = ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
    #     return self.gamma


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
        # question which p should I calculate the gamma
        #gamma = ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
        gamma = ti.sqrt(1 + p_minus.norm()**2 /(self.m**2 * cst.C**2))
        #print('gamma',gamma)
        #print(self.q)
        t = self.q * dt * B / (2 * gamma* cst.C * self.m) # Gaussian unit
        
        p_p = p_minus + p_minus.cross(t)

        s = 2 * t /(1 + t.dot(t))

        p_plus = p_minus + p_p.cross(s)

        self.p = p_plus + self.q * E * dt / 2.0
        #print(self.p)
        

    @ti.func
    def leap_frog(self, dt, E, B, L ):
        self.boris_push(dt, E, B)
        # boris update the momentum

        # gamma = self.getgamma()
        # gammam = gamma * self.m
        gammam = self.m * ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
        
        v_xyz = self.p / gammam
        lat = self.r[2]
        #print('xyz',v_xyz)
        ratio = ti.Vector([1,1, (L * cst.Planet_Radius * \
            ti.sqrt((1 + 3 * ti.sin(lat) * ti.sin(lat))) * ti.cos(lat))])

        #print('ratio',self.m)
        self.r = self.r + dt * v_xyz * ratio





ti.init(default_fp=ti.f64)


N = 1
charge = cst.Charge
mass = cst.Mp
print(mass)
E0 = ti.Vector([0,0.0,0])
B0 = cst.B0

#print('Earth field at L = 5', B_L)
#w0 = cst.Charge * B0 /(mass * cst.C)
L = 5
print('magnetic field at L = 5',B0/(L**3))
print('gyrofrequency is ',cst.Charge * B0/(L**3) /(mass * cst.C),'rad/s')
print('the test frequency at L = 5 is', 0.002496 * 1.76e7)
wc0 = cst.Charge * B0/(L**3) /(mass * cst.C)
Np = 100             # Number of cyclotronic periods

Tf =2 * 3.1415926 / (cst.Charge * B0/(L**3) /(mass * cst.C))
print('Tf',Tf)
Nt = 2000
dt = Tf/50
#Nt = 200
#print('time step',Nt)
r_taichi = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N,Nt))
p_taichi = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N,Nt))


particles = Particle.field(shape = (N,))

@ti.kernel
def init():
    for n in range(N):
        particles[n].initParticles(cst.Mp, cst.Charge)
        #v = particles[1].q
        
        #print(particles[n].q)
        particles[n].initPos(0.0,-0.3,0.0)

        particles[n].initMomentum(1.2e-23,0.0,0.0)
        #print('init!')
        B = dipole_field(5, [0,0.3,0.0],B0)
        print('the init B field is',B)
        particles[n].boris_push(-dt/2.0, E0, B)

@ti.kernel
def simulate():
    for n in range(N):
        B = dipole_field(L, particles[n].r,B0)
        particles[n].leap_frog(dt,E0,B,L)
        #r_taichi[n,i] = particles[n].r
        #p_taichi[n,i] = particles[n].p
        print('n = ',n,'location',particles[n].r)
        #print(particles[1].r.norm())
        print('momentum * e20',particles[n].p.norm()*1e20)
            #print(B)

# init()
# simulate()
# #print(particles[1].p.norm()**2 /(particles[1].m**2 * cst.C**2))
# print('pnorm',particles[1].p.norm()**2 )
# print('pnorm2',particles[1].p.norm()**2 )


# # r = ti.Vector([1,2,3])
# # BB = dipole_field(5,r,B0)
# print('location',r_taichi[3,10])
# print(p_taichi[3,100].norm())
# print(p_taichi[3,10].norm())

gui = ti.GUI("Solar System", background_color=0x25A6D9)

while True:
    if gui.get_event():
        if gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
            init()
            #print(particles[1].r)
    simulate()
        #print(particles[1].r)
    
    #gui.circle([, 0.5], radius=20, color=0x8C274C)
    gui.circle([0.5, 0.5], radius=20, color=0x8C274C)
    gui.circles(particles[0].r.to_numpy()[:2].reshape(1,2) +[0.5,0.5], radius=5, color=0xFFFFFF)
    gui.show()
