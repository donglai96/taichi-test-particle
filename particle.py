"""
@author Donglai Ma
@email donglaima96@gmail.com
@create date 2022-12-06 22:14:44
@modify date 2022-12-06 22:14:44
@desc test particle simulation using taichi
"""

from curses import tigetnum
from unicodedata import name
import taichi as ti
import constants as cst
from dipolefield import dipole_field
from taichiphysics import *
import matplotlib.pyplot as plt
from wave import Waves
from taichiphysics import *
from wavepacket import Wave

#ti.init(arch = ti.gpu)


""" 
Full relativistic Lorentz equation

https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2011GL046787
"""
#ti.init(arch= ti.gpu)



@ti.dataclass
class Particle(object):
    r: ti.types.vector(3,ti.f64)
    p: ti.types.vector(3,ti.f64) #momentum
    #gamma: ti.f64
    m: ti.f64
    q: ti.f64
    alpha: ti.f64 #pitch angle

    alpha0: ti.f64 #equator pitch angle
    # detect_phiz: ti.types.vector(1,ti.f64)
    # detect_k:ti.types.vector(1,ti.f64)
    #phiz: ti.field(ti.f64,shape = 1)
    # def __init__(self,nw):
        
    #     self.phi = ti.types.vector(nw,ti.f64)
    #phi:ti.f64

    @ti.func
    def initParticles(self, mass, charge):
        self.m = mass
        self.q = charge
    @ti.func
    def initPos(self, x, y, z):
        self.r = ti.Vector([x, y, z]) #z would be latitude
        #self.detect_phiz = ti.Vector([0])
        #self.phiz = ti.field(ti.f64,shape = 1)
    @ti.func
    def initMomentum(self, px, py, pz):
        self.p = ti.Vector([px, py, pz])
    
    # @ti.func
    # def getgamma(self):
    #     self.gamma = ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
    #     return self.gamma
    @ti.func
    def get_pitchangle(self):
        self.alpha = ti.acos(self.p[2]/self.p.norm())
    

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
    def leap_frog(self, dt, E, B, L):

        #self.boris_push(dt, E, B)
        # boris update the momentum

        # gamma = self.getgamma()
        # gammam = gamma * self.m
        
        gammam = self.m * ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
        
        v_xyz = self.p / gammam
        lat = self.r[2]
        #print('xyz',v_xyz)
        ratio = ti.Vector([1,1, 1/(L * cst.Planet_Radius * \
            ti.sqrt((1 + 3 * ti.sin(lat) * ti.sin(lat))) * ti.cos(lat))])

        # #print('ratio',self.m)
        self.r = self.r + 0.5 * dt * v_xyz * ratio
        #dr1 = 0.5 * dt * v_xyz * ratio
        #print('dr1')
        #self.detect_phiz+= 0.5 *self.detect_k* dr1[2]
        self.boris_push(dt, E, B)
        gammam = self.m * ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
        
        v_xyz = self.p / gammam
        lat = self.r[2]
        #print('xyz',v_xyz)
        ratio = ti.Vector([1,1, 1/(L * cst.Planet_Radius * \
            ti.sqrt((1 + 3 * ti.sin(lat) * ti.sin(lat))) * ti.cos(lat))])
        self.r = self.r + 0.5 * dt * v_xyz * ratio
        #dr2 = 0.5 * dt * v_xyz * ratio
        #self.detect_phiz+= 0.5 *self.detect_k* dr2[2]
        

if __name__ == "__main__" :
    N =3
    particle_ev = 9800 #ev
    particle_eg = ev2erg(particle_ev)

    pitch_angle = get_pitchanlge_numpy(np.deg2rad(30),np.deg2rad(15),-1)
    
    ppp = erg2p(particle_eg , cst.Me,cst.C)
    pz = ppp * np.cos(pitch_angle)
    p_perp = ppp*np.sin(pitch_angle)
    #px_numpy = np.zeros(N)
    #py_numpy = np.zeros(N)
    phi_particle = np.linspace(0, 2 * np.pi, N)
    px_numpy = np.cos(phi_particle) * p_perp
    py_numpy = np.sin(phi_particle) * p_perp
    print('px and py',px_numpy,py_numpy)
    Bw0 = 0.0# Gauss
    fwave = 980  #Hz
    nlat = 100000
    latmax = 15
    latmin = 0
    n0 =10
    L = 5
    nw = 1
    waves_numpy = Waves( nw, L, nlat, 0, latmax*np.pi/180, n0)
    Bw_lat = np.zeros((nw, nlat)) + Bw0
    waves_numpy.generate_parallel_wave(np.array([fwave *2 * np.pi]),Bw_lat)
    print(waves_numpy.k.shape)
    # plt.plot(Bw_lat[0])
    # plt.show()
    # plt.plot(np.rad2deg(waves.lats),waves.k[0,:])
    # plt.plot(np.rad2deg(waves.lats),waves.phi_z[0,:])
    # plt.show()
    ti.init(default_fp=ti.f64)

    lat_launch = 15 * ti.math.pi/180


    phiz_interp = ti.field(dtype = ti.f64,shape = (nw,nlat))
    lat_interp = ti.field(dtype = ti.f64,shape = (nlat,))

    phiz_interp.from_numpy(waves_numpy.phi_z)
    lat_interp.from_numpy(waves_numpy.lats)
    
    px = ti.field(dtype = ti.f64,shape = (N,))
    py = ti.field(dtype = ti.f64,shape = (N,))
    px.from_numpy(px_numpy)
    py.from_numpy(py_numpy)
    

    charge = -1 * cst.Charge
    mass = cst.Me
    print(mass)
    E0 = ti.Vector([0,0.0,0])
    B0 = cst.B0
    
    #lat_max = 15
    Nw = 1
    #nlats = 100000

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
    Nt = 5000000
    dt = Tf/500
    record_num = 100
    #Nt = 200
    #print('time step',Nt)
    r_taichi = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N,Nt//record_num))
    p_taichi = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N,Nt//record_num))
    phi_taichi = ti.Vector.field(n = Nw,dtype = ti.f64,shape = (N,Nt//record_num))
    Bw_taichi = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N,Nt//record_num))

    k_taichi = ti.Vector.field(n = Nw,dtype = ti.f64,shape = (N,Nt//record_num))
    particles = Particle.field(shape = (N,))
    waves = Wave.field(shape = (Nw,))
    print('The init pz is',pz)
    # phiz_taichi = ti.field(dtype = ti.f64,shape = (Nw,))
    # phiz_taichi[0] = 0
    # k_taichi = ti.field(dtype = ti.f64,shape = (Nw,))
    # k_taichi[0] = 0
    # k_nw = ti.Vector([0.0])
    #k_nw = ti.field(dtype = ti.f64,shape = (Nw,))
    #print('k_field,',0.5 * k_nw)
    @ti.kernel
    def init():
        for n in range(N):
            particles[n].initParticles(mass, charge)
            #print('init phi',phiz_taichi[0])
            #v = particles[1].q
            
            #print(particles[n].q)
            particles[n].initPos(0.0,0.0,lat_launch)

            particles[n].initMomentum(1e-18,0,-1.2e-18)
            #print('init!')
            B = dipole_field(5, particles[n].r,B0)
            #print('the init B field is',B)
            #print('init the wave')
            for m in range(Nw):
                waves[m].initialize(fwave * 2 *ti.math.pi, L, n0, Bw0,
                    latmin*ti.math.pi/180, latmax*ti.math.pi/180)
            #particles[n].boris_push(-dt/2.0, E0, B)

    @ti.kernel
    def simulate():
        """This is for one time step, for video"""
        for n in range(N):
            B = dipole_field(L, particles[n].r,B0)
            #Bw = waves[0].Bw
            # for m in range(Nw):
            #     phiz_minus = waves[m].phiz
            #     k = waves[m].k
            #     #update the phase
            #     waves[m].phiz = particles[n].leap_frog(dt,E0,B,L,phiz_minus, k)
            #     waves[m].get_field(particles[n].r,)
            #particles[n].get_pitchangle()
            #r_taichi[n,i] = particles[n].r
            #p_taichi[n,i] = particles[n].p
            #print('n = ',n,'location',particles[n].r)
            #print(particles[1].r.norm())
            #print('momentum * e20',particles[n].p.norm()*1e20)
            #print('equator pitch angle',particles[n].alpha)
    @ti.kernel
    def simulate_t():
        """This is for one time step, for video"""
        for t in range(Nt):
            for n in range(N):
                #get the field
                Bw = ti.Vector([0.0,0.0,0.0])
                Ew = ti.Vector([0.0,0.0,0.0])
                # based on lat get the wave info : phiz,

                # for m in range(Nw):
                #     pass
                    # if particles[n].r[2] < latmax*ti.math.pi/180 :
                    #     phiz_index = interp(lat_interp,particles[n].r[2],nlat)
                    #     index1 = ti.floor(phiz_index,dtype = ti.i32)
                    #     index2 = index1 + 1
                    #     phiz = ti.math.mix(phiz_interp[m,index1],phiz_interp[m,index2],phiz_index-index1)
                    #     #print('index',index1)
                    #     #print('phiz',phiz)
                    #     waves[m].get_field(particles[n].r,t * dt, phiz)
                    #     Bw += waves[m].Bw
                    #     Ew += waves[m].Ew
                

                B = dipole_field(L, particles[n].r,B0) 
                #print(B.norm())
                # Bw = waves[0].get_Bfield(lat,t)
                # Ew = waves[0].get_Efield(lat,t)
                # B =B + Bw
                # E = Ew
                # print('Bw',Bw)
                # print('Ew',Ew)
                particles[n].leap_frog(dt,E0,B,L)
                
                particles[n].get_pitchangle()
                if t%record_num ==0:
                    r_taichi[n,t//record_num] = particles[n].r
                    p_taichi[n,t//record_num] = particles[n].p
                    Bw_taichi[n,t//record_num] = Ew
                    #print()
                    # phi_taichi[n,t//record_num] = particles[n].detect_phiz
                    # k_taichi[n,t//record_num] = particles[n].detect_k
                # print('n = ',n,'location',particles[n].r)
                # #print(particles[1].r.norm())
                # print('momentum * e20',particles[n].p.norm()*1e20)
                # print('equator pitch angle',
                # get_equator_pitchangle(particles[n].alpha,particles[n].r[2])* 180/ti.math.pi)
                #print(B)
        print(Nt)

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
# simulation of particle traj
    # gui = ti.GUI("Solar System", background_color=0x25A6D9)

    # while True:
    #     if gui.get_event():
    #         if gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
    #             init()
    #             #print(particles[1].r)
    #     simulate()
    #         #print(particles[1].r)
        
    #     #gui.circle([, 0.5], radius=20, color=0x8C274C)
    #     gui.circle([0.5, 0.5], radius=20, color=0x8C274C)
    #     gui.circles(particles[0].r.to_numpy()[:2].reshape(1,2) +[0.5,0.5], radius=5, color=0xFFFFFF)
    #     gui.show()
    init()
    simulate_t()
    print('finshed')
    r_taichi_result = r_taichi.to_numpy()
    p_taichi_result = p_taichi.to_numpy()
    Bw_taichi_result = Bw_taichi.to_numpy()
    print(r_taichi_result.shape)
    alpha_result = np.zeros((N,Nt//record_num))
    p_numpy = np.sqrt((p_taichi_result[:,:,0])**2 + \
        (p_taichi_result[:,:,1])**2 \
        + (p_taichi_result[:,:,2])**2)
    alpha_result = np.arccos(p_taichi_result[:,:,2] \
        /np.sqrt((p_taichi_result[:,:,0])**2 + \
        (p_taichi_result[:,:,1])**2 \
        + (p_taichi_result[:,:,2])**2))

    phi_result = phi_taichi.to_numpy()[:,:,0]
    k_result = k_taichi.to_numpy()[:,:,0]
    alpha_eq_result = get_equator_pitchangle_numpy(alpha_result,r_taichi_result[:,:,2])

    fig,axs = plt.subplots(5,sharex=False)
    for i in range(N):
        axs[0].plot(np.rad2deg(r_taichi_result[i,:,2]),np.rad2deg(alpha_eq_result[i,:]))
        axs[0].set_ylim(0,90)
        axs[1].plot(np.rad2deg(r_taichi_result[i,:,2]),1e20*p_taichi_result[i,:,2])
        #axs[1].set_ylim(100,200)

        axs[0].set_xlim([np.rad2deg(r_taichi_result[0,:,2])[0],np.rad2deg(r_taichi_result[0,:,2])[-1]])

        axs[1].set_xlim([np.rad2deg(r_taichi_result[0,:,2])[0],np.rad2deg(r_taichi_result[0,:,2])[-1]])
        axs[2].plot(np.rad2deg(alpha_result[i,:]))
        axs[2].axhline(90,color = 'green')
        axs[3].plot(np.rad2deg(r_taichi_result[i,:,2]))

        axs[4].plot(np.rad2deg(r_taichi_result[i,:,2]),np.rad2deg(Bw_taichi_result[i,:,0]),color = 'red')
        axs[4].plot(np.rad2deg(r_taichi_result[i,:,2]),np.rad2deg(Bw_taichi_result[i,:,1]),color = 'green')
        axs[4].plot(np.rad2deg(r_taichi_result[i,:,2]),np.rad2deg(Bw_taichi_result[i,:,2]),color = 'blue')
        axs[4].set_xlim([np.rad2deg(r_taichi_result[0,:,2])[0],np.rad2deg(r_taichi_result[0,:,2])[-1]])

    # axs[4].plot(np.rad2deg(r_taichi_result[0,:,2]),phi_result[0,:])
    # axs[5].plot(np.rad2deg(r_taichi_result[0,:,2]),k_result[0,:])
    # axs[4].set_xlim([np.rad2deg(r_taichi_result[0,:,2])[0],np.rad2deg(r_taichi_result[0,:,2])[-1]])
    # axs[5].set_xlim([np.rad2deg(r_taichi_result[0,:,2])[0],np.rad2deg(r_taichi_result[0,:,2])[-1]])

    #axs[2].axhline(90,color = 'green')
    plt.show()