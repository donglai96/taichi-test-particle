

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
from wavepacket import Wave
from wave import Waves
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
    alpha: ti.f64 #pitch angle
    alpha0: ti.f64 #equator pitch angle
    t:ti.f64

    @ti.func
    def initParticles(self, mass, charge):
        self.m = mass
        self.q = charge
        self.t = 0
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
    def leap_frog(self, dt, E, B, L ):

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
        self.boris_push(dt, E, B)
        gammam = self.m * ti.sqrt(1 + self.p.norm()**2 /(self.m**2 * cst.C**2))
        
        v_xyz = self.p / gammam
        lat = self.r[2]
        #print('xyz',v_xyz)
        ratio = ti.Vector([1,1, 1/(L * cst.Planet_Radius * \
            ti.sqrt((1 + 3 * ti.sin(lat) * ti.sin(lat))) * ti.cos(lat))])
        self.r = self.r + 0.5 * dt * v_xyz * ratio


if __name__ == "__main__" :
    


    N = 24
    charge = -1 * cst.Charge
    mass = cst.Me
    print(mass)
    E0 = ti.Vector([0,0.0,0])
    B0 = cst.B0
    particle_ev = 9442.1022112 #ev
    particle_eg = ev2erg(particle_ev)
    
    
    pitch_angle = get_pitchanlge_numpy(np.deg2rad(30),np.deg2rad(15),-1)
    print('Initial pitch angle!!', pitch_angle)
    #pitch_angle = np.deg2rad(4)
    ppp = e2p(particle_eg )
    pz = ppp * np.cos(pitch_angle)
    p_perp = ppp*np.sin(pitch_angle)
    phi_particle = np.linspace(0.0,   2 * np.pi, N)
    px_numpy = np.cos(phi_particle) * p_perp
    py_numpy = np.sin(phi_particle) * p_perp

    
    print(px_numpy)
    print('px and py',px_numpy,py_numpy)
    print('initial p',pz,p_perp)
    #print('Earth field at L = 5', B_L)
    #w0 = cst.Charge * B0 /(mass * cst.C)
    L = 6.4
    print('magnetic field at L = 6.4',B0/(L**3))
    print('gyrofrequency is ',cst.Charge * B0/(L**3) /(mass * cst.C),'rad/s')
    print('the test frequency at L = 6.4 is', 0.002496 * 1.76e7)
    wc0 = cst.Charge * B0/(L**3) /(mass * cst.C)
    Np = 72           # Number of cyclotronic periods

    Tf =2 * 3.1415926 / (cst.Charge * B0/(L**3) /(mass * cst.C))
    print('Tf',Tf)
    Nt = 200000
    dt = Tf/500
    print(dt,'dt outof scope')
    record_num = 100
    #Nt = 200
    #print('time step',Nt)
    ti.init(arch = ti.cpu,default_fp=ti.f64)

    px_init = ti.field(dtype = ti.f64,shape = (N,))
    py_init= ti.field(dtype = ti.f64,shape = (N,))
    px_init.from_numpy(px_numpy)
    py_init.from_numpy(py_numpy)
    print('px_init',1e20*px_numpy)
    r_taichi = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N,Nt//record_num))
    p_taichi = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N,Nt//record_num))
    r_taichi_single = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N))
    p_taichi_single = ti.Vector.field(n = 3,dtype = ti.f64,shape = (N))


    particles = Particle.field(shape = (N,))

    # init the wave
    # first init the numpy
    NW = 1
    nlat = 10000
    latmax = 30
    n0 = 10 #cm^-3
    Bw0 = 3e-6# Gauss
    #Bw0 = 0
    fwave = 980 * 2 * np.pi
    
    waves_numpy = Waves( NW, L, nlat, 0, latmax*np.pi/180, n0)
    #Bw_lat = np.zeros((NW, nlat)) + Bw0

    #modulate the amplitude 
    # following https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085987
    # Bw = Bw0 tanh(\lambda) tanh(30 - \lambda)

    #ratio_Bw_lat = Bw0 * np.tanh(np.rad2deg(waves_numpy.lats)) * np.tanh(30 - np.rad2deg(waves_numpy.lats))
    ratio_Bw_lat = Bw0
    Bw_lat = np.zeros((NW, nlat)) + ratio_Bw_lat
    waves_numpy.generate_parallel_wave(np.array([fwave]),Bw_lat)

    # plt.plot(np.rad2deg(waves_numpy.lats),Bw_lat[0,:])
    # plt.show()
    # plt.plot(waves_numpy.lats[:],waves_numpy.phi_z[0,:])
    # plt.show()
    phiz_interp = ti.field(dtype = ti.f64,shape = (NW,nlat))
    Bw_interp = ti.field(dtype = ti.f64,shape = (NW,nlat))
    lat_interp = ti.field(dtype = ti.f64,shape = (nlat,))

    phiz_interp.from_numpy(waves_numpy.phi_z)
    Bw_interp.from_numpy(waves_numpy.Bw0s)
    lat_interp.from_numpy(waves_numpy.lats)

    #init waves in taichi
    waves = Wave.field(shape = (NW,))



    lat_init = 15


    @ti.kernel
    def init():
        for n in range(N):
            particles[n].initParticles(mass, charge)
            #v = particles[1].q
            
            #print(particles[n].q)
            particles[n].initPos(0.0,0.0,lat_init*ti.math.pi/180)

            particles[n].initMomentum(px_init[n],py_init[n],pz)
            #print('init!')
            B = dipole_field(L, particles[n].r,B0)
            print('the init B field is',B)
            print('the init p is',1e20*particles[n].p)
            for m in range(NW):
                waves[m].initialize(fwave , L, n0, 
                    0.0*ti.math.pi/180, latmax*ti.math.pi/180)
            #particles[n].boris_push(-dt/2.0, E0, B)

    @ti.kernel
    def simulate():
        """This is for one time step, for video"""
        for n in range(N):
            B = dipole_field(L, particles[n].r,B0)
            Bw = ti.Vector([0.0,0.0,0.0])
            Ew = ti.Vector([0.0,0.0,0.0])
            #get Bw
            for m in (range(NW)):
                    
                    if particles[n].r[2] < latmax*ti.math.pi/180 :
                        phiz_index = interp(lat_interp,particles[n].r[2],nlat)
                        #print('phiz_index is', phiz_index)
                        #print('lat_interp is', particles[n].r[2])
                        index1 = ti.floor(phiz_index,dtype = ti.i32)
                        index2 = index1 + 1
                        phiz = ti.math.mix(phiz_interp[m,index1],phiz_interp[m,index2],phiz_index-index1)
                        # print('interp process',phiz_interp[m,index1],phiz_interp[m,index2],phiz_index)
                        # print('interp result',phiz)
                        #print('index',index1)
                        #print('phiz',phiz)
                        Bw_interp = ti.math.mix(Bw_interp[m,index1],Bw_interp[m,index2],phiz_index-index1)

                        waves[m].get_field(particles[n].r,particles[n].t, phiz, Bw_interp)
                        
                        #print('t_dt in scope',dt)n
                        #print('Bw',Bw_interp)
                        #print('t!',t)
                        #print(Bw_interp)
                        Bw += waves[m].Bw
                        Ew += waves[m].Ew
            E = Ew
            B = B + Bw
            #print('E field is :',E)
            #print(Bw)
            particles[n].leap_frog(dt,E,B,L)
            particles[n].t += dt
            r_taichi_single[n] = particles[n].r
            p_taichi_single[n] = particles[n].p
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
        
        for n in ti.static(range(N)):
            for t in (range(Nt)):
                B = dipole_field(L, particles[n].r,B0)
                particles[n].leap_frog(dt,E0,B,L)
                particles[n].get_pitchangle()
                if t%record_num ==0:
                    #print(t/Nt * 100,'percent finished')
                    r_taichi[n,t//record_num] = particles[n].r
                    p_taichi[n,t//record_num] = particles[n].p
            
                # print('n = ',n,'location',particles[n].r)
                # #print(particles[1].r.norm())
                # print('momentum * e20',particles[n].p.norm()*1e20)
                # print('equator pitch angle',
                # get_equator_pitchangle(particles[n].alpha,particles[n].r[2])* 180/ti.math.pi)
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
    p_record = np.zeros((Nt//record_num,N,3))
    r_record = np.zeros((Nt//record_num,N,3))
    for t in range(Nt):
        simulate()
        if t % record_num ==0:
            #print(100 * t//Nt, "percent finshed")
            for n in range(N):
                p_record[t //record_num,n,:] = particles[n].p.to_numpy()
                #print('this is p',particles[n].p.to_numpy())
                r_record[t //record_num,n,:] = particles[n].r.to_numpy()
                #print('this is r',particles[n].r.to_numpy())
    print('finshed')
    # r_taichi_result = r_taichi.to_numpy()
    # p_taichi_result = p_taichi.to_numpy()

    r_taichi_result_final = r_taichi_single.to_numpy()
    p_taichi_result_final = p_taichi_single.to_numpy()
    # get index where p < 0


    print('final r')
    print(r_taichi_result_final)
    print('final p')
    print(1e20 * p_taichi_result_final)
    
    r_taichi_result = r_record[:,:,:]
    p_taichi_result = p_record[:,:,:]

    print(r_taichi_result.shape)
    alpha_result = np.zeros((N,Nt//record_num))
    p_numpy = np.sqrt((p_taichi_result[:,:,0])**2 + \
        (p_taichi_result[:,:,1])**2 \
        + (p_taichi_result[:,:,2])**2)
    alpha_result = np.arccos(p_taichi_result[:,:,2] \
        /np.sqrt((p_taichi_result[:,:,0])**2 + \
        (p_taichi_result[:,:,1])**2 \
        + (p_taichi_result[:,:,2])**2))

    alpha_eq_result = get_equator_pitchangle_numpy(alpha_result,r_taichi_result[:,:,2])
    energy_result = erg2ev(p2e(p_numpy))/1000

    alpha_result = r_taichi_result[:,:,2]

    #plot the result 
    fig, axs = plt.subplots(5, sharex= False)
    for i in range (N):
        axs[0].plot(np.rad2deg(alpha_result[:,i]),np.rad2deg(alpha_eq_result[:,i]))
        axs[0].set_ylim(-5,70)
        axs[0].set_xlim(np.rad2deg(alpha_result[:,i][0]),0)
        axs[1].plot(np.rad2deg(alpha_result[:,i]),energy_result[:,i])
        axs[1].set_ylim(95,130)
        axs[1].set_xlim(np.rad2deg(alpha_result[:,i][0]),0)
        #plot of time

        axs[2].plot(np.rad2deg(alpha_result[:,i]))
        
        axs[3].plot(np.rad2deg(alpha_eq_result[:,i]))
        axs[3].set_ylim(-5,70)
        axs[4].plot(energy_result[:,i])
        #axs[2].set_xlim()
        #axs[1].plot(np.rad2deg(alpha_result[:,i]),1e20 * p_numpy[:,i])
        #axs[1].set_ylim(1000,1400)
        #axs[2].plot(np.rad2deg(alpha_result[:,i]),np.rad2deg(alpha_result[:,i]))
    #plt.xlim(np.rad2deg(alpha_result[:,i][0]),90)
    axs[0].set_ylabel('Equator pitch angle')
    axs[1].set_ylabel('Energy(keV)')
    axs[2].set_ylabel('pitch angle_with time')
    axs[3].set_ylabel('Equator pitch angle with time')
    axs[4].set_ylabel('Energy with time')
    plt.show()