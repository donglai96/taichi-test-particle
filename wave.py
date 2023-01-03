import numpy as np
import taichi as ti
import constants as cst
from scipy import integrate
import matplotlib.pyplot as plt

# @ti.data_oriented
# class Waves:
#     def __init__(self, direction, L, nlat, nw,lat_min,lat_max):
#         self.direction = direction
#         self.nlat = nlat
#         self.nw = nw
#         self.L = L 
#         self.lat_min = lat_min
#         self.lat_max = lat_max
#         self.wce = ti.field(ti.f64, shape=nlat)
#         self.wpe = ti.field(ti.f64, shape=nlat)
#         self.Bw = ti.Vector.field(3,shape=(nlat, nw))
#         self.Ew = ti.Vector.field(3,shape=(nlat, nw))

#     @ti.func
#     def
    
def get_dipole_numpy(L, lat, B0):
    cos_lat= np.cos(lat)
    sin_lat = np.sin(lat)
    Bz = B0/(L**3 * cos_lat**6) * np.sqrt(1 + 3 * sin_lat ** 2)
    return Bz
def get_density_numpy(n0,lat):
    return n0 * (np.cos(lat)**-4)
def s_numpy(L, R, lat):
    x = np.sin(lat)
    tt = np.sqrt(x**2 + 1.0 / 3)
  
    return L * R * np.sqrt(3) * (0.5 * x * tt + np.log(np.sqrt(3) * abs(x + tt)) / 6.0)
        
# The wave can be written in a numpy regime
# The locate function should be in taichi regime
class Waves(object):
    def __init__(self, direction, L, nlat, lat_min, lat_max, n0):
        self.direction = direction
        self.nlat = nlat
        #self.nw = nw
        self.L = L
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.n0 = n0

        self.lats, self.dlat = np.linspace(lat_min, lat_max,num = self.nlat, retstep= True)# generate lats
        print(self.lats)
        self.wces = get_dipole_numpy(self.L, self.lats,cst.B0)* cst.Charge/ (cst.Me * cst.C) # wce from latmin to latmax
        self.ne_lats = get_density_numpy(self.n0, self.lats)
        self.wpes = np.sqrt(4 * np.pi * self.ne_lats * cst.Charge**2 / cst.Me)
        #ti.sqrt(4 * ti.math.pi *n0*cst.Charge**2 /cst.Me)
    def generate_parallel_wave(self,ws,Bw0s):
        """This is for parallel chorus waves

        Args:
            ws (_type_): _description_
            Bw0s (_type_): _description_
        """
        self.ws = ws
        self.nw = ws.shape[0]
        if Bw0s.shape != (self.nw, self.nlat):
            raise ValueError("Please check the inital Bw0!")
        self.Bw0s = Bw0s
        kk = np.zeros((self.nw, self.nlat))
        phi_z = np.zeros((self.nw, self.nlat))
        for i in range(self.nw):
            RR = 1 + self.wpes**2 / ((self.wces - self.ws[i]) * self.ws[i])
            kk[i,:] = self.ws[i]* np.sqrt(RR)/cst.C
        self.k = kk # get the k of function (nw, nlat)
        # get the phase kz as a function of (nw, nlat)

        self.ss = s_numpy(self.L, cst.Planet_Radius,self.lats)

        for i in range(self.nw):
            phi_z[i,:] = integrate.cumulative_trapezoid(self.k[i,:], self.ss, initial=0)
        self.phi_z= phi_z

if __name__ == '__main__':
    Bw0 = 3e-6 # Gauss
    fwave = 980*np.pi #Hz
    nlat = 10000
    latmax = 15
    n0 =10
    L = 5
    nw = 1
    waves = Waves( 1, L, nlat, 0, latmax*np.pi/180, n0)
    Bw_lat = np.zeros((nw, nlat)) + Bw0
    waves.generate_parallel_wave(np.array([fwave *2 * np.pi]),Bw_lat)
    print(waves.k.shape)
    plt.plot(np.rad2deg(waves.lats),waves.k[0,:])
    plt.plot(np.rad2deg(waves.lats),waves.phi_z[0,:])
    plt.show()
        # get k
    # The following function would be written in taichi regime
    # def get_wavefield(lat, t):

    # 3 arrays
    # phi_interp
    # k_interp
    #
    