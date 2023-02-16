# rewrite this code in waves.py
# the reason is that this way of calcilating
# phase is not easy
import taichi as ti
from taichiphysics import *
import constants as cst
from dipolefield import *
from dipolefield import dipole_field


@ti.dataclass

class Wave:
    w:ti.f64 #frequency
    #lat0:ti.f64 #start lat
    #always start at equator for ver.0
    #right now only consider electrons
    L: ti.f64 # location
    wpe0: ti.f64
    wce0: ti.f64
    wpe:ti.f64
    wce:ti.f64
    n0: ti.f64
    ne:ti.f64
    Bw:ti.types.vector(3,ti.f64)
    Ew:ti.types.vector(3,ti.f64)
    Bw0:ti.f64
    k:ti.f64
    n:ti.f64
    phi0: ti.f64
    phi:ti.f64
    phit:ti.f64
    phiz:ti.f64
    lat_max:ti.f64
    B_back:ti.types.vector(3,ti.f64)




    @ti.func
    def initialize(self, w, L, n0, phi0, lat_max):
        self.w = w
        self.L = L 
        self.n0 = n0
        self.wpe0 = ti.sqrt(4 * ti.math.pi *n0*cst.Charge**2 /cst.Me)
        #self.B = dipole_field(L, [0,0,0], cst.B0)
        #self.wce0 = cst.Charge * self.B / (cst.Me * cst.C)
        #self.Bw0 = Bw0
        self.phi0 = 0
        self.phi = 0
        self.phit = 0
        self.phiz = 0 
        self.lat_max = lat_max
    
    @ti.func
    def get_field(self, r, t, phi_z,Bw0):
        """Only after get field can get the E field and B field

        Args:
            lat (_type_): _description_
            t (_type_): _description_
        """
        # based on the latitude get the density and the wce
        lat = r[2]
        #self.ne = self.n0 * ti.cos(lat) ** -4
        self.ne = self.n0 
        self.wpe = ti.sqrt(4 * ti.math.pi *self.ne*cst.Charge**2 /cst.Me)
        self.B_back = dipole_field(self.L, [0,0,lat], cst.B0)
        self.wce = cst.Charge * self.B_back.norm() / (cst.Me * cst.C)

        #phiz = get_interp(phi_list,lat_list,lat)



        # get the phi
        #first get the k_plus
        # use the k_minus and k_plus get the k
        #get the integral of z part
        # wce is positive value
        RR = 1 + self.wpe**2 / ((self.wce - self.w) * self.w) 
        #k_plus = self.w * np.sqrt(RR)/cst.C
        #self.phiz += (self.k + k_plus) / 2   * dz
        self.k = self.w * ti.sqrt(RR)/cst.C
        self.phiz = phi_z
        self.phi = self.phiz - self.w * t

        # get the field
        cosp = ti.cos(self.phi)
        sinp = ti.sin(self.phi)
        if lat < self.lat_max:
            self.Bw = [Bw0 * cosp,  - Bw0 * sinp, 0.0]

            Ew = Bw0 / (cst.C * self.k / self.w)
            self.Ew= [- Ew * sinp, - Ew * cosp,  0.0]
        else:
            self.Bw = [0.0,0.0,0.0]
            self.Ew = [0.0,0.0,0.0]



