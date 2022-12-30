import taichi as ti
from taichiphysics import *
import constants as cst

@ti.dataclass

class Wave:
    w:ti.f64 #frequency
    lat0:ti.f64 #start lat
    L: ti.f64 # location
    wpe0: ti.f64
    wce0: ti.f64
    


