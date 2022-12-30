""" Some useful physic equations in taichi"""
import constants as cst
import numpy as np
import taichi as ti
def ev2erg(ev):
    erg = 1.60218e-12 * ev
    return erg



# def plasma_frequency(n, m ,q):
#     """
#     return the plasma frequency rad/s
#     """
#     return np.sqrt(4 * np.pi *n*q**2 /m) 

@ti.func
def plasma_frequency(n, m ,q):
    return ti.sqrt(4 * ti.math.pi *n*q**2 /m)

# ti.init()
# plasma_frequency(1., 1. ,1.)
@ti.func
def get_equator_pitchangle(alpha,lat):
    return ti.asin( (1 + 3 * ti.sin(lat)**2)**( -0.25) * ti.cos(lat)**3 * ti.sin(alpha) )


def get_equator_pitchangle_numpy(alpha,lat):
    return np.arcsin( (1 + 3 * np.sin(lat)**2)**( -0.25) * np.cos(lat)**3 * np.sin(alpha) )