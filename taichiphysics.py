""" Some useful physic equations in taichi"""
import constants as cst
import numpy as np
import taichi as ti
import taichi.math as tm

def ev2erg(ev):
    return ev * 1.60218e-12
def erg2ev(erg):
    return erg / 1.60218e-12
def e2p(e, E0=cst.Me*cst.C**2):
    return np.sqrt(e * (e + 2 * E0)) / cst.C 

def p2e(p, E0=cst.Me*cst.C**2):
    return np.sqrt(p**2 * cst.C**2 + E0**2) - E0 

def p2v(p,m= cst.Me):
    gamma_m = m * np.sqrt(1 + p**2 /(m * m*cst.C**2)) 
    return p/gamma_m

def v2p(v,m = cst.Me):
    gamma_m = m * 1 / np.sqrt(1 - v**2 / cst.C**2)
    return gamma_m * v



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

@ti.func
def interp(xx,x,n):
    jl = -1
    ju = n
    jm = 0
    j = 0
    frac = 0.0
    
    ascnd = (xx[n - 1]>= xx[0])
    #print(ascnd)
    while ((ju - jl) > 1):
        jm = ti.floor((ju + jl)/2,dtype = ti.i32)
    
        if ((x >= xx[jm]) == ascnd):
            jl = jm
        else:
            ju = jm
    #print(jm)
            
    if (x == xx[0]):
        j = 0
    elif (x == xx[n-1]):
        j = n-2
    else:
        j = jl
    #print('find',j)
    if (j == -1):
        frac = -1
    elif (j == n-1):
        frac = n-1
    else:
        frac = j + (x - xx[j]) / (xx[j + 1] - xx[j])
        
    
        
    return frac


def get_equator_pitchangle_numpy(alpha,lat):
    return np.arcsin( (1 + 3 * np.sin(lat)**2)**( -0.25) * np.cos(lat)**3 * np.sin(alpha) )
def get_pitchanlge_numpy(alpha0,lat,signpb):
    tmp = (1 + 3 * np.sin(lat)**2)**0.25 / np.cos(lat)**3 * np.sin(alpha0)
    alpha = np.arcsin(tmp)
    if signpb< 0:
        alpha = np.pi - alpha
    return alpha