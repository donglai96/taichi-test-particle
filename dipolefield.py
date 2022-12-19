import taichi as ti
import constants as cst


@ti.func
def dipole_field(L, r, B0):
    """get dipole field vector
    The coordinates is z along the field line, x and y satisfy the
    divergence free law

    Args:
        L (ti.f32): L shell
        r (ti vector): location of particle
        B0 (ti.f32): magnetic field at equator
    """
    lat = r[2]
    x = r[0]
    y = r[1]
    cos_lat= ti.cos(lat)
    sin_lat = ti.sin(lat)
    Bz = B0/(L**3 * cos_lat**6) * ti.sqrt(1 + 3 * sin_lat ** 2)

    dBdz = 3 * Bz * sin_lat / (L * cst.Planet_Radius * ti.sqrt(1 + 3 * sin_lat ** 2)) \
        * (1 / (1 + 3 * sin_lat ** 2) + 2 /(cos_lat ** 2))

    Bx = -dBdz * (x / 2.0)
    By = -dBdz * (y / 2.0)
    B = ti.Vector([Bx,By,Bz])
    return B
# def dipole_field(B0, L, R, x, y, latitude):
#     #get the latitude, this is a vector
#     lat_sin = ti.sin(latitude)
#     lat_cos = ti.cos(latitude)
    
#     b_align = B0/(L **3 * lat_cos**6) * ti.sqrt(1 + 3 * lat_sin**2)

#     #get bx and by
#     dBdz = 3 * b_align * lat_sin/ (L * R * ti.sqrt(1 + 3 * lat_sin**2)) \
#         * (1 / (1 + 3 * lat_sin**2) + 2 /(lat_cos **2))
#     b_x = -dBdz * (x / 2.0)
#     b_y = -dBdz * (y / 2.0)
#     return b_x, b_y, b_align
