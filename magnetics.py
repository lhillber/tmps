#! user/bin/python3
#
# magnetics.py
#
# DESCRIPTION
# ===========
#
# This script enables the exact evaluation magnetic field vectors in 3D space
# for various combinations of current loops and lines of current. For faster
# evaluation, the fields are evaluated on a uniform 3D grid and values between
# the grid points are found by linear interpolation.
#
# REQUIREMENTS
# ============
#
# python3, numpy, scipy, and matplotlib
#
# USAGE
# =====
# Assuming a Bash-like shell:
# Wherever you have saved this script create the folowing directories:
#
#       mkdir -p data/fields
#       mkdir -p plots/fields
#
# To run the default behavior (defined at the bottom of this file), run
#
#       python3 magnetics.py
#
# Try editing the AH_geometry dictionary to change the coil parameters.
#
# OUTPUT
# ======
# The current default behavior creates an antihelmholtz field.
# The data is saved to data/fields. The file name is a hash of the
# AH_geometry dictionary, so a unique name is created for any unique
# set of geometry parameters.
#
# The default plotting functions are executed, and the results saved to
# plots/fields/AH_field.pdf.
#
# Then, an additinal bit of code is executed to demonstrate how to acess the
# field and gradient data.
#
# By Logan Hillberry
#
# lhillberry@gmail.com
#
# Last updated 15 March 2018
#

import os
from itertools import cycle
import traceback
import pickle
import hashlib
import json
import copy

import numpy as np
from numpy import pi
from numpy.linalg import norm, inv

import scipy.special as sl
from scipy.interpolate import interpn, interp1d, RegularGridInterpolator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from matplotlib.ticker import NullFormatter, FormatStrFormatter

def curr_pulse(t, t0, tau, shape, scale, **kwargs):
    ps =[-3.4162938439228601e-39, 2.5583642117992684e-35,
         -8.6114257981640124e-32, 1.7195168964982026e-28,
         -2.2632215792992635e-25, 2.0622289076608083e-22,
         -1.3280636526982561e-19, 6.0473267830240955e-17,
         -1.9059936559084898e-14, 3.9198420604958593e-12,
         -4.4115606443755746e-10, 3.5928719421049858e-09,
          5.9195297664400324e-06, -0.00070221256261507965,
          0.024339393545649748, 0.68727142284634279] 
    poly = np.poly1d(ps)
    shapes = {'sin'   : np.sin(pi * (t - t0) / tau),
              'square': 1,
              'ramp'  : (t - t0) / tau,
              'poly'   : poly(t-t0)}
    if t0 <= t <= t0 + tau:
        return scale * shapes[shape]
    else:
        return 0.0

def hash_state(self_dict, exclude_keys):
    name_dict = copy.deepcopy(self_dict)
    for k, v in self_dict.items():
        if k in exclude_keys:
            del name_dict[k]

        if type(v) == np.ndarray:
            for i, vi in enumerate(v):
                if isinstance(vi, np.float64):
                    name_dict[k][i] = np.asscalar(vi)
            name_dict[k] = list(v)

    for k, v in name_dict.items():
        uid = hashlib.sha1(
                 json.dumps(
                 {k:v},
                 sort_keys=True).encode('utf-8')).hexdigest()
    uid = hashlib.sha1(
             json.dumps(
             name_dict,
             sort_keys=True).encode('utf-8')).hexdigest()
    return uid


# MAGNETIC FIELDS
# ===============

def Bloop_axis(z, I, z0, R, u0=1.257e-6*1e2*1e-12):
    return u0*I/2 * R*R/((z-z0)**2 + R*R)**(3/2)


def Bcoil_axis(z, I, n, z0, d, R, M, N, **kwargs):
    B = np.zeros_like(z)
    for k in range(N):
        for j in range(M):
            B += Bloop_axis(z, I, z0 + n*(j+1/2)*d, R + (k+1/2)*d)
    return B

def BAH_axis(z, I, n, z0, d, R, M, N, A, **kwargs):
    z0a = z0 + n*A
    z0b = z0 - n*A
    return Bcoil_axis(z,  I, n, z0a, d, R, M, N) +\
           Bcoil_axis(z, -I, -n, z0b, d, R, M, N)

def Bcoil_pair_axis(z, I1, I2, n, z0, d, R1, M1, N1, A1,
                    R2, M2, N2, A2, **kwargs):
    z01 = z0 + n*A1
    z02 = z0 - n*A2
    return Bcoil_axis(z,  I1, n, z01,  d, R1, M1, N1) +\
           Bcoil_axis(z, -I2, -n, z02, d, R2, M2, N2)

def BHH_axis(z, I, n, z0, d, R, M, N, A, **kwargs):
    z0a = z0 + n*A
    z0b = z0 - n*A
    return Bcoil_axis(z, I,  n, z0a, d, R, M, N) +\
           Bcoil_axis(z, I, -n, z0b, d, R, M, N)


# Field of anti-Helmholtz and Helmholtz coils for creating a biased gradient
def Bmop_axis(z, IAH, IHH, n, z0, d, RAH, MAH, NAH, AAH,
            RHH, MHH, NHH, AHH, **kwargs):
    return BHH_axis(z, IHH, n, z0, d, RHH, MHH, NHH, AHH) +\
            BAH_axis(z, IAH, n, z0, d, RAH, MAH, NAH, AAH)

def Bmop3_axis(z, I1, I2, n, z0, d, R1, M1, N1, A1,
            R2, M2, N2, A2,**kwargs):
    z0a = z0 + n * A1
    z0b = z0 - n * A1
    B = BAH_axis(z, I2, n, z0, d, R1, M1, N1, A2)
    if I1 >= 0:
        B += Bcoil_axis(z, I1, n, z0a, d, R1, M1, N1)
    elif I1 < 0:
        B += Bcoil_axis(z, -I1, -n, z0b, d, R2, M2, N2)
    return B



# Field at point r due to current I running in loop raidus R and normal vector
# n centered at point r0.
class Field():
    def __init__(self, geometry, recalc_B=True, data_dir='data/fields', load_fname=None):
        for k in geometry:
            if k[:2] == 'r0' or k[0] == 'n':
                geometry[k] = np.array(geometry[k], dtype=float)
        self.geometry = geometry
        self.uid = hash_state(geometry, exclude_keys=['field_name'])
        self.fname = os.path.join(data_dir, self.uid)

        vsamp = [-800, -600, -500, -400, -300, 300, 400, 500, 600, 800]
        HH1 = [325.4, 273.9, 273.9, 273.9, 279.0, 751.4, 899.8, 1047.0, 1222.4, 1637.1]
        HH2 = [1552.0, 1162.4, 1015.0, 866.6, 723.2, 240.6, 240.6, 235.5, 241.9, 276.]
        AH1 = [1618.4, 1557.8, 1542.4, 1536.0, 1524.5, 1349.1, 1283.8, 1207.0, 1135.4, 1058.0]
        AH2 = [1231.7, 1272.3, 1327.4, 1392.6, 1452.8, 1623.0, 1623.0, 1634.6, 1634.6, 1684.9]
        self.Ifuncs = [interp1d(vsamp, Isamp, kind=2) for
            Isamp in [AH1, AH2, HH1, HH2]]

        if not recalc_B:
            try:
                print('Attempting to load field...')
                self.load(fname=load_fname)
            except(FileNotFoundError):
                print('Field data not found. Generating now...')
                recalc_B = True
        if recalc_B:
            self.Bmap = {'w'     : self.Bwire,
                         'sl'    : self.Bsquare_loop,
                         'sc'    : self.Bsquare_coil,
                         'sAH'   : self.BsquareAH,
                         'sHH'   : self.BsquareHH,
                         'smop'  : self.Bsquare_mop,
                         'smop3' : self.Bsquare_mop3,
                         'smop4' : self.Bsquare_mop4,
                         'l'     : self.Bloop,
                         'c'     : self.Bcoil,
                         'AH'    : self.BAH,
                         'HH'    : self.BHH,
                         'mop'   : self.Bmop,
                         'mop3'  : self.Bmop3}
            self.make_grid(**self.geometry)
            self.make_B(self.geometry)
            self.save()

    # save state of the class instance with name given by unique hash of params
    def save(self):
        print('Saving field data to {}'.format(self.fname))
        file = open(self.fname,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    # load state...
    def load(self, fname=None):
        if fname is None:
            fname = self.fname
        file = open(fname,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        print('Field data loaded from {}'.format(self.fname))

    def make_grid(self, xmin, xmax, Nxsteps,
                        ymin, ymax, Nysteps,
                        zmin, zmax, Nzsteps, **kwargs):
        print('Making solution grid...')
        # grid spacing
        dx = (xmax - xmin)/Nxsteps
        dy = (ymax - ymin)/Nysteps
        dz = (zmax - zmin)/Nzsteps
        # axes
        x = np.linspace(xmin, xmax, Nxsteps)
        y = np.linspace(ymin, ymax, Nysteps)
        z = np.linspace(zmin, zmax, Nzsteps)
        # The grid of points on which we want to evaluate the field
        grid_list = np.meshgrid(x, y, z, indexing='ij')
        X, Y, Z = grid_list
        # grid shapes
        Xshape = X.shape
        Yshape = Y.shape
        Zshape = Z.shape
        # convert grid coordinates to point form
        r = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        # cubic limits for square aspect ratio in 3D plots
        max_range = np.array([X.max()-X.min(),
            Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        xlim = mid_x - max_range, mid_x + max_range
        ylim = mid_y - max_range, mid_y + max_range
        zlim = mid_z - max_range, mid_z + max_range
        self.xyz = [x, y, z]
        self.XYZ = [X, Y, Z]
        self.r = r
        self.dxyz = [dx, dy, dz]
        self.XYZshape = [Xshape, Yshape, Zshape]
        self.xyzlim = [xlim, ylim, zlim]
        return self.xyz, self.XYZ, self.r, self.dxyz, self.XYZshape, self.xyzlim

    def vector_interpolation(self, V):
        VX, VY, VZ = V
        x, y, z = self.xyz
        VX_interp = RegularGridInterpolator((x,y,z), VX,
            method='linear', bounds_error=False, fill_value=0)
        VY_interp = RegularGridInterpolator((x,y,z), VY,
            method='linear', bounds_error=False, fill_value=0)
        VZ_interp = RegularGridInterpolator((x,y,z), VZ,
            method='linear', bounds_error=False, fill_value=0)
        return VX_interp, VY_interp, VZ_interp

    def make_B(self, geometry, **kwargs):
        print('Calculating field and gradient...')
        # grid spacing, shape, and valuse
        dx, dy, dz = self.dxyz
        Xshape, Yshape, Zshape =self.XYZshape
        r = self.r
        # mop field, shaped like r
        config = geometry['config']
        B = self.Bmap[config](r, **geometry)   # kg A^-1 us^-2
        # grids of mop field components
        BX = B[::, 0]
        BY = B[::, 1]
        BZ = B[::, 2]
        BX.shape = Xshape
        BY.shape = Yshape
        BZ.shape = Zshape
        BXYZ = [BX, BY, BZ]
        BXYZ_interp = self.vector_interpolation(BXYZ)
        # grid of field magnitude and grids of its gradient
        norm_B = np.sqrt(BX**2 + BY**2 + BZ**2)
        grad_norm_BXYZ = np.gradient(norm_B, dx, dy, dz) # kg A^-1 us^-2 cm^-1
        grad_norm_BXYZ_interp = self.vector_interpolation(grad_norm_BXYZ)
        self.BXYZ_interp = BXYZ_interp
        self.grad_norm_BXYZ_interp = grad_norm_BXYZ_interp
        return

    # units of u0 are kg m A^-2 s^-2 cm/m s^2/us^2
    def Bwire(self, r, I, n, r0, L, u0=1.257e-6*1e2*1e-12, **kwargs):
                # coil frame to lab frame transformation and its inverse
        r0 = np.array(r0)
        l, m, n = coil_vecs2(n)
        trans = np.vstack((l, m, n))
        inv_trans = inv(trans)
        # Shift origin to wire center
        r = r - r0
        # transform field points to wire frame
        r = np.dot(r, inv_trans)
        # move to cylindrical coordinates for the coil
        x = r[:,0]
        y = r[:,1]
        z = r[:,2]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        # field values
        radm = 1.0 / np.sqrt(4*rho**2 + (L - 2*z)**2)
        radp = 1.0 / np.sqrt(4*rho**2 + (L + 2*z)**2)
        Bphi = u0 * I / (4 * pi * rho) * (radm * (-2*z + L) + radp * (2*z + L))
        Bz = np.zeros_like(Bphi)
        # Set field field values on wire to zero instead of NaN or Inf
        # (caused by a divide by 0)
        Bphi[np.isnan(Bphi)] = 0.0
        Bphi[np.isinf(Bphi)] = 0.0
        # covert back to coil-cartesian coordinates then to lab coordinates
        B = np.c_[-np.sin(phi) * Bphi, np.cos(phi) * Bphi, Bz]
        B = np.dot(B, trans)
        return B

    # units of u0 are kg m A^-2 s^-2 cm/m s^2/us^2
    def Bloop(self, r, I, n, r0, R, u0=1.257e-6*1e2*1e-12, **kwargs):
        # coil frame to lab frame transformation and its inverse
        r0 = np.array(r0)
        l, m, n = coil_vecs2(n)
        trans = np.vstack((l, m, n))
        inv_trans = inv(trans)
        # Shift origin to coil center
        r = r - r0
        # transform field points to coil frame
        r = np.dot(r, inv_trans)
        # move to cylindrical coordinates for the coil
        x = r[:,0]
        y = r[:,1]
        z = r[:,2]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        # elliptic integrals of first and second kind
        E = sl.ellipe(4 * R * rho / ((R + rho )**2 + z**2))
        K = sl.ellipk(4 * R * rho / ((R + rho )**2 + z**2))
        # field values
        Bz = u0 * I / (2 * pi * np.sqrt((R + rho)**2 + z**2)) * (
                    K + E * (R**2 - rho**2 - z**2)/((R - rho)**2 + z**2) )

        Brho = u0 * I * z / (2 * pi * rho * np.sqrt((R + rho)**2 + z**2)) * (
                   -K + E * (R**2 + rho**2 + z**2)/((R - rho)**2 + z**2) )
        # Set field values on coil axis to zero instead of NaN or Inf
        # (caused by a divide by 0)
        Brho[np.isnan(Brho)] = 0.0
        Brho[np.isinf(Brho)] = 0.0
        Bz[np.isnan(Bz)]     = 0.0
        Bz[np.isinf(Bz)]     = 0.0
        # covert back to coil-cartesian coordinates then to lab coordinates
        B = np.c_[np.cos(phi) * Brho, np.sin(phi) * Brho, Bz]
        B = np.dot(B, trans)
        return B

    def Bsquare_loop(self, r, I, n, r0, L, W, ang, **kwargs):
        r0 = np.array(r0)
        l, m, n = coil_vecs2(n)
        trans = rotation_matrix(n*np.sin(ang))
        inv_trans = inv(trans)
        r = r - r0
        r = np.dot(r, inv_trans)
        B  = self.Bwire(r, I,  l, -W/2 * m, L) # >
        B += self.Bwire(r, I, -l,  W/2 * m, L) # v
        B += self.Bwire(r, I, -m, -L/2 * l, W) # ^
        B += self.Bwire(r, I,  m,  L/2 * l, W) # <
        B = np.dot(B, trans)
        return B

    def Bsquare_coil(self, r, I, n, r0, L, W, d, M, N, ang, **kwargs):
        B = np.zeros_like(r)
        for k in range(N):
            for j in range(M - k%2):
                B += self.Bsquare_loop(r, I, n,
                        r0 + n*(j + 1/2 + k%2)*d,
                        L + (k+1/2)*d,
                        W + (k+1/2)*d, ang)
        return B

    # Field of two identical coils situated co-axially along loop-normal n, with the
    # closest loops of the two coils separated by 2A. Then, r0 is axial point
    # midway between the two coils. Current in the two coils flows in opposite
    # directions (anti-Helmholtz configuration)
    def BsquareAH(self, r, I, n, r0, L, W, d, M, N, ang, A, **kwargs):
        r0a = r0 + n*A
        r0b = r0 - n*A
        return self.Bsquare_coil(r, I, n, r0a, L, W, d, M, N, ang) +\
               self.Bsquare_coil(r, I, -n, r0b, L, W, d, M, N, ang)


    def Bsquare_mop4(self, r, V, n, r0, d, ang,
                        LAH, WAH, MAH, NAH, AAH,
                        LHH, WHH, MHH, NHH, AHH, **kwargs):
        if V == None:
            IAH1, IAH2, IHH1, IHH2 = 0.0, 0.0, 0.0, 0.0
        else:
            IAH1, IAH2, IHH1, IHH2 = (np.array([Ifunc(V) for Ifunc in self.Ifuncs]).T).tolist()
        r0aAH = r0 + n*AAH
        r0bAH = r0 - n*AAH
        r0aHH = r0 + n*AHH
        r0bHH = r0 - n*AHH
        return self.Bsquare_coil(r,  IAH1,  n, r0aAH, LAH, WAH, d, MAH, NAH, ang) +\
               self.Bsquare_coil(r,  IAH2, -n, r0bAH, LAH, WAH, d, MAH, NAH, ang) +\
               self.Bsquare_coil(r,  IHH1,  n, r0aHH, LHH, WHH, d, MHH, NHH, ang) +\
               self.Bsquare_coil(r,  IHH2, -n, r0bHH, LHH, WHH, d, MHH, NHH, ang)

    # Same as BHH but with currents of the two coils flowing in the same direction
    # (Helmholtz configuration)
    def BsquareHH(self, r, I, n, r0, L, W, d, M, N, ang, A, **kwargs):
        r0a = r0 + n * A
        r0b = r0 - n * A
        return self.Bsquare_coil(r, I, n, r0a, L, W, d, M, N, ang) +\
               self.Bsquare_coil(r, -I, -n, r0b, L, W, d, M, N, ang)

    # Field of anti-Helmholtz and Helmholtz coils for creating a biased gradient
    def Bsquare_mop(self, r, IAH, nAH, r0AH, LAH, WAH, dAH, MAH, NAH, ang, AAH,
                IHH, nHH, r0HH, LHH, WHH, dHH, MHH, NHH, AHH, **kwargs):
        return self.BsquareAH(
                        r, IAH, nAH, r0AH, LAH, WAH, dAH, MAH, NAH, ang, AAH) +\
               self.BsquareHH(
                        r, IHH, nHH, r0HH, LHH, WHH, dHH, MHH, NHH, ang, AHH)

    def Bsquare_mop3(self, r, I1, I2, n, r0, L1, W1, d, M1, N1, ang, A,
                L2, W2, M2, N2, **kwargs):
        r0a = r0 + n * A
        r0b = r0 - n * A
        B = self.BsquareAH(r, I2, n, r0, L2, W2, d, M2, N2, ang, A)
        if I1 >= 0:
            B +=  self.Bsquare_coil(r,  I1, n, r0a, L1, W1, d, M1, N1, ang)
        elif I1 < 0:
            B +=  self.Bsquare_coil(r, -I1, -n, r0b, L1, W1, d, M1, N1, ang)
        return B

    # Field of N layers of M wire loops (wire diameter d) centered at r0
    # such that the first of M loops has its center at r0 and the Mth loop has its
    # center at r0 + M*d*n where n is the normal to the loops. The first loop of N
    # layers has radius R.
    def Bcoil(self, r, I, n, r0, d, R, M, N, **kwargs):
        B = np.zeros_like(r)
        for k in range(N):
            for j in range(M - k%1):
                B += self.Bloop(r, I, n, r0 + n*(j+1/2+k%1)*d, R + (k+1/2)*d)
        return B


    # Field of two identical coils situated co-axially along loop-normal n, with the
    # closest loops of the two coils separated by 2A. Then, r0 is axial point
    # midway between the two coils. Current in the two coils flows in opposite
    # directions (anti-Helmholtz configuration)
    def BAH(self, r, I, n, r0, d, R, M, N, A, **kwargs):
        #r0a = r0 + n*A
        #r0b = r0 - n * (A + M * d)
        r0a = r0 + n*A
        r0b = r0 - n*A
        return self.Bcoil(r,  I,  n, r0a, d, R, M, N) +\
               self.Bcoil(r, I, -n, r0b, d, R, M, N)

    # Same as BHH but with currents of the two coils flowing in the same direction
    # (Helmholtz configuration)
    def BHH(self, r, I, n, r0, d, R, M, N, A, **kwargs):
        #r0a = r0 + n * A
        #r0b = r0 - n * (A + M * d)
        r0a = r0 + n*A
        r0b = r0 - n*A
        return self.Bcoil(r, I, n, r0a, d, R, M, N) +\
               self.Bcoil(r, -I, -n, r0b, d, R, M, N)

    # Field of anti-Helmholtz and Helmholtz coils for creating a biased gradient
    def Bmop(self, r, IAH, IHH, n, r0, d, RAH, MAH, NAH, AAH,
                RHH, MHH, NHH, AHH, **kwargs):
        return self.BAH(r, IAH, n, r0, d, RAH, MAH, NAH, AAH) +\
               self.BHH(r, IHH, n, r0, d, RHH, MHH, NHH, AHH)


    def Bmop3(self, r, I1, I2, n, r0, d, R1, M1, N1, A1,
                R2, M2, N2, A2,**kwargs):
        r0a = r0 + n * A1
        r0b = r0 - n * A1
        B = self.BAH(r, I2, n, r0, d, R1, M1, N1, A2)
        if I1 >= 0:
            B += self.Bcoil(r, I1, n, r0a, d, R1, M1, N1)
        elif I1 < 0:
            B += self.Bcoil(r, -I1, -n, r0b, d, R2, M2, N2)
        return B

def coil_vecs2(n):
    if  np.abs(n[2]) == 1:
        l = np.array([1, 0, 0])
    else:
        l = np.cross(n, np.array([0, 0, 1]))
    # normalize coil's normal vector
    l = l/norm(l)
    m = np.cross(n, l)
    return l, m, n

# Circuit estimates for Circular coils
# ------------------------------------

def dimensions(R, d, M, N, **kwargs):
    Rmin = R-d/2
    Rmax = R + (N - 1/2)*d
    Rav = (Rmin + Rmax) / 2
    width = M*d
    return Rmin, Rmax, Rav, width

def resistance(Rav, d, M, N, rho=16.78e-9 * 1e2):
    return 8*M*N*Rav*rho / (d*d)

def inductance(Rav, d, M, N, u0=1.257e-6*1e2):
    return (M*N)**2 * u0 * Rav * (np.log(4*Rav / d) - 2) * 1e-4

def capacitance(RR, LL, tau):
    return 4 * LL * tau**2 / ( (tau*RR)**2 + (2*pi*LL)**2)

def RLC_values(R, d, M, N, tau, **kwargs):
    _, _, Rav, _ = dimensions(R, d, M, N)
    RR = resistance(Rav, d, M, N)
    LL = inductance(Rav, d, M, N)
    CC = capacitance(RR, LL, tau)
    return RR, LL, CC

def RLC_model(RR, LL, CC):
    alpha = RR / (2 * LL)
    omega0 = 1 / np.sqrt(LL * CC)
    beta = np.sqrt(omega0**2 - alpha**2)
    zeta = RR/2 * np.sqrt(CC/LL)
    if zeta>1:
        print('over damped circuit')
    return alpha, beta

def current(t, V0, tau, R, d, M, N, **kwargs):
    RR, LL, CC = RLC_values(R, d, M, N, tau)
    alpha, beta = RLC_model(RR, LL, CC)
    return V0 * np.exp(-alpha * t) * np.sin(beta * t) / (beta * LL)

def split_geometry(geometry):
    split_keys = []
    for k,v in geometry.items():
        if k[0] == 'I':
            split_keys += [k[1:]]
    geometries = {}
    for sk in split_keys:
        g = copy.deepcopy(geometry)
        for k, v in geometry.items():
            n = len(sk)
            if k[-n:] == sk:
                slimk = k[:-n]
                for sk2 in split_keys:
                    g[slimk] = geometry[k]
                    try:
                        del g[slimk+sk2]
                    except:
                        pass
        geometries[sk] = g
    return geometries

# PLOTTING
# ========

# visualize wires in 3D
def wire_viz(ax, n, r0, L, d, color='k', **kwargs):
    a = [[0, -L/2],[d/2, -L/2],[d/2, L/2],[0, L/2]]
    a = np.array(a)
    r, theta = np.meshgrid(a[:,0], np.linspace(0, 2*np.pi, 30))
    z = np.tile(a[:,1], r.shape[0]).reshape(r.shape)
    x = r*np.sin(theta)
    y = r*np.cos(theta)
    l, m, n = coil_vecs2(n)
    trans = np.vstack((l, m, n))
    rxyz = np.c_[x.flatten(), y.flatten(), z.flatten()]
    rxyz = np.dot(rxyz, trans)
    wire_data = rxyz + r0
    x = wire_data[:,0].reshape(r.shape);
    y = wire_data[:,1].reshape(r.shape);
    z = wire_data[:,2].reshape(r.shape);
    ax.plot_surface(x,y,z, color=color)

def square_loop_viz(ax, n, r0, L, W, d, ang, color='k', **kwargs):
        l, m, n = coil_vecs2(n)
        rot = rotation_matrix(n*np.sin(ang))
        l = np.dot(l, rot)
        m = np.dot(m, rot)
        wire_viz(ax,  l, r0 - W/2 * m, L, d, color=color)
        wire_viz(ax, -l, r0 + W/2 * m, L, d, color=color)
        wire_viz(ax, -m, r0 - L/2 * l, W, d, color=color)
        wire_viz(ax,  m, r0 + L/2 * l, W, d, color=color)

def square_coil_viz(ax, n, r0, L, W, d, ang, M, N, color=None, **kwargs):
    if color is None:
       c = cycle(['C' + str(i) for i in range(10)][:M*N])
    else:
       c = cycle([color])
    for k in range(N):
        for j in range(M - k%2):
            shift  = r0 + n*(j + 1/2 + k%2)*d
            Llayer = L + (k+1/2)*d
            Wlayer = W + (k+1/2)*d
            square_loop_viz(ax, n, shift, Llayer, Wlayer, d, ang, color=next(c))

def square_coil_pair_viz(ax, n, r0, L, W, d, ang, M, N, A, color=None, **kwargs):
    r0a = r0 + n * A
    r0b = r0 - n * A
    square_coil_viz(ax, n, r0a, L, W, d, ang, M, N, color=color)
    square_coil_viz(ax, -n, r0b, L, W, d, ang, M, N, color=color)

def square_mop_viz(ax, nAH, r0AH, LAH, WAH, dAH, ang, MAH, NAH, AAH,
                nHH, r0HH, LHH, WHH, dHH, MHH, NHH, AHH,
                colorAH=None, colorHH=None, **kwargs):
    square_coil_pair_viz(ax, nAH, r0AH, LAH, WAH, dAH, ang, MAH, NAH, AAH,
        color=colorAH)
    square_coil_pair_viz(ax, nHH, r0HH, LHH, WHH, dHH, ang, MHH, NHH, AHH,
        color=colorHH)

def square_mop3_viz(ax, n, I1, r0, L1, W1, d, M1, N1, A, L2, W2, ang, M2, N2,
        colorpair=None, colorcoil=None, **kwargs):
    r0a = r0 + n * A
    r0b = r0 - n * A
    square_coil_pair_viz(ax, n, r0, L2, W2, d, ang, M2, N2, A, color=colorpair)
    if I1 >= 0:
        square_coil_viz(ax, n, r0a, L1, W1, d, ang, M1, N1, color=colorcoil)
    elif I1 < 0:
        square_coil_viz(ax, -n, r0b, L1, W1, d, ang, M1, N1, color=colorcoil)


def loop_viz(ax, n, r0, d, R, color='k', **kwargs):
    angle = np.linspace(0, pi, 16)
    theta, phi = np.meshgrid(3*2*angle/4, 2*angle)
    x = (R + d/2.0 * np.cos(phi)) * np.cos(theta)
    y = (R + d/2.0 * np.cos(phi)) * np.sin(theta)
    z = d/2.0 * np.sin(phi)
    l, m, n = coil_vecs2(n)
    trans = np.vstack((l, m, n))
    rxyz = np.c_[x.flatten(), y.flatten(), z.flatten()]
    rxyz = np.dot(rxyz, trans)
    loop_data = rxyz + r0
    x = loop_data[:,0].reshape(theta.shape);
    y = loop_data[:,1].reshape(theta.shape);
    z = loop_data[:,2].reshape(theta.shape);
    ax.plot_surface(x, y, z, color=color)

def coil_viz(ax, n, r0, d, R, M, N, color=None, **kwargs):
    if color is None:
        c = cycle(['C' + str(i) for i in range(10)][:M*N])
    else:
       c =  cycle([color])
    for j in range(M):
        for k in range(N):
            shift = r0 + n*(j+1/2)*d
            rad = R + (k+1/2)*d
            loop_viz(ax, n, shift, d, rad, color=color)

def coil_pair_viz(ax, n, r0, d, R, M, N, A, color=None, **kwargs):
    r0a = r0 + n * A
    r0b = r0 - n * A
    coil_viz(ax, n, r0a, d, R, M, N, color=color)
    coil_viz(ax, n, r0b, d, R, M, N, color=color)

def mop_viz(ax, n, r0, d, RAH, MAH, NAH, AAH,
                RHH, MHH, NHH, AHH,
                colorAH=None, colorHH=None, **kwargs):
    coil_pair_viz(ax, n, r0, d, RAH, MAH, NAH, AAH, color=colorAH)
    coil_pair_viz(ax, n, r0, d, RHH, MHH, NHH, AHH, color=colorHH)

def mop3_viz(ax, n, r0, R1, d, M1, N1, A, R2, M2, N2, 
        colorpair=None, colorcoil=None, **kwargs):
    r0b = r0 - n * (A + M2 * d)
    coil_viz(ax, n, r0b, d, R2, M2, N2, color=colorcoil)
    coil_pair_viz(ax, n, r0, d, R1, M1, N1, A, color=colorpair)

# dictionary mapping `config` to a viz function
def geometry_viz(ax, geometry):
    viz_map = {'w'     : wire_viz,
               'sl'    : square_loop_viz,
               'sc'    : square_coil_viz,
               'sAH'   : square_coil_pair_viz,
               'sHH'   : square_coil_pair_viz,
               'smop'  : square_mop_viz,
               'smop3' : square_mop3_viz,
               'l'     : loop_viz,
               'c'     : coil_viz,
               'p'     : coil_pair_viz,
               'AH'    : coil_pair_viz,
               'HH'    : coil_pair_viz,
               'mop'   : mop_viz,
               'mop3'  : mop3_viz}
    config = geometry['config']
    viz_map[config](ax, **geometry)


# https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normalsu
def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.
    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = norm(d)
    if sin_angle == 0:
        return np.identity(3)
    d = d/sin_angle
    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[   0,   d[2], -d[1]],
                     [-d[2],    0,   d[0]],
                     [ d[1], -d[0],    0 ]], dtype=np.float64)
    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

# replace field components > vclip and < min_frac * min() with 0
def mask_vector_field(VX, VY, VZ, Vclip, min_frac):
    norm_V = np.sqrt(VX**2 + VY**2 + VZ**2)
    VX[norm_V > Vclip] = 0
    VY[norm_V > Vclip] = 0
    VZ[norm_V > Vclip] = 0
    norm_V[norm_V > Vclip] = Vclip
    Vmin = np.min(norm_V) * min_frac
    VX[norm_V < Vmin] = 0
    VY[norm_V < Vmin] = 0
    VZ[norm_V < Vmin] = 0
    norm_V[norm_V < Vmin] = Vmin
    return VX, VY, VZ, norm_V

# plot a field in 3D
def plot_3d(ax, field, grad_norm=False,
        skip=20, Vclip=2, min_frac=1, **kwargs):
    skip_slice = [slice(None, None, skip)]*3
    r = field.r
    X, Y, Z = field.XYZ
    Xshape, Yshape, Zshape = field.XYZshape
    if grad_norm:
        print('Plotting 3D gradient vectors...')
        VX, VY, VZ = [1e12 * interp(r) for interp in field.grad_norm_BXYZ_interp]
        title = r'$\nabla B$'
    else:
        print('Plotting 3D field vectors...')
        VX, VY, VZ = [1e12 * interp(r) for interp in field.BXYZ_interp]
        title = r'$\bf{B}$'
    VX.shape = Xshape
    VY.shape = Yshape
    VZ.shape = Zshape
    VX, VY, VZ, norm_V = mask_vector_field(VX, VY, VZ, Vclip, min_frac)
    ax.quiver(X[skip_slice], Y[skip_slice], Z[skip_slice],
        VX[skip_slice], VY[skip_slice], VZ[skip_slice],
        length=0.6, pivot='middle', lw=0.5)
    ax.set_ylabel('y [cm]')
    ax.set_xlabel('x [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_title(title)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])

# plot field magnitude contour slices
# NOTE: need to play with subplots_adjust to make pretty
def plot_contour(fignum, field, grad_norm=False,
        Vclip=2, min_frac=1,  **kwargs):
    r = field.r
    xyz = field.xyz
    Xshape, Yshape, Zshape = field.XYZshape
    if grad_norm:
        print('Plotting 2D contour slice of gradient...')
        VX, VY, VZ = [1e12 * interp(r) for interp in field.grad_norm_BXYZ_interp]
        title = r'$\lvert \nabla B(%s = %.2f) \rvert$'
    else:
        print('Plotting 2D contour slice of field...')
        VX, VY, VZ = [1e12 * interp(r) for interp in field.BXYZ_interp]
        title = r'$B(%s = %.2f)$'
    fig = plt.figure(fignum, figsize=(6,9.5))
    VX.shape = Xshape
    VY.shape = Yshape
    VZ.shape = Zshape
    VX, VY, VZ, norm_V = mask_vector_field(VX, VY, VZ, Vclip, min_frac)
    VXYZ = [VX, VY, VZ]
    vmin, vmax = [np.min(norm_V), np.max(norm_V)]
    # plot the field and coils
    coords = np.array([0 , 1 , 2]) # x y z
    coord_labels = ['x', 'y', 'z']
    I_slice = [slice(None, None, None)]*3
    Nxyz = [field.geometry['Nxsteps'],
            field.geometry['Nysteps'],
            field.geometry['Nzsteps']]

    xyzis = [inds for inds in map(
                lambda N: [int(N / 4), int(N / 2), int(3 * N / 4)], Nxyz)]
    c =  1
    for coord in coords:
        for i, zi in enumerate(xyzis[coord]):
            ax = fig.add_subplot(3, 3, c)
            xbase, ybase = np.roll(coords, -coord)[1:]
            # xbase and ybase are fliped when coord == y
            # (like minus sign on y component of cross product)
            if coord == 1:
                xbase_tmp = xbase
                xbase = ybase.copy()
                ybase = xbase_tmp.copy()
            x = xyz[xbase]
            y = xyz[ybase]
            z = xyz[coord]
            VX = VXYZ[xbase]
            VY = VXYZ[ybase]
            VZ = VXYZ[coord]
            x, y = np.meshgrid(x, y, indexing='ij')
            slicei = I_slice.copy()
            slicei[coord] = zi
            b = norm_V[slicei]
            cp = plt.contourf(x, y, b)
            #bx = VX[slicei]
            #by = VY[slicei]
            #bz = VZ[slicei]
            #sp = ax.streamplot(x, y, bx, by)
            #plt.clim(*clim)
            ax.set_ylabel(coord_labels[ybase] + ' [cm]')
            ax.set_xlabel(coord_labels[xbase] + ' [cm]')
            ax.set_title(title % (coord_labels[coord], z[zi]), fontsize=12)
            c += 1
            ax.axis('scaled')
    cax = fig.add_axes([0.87, 0.23, 0.02, 0.14])
    fig.colorbar(cp, cax=cax)
    #ax.set_xlim([-2,2])
    #ax.set_ylim([-2,2])
    plt.subplots_adjust(hspace=-0.6, wspace=0.5,
            top=0.92, bottom=0.08, left=0.10, right=0.85)
    fignum += 1
    return fignum

# plot 1D slices of fields
# NOTE: assumes radial symmetry.
# TODO: generalize
def plot_slices(fignum, field, grad_norm=False):
    if grad_norm:
        print('Plotting 1D gradient slices...')
        xinterp, yinterp, zinterp = field.grad_norm_BXYZ_interp
        ylabel = r'$\partial B \partial %s$ [T/cm]'
    else:
        print('Plotting 1D field slices...')
        xinterp, yinterp, zinterp = field.BXYZ_interp
        ylabel = '$B_%s$ [T]'
    fig = plt.figure(fignum, figsize=(6,4))
    x, y, z = field.xyz
    # plot the field and coils
    for coordi, (coord, interp) in enumerate(zip(
            [r'\rho', 'z'], [xinterp, zinterp])):
        ax = fig.add_subplot(2, 1, coordi+1)
        ax.set_ylabel(ylabel % coord)
        for slicei, rho in enumerate([-0.2, 0.0, 0.2]):
            label = r'${}$'.format(str(rho))
            on = np.ones(len(z))*rho
            off = np.zeros(len(z))
            if coord in ('y'):
                xs = np.vstack((off, on, z)).T
            if coord in ('x', 'z', r'\rho'):
                xs = np.vstack((on, off, z)).T
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if coordi == 0:
                ax.xaxis.set_major_formatter(NullFormatter())
            if coordi != 0:
                ax.set_xlabel(r'$z$ [cm]')
            mask = np.logical_and(z>-2, z<2)
            ax.plot(z[mask], 1e12 * interp(xs[mask]),label=label)
    ax.legend(ncol=1, loc='lower left', bbox_to_anchor=[1,0.2],
            handlelength=1, labelspacing=0.2, handletextpad=0.2)
    ax.text(1.04, 1.2, r'$\rho$ [cm]', transform=ax.transAxes)
    plt.subplots_adjust(hspace=0.0)
    fignum += 1
    return fignum

# wrapper for 3D field plotting and wire visualization
def show_field_and_wires(fignum, field,
        grad_norm=False, skip=10, Vclip=2, min_frac=1):
    geometry = field.geometry
    fig = plt.figure(fignum, figsize=(6,4))
    ax = fig.add_subplot(1,1,1, projection='3d')
    plot_3d(ax, field, grad_norm=grad_norm, skip=skip, Vclip=Vclip, min_frac=min_frac)
    geometry_viz(ax, geometry)
    fignum += 1
    return fignum

# save many plots in single pdf
def multipage(fname, figs=None, clf=True, dpi=300, clip=True, extra_artist=False):
    pp = PdfPages(fname)
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        if clip is True:
            fig.savefig(pp, format='pdf', bbox_inches='tight',
                        bbox_extra_artist=extra_artist)
        else:
            fig.savefig(pp, format='pdf', bbox_extra_artist=extra_artist)
        if clf==True:
            fig.clf()
    pp.close()
    return

# make input params into numpy arrays and normalize where needed
def format_geometry(geometry):
    for k, v in geometry.items():
        if k[0] == 'n':
            n = np.array(v)
            n = n / norm(n)
            geometry[k] = n
        if k[:2] == 'r0':
            geometry[k] = np.array(v)
    return geometry

# main plotting function. calls all 
def field_analysis(field, fname,
        fignum=0, show=False, save=True,
        skip=10, min_frac=1, Bclip=2, grad_norm_B_clip=2):
    fignum = show_field_and_wires(fignum, field,
            skip=skip, Vclip=Bclip, min_frac=min_frac)
    fignum = show_field_and_wires(fignum, field, grad_norm=True,
            skip=skip, Vclip=grad_norm_B_clip, min_frac=min_frac)
    fignum = plot_contour(fignum, field,
            Vclip=Bclip, min_frac=min_frac)
    fignum = plot_contour(fignum, field, grad_norm=True,
            Vclip=grad_norm_B_clip, min_frac=min_frac)
    fignum = plot_slices(fignum, field)
    fignum = plot_slices(fignum, field, grad_norm=True)
    if show:
        plt.show()
    if save:
        if show:
            print('Cannot show and save plots.')
        else:
            save_loc = os.path.join('plots/fields', fname + '.pdf')
            multipage(save_loc)
            print('\n Plots saved to {}'.format(save_loc))


# A few test case geometry dictionaries
mop_geometry = dict(
    config = 'mop',
    RAH = 1.0,
    RHH = 1.2,
    AAH = 1.0/2,
    AHH = 1.0/2,
    r0AH = [0.0, 0.0, 0.0],
    r0HH = [0.0, 0.0, 0.0],
    dAH = 0.081,
    dHH = 0.081,
    MAH = 6,
    MHH = 6,
    NAH = 2,
    NHH = 2,
    nAH = [0, 0, 1.0],
    nHH = [0, 0, 1.0],
    # current pulse params
    IAH = 1000, # max current, A
    IHH = -1000,
    # B-field solution grid
    Nzsteps  =  100,
    Nysteps  =  100,
    Nxsteps  =  100,
    xmin     = -2,
    xmax     =  2,
    ymin     = -2,
    ymax     =  2,
    zmin     = -2,
    zmax     =  2,
    )

smop_geometry = dict(
    config = 'smop',
    nHH = [0, 0, 1],
    nAH = [0, 0, 1],
    r0HH = [0, 0, 0],
    r0AH = [0, 0, 0],
    LHH = 4.75,
    WHH = 2.84,
    dHH  = 0.086,
    dAH  = 0.086,
    MHH = 5,
    NHH = 2,
    ang = 0.0,
    AHH = 2.34 / 2.0,
    AAH = 2.08 / 2.0,
    LAH = 4.29,
    WAH = 2.03,
    MAH = 5,
    NAH = 2,
    IAH = 1000,
    IHH = 1000,
    # B-field solution grid
    Nzsteps  =  100,
    Nysteps  =  100,
    Nxsteps  =  100,
    xmin     = -5,
    xmax     =  5,
    ymin     = -3,
    ymax     =  3,
    zmin     = -3,
    zmax     =  3
    )

sAH_geometry = dict(
    config = 'sAH',
    n = [0, 0, 1],
    r0 = [0, 0, 0],
    WHH = 2.84,
    d  = 0.086,
    ang = 0.0,
    A = 2.08 / 2.0,
    L = 4.29,
    W = 2.03,
    M = 5,
    N = 2,
    I = 1000,
    # B-field solution grid
    Nzsteps  =  100,
    Nysteps  =  100,
    Nxsteps  =  100,
    xmin     = -5,
    xmax     =  5,
    ymin     = -3,
    ymax     =  3,
    zmin     = -3,
    zmax     =  3)

sHH_geometry = dict(
    config = 'sHH',
    n = [0, 0, 1],
    r0 = [0, 0, 0],
    L = 4.75,
    W = 2.84,
    d  = 0.086,
    M = 5,
    N = 2,
    ang = 0.0,
    A = 2.34 / 2.0,
    I = 1000,
    # B-field solution grid
    Nzsteps  =  100,
    Nysteps  =  100,
    Nxsteps  =  100,
    xmin     = -5,
    xmax     =  5,
    ymin     = -3,
    ymax     =  3,
    zmin     = -3,
    zmax     =  3
    )

smop3_geometry = dict(
    # coil params
    config = 'smop3',
    I2 = 1500,
    I1 = 1500,
    n = [0, 0, 1],
    r0 = [0, 0, 0],
    L1 = 4.75,
    W1 = 2.84,
    d  = 0.086,
    M1 = 5,
    N1 = 2,
    ang = 0.0,
    A = 2.34 / 2.0,
    #A = 2.08 / 2.0,
    L2 = 4.29,
    W2 = 2.03,
    M2 = 5,
    N2 = 2,
    # B-field solution grid
    Nzsteps  =  100,
    Nysteps  =  100,
    Nxsteps  =  100,
    xmin     = -5,
    xmax     =  5,
    ymin     = -3,
    ymax     =  3,
    zmin     = -3,
    zmax     =  3,
        )

AH_geometry = dict(
    config = 'AH',
    R = 1.0,                # radius, cm
    A = np.sqrt(3)*1.0/2,   # half distance between coils
    r0 = [0.0, 0.0, 0.0],   # center of coil pair, cm
    d = 0.081,              # wire diameter, cm
    M = 1,                  # number of turns in a layer
    N = 1,                  # number of layers
    n = [0, 0, 1.0],        # normal vector of coil pair
    I = 5000,               # current, A
    # spatial discretization of solution
    Nzsteps  =  100,
    Nysteps  =  100,
    Nxsteps  =  100,
    # spatial extent of solution
    xmin     = -2,
    xmax     =  2,
    ymin     = -2,
    ymax     =  2,
    zmin     = -2,
    zmax     =  2,
    )

def AH_example():
    # make the field
    # recalc_B = True will always recalculate the field
    # recalc_B = False will attempt to load saved field data
    AH_field = Field(AH_geometry, recalc_B=False)
    # run default plotting
    field_analysis(AH_field, 'AH_field',
        fignum=0,
        show=False,
        save=True,
        skip=10,
        min_frac=1,
        Bclip=2,
        grad_norm_B_clip=2)

    # Here is some code showing how to acess the field data at any point
    # Simulations uses microseconds time unit and cm as space unit.
    # Multiply field by 1e12 to convert to seconds since u0 ~ time^-2.
    # This puts B in units of Tesla
    # Multiply gradient by 1e12 for units of T/cm
    BXYZ_interp = AH_field.BXYZ_interp # list of three interpolation functions
    point = [0, 0, 0.5] # along z axis, field and gradient should be z-directed
    BXYZ = [1e12 * interp(point)[0] for interp in BXYZ_interp] # xyz components of of the field at point
    print('x, y, z [cm] components of field [T] at {} are \n {}'.format(point, BXYZ ))

    # The gradient of the norm of the field is also available
    grad_norm_BXYZ_interp = AH_field.grad_norm_BXYZ_interp # list of three interpolation functions
    grad_norm_BXYZ = [1e12 * interp(point)[0] for interp in BXYZ_interp] # xyz components of of the field at point
    print('x, y, z [cm] components of grad norm field [T/cm] at {} are \n {}'.format(point, grad_norm_BXYZ ))

    # here is a slightly more complicated use case
    # lets calculate the field along the z axis and compare to analytic result
    # exact axial solution
    def Bloop_axis(z, I, R, u0=1.257e-6*1e2*1e-12):
        return u0*I/2 * R*R/(z*z + R*R)**(3/2)

    def BAH_axis(z):
        I=AH_geometry['I']
        R=AH_geometry['R']
        A=AH_geometry['A']
        d=AH_geometry['d']
        return Bloop_axis(z - A - d/2,  I, R + d/2) +\
               Bloop_axis(z + A + d/2, -I, R + d/2)
    # prepare the 3D vectors for evaluation
    Npoints = 600
    r = np.zeros((Npoints, 3))
    r[::, 2] = np.linspace(-2, 2, Npoints)
    BXYZ = [1e12 * interp(r) for interp in BXYZ_interp]
    plt.plot(r[::, 2], BXYZ[2], label='3D analytic slice')
    plt.plot(r[::, 2], 1e12 * BAH_axis(r[::, 2]), '--', label='axis analytic')
    plt.ylabel(r'$B_z$ [T]')
    plt.xlabel('z [cm]')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    geometry = dict(
        # coil params
        config = 'mop3',
        I1      = 1500.0,  # bias max current, A
        I2      = 1500.0,  # kick max current, A
        n = [0, 0, 1],
        r0 = [0.0, 0.0, 0.0],
        d  = 0.086,

        R1 = 3.678,
        A1 = 1.580,
        M1 = 7,
        N1 = 2,

        R2 = 3.145,
        A2 = 1.717,
        M2 = 7,
        N2 = 2,

        # B-field solution grid
        Nzsteps  =  100,
        Nysteps  =  100,
        Nxsteps  =  100,
        xmin     = -5,
        xmax     =  5,
        ymin     = -3,
        ymax     =  3,
        zmin     = -3,
        zmax     =  3,
        )

    sg = split_geometry(geometry)
    ts = np.linspace(0, 500e-6, 200)
    Imaxs = {}
    for k, g in sg.items():
        Is = current(ts, V0=1000, tau=250e-6, **g)
        print(RLC_values(tau=250e-6, **g))
        Imax = np.max(Is)
        Imaxs[k] = Imax
        plt.plot(ts, Is)
    plt.show()
    print(Imaxs)

    zs = np.linspace(-0.5,0.5,200)
    geometry['z0']=0.0
    geometry['n']=1.0
    geometry['IAH']=np.min(list(Imaxs.values()))
    geometry['IHH']=np.min(list(Imaxs.values()))

    geometry['I1'] = 1500
    geometry['I2'] = 1500
    Bs = Bmop3_axis(zs, **geometry) * 1e12

    plt.clf()
    plt.plot(zs, Bs)
    plt.show()
    plt.clf()
    dBs = np.gradient(Bs, zs[1]-zs[0])
    plt.plot(zs, dBs)
    plt.show()
    print(np.mean(dBs), np.std(dBs))
    #smop3_field = Field(smop3_geometry, recalc_B=True)
    # run default plotting
    #show_field_and_wires(1, smop3_field)
    #plt.show()
    #field_analysis(smop3_field, 'smop3_pos_field',
    #    fignum=0,
    #    show=True,
    #    save=Falsd,
    #    skip=10,
    #    min_frac=1,
    #    Bclip=2,
    #    grad_norm_B_clip=2)

    #smop3_geometry = dict(
    #    # coil params
    #    config = 'smop3',
    #    I2 = 1500,
    #    I1 = -1500,
    #    n = [0, 0, 1],
    #    r0 = [0, 0, 0],
    #    L1 = 4.75,
    #    W1 = 2.84,
    #    d  = 0.086,
    #    M1 = 5,
    #    N1 = 2,
    #    ang = 0.0,
    #    A = 2.34 / 2.0,
    #    #A = 2.08 / 2.0,
    #    L2 = 4.29,
    #    W2 = 2.03,
    #    M2 = 5,
    #    N2 = 2,
    #    # B-field solution grid
    #    Nzsteps  =  100,
    #    Nysteps  =  100,
    #    Nxsteps  =  100,
    #    xmin     = -5,
    #    xmax     =  5,
    #    ymin     = -3,
    #    ymax     =  3,
    #    zmin     = -3,
    #    zmax     =  3,
    #        )
    #smop3_field = Field(smop3_geometry, recalc_B=True)
    ## run default plotting
    #field_analysis(smop3_field, 'smop3_neg_field',
    #    fignum=0,
    #    show=False,
    #    save=True,
    #    skip=10,
    #    min_frac=1,
    #    Bclip=2,
    #    grad_norm_B_clip=2)

