#! user/bin/python3
#
# tmps.py
#
# By Logan Hillberry
#
# lhillberry@gmail.com
#
# Last updated 30 October 2017
#
# DESCRIPTION
# ===========
# This script enables dynamical simulation of Natoms = 100000+ particles with
# dipole moment mu and mass m interacting with an applied magnetic field B.
# The particular field considered in this file is that of Helmholtz and
# anti-Helmoltz coil pairs. This field is of interest because it provides an
# approximatly constant gradient superposed with a uniform, directional bias.
# The field is used to provide a one-dimensional kick to the dipolar particles.
#
# The magneto-static equations for the field vectors come from the exact
# solution to a current carrying loop. The unique components (radial and z) are
# proportional to linear combinations of the complete elliptic integrals K and
# E. Supported 3D geometries include single current loops, MxN current coils,
# Helmoltz (HH)/ anti-Helmholtz (AH) coil pairs, and the HH + AH configuration
# used for magnetic-optical (MOP) cooling, called the MOP configuration.
# Additionally, the script supplies useful coil vizualization functions which
# orient a closed tube in three dimensional space with a thickness given in the
# same units as the axes. Several functions drew heavily from existing online
# resources, urls for which are given contextually in the source code.
#
# Dynamics are computed with an rk4 algorithm available as a method to the
# Cloud class. The acceleration used in rk4 is calculated at any point in
# spacetime by multiplying the gradient of the norm of the normalized magnetic
# field evaluated at point r, by a time dependent current signal evaluated at
# time t.
#
#                   a = +/- mu / m * I(t) * grad |B(r, I=1))|
#
# The sign is given by the internal magnetic state of the atom (+1 or -1) which
# is tracked as a seventh dimension of phase space. We assume a quantization
# axis of z. This means the mop coils are extpected to have a normal of
# [0,0,1] even though the feild calculator can handle arbitrary orientation.
# Factoring the time dependent current past the spatial derivative is only
# possible in the mop configuration if all coils have current signals
# proportional to one another. Furthermore, since evaluating the complete
# elliptic integrals can become expensive, we first evalute the field everywhere
# on a spatial grid, take its magnitude, then take the gradient of the result.
# Then, during time evolution, the acceleration vector of each particle is given
# component wise by three independent linear interpolations of the grad-norm-B
# grid.
#
# At the end of the simulation, the full state of the cloud (including
# trajectories, velocities, temperatures, densities, and phase space density,
# as well as the B field grid used define the acceleration function) is saved to
# disk as a binary file. The default behavior also generates several plots saved
# to a single file called mop_sim_default.pdf inside a user created directory c
# alled `plots`. If `animate_traj` is uncommented inside the default behavior
# the script will also generate an animated trajectory of the simulation, **but
# this is very slow and shold not be used as is for more than a couple hundered
# particles**, possibly running over night.
#
# Once a simulation is saved, it can be re loaded. this can be particularly
# useful to avaoid recalculating the B-field grid while changing cloud and pulse
# parameters. See the function `run_sim` for more loading flags.
#
# USAGE
# =====
# (Assuming a bash-like shell)
#
# Create a folder called `plots` in the directory containing this script
#
#                               mkdir plots
#
# To execute the default behavior, run
#
#                             python3 mop_sim.py
#
# in the directory containing this script. It will generate a file of several
# plot called plots called mop_sim.pdf in the `plots` directory.
#
# REQUIREMENTS
# ============
#
# This script requires python3, numpy, scipy, and matplotlib
#
# COIL GEOMETRY
# =============
#
# n, loop normal, current flow given by right hand rule
# r0, center-of-loop or center-of-coil-pair position vector [x0, y0, z0], cm
# R, loop radius, cm
# d, diameter of wire, cm
# M, number of loops per coil layer, int
# N, number of layers per coil, int
# A, coil pair half-separation, cm
# r, field points [x, y, z] or [[x1, y1, z1], [x2, y2, z2], ...]
# B* functions return [Bx, By, Bz] or [[Bx1, By1, Bz1], [Bx2, By2, Bz2], ...]
# Vacuum permiability sets units for the simulation, defined here globally
#
#               n
#               ^
#               |
#    d = 00     .     XX  ^
#        00  R  .     XX  |
#        00<----.     XX  M
#        00     .     XX  |
#        00     .     XX  v
#     -> N <-   ^
#               |
#               | A
#               |
#               V
#               .  r0
#
#
#
#
#
#        00<----.     XX
#        00     .     XX
#        00     .     XX
#        00     .     XX
#        00     .     XX
#

# Import required libraries
from itertools import cycle, product
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter, FormatStrFormatter

import numpy as np
from numpy import pi
from numpy.linalg import norm, inv

import scipy
from scipy.interpolate import interpn, interp1d, RegularGridInterpolator
import scipy.special as sl
import matplotlib.animation as animation

import sys
import os
import pickle
import traceback

# The following two lines ensure type 1 fonts are used in saved pdfs
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# plotting defaults
plt_params = {
          'font.size'   : 14,
          }
plt.rcParams.update(plt_params)

# UNITS
# =====
# Input allnumbers in uits of: cm, us, mK, A, kg
u0 = 1.257e-6 * 1e2 * 1e-12
    # magnetic permiability, kg m A^-2 s^-2 cm/m s^2/us^2
kB = 1.381e-23 * 1e4 * 1e-12 * 1e-3
    # Boltzmann's constant, kg m^2 s^-2 K^-1 cm^2/m^2 s^2/us^2 K/mk
h  = 6.62607004e-34 * 1e4 * 1e-6
    # Planck's Constant, m^2 kg s^-1 cm^2/m^2 s/us
rtube = 0.75 * 2.54 / 2
    # radius of slower tube, cm

# MAGNETIC FIELDS
# ===============
# Field at point r due to current I running in loop raidus R centered at r0 with
# normal vector n
def Bloop(r, I, n, r0, R):
    # coil frame to lab frame transformation and its inverse
    r0 = np.array(r0)
    n, l, m = coil_vecs(n)
    trans = np.vstack((l,m,n))
    inv_trans = inv(trans)

    # Shift origin to coil center
    r = r-r0

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
    Bz =  u0 * I / (2 * pi * np.sqrt((R + rho)**2 + z**2)) * (
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
    B = np.c_[np.cos(phi)*Brho, np.sin(phi)*Brho, Bz]
    B = np.dot(B, trans)
    return B

# Field of N layers of M wire loops (wire diameter d) centered at r0
# such that the first of M loops has its center at r0 and the Mth loop has its
# center at r0 + M*d*n where n is the normal to the loops. The first loop of N
# layers has radius R.
def Bcoil(r, I, n, r0, R, d, M, N):
    B = np.zeros_like(r)
    for j in range(M):
        for k in range(N):
            B += Bloop(r, I, n, r0 + n*(j+1/2)*d, R + (k+1/2)*d)
    return B

# Field of two identical coils situated co-axially along loop-normal n, with the
# closest loops of the two coils separated by 2A. Then, r0 is axial point
# midway between the two coils. Current in the two coils flows in opposite
# directions (anti-Helmholtz configuration)
def BAH(r, I, n, r0, R, d, M, N, A):
    r0a = r0 + n*A
    r0b = r0 - n * (A + M * d)
    return Bcoil(r, I, n, r0a, R, d, M, N)+\
           Bcoil(r, -I, n, r0b, R, d, M, N)

# Same as BHH but with currents of the two coils flowing in the same direction
# (Helmholtz configuration)
def BHH(r, I, n, r0, R, d, M, N, A):
    r0a = r0 + n * A
    r0b = r0 - n * (A + M * d)
    return Bcoil(r, I, n, r0a, R, d, M, N)+\
           Bcoil(r, I, n, r0b, R, d, M, N)

# Field of anti-Helmholtz and Helmholtz coils for creating a biased gradient
def Bmop(r, IAH, nAH, r0AH, RAH, dAH, MAH, NAH, AAH,
            IHH, nHH, r0HH, RHH, dHH, MHH, NHH, AHH, **kwargs):
    return BAH(r, IAH, nAH, r0AH, RAH, dAH, MAH, NAH, AAH) +\
           BHH(r, IHH, nHH, r0HH, RHH, dHH, MHH, NHH, AHH)


# HELPERS
# =======

# shape a vector field in grid form to point form
def vec_shape(V, xshape, yshape, zshape):
    Vx = V[::, 0]
    Vy = V[::, 1]
    Vz = V[::, 2]
    Vx.shape = xshape
    Vy.shape = yshape
    Vz.shape = zshape
    return Vx, Vy, Vz

# generate 3 orthonormal basis vectors for the coil, the first being colinear to
# the supplied vector n
def coil_vecs(n):
    # create two vectors perpindicular to n
    if  np.abs(n[0]) == 1:
        l = np.r_[n[2], 0, -n[0]]
    else:
        l = np.r_[0, n[2], -n[1]]
    # normalize coil's normal vector
    l = l/norm(l)
    m = np.cross(n, l)
    return n, l, m

def format_params(params):
    params['r0AH'] = np.array(params['r0AH'])
    params['r0HH'] = np.array(params['r0HH'])
    params['nAH'] = np.array(params['nAH'])
    params['nAH'] = params['nAH']/norm(params['nAH'])
    params['nHH'] = np.array(params['nHH'])
    params['nHH'] = params['nHH']/norm(params['nHH'])
    if params['delay'] == None:
        if params['v0'][0] == 0:
            params['delay'] = 0.0
        else:
            params['delay'] = (abs(
                    params['r0_cloud'][0])) / params['v0'][0]  # us
    if params['dt'] == None:
        params['dt'] = params['tau']/100  # us
    if params['tmax'] == None:
        params['tmax'] = (params['Npulse']) * (
            params['tau'] + params['tcharge']) + params['delay'] # us
    return params


# PLOTTING
# ========
def coil_viz(ax, n, r0, R, d, M, N, color=None):
    if color is None:
        c = cycle(['C' + str(i) for i in range(10)][:M*N])
    else:
       c =  cycle([color])
    for j in range(M):
        for k in range(N):
            shift = r0 + n*(j+1/2)*d
            rad = R + (k+1/2)*d
            lw = linewidth_from_data_units(d, ax)
            p = Circle((0,0), rad, edgecolor=next(c), lw=lw, ls='solid', fill=False)
            ax.add_patch(p)
            pathpatch_2d_to_3d(p, z = 0, normal = n)
            pathpatch_translate(p, shift)

def coil_pair_viz(ax, n, r0, R, d, M, N, A, color=None):
    r0a = r0 + n * A
    r0b = r0 - n * (A + M * d)
    coil_viz(ax, n, r0a, R, d, M, N, color=color)
    coil_viz(ax, n, r0b , R, d, M, N, color=color)

def mop_viz(ax, nAH, r0AH, RAH, dAH, MAH, NAH, AAH,
                nHH, r0HH, RHH, dHH, MHH, NHH, AHH,
                colorAH=None, colorHH=None, **kwargs):
    r0AHa = r0AH + nAH * AAH
    r0AHb = r0AH - nAH * (AAH + MAH * dAH)
    r0HHa = r0HH + nHH * AHH
    r0HHb = r0HH - nHH * (AHH + MHH * dHH)
    coil_pair_viz(ax, nAH, r0AH, RAH, dAH, MAH, NAH, AAH, color=colorAH)
    coil_pair_viz(ax, nHH, r0HH, RHH, dHH, MHH, NHH, AHH, color=colorHH)


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
    d /= sin_angle
    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)
    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.
    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)
    normal /= norm(normal) #Make sure the vector is normalised
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()
    path = trans.transform_path(path) #Apply the transform
    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color
    verts = path.vertices #Get the vertices in 2D
    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix
    pathpatch._segment3d = np.array([
        np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

#https://stackoverflow.com/questions/19394505/matplotlib-expand-the-line-with-specified-width-in-data-unit
def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.
    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.
    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)

def plot_grad_norm_B(ax, grad_norm_B, X, Y, Z, I0,
        skip=20, Bclip=2, min_frac=1, **kwargs):
    print('plotting 3D gradient vectors...')
    skip_slice = [slice(None, None, skip)]*3
    dBdx, dBdy, dBdz = I0*np.asarray(grad_norm_B)*1e12
    norm_gnB = np.sqrt(dBdx**2 + dBdy**2 + dBdz**2)
    dBdx[norm_gnB > Bclip] = 0
    dBdy[norm_gnB > Bclip] = 0
    dBdz[norm_gnB > Bclip] = 0
    norm_gnB[norm_gnB > Bclip] = Bclip
    Bmin = np.min(norm_gnB) * min_frac
    dBdx[norm_gnB < Bmin] = 0
    dBdy[norm_gnB < Bmin] = 0
    dBdz[norm_gnB < Bmin] = 0
    norm_gnB[norm_gnB < Bmin] = Bmin
    # plot the field and coils
    ax.set_ylabel('y [cm]')
    ax.set_xlabel('x [cm]')
    ax.set_zlabel('z [cm]')
    ax.quiver(X[skip_slice], Y[skip_slice], Z[skip_slice],
        dBdx[skip_slice], dBdy[skip_slice], dBdz[skip_slice],
        length=0.6, pivot='middle', lw=0.5)

def plot_grad_norm_B_slices(fignum, cloud,
        Bclip=3, min_frac=1, skip=25,**kwargs):
    print('plotting gradient slices...')
    grad_norm_B = cloud.grad_norm_B
    x, y, z = cloud.xyz
    I0 = cloud.params['I0']
    fig = plt.figure(fignum, figsize=(6,4))
    dBdx, dBdy, dBdz = I0*np.asarray(grad_norm_B)*1e12
    # plot the field and coils
    xinterp = RegularGridInterpolator((x,y,z), dBdx,
        method='linear', bounds_error=False, fill_value=0)
    yinterp = RegularGridInterpolator((x,y,z), dBdy,
        method='linear', bounds_error=False, fill_value=0)
    zinterp = RegularGridInterpolator((x,y,z), dBdz,
        method='linear', bounds_error=False, fill_value=0)
    for coordi, (coord, interp) in enumerate(zip(
            [r'\rho', 'z'], [xinterp, zinterp])):
        ax = fig.add_subplot(2,1,coordi+1)
        ax.set_ylabel(r'$\partial |B|/\partial %s$ [T/cm]' % coord)
        for slicei, rho in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
            label = r'${}$'.format(str(rho))
            on = np.ones(len(z))*rho
            off = np.zeros(len(z))
            if coord in ('y'):
                xs = np.vstack((off, on, z)).T
            if coord in ('x', 'z', r'\rho'):
                xs = np.vstack((on, off, z)).T
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if coordi == 0:
                ax.xaxis.set_major_formatter(NullFormatter())
            if coordi != 0:
                ax.set_xlabel(r'$z$ [cm]')
            mask = np.logical_and(z>-0.75, z<0.75)
            ax.plot(z[mask], interp(xs[mask]),label=label)
    ax.legend(ncol=1, loc='lower left', bbox_to_anchor=[1,0.2],
            handlelength=1, labelspacing=0.2, handletextpad=0.2)
    ax.text(1.04, 1.65, r'$\rho$ [cm]', transform=ax.transAxes)
    plt.subplots_adjust(hspace=0.0)
    fignum += 1
    return fignum

def plot_B(ax, B, X, Y, Z, I0,
        skip=20, Bclip=2, min_frac=1, **kwargs):
    print('plotting 3D field vectors...')
    skip_slice = [slice(None, None, skip)]*3
    Bx, By, Bz = I0*np.asarray(grad_norm_B)*1e12
    norm_gnB = np.sqrt(B**2 + B**2 + B**2)
    Bx[norm_gnB > Bclip] = 0
    By[norm_gnB > Bclip] = 0
    Bz[norm_gnB > Bclip] = 0
    norm_gnB[norm_gnB > Bclip] = Bclip
    Bmin = np.min(norm_gnB) * min_frac
    Bx[norm_gnB < Bmin] = 0
    By[norm_gnB < Bmin] = 0
    Bz[norm_gnB < Bmin] = 0
    norm_gnB[norm_gnB < Bmin] = Bmin
    # plot the field and coils
    ax.set_ylabel('y [cm]')
    ax.set_xlabel('x [cm]')
    ax.set_zlabel('z [cm]')
    ax.quiver(X[skip_slice], Y[skip_slice], Z[skip_slice],
        Bx[skip_slice], By[skip_slice], Bz[skip_slice],
        length=0.2, pivot='middle')
    mop_viz(ax, **params)

def plot_phase_space(fignum, cloud, time_indicies=[0, -1]):
    print('plotting phase space slices...')
    traj = cloud.traj
    vels = cloud.vels
    for i, (coord, letter) in enumerate(
            zip(('x', 'y', 'z'), ('(a)', '(b)', '(c)'))):
        fig = plt.figure(fignum, figsize=(3,3))
        for j, (ti, label) in enumerate(zip(time_indicies, ['initial', 'final'])):
            # the random data
            x = traj[ti, ::, i]
            y = vels[ti, ::, i] * 1e6 * 1e-2  # m/s
            nullfmt = NullFormatter()         # no labels
            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            bottom_h = left_h = left + width + 0.04
            rect_hist2d = [left, bottom, width, height]
            rect_histx = [left, bottom_h, width, 0.2]
            rect_histy = [left_h, bottom, 0.2, height]
            axScatter = fig.add_axes(rect_hist2d)
            axHistx = fig.add_axes(rect_histx)
            axHisty = fig.add_axes(rect_histy)
            # no labels
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHisty.yaxis.set_major_formatter(nullfmt)
            # the scatter plot:
            axScatter.scatter(x, y, s=0.25, label=label, rasterized=True,
            alpha=0.5)
            # now determine nice limits by hand:
            xmax, ymax = np.max(np.fabs(x)), np.max(np.fabs(y))
            binwidthx = 0.08
            binwidthy = binwidthx * (ymax/xmax)
            limx = (int(xmax/binwidthx) + 1) * binwidthx
            limy = (int(ymax/binwidthy) + 1) * binwidthy
            axScatter.set_xlim((-limx, limx))
            axScatter.set_ylim((-limy, limy))
            binsx = np.arange(-limx, limx + binwidthx, binwidthx)
            binsy = np.arange(-limy, limy + binwidthy, binwidthy)
            axHistx.hist(x, bins=binsx, alpha=0.7)
            axHisty.hist(y, bins=binsy, orientation='horizontal', alpha=0.7)
            axHistx.set_xlim(axScatter.get_xlim())
            axHisty.set_ylim(axScatter.get_ylim())
            axHistx.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
            axHisty.ticklabel_format(axis='x',style='sci',scilimits=(1,4))
            axHistx.yaxis.major.formatter._useMathText = True
            axHisty.xaxis.major.formatter._useMathText = True
            axScatter.set_xlabel('${}$ [cm]'.format(coord))
            axScatter.set_ylabel(r'$v_{}$ [m/s]'.format(coord))
        axScatter.legend(
            bbox_to_anchor=[1.52,1.42], markerscale=10, labelspacing=0.1,
            handletextpad=0.0, framealpha=0.0)
        #axHistx.set_title("{} phase space slice".format(coord))
        axHistx.text(-0.35, 0.5, letter,
            weight='bold', transform=axHistx.transAxes)
        fignum+=1
    return fignum

def plot_temps(fignum, cloud, include_names=['Tx','Ty','Tz','T']):
    ts = cloud.ts
    temps = cloud.temps
    temp_names = cloud.temp_names
    figT = plt.figure(fignum, figsize=(3,3))
    axT = figT.add_subplot(1,1,1)
    colors = ['C3', 'C7', 'k', 'C2']
    lines = ['--','-.', '-',':']
    for label, temp, color, line in zip(temp_names, temps.T, colors, lines):
        if label in include_names:
            use_label = '$T_{}$'.format(label[1])
            if label == 'T':
                use_label = '$T$'
            axT.semilogy(ts, temp, label=use_label, c=color, ls=line)
        axT.set_xlabel('$t$ [$\mu$s]')
        axT.set_ylabel('$T$ [mK]')
    axT.legend()
    fignum += 1
    #axT.text(-0.3, 1.01, '(d)', weight='bold', transform=axT.transAxes)
    return fignum

def plot_psd(fignum, cloud):
    ts = cloud.ts
    psd = cloud.psd
    figPSD = plt.figure(fignum, figsize=(3,3))
    axPSD = figPSD.add_subplot(1,1,1)
    axPSD.plot(ts, psd, c='k')
    axPSD.set_xlabel('$t$ [$\mu$s]')
    axPSD.set_ylabel('phase space density')
    fignum += 1
    return fignum

def plot_scalar_summary(fignum, cloud):
    ts = cloud.ts
    temps = cloud.temps
    dens = cloud.dens
    psd  = cloud.psd
    Is   = cloud.Is
    temp_names = cloud.temp_names
    dens_names = cloud.dens_names
    fig = plt.figure(fignum, figsize=(5,5))
    axI = fig.add_subplot(2,2,1)
    axT = fig.add_subplot(2,2,2)
    axD = fig.add_subplot(2,2,3)
    axR = fig.add_subplot(2,2,4)
    axI.plot(ts, Is)
    axI.plot(ts,Is, label='current')
    axI.legend()
    for label, den in zip(dens_names, dens.T):
        axD.plot(ts, den, label=label)
    axD.legend()
    for label, temp in zip(temp_names, temps.T):
        axT.plot(ts, temp, label=label)
    axT.legend()
    axR.plot(ts, psd, label='phase space density')
    return fignum

def plot_traj(fignum, cloud, seglen=2):
    traj             = cloud.traj
    spins            = cloud.spins
    xlim, ylim, zlim = cloud.xyzlim
    X, Y, Z          = cloud.grid_list
    grad_norm_B = cloud.grad_norm_B
    print('plotting 3D trajectory...')
    spin_color = ['red','black']
    fig = plt.figure(fignum, figsize=(8,8))
    ax = fig.gca(projection='3d')
    ax.set_ylabel('y [cm]')
    ax.set_xlabel('x [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
    ax.view_init(elev=10., azim=30)
    marker_dict={1:u'$\u2193$', -1:u'$\u2191$'}
    Ntsteps = len(traj[::,0,0])
    alphas = np.linspace(1/Ntsteps, 1, int(Ntsteps))
    for j, spin in enumerate(spins):
        if j < 1000:
            x, y, z = traj[::,j,0], traj[::,j,1], traj[::,j,2]
            ax.plot(x, y, z, color=spin_color[int((spin) + 1/2)],
                    lw=1.2)
            ax.plot([x[-1]],[y[-1]],[z[-1]], color='k',
                mew=0., alpha=0.9, marker=marker_dict[spin], ms=10)

            for f in range(0, Ntsteps-seglen, seglen):
                x, y, z = traj[f:f+seglen+1,j,0], traj[f:f+seglen+1,j,1],\
                        traj[f:f+seglen+1,j,2]
                #ax.plot(x, y, z, color=spin_color[int((spin) + 1/2)],
                #    lw=1.2, alpha = alphas[f])
            #ax.plot([x[-1]],[y[-1]],[z[-1]], color='k',
            #    mew=0., alpha=0.9, marker=marker_dict[spin], ms=10)
        elif j > 100:
            print('Only plotting trajectory for the first 1000 atoms')
            break

    if not grad_norm_B is None:
        plot_grad_norm_B(ax, grad_norm_B, X, Y, Z, **cloud.params)
    mop_viz(ax, **cloud.params)
    fignum+=1
    return fignum

def animate_traj(fignum, cloud):
    ts = cloud.ts
    traj = cloud.traj
    spins = cloud.spins
    xlim, ylim, zlim  = cloud.xyzlim
    plot_dir = cloud.params['plot_dir']
    suffix = cloud.params['suffix']
    print('animating trajectory...')
    # Set up formatting for the movie files
    # Attaching 3D axis to the figure
    anifig = plt.figure(fignum)
    aniax = axes3d.Axes3D(anifig)
    aniax.set_xlim(*xlim)
    aniax.set_ylim(*ylim)
    aniax.set_zlim(*zlim)
    # TODO: show scaled vector plot over time
    # plot_grad_norm_B(aniax, grad_norm_B, X, Y, Z, **params)
    # animate no more than 100 lines
    traj_transpose = np.transpose(traj,(1,2,0))[::,::,::]
    tail_length = (4/5)*len(ts)
    rgb_color = {1 : [0., 0.5, 1.0],
                -1 : [1.0, 0.5, 0.0]}
    vls = [Vanishing_Line(len(ts), tail_length, rgb_color[s]) for _, s in
            enumerate(spins)]
    for vl in vls:
        aniax.add_collection(vl.get_LineCollection())
    pts = [aniax.plot(trj[0, 0:1], trj[1, 0:1], trj[2, 0:1],
            markeredgecolor=[0,0,0,0.7])[0] for trj in traj_transpose]
    # Creating the Animation object
    #  may need blit=False on mac OSX
    line_ani = animation.FuncAnimation(anifig, update_lines, len(ts),
        fargs=(traj_transpose, vls, pts, spins), interval=50, blit=False,
        repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30,
            metadata=dict(artist='Logan Hillberry'), bitrate=-1)
    save_loc = os.path.join(plot_dir,'traj'+suffix+'.mp4')
    line_ani.save(save_loc, writer=writer)
    fignum += 1
    return fignum

# update function for trajactory animation
def update_lines(num, data, lines, pts, spins):
    Natom, Ncoord, Ntsteps = data.shape
    sys.stdout.write( 'animating step {}/{}\r'.format(num+1, Ntsteps))
    sys.stdout.flush()
    for j, (line, pt, spin, dat) in enumerate(zip(lines, pts, spins, data)):
        tail_length = max(0, num - line.tail_length)
        xs, ys, zs = dat[::, tail_length:num]
        for x, y, z in zip(xs, ys, zs):
            line.add_point(x, y, z)
        pt.set_data(xs[-1:], ys[-1:])
        pt.set_3d_properties(zs[-1:])
        if spin == -1:
            #line.set_color('cyan')
            pt.set_marker(u'$\u2193$') # down arrow unicode
        elif spin == 1:
            #line.set_color('magenta')
            pt.set_marker(u'$\u2191$') # up arrow unicode
    return lines


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

#https://stackoverflow.com/questions/43589232/how-can-i-change-the-alpha-value-dynamically-in-matplotlib-python
class Vanishing_Line(object):
    def __init__(self, n_points, tail_length, rgb_color):
        self.n_points = int(n_points)
        self.tail_length = int(tail_length)
        self.rgb_color = rgb_color

    def set_data(self, x=None, y=None, z=None):
        if x is None or y is None or z is None:
            self.lc = Line3DCollection([])
        else:
            # ensure we don't start with more points than we want
            x = x[-self.n_points:]
            y = y[-self.n_points:]
            z = z[-self.n_points:]
            # create a list of points with shape (len(x), 1, 3)
            # array([[[  x0  ,  y0  , z0 ]],
            #        [[  x1  ,  y1  , z1 ]],
            #        ...,
            #        [[  xn  ,  yn  , zn ]]])
            self.points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            # group each point with the one following it (shape (len(x)-1, 2, 2)):
            # array([[[  x0  ,   y0  , z0 ],
            #         [  x1  ,   y1  , z1 ]],
            #        [[  x1  ,   y1  , z1 ],
            #         [  x2  ,   y2  , z2]],
            #         ...
            self.segments = np.concatenate(
                    [self.points[:-1], self.points[1:]], axis=1)
            if hasattr(self, 'alphas'):
                del self.alphas
            if hasattr(self, 'rgba_colors'):
                del self.rgba_colors
            #self.lc = LineCollection(self.segments, colors=self.get_colors())
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())

    def get_LineCollection(self):
        if not hasattr(self, 'lc'):
            self.set_data()
        return self.lc

    def add_point(self, x, y, z):
        if not hasattr(self, 'points'):
            self.set_data([x],[y],[z])
        else:
            # TODO: could use a circular buffer to reduce memory operations...
            self.segments = np.concatenate(
                    (self.segments, [[self.points[-1][0], [x,y,z]]]))
            self.points = np.concatenate((self.points, [[[x,y,z]]]))
            # remove points if necessary:
            while len(self.points) > self.n_points:
                self.segments = self.segments[1:]
                self.points = self.points[1:]
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())

    def get_alphas(self):
        n = len(self.points)
        if n < self.n_points:
            rest_length = self.n_points - self.tail_length
            if n <= rest_length:
                return np.linspace(1.0/n, 1.0, n)
            else:
                tail_length = n - rest_length
                tail = np.linspace(1.0/tail_length, 1.0, tail_length)
                rest = np.zeros(rest_length)
                return np.concatenate((tail, rest))
        else: # n == self.n_points
            if not hasattr(self, 'alphas'):
                tail = np.linspace(1.0/self.tail_length, 1.0, self.tail_length)
                rest = np.zeros(self.n_points - self.tail_length)
                self.alphas = np.concatenate((tail, rest))
            return self.alphas

    def get_colors(self):
        n = len(self.points)
        if  n < 2:
            return [self.rgb_color+[1.] for i in range(n)]
        if n < self.n_points:
            alphas = self.get_alphas()
            rgba_colors = np.zeros((n, 4))
            # first place the rgb color in the first three columns
            rgba_colors[:,0:3] = self.rgb_color
            # and the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas
            return rgba_colors
        else:
            if hasattr(self, 'rgba_colors'):
                pass
            else:
                alphas = self.get_alphas()
                rgba_colors = np.zeros((n, 4))
                # first place the rgb color in the first three columns
                rgba_colors[:,0:3] = self.rgb_color
                # and the fourth column needs to be your alphas
                rgba_colors[:, 3] = alphas
                self.rgba_colors = rgba_colors
            return self.rgba_colors

# THERMODYNAMICS AND MECHANICS
# ============================
class Cloud():
    def __init__(self, Natom, T, width, r0_cloud, m, mu, suffix,
            v0, init_spins=None, seed=None, load=False, **kwargs):

        # http://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init
        filename, line_number, function_name, text =\
                traceback.extract_stack()[-2]

        def_name    = text[:text.find('=')].strip()
        self.name   = def_name
        self.suffix = suffix

        if load:
            self.load()

        else:
            self.m         = m
            self.mu        = mu
            self.N         = Natom
            self.T         = T
            self.width     = width
            self.r0        = r0_cloud
            self.particles = np.zeros((self.N, 8))
            self.xs        = self.particles[:, 0:3]
            self.vs        = self.particles[:, 3:6]
            self.spins     = self.particles[:, 6]
            self.drop_mask = [False]*self.N
            self.keep_mask = np.logical_not(self.drop_mask)

            for i in range(3):
                np.random.seed(seed)
                self.xs[:,i]   = np.random.normal(
                        self.r0[i], self.width[i], self.N)
                self.vs[:,i]  = maxwell_velocity(
                        self.T, self.m, nc=self.N) + v0[i]

            if init_spins == None:
                np.random.seed(seed)
                self.spins[::] = np.random.choice([-1,1], size=self.N)
            else:
                self.spins[::] = np.array([init_spins]*self.N)

            # names of coordinates and their slices, callable from coord_dict
            xs_c            = ((0,3,None), 'xs', 'pos')
            vs_c            = ((3,6,None), 'vs', 'vel', 'vels')
            s_c             =  (7, 's', 'spin', 'spins')
            coords          = [xs_c, vs_c, s_c]
            self.coord_dict = {k:coord[0] for coord in coords for k in coord}

            # names of variables returned by get_temp and get_density.
            self.temp_names = ('Tx', 'Ty', 'Tz', 'T ')
            self.dens_names = ('Dx', 'Dy', 'Dz', 'D ')
            # available optical pumping in the z direction, position or velocity
            self.typs       = ['vs', 'xa']

    # save state of the class instance with name given by the class instance
    # variable name (`cloud` by default)
    def save(self):
        # cloud.bin
        name = self.name + self.suffix + '.bin'
        print('saving simulation data to {}'.format(name))
        file = open(name,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    # load state...
    def load(self):
        name = self.name + self.suffix + '.bin'
        print('loading simulation data from {}'.format(name))
        file = open(name,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)

    def get_particles(self, coord):
        return self.particles[::, slice(*self.coord_dict[coord])]

    def set_particles(self, coord, val):
        self.particles[::, slice(*self.coord_dict[coord])] = val

    def rk4(self, a, t, dt):
        mz = np.vstack([self.spins]*3).T
        k1 = dt * mz *  a(self.xs, t)
        l1 = dt * self.vs
        k2 = dt * mz * a(self.xs + l1/2, t + dt/2)
        l2 = dt * (self.vs + k1/2)
        k3 = dt * mz * a(self.xs + l2/2, t + dt/2)
        l3 = dt * (self.vs + k2/2)
        k4 = dt * mz * a(self.xs + l3, t + dt)
        l4 = dt * (self.vs + k3)
        new_xs = self.xs + 1/6 * (l1 + 2*l2 + 2*l3 + l4)
        old_xs = self.xs.copy()
        new_vs = self.vs + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        old_vs = self.vs.copy()
        self.xs[::] = new_xs
        # drop particles outside the following simulation volume
        #self.drop_mask = np.all(np.vstack([np.abs(new_xs[::,0]) > rtube,\
        #    new_xs[:,1]**2 +  new_xs[:,2] > rtube**2]), axis=0)
        #self.xs[self.drop_mask] = old_xs[self.drop_mask]
        self.vs[::] = new_vs
        #self.vs[self.drop_mask] = old_vs[self.drop_mask]

    def optical_pump(self, mode):
        if mode in ('vs', 'vel', 'v'):
            self.spins[::] = np.sign(self.vs[::,2])
        if mode in ('xs', 'pos', 'x'):
            self.spins[::] = np.sign(self.xs[::,2])

    def get_temp(self):
        c = self.m/kB
        Tx = c*np.var(self.vs[self.keep_mask,0])
        Ty = c*np.var(self.vs[self.keep_mask,1])
        Tz = c*np.var(self.vs[self.keep_mask,2])
        T  = (Tx  + Ty + Tz)/3
        return Tx, Ty, Tz, T

    def get_number(self):
        n = np.sum(self.keep_mask)
        return n

    def get_density(self, correction=1e7):
        n = self.get_number()
        Dx = n/(2*pi*np.var(self.xs[self.keep_mask, 0]))**(3/2)
        Dy = n/(2*pi*np.var(self.xs[self.keep_mask, 1]))**(3/2)
        Dz = n/(2*pi*np.var(self.xs[self.keep_mask, 2]))**(3/2)
        D = n / ( 2/3 * pi *
            np.sum(np.var(self.xs[self.keep_mask,::], axis=0), axis=0))**(3/2)
        return Dx, Dy, Dz, D*correction

    def get_psd(self):
        Tx, Ty, Tz, T = self.get_temp()
        Dx, Dy, Dz, D = self.get_density()
        rho = D * (h**2/(2 * pi * self.m * kB * T))**(3/2)
        return rho


def curr_pulse(t, Npulse, tcharge, delay, tau, I0, t0, shape, decay, **kwargs):
    tprime = t - delay
    limda = t0
    limdb = t0 + tau
    for p in range(Npulse):
        shapes = {'sin': np.sin(pi * (tprime - limda) / tau), 'square': 1 }
        if limda < tprime < limdb:
            return I0/(decay)**p * shapes[shape]
        limda += tau + tcharge
        limdb += tau + tcharge
    else:
        return 0.0

def maxwell_velocity(T, m, seed=None, nc=3):
    np.random.seed(seed)
    # characteristic velocity of temperature T
    v_ = (kB * T / m)**(1/2) # cm/us
    if nc == 1:
        return v_ * np.random.normal(0, 1)
    else:
        return v_ * np.random.normal(0, 1, nc)


def make_grid(xmin, xmax, Nxsteps, ymin, ymax, Nysteps, zmin, zmax, Nzsteps,
        **kwargs):
    print('making solution grid...')
    # grid spacing
    dx = (xmax - xmin)/Nxsteps
    dy = (ymax - ymin)/Nysteps
    dz = (zmax - zmin)/Nzsteps
    # axes
    x = np.linspace(xmin, xmax, Nxsteps)
    y = np.linspace(ymin, ymax, Nysteps)
    z = np.linspace(zmin, zmax, Nzsteps)
    # The grid of points on which we want to evaluate the field
    grid_list = np.meshgrid(x, y, z)
    X, Y, Z = grid_list
    # grid shapes
    xshape = X.shape
    yshape = Y.shape
    zshape = Z.shape
    r = np.vstack(grid_list).reshape(3, -1).T
    # convert grid coordinates to point form
    #r = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    # cubic limits for square aspect ratio in 3D plots
    max_range = np.array([X.max()-X.min(),
        Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    xlim = mid_x - max_range, mid_x + max_range
    ylim = mid_y - max_range, mid_y + max_range
    zlim = mid_z - max_range, mid_z + max_range
    rr = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T
    return x,y,z, X,Y,Z, r, dx,dy,dz, xshape,yshape,zshape, xlim,ylim,zlim

# unscaled for current
def make_B(r, xshape, yshape, zshape, dx, dy, dz, B_statistics=False, **kwargs):
    print('calculating field and gradient...')
    # grid spacing
    # mop field, shaped like r
    B = Bmop(r, IAH=1, IHH=kwargs['HH_scale'], **kwargs)  # kg A^-1 us^-2

    # grids of mop field components
    Bx, By, Bz = vec_shape(B, xshape, yshape, zshape)

    # grid of field magnitude and grids of its gradient
    norm_B = np.sqrt(Bx**2 + By**2 + Bz**2)
    grad_norm_B = np.gradient(norm_B, dx, dy, dz) # kg A^-1 us^-2 cm^-1
    dBdx, dBdy, dBdz = grad_norm_B

    if B_statistics:
        # z gradient in control volume
        dBdz_inside = dBdz[
            int(xshape[0]/2)-int(rtube/dx):int(xshape[0]/2)+int(rtube/dx),
            int(ysgape[0]/2)-int(rtube/dy):int(ysgape[0]/2)+int(rtube/dy),
            int(zshape[0]/2)-int(rtube/dz):int(zshape[0])+int(rtube/dz)]

        # gradient statistics inside control volume
        grad_opt = (kB*params['T']*m)**(1/2)/(mu*params['tau'])
        grad_center = dBdz[int(Nxsteps/2), int(Nxsteps/2), 0]
        grad_mean = np.mean(dBdz_inside)
        grad_max = np.max(dBdz_inside)

        print( 'avg gradient supplied {} T/cm'.format(
                params['I0'] * grad_mean * 1e12))

        print('expected delta vz {} (central grad)'.format(
            params['I0'] * grad_center * params['tau'] * mu / m * 1e6 * 1e-2))
        print('expected delta vz {} (mean grad)'.format(
            params['I0'] * grad_mean * params['tau'] * mu / m * 1e6 * 1e-2))
        print('expected delta vz {} (max grad)'.format(
            params['I0'] * grad_max  * params['tau'] * mu / m * 1e6 * 1e-2))
        print('ideal delta vz {} (optimal grad)'.format(
            grad_opt * params['tau'] * mu / m * 1e6 * 1e-2))
        print('observed delta vz = {}'.format(
            (np.mean(vels[-1,::,2]) - np.mean(vels[0,::,2])) * 1e-2 * 1e6))
    return Bx, By, Bz, dBdx, dBdy, dBdz

def make_acceleration(curr_pulse, xyz, grad_norm_B, params):
    x, y, z = xyz
    dBdx,dBdy, dBdz = grad_norm_B
    def a(xs, t):
        xinterp = RegularGridInterpolator((x,y,z), dBdx,
            method='linear', bounds_error=False, fill_value=0)
        yinterp = RegularGridInterpolator((x,y,z), dBdy,
            method='linear', bounds_error=False, fill_value=0)
        zinterp = RegularGridInterpolator((x,y,z), dBdz,
            method='linear', bounds_error=False, fill_value=0)
        dBdx_interp = xinterp(xs)
        dBdy_interp = yinterp(xs)
        dBdz_interp = zinterp(xs)
        a_xyz = - params['mu'] / params['m'] * curr_pulse(t, **params) * np.c_[
                dBdx_interp, dBdy_interp, dBdz_interp]
        return a_xyz
    return a

# Default behavior
# set load true to import previously saved data. If False, all other import
# flags are ignored
def run_sim(params_tmp,
        load=False, save=True,
        recalculate_B=False, resimulate=True, replot=True):
    # process supplied params dictonary which may contain lists of parameters,
    # signifying a parameter sweep.
    params_list = []
    sweep_vals_list = []
    sweep_params=[]
    for k, v in params_tmp.items():
        # these parameters are a vectors, check if a list of them is supplied
        if k in ('r0AH', 'r0HH', 'width', 'nAH', 'nHH', 'r0_cloud', 'v0'):
            if type(v[0]) == list:
                sweep_params.append(k)
                sweep_vals_list.append(v)
        # Most parameters are scalars, check if a list of them is supplied
        else:
            if type(v) == list:
                sweep_params.append(k)
                sweep_vals_list.append(v)
    # dictinary of sweeped parameters as keys, their length as values
    sweep_shape = {sweep_param:len(sweep_vals) for
        sweep_param, sweep_vals in zip(sweep_params, sweep_vals_list)}
    # Total number of simulations requested
    num_sims = np.product([ i for i in sweep_shape.values()], dtype=int)
    # initialize final/initial ratios of temperatures, x, y, z, and average
    temp_ratios = np.zeros((4, num_sims))
    if num_sims > 1:
        sim_num = 0
        # ensure every a set of parameters is generated for every combination of
        # simulation parameters
        sweep_vals_list_product = product(*sweep_vals_list)
        sweep_params_product = [sweep_params] * num_sims
        for sweep_vals, sweep_params in zip(
                sweep_vals_list_product, sweep_params_product):
            new_params = params_tmp.copy()
            for sweep_val, sweep_param in zip(sweep_vals, sweep_params):
                new_params[sweep_param] = sweep_val
                new_params['suffix'] = params_tmp['suffix'] + '-' + str(sim_num)
            params_list.append(new_params)
            sim_num += 1

    else:
       params_list = [params_tmp]

    # Run each simulation
    for sim_num, params in enumerate(params_list):
        print(params['delay'], params['v0'])
        print('BEGINING SIMULATION {} of {}'.format(sim_num, num_sims))
        print()
        params = format_params(params)
        for param, val in params.items():
            print('{} = {}'.format(param, val))
        print()
        # instance of cloud class for each simulation
        cloud = Cloud(**params, init_spins=None, load=load)
        # time axis
        ts = np.arange(params['t0'], params['tmax'], params['dt'])
        Ntsteps = len(ts)
        if not hasattr(cloud, 'grad_norm_B') or recalculate_B:
            # electric current evaluated at all simulation times
            Is = [curr_pulse(t, **params) for t in ts]
            # make and characterize 3D grid
            x,y,z, X,Y,Z, r, dx,dy,dz, xshape,yshape,zshape, xlim,ylim,zlim =\
                make_grid(**params)
            # get B field and grad of norm of B field (unscaled for current)
            Bx, By, Bz, dBdx, dBdy, dBdz = make_B(r, xshape, yshape, zshape, dx, dy,
            dz, **params)

            # load simulation data into the cloud class
            cloud.params            = params
            cloud.grid_list         = [X, Y, Z]
            cloud.xyz               = [x, y, z]
            cloud.r                 = r
            cloud.dxyz              = [dx, dy, dz]
            cloud.xyzshape          = [xshape, yshape, zshape]
            cloud.xyzlim            = [xlim, ylim, zlim]
            cloud.ts                = ts
            cloud.Is                = Is
            cloud.grad_norm_B       = [dBdx, dBdy, dBdz]

        if not hasattr(cloud, 'traj') or resimulate:
            # make an acceleration function a(xs, t) for use in rk4
            a = make_acceleration(curr_pulse, cloud.xyz,
                    cloud.grad_norm_B, params)
            # initialize memory for simulation measures
            # real space trajectory
            cloud.traj = np.zeros((Ntsteps, params['Natom'], 3))
            # velocity space trajectory
            cloud.vels = np.zeros((Ntsteps, params['Natom'], 3))
            # x, y, z, and average temperatures
            cloud.temps = np.zeros((Ntsteps, 4))
            # x, y, z, and average densities
            cloud.dens = np.zeros((Ntsteps, 4))
            # phase space density
            cloud.psd  = np.zeros(Ntsteps)

            # run the time evolution
            print()
            ti = 0
            # step forward through initial delay
            # TODO: no need to rk4 this since there are no forces
            for t in np.arange(params['t0'], params['delay'], params['dt']):
                if ti < Ntsteps:
                    cloud.rk4(a, t, params['dt'])
                    cloud.traj[ti, ::, ::] = cloud.get_particles('xs')
                    cloud.vels[ti, ::, ::] = cloud.get_particles('vs')
                    cloud.temps[ti,::] = cloud.get_temp()
                    cloud.dens[ti] = cloud.get_density()
                    cloud.psd[ti] = cloud.get_psd()
                    ti+=1

            # step forward through pulse sequence
            ta = sum(params[tname] for tname in ('t0', 'delay'))
            tb = sum(params[tname] for tname in ('t0', 'delay', 'tau', 'tcharge'))
            for p in range(params['Npulse']):
                for t in np.arange(ta, tb, params['dt']):
                    sys.stdout.write(' '*45 + 'simulating t = {:.2f} of {}\r'.format(t, params['tmax']))
                    sys.stdout.flush()
                    if  ta - params['dt']/2 < t < ta + params['dt']/2:
                        typ = cloud.typs[0]
                        print('PULSE {} BEGINS t = {}'.format(p+1, t))
                        print('optically pumped t = {}'.format(t))
                        cloud.optical_pump(typ)
                    if ti < Ntsteps:
                        cloud.rk4(a, t, params['dt'])
                        cloud.traj[ti, ::, ::] = cloud.get_particles('xs')
                        cloud.vels[ti, ::, ::] = cloud.get_particles('vs')
                        cloud.temps[ti, ::] = cloud.get_temp()
                        cloud.dens[ti] = cloud.get_density()
                        cloud.psd[ti] = cloud.get_psd()

                    if ta + params['tau']/2 - params['dt']/2 < t < ta + params['tau']/2 + params['dt']/2:
                        print('current peaks t = {}\r'.format(t))

                    if ta + params['tau'] - params['dt']/2 < t < ta + params['tau'] + params['dt']/2:
                        print('Pulse {} ends t = {}\r'.format(p+1, t))
                        print( 'temp | initl | final | ratio')
                        print( '----------------------------')
                        for label, temp in zip(cloud.temp_names, cloud.temps.T):
                            print(  '  {} | {:>5.1f} | {:>5.1f} | {:<5.3f}'.format(
                                label, temp[0], temp[ti], temp[ti]/temp[0]))
                        print()
                    ti += 1
                ta = tb
                tb = ta + params['tau'] + params['tcharge']

            cloud.Nremain = np.sum(cloud.keep_mask)
            print('{}/{} atoms remain'.format(
                    cloud.Nremain, params['Natom']))
            print()

        # plotting
        if not hasattr(cloud, 'traj') or\
                not hasattr(cloud, 'grad_norm_B') or replot:
            fignum = 0
            #fignum = animate_traj(fignum, cloud)
            fignum = plot_grad_norm_B_slices(fignum, cloud)
            fignum = plot_traj(fignum, cloud)
            fignum = plot_phase_space(fignum, cloud)
            fignum = plot_temps(fignum, cloud, include_names=['Tx','Ty','Tz'])
            fignum = plot_psd(fignum, cloud)
            fignum = plot_scalar_summary(fignum, cloud)

            # show or save plots
            #plt.show()
            save_loc = os.path.join(params['plot_dir'],'mop_sim' +\
                    params['suffix'] + '.pdf')
            print('saving plots...')
            multipage(save_loc, dpi=600)
            print('plots saved to {}'.format(save_loc))
        # Save the state of the cloud to disk
        if save == True:
            cloud.save()
        temp_ratios[::, sim_num] = cloud.temps[-1]/cloud.temps[0]
    return sweep_shape, sweep_vals_list, temp_ratios

def process_sweep(sweep_shape, sweep_vals_list, temp_ratios):
    shaper = [i for i in sweep_shape.values()]
    sweep_params = [i for i in sweep_shape.keys()]
    sweep_vals_list_flat = []

    # grab the first component of the v0 vector to use as an axis
    for sweep_vals in sweep_vals_list:
        sv = [v if type(v) != list else v[0] for v in sweep_vals]
        sweep_vals_list_flat.append(sv)

    #sweep_vals_list_flat = np.asarray(sweep_vals_list_flat)
    X, Y = np.meshgrid(*sweep_vals_list_flat)
    ys, xs = sweep_vals_list_flat
    Txs = temp_ratios[0].reshape(shaper)
    Tys = temp_ratios[1].reshape(shaper)
    Tzs = temp_ratios[2].reshape(shaper)
    Ts  = temp_ratios[3].reshape(shaper)
    TTs = [Txs, Tys, Tzs]

    fignum=100
    figs=[]
    for label, T in zip(['Tx/Tx0', 'Ty/Ty0', 'Tz/Tz0'], TTs):
        fig = plt.figure(fignum, figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(T, interpolation=None, origin='lower', aspect='auto')

        xticks = range(0, len(xs))
        yticks = range(len(ys))

        xticklabels = ["{:6.0f}".format(x) for x in xs[::4]]
        yticklabels = ["{:6.0f}".format(y*1e-2*1e6) for y in ys]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        ax.set_title(label)
        ax.set_xlabel('delay [us]')
        ax.set_ylabel('V0 [m/s]')
        plt.colorbar(im)
        fignum += 1
        figs.append(fig)
    save_loc = os.path.join('plots', 'mop_sim_T-ratios' + '.pdf')
    print('saving plots...')
    multipage(save_loc, figs=figs, dpi=600)

if __name__ == '__main__':
    pass
