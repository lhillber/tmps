#! user/bin/python3
#
# tmps.py
#
# By Logan Hillberry
#
# lhillberry@gmail.com
#
# Last updated 17 February 2018
#
# DESCRIPTION
# ===========
# This script enables dynamical simulation of Natom = 100000+ particles with
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
# axis of z. This means the     mop coils are extpected to have a normal of
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
# alled `plots`.
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
from itertools import product, cycle
import hashlib
import json
import copy
from functools import reduce  # forward compatibility for Python 3
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from numpy import pi
from numpy.linalg import norm

import scipy
from scipy.stats import gaussian_kde, skew
from mpi4py import MPI
from fastkde import fastKDE

import sys
import os
import pickle
import traceback

import magnetics as mag

# The following two lines ensure type 1 fonts are used in saved pdfs
mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype']  = 42

# plotting defaults
plt_params = {
          'font.size'   : 14,
          }
plt.rcParams.update(plt_params)

#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


# UNITS
# =====
# Input allnumbers in uits of: cm, us, mK, A, kg
kB = 1.381e-23 * 1e4 * 1e-12 * 1e-3
    # Boltzmann's constant, kg m^2 s^-2 K^-1 cm^2/m^2 s^2/us^2 K/mk
h  = 6.62607004e-34 * 1e4 * 1e-6
    # Planck's Constant, m^2 kg s^-1 cm^2/m^2 s/us
rtube = 0.75 * 2.54 / 2
    # radius of slower tube, cm


# PLOTTING
# ========
def plot_phase_space(fignum, sim, cloud, time_indicies=[0, -1]):
    print('Plotting phase space slices...')
    traj = sim.measures['traj'][::, cloud.keep_mask, ::]
    vels = sim.measures['vels'][::, cloud.keep_mask, ::]
    for i, (coord, letter) in enumerate(
            zip(('x', 'y', 'z'), ('', '', ''))):
        fig = plt.figure(fignum, figsize=(3,3))
        for j, (ti, label) in enumerate(zip(time_indicies, ['initial', 'final'])):
            x = traj[ti, ::, i] - cloud.r0[i]
            if coord == 'x':
                x = x - np.mean(x)
            y = (vels[ti, ::, i] - cloud.v0[i])* 1e6 * 1e-2  # m/s
            nullfmt = NullFormatter()         # no labels
            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            bottom_h = left_h = left + width + 0.04
            rect_hist2d = [left, bottom, width, height]
            rect_histx = [left, bottom_h, width, 0.2]
            rect_histy = [left_h, bottom, 0.2, height]
            axScatter = fig.add_axes(rect_hist2d, label='scatter')
            axHistx = fig.add_axes(rect_histx, label='histx')
            axHisty = fig.add_axes(rect_histy, label='histy')
            # no labels
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHisty.yaxis.set_major_formatter(nullfmt)
            # the scatter plot:
            axScatter.scatter(x, y, s=0.25, label=label, rasterized=True,
            alpha=0.5)
            # now determine nice limits by hand:
            xmax, ymax = np.max(np.fabs(x)), np.max(np.fabs(y))
            binwidthx = 0.025
            binwidthy = binwidthx * (ymax/xmax)
            limx = (int(xmax/binwidthx) + 1) * binwidthx
            limy = (int(ymax/binwidthy) + 1) * binwidthy
            mx = np.mean(x)
            my = np.mean(y)
            axScatter.set_xlim((mx-limx, mx+limx))
            axScatter.set_ylim((my-limy, my+limy))
            binsx = np.arange(-limx, limx + binwidthx, binwidthx)
            binsy = np.arange(-limy, limy + binwidthy, binwidthy)
            axHistx.hist(x, bins=binsx, alpha=0.7)
            axHisty.hist(y, bins=binsy, orientation='horizontal', alpha=0.7)
            axHistx.set_xlim(axScatter.get_xlim())
            axHisty.set_ylim(axScatter.get_ylim())
            #axHistx.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
            #axHisty.ticklabel_format(axis='x',style='sci',scilimits=(1,4))
            #axHistx.yaxis.major.formatter._useMathText = True
            #axHisty.xaxis.major.formatter._useMathText = True
            axScatter.set_xlabel(r'$%s$ [cm]' % coord)
            axScatter.set_ylabel(r'$v_%s$ [m/s]' % coord)
        axScatter.legend(
            bbox_to_anchor=[1.52,1.42], markerscale=10, labelspacing=0.1,
            handletextpad=0.0, framealpha=0.0)
        #axHistx.set_title("{} phase space slice".format(coord))
        axHistx.text(-0.35, 0.5, letter,
            weight='bold', transform=axHistx.transAxes)
        fignum+=1
    return fignum


def plot_phase_space2(fignum, sim, cloud,
        time_indicies=[0, -1], remove_mean=False, Nkde=-1):
    print('Plotting phase space slices...')
    for ti in time_indicies:
        xs = sim.measures['traj'][ti, cloud.keep_mask, ::]
        vs = sim.measures['vels'][ti, cloud.keep_mask, ::] * 1e6 * 1e-2 
        fig, axs = plt.subplots(4, 4, num=fignum, figsize=(7,7), sharex=False, sharey=False)
        coordi = [[(None, None), (1, 0), (2, 1), (0, 2)],
                  [   (3, 4),    (3, 0), (3, 1), (3, 2)],
                  [   (4, 5),    (4, 0), (4, 1), (4, 2)],
                  [   (5, 3),    (5, 0), (5, 1), (5, 2)]]
        coords = ['$x$', '$y$', '$z$', '$v_x$', '$v_y$', '$v_z$']
        ps = np.hstack([xs, vs])
        for i in range(4):
            for j in range(4):
                ax = axs[i, j]
                n, m = coordi[i][j]
                if (m, n) == (None, None):
                    ax.axis('off')
                    continue
                x = ps[:Nkde, m]
                y = ps[:Nkde, n]
                xm = np.mean(x)
                ym = np.mean(y)
                if remove_mean:
                    x = x - xm
                    y = y - ym
                    xm = 0.0
                    ym = 0.0
                xname = coords[m]
                yname = coords[n]
                #xy = np.vstack([x, y])
                #z = gaussian_kde(xy)(xy)
                Z, [xax, yax] = fastKDE.pdf(x, y)
                X, Y = np.meshgrid(xax, yax)
                x, y, z = [c.flatten() for c in (X, Y, Z)]
                zm = np.mean(z)*2
                x[z<zm] = np.nan
                y[z<zm] = np.nan
                z[z<zm] = np.nan
                #Sort the points by density
                idx = z.argsort()
                idx = np.random.choice(idx, 10000)
                x, y, z = x[idx], y[idx], z[idx]
                if xname[1] == 'v':
                    ax.set_xlim(xm-20, xm+20)
                else:
                    ax.set_xlim(xm-2, xm+2)
                if yname[1] == 'v':
                    ax.set_ylim(ym-20, ym+20)
                else:
                    ax.set_ylim(ym-2, ym+2)
                if i==0:
                    ax.set_ylabel(yname)
                if i == 3:
                    ax.set_xlabel(xname)
                if j==0:
                    ax.set_xlabel(xname)
                    ax.set_ylabel(yname)
                if i != 3:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if j != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.scatter(x, y, c=z, s=15, rasterized=True,
                        alpha=1, edgecolors='none')
        fignum += 1
    return fignum

def plot_integrated_density(fignum, sim, cloud):
    print('Plotting real space slices...')
    traj = sim.measures['traj'][::, cloud.keep_mask, ::]
    vels = sim.measures['vels'][::, cloud.keep_mask, ::]
    coords = ('x', 'y', 'z')
    fig = plt.figure(fignum, figsize=(6,3))
    inds_list = [(0,2),(0,1)]
    xyzs = [0, 0]
    zmax = 0
    for i, inds in enumerate(inds_list):
        x, y = [traj[-1, 1:1000, inds[0]], traj[-1, 1:1000, inds[1]]]
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Sort the points by density
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        xyzs[i] = [x,y,z]
        zmax_tmp = np.max(z)
        if zmax_tmp > zmax:
            zmax = zmax_tmp
    for i, (xyz, inds) in enumerate(zip(xyzs, inds_list)):
        x, y, z = xyz
        ax = fig.add_subplot(1,2,i+1)
        acoord, bcoord = coords[inds[0]], coords[inds[1]]
        scatter = ax.scatter(x, y, c=z, s=10, edgecolor='', vmax=zmax)
        ax.axhline(y=np.mean(y))
        ax.set_xlabel(acoord)
        ax.set_ylabel(bcoord)
        x0, x1 = ax.get_xlim()
        y0, y1 = [-1, 1]
        ax.set_ylim([y0, y1])
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    fig.subplots_adjust(wspace=0.5, right=0.8, top=0.9, bottom=0.1)
    cbar_ax = fig.add_axes([0.83, 0.23, 0.03, 0.53])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    #old_ticks = cbar.ax.get_yticklabels()
    #cbar_ticks = [ i for i in
    #        map(lambda x: "{:1.1f}".format(float(x.get_text())), old_ticks)]
    #cbar.ax.set_yticklabels(cbar_ticks)
    fignum+=1
    return fignum

def plot_integrated_density2(fignum, sim, cloud, time_indicies=[-1]):
    print('Plotting real space slices...')
    traj = sim.measures['traj'][::, cloud.keep_mask, ::]
    coords = ('x', 'y', 'z')
    inds_list = [(0,2),(0,1)]
    xyzs = [0, 0]
    zmax = 0
    nullfmt = NullFormatter()         # no labels
    for i, inds in enumerate(inds_list):
        fig = plt.figure(fignum, figsize=(6,3))
        for j, (ti, label) in enumerate(zip(time_indicies, ['initial', 'final'])):
            x, y = [traj[ti, ::, inds[0]], traj[ti, ::, inds[1]]]
            x = x - np.mean(x)
            y = y - np.mean(y)
            xname = coords[inds[0]]
            yname = coords[inds[1]]
            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            bottom_h = left_h = left + width + 0.04
            rect_hist2d = [left, bottom, width, height]
            rect_histx = [left, bottom_h, width, 0.2]
            rect_histy = [left_h, bottom, 0.2, height]
            axScatter = fig.add_axes(rect_hist2d, label='scatter')
            axHistx = fig.add_axes(rect_histx, label='histx')
            axHisty = fig.add_axes(rect_histy, label='histy')
            # no labels
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHisty.yaxis.set_major_formatter(nullfmt)
            # the scatter plot:
            axScatter.scatter(x, y, s=0.25, label=label, rasterized=True,
            alpha=0.5)
            # now determine nice limits by hand:
            xmax, ymax = np.max(np.fabs(x)), np.max(np.fabs(y))
            binwidthx = 0.05
            binwidthy = binwidthx * (ymax/xmax)
            limx = (int(xmax/binwidthx) + 1) * binwidthx
            limy = (int(ymax/binwidthy) + 1) * binwidthy
            mx = np.mean(x)
            my = np.mean(y)
            axScatter.set_xlim((mx-limx, mx+limx))
            axScatter.set_ylim((my-limy, my+limy))
            binsx = np.arange(-limx, limx + binwidthx, binwidthx)
            binsy = np.arange(-limy, limy + binwidthy, binwidthy)
            axHistx.hist(x, bins=binsx, alpha=0.7)
            axHisty.hist(y, bins=binsy, orientation='horizontal', alpha=0.7)
            axHistx.set_xlim(axScatter.get_xlim())
            axHisty.set_ylim(axScatter.get_ylim())
            #axHistx.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
            #axHisty.ticklabel_format(axis='x',style='sci',scilimits=(1,4))
            #axHistx.yaxis.major.formatter._useMathText = True
            #axHisty.xaxis.major.formatter._useMathText = True
            axScatter.set_xlabel(xname + ' [cm]')
            axScatter.set_ylabel(yname + ' [cm]')
        axScatter.legend(
            bbox_to_anchor=[1.52,1.42], markerscale=10, labelspacing=0.1,
            handletextpad=0.0, framealpha=0.0)
        #axHistx.set_title("{} phase space slice".format(coord))
        #axHistx.text(-0.35, 0.5, letter,
        #    weight='bold', transform=axHistx.transAxes)
        fignum+=1
    return fignum



def plot_current(fignum, sim):
    for pulse in sim.pulses:
        ts = sim.ts
        Is = pulse['Is']
        plt.plot(ts, Is)

    temps = cloud.temps
    dens  = cloud.dens
    psd   = cloud.psd
    Is    = cloud.params['IAH'] * np.array([
                mag.curr_pulse(t, **cloud.params) for t in ts])
    temp_names = ('Tx', 'Ty', 'Tz', 'T ')
    dens_names = ('Dx', 'Dy', 'Dz', 'D ')
    fig = plt.figure(fignum, figsize=(5,5))
    axI = fig.add_subplot(2,2,1)
    axT = fig.add_subplot(2,2,2)
    axD = fig.add_subplot(2,2,3)
    axR = fig.add_subplot(2,2,4)
    axI.plot(ts, Is)
    axI.plot(ts,Is, label='current')
    axI.legend(loc='upper right',
            handlelength=1, labelspacing=0.2, handletextpad=0.2)
    for label, den in zip(dens_names, dens.T):
        axD.plot(ts, den, label=label)
    axD.legend(ncol=2, loc='upper right',
            handlelength=1, labelspacing=0.2,
            handletextpad=0.2, columnspacing=0.4)
    for label, temp in zip(temp_names, temps.T):
        axT.plot(ts, temp, label=label)
    axT.legend(ncol=2, loc='upper right',
            handlelength=1, labelspacing=0.2,
            handletextpad=0.2, columnspacing=0.4)
    axR.plot(ts, psd, label='PSD')
    axR.legend(loc='upper right',
         handlelength=1, labelspacing=0.2, handletextpad=0.2)
    fig.subplots_adjust(wspace=0.4, hspace=0.4, right=0.9, top=0.9, bottom=0.1)
    return fignum

def plot_traj(fignum, sim, cloud, field, seglen=2):
    traj             = sim.measures['traj'][::, cloud.keep_mask, ::]
    spins            = cloud.spins[cloud.keep_mask]
    xlim, ylim, zlim = field.xyzlim
    X, Y, Z          = field.XYZ
    print('Plotting 3D trajectory...')
    spin_color = ['red','black']
    fig = plt.figure(fignum, figsize=(8,8))
    ax = fig.gca(projection='3d')
    ax.set_ylabel('y [cm]')
    ax.set_xlabel('x [cm]')
    ax.set_zlabel('z [cm]')
    #ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
    ax.view_init(elev=10., azim=80)
    marker_dict = {1:r'$\uparrow$', -1:r'$\downarrow$'}
    Ntsteps = len(traj[::,0,0])
    alphas = np.linspace(1/Ntsteps, 1, int(Ntsteps))
    mag.plot_3d(ax, field, grad_norm=True)
    mag.geometry_viz(ax, field.geometry)
    if cloud.pin_hole:
        mag.loop_viz(ax,
            n=np.array([1,0,0]),
            r0=np.array(cloud.r0_ph),
            R=cloud.D_ph/2,
            d=0.05,
            color='r'
            )
    for j, spin in enumerate(spins):
        if j < 100:
            x, y, z = traj[::,j,0], traj[::,j,1], traj[::,j,2]
            ax.plot(x, y, z, color=spin_color[int((spin) + 1/2)],
                    lw=1.2)
            ax.plot([x[-1]],[y[-1]],[z[-1]], color='k',
                mew=0., alpha=0.9, marker=marker_dict[spin], ms=10)

            #for f in range(0, Ntsteps-seglen, seglen):
            #    x, y, z = traj[f:f+seglen+1,j,0], traj[f:f+seglen+1,j,1],\
            #            traj[f:f+seglen+1,j,2]
        elif j > 100:
            print('Only plotting trajectory for the first 100 atoms')
            break
    fignum+=1
    return fignum

# save many plots in single pdf
def multipage(fname, figs=None, clf=True, dpi=100, clip=True, extra_artist=False):
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

# THERMODYNAMICS AND MECHANICS
# ============================

# characteristic velocity of temperature T
def maxwell_velocity(T, m, nc=3):
    v_ = (kB * T / m) ** (1/2) # cm/us
    if nc == 1:
        return np.random.normal(0, v_)
    else:
        return np.random.normal(0, v_, nc)

# Thermal cloud or beam of atoms
class Cloud():
    def __init__(self, Natom=10000,
                       m=1.16e-26,
                       mu=9.27e-20,
                       vrecoil=0.0,
                       max_init=1,
                       Tt=100.0,
                       Tl=100.0,
                       width=[0.35, 0.25, 0.25],
                       r0_cloud=[0.0, 0.0, 0.0],
                       v0=[0.0, 0.0, 0.0],
                       tag=False,
                       r0_tag=[0.0, 0.0, 0.0],
                       t0_tag=0.0,
                       dt_tag=15.0,
                       pin_hole=False,
                       r0_ph=[0.0, 0.0, 0.0],
                       D_ph=0.5,
                       reinit=True,
                       save_cloud=True,
                       data_dir = 'data/clouds',
                       **kwargs):
        self.Natom      = Natom
        self.m          = m
        self.mu         = mu
        self.vrecoil    = vrecoil
        self.max_init   = max_init
        self.Tt         = Tt
        self.Tl         = Tl
        self.width      = width
        self.r0         = r0_cloud
        self.v0         = v0
        self.pin_hole   = pin_hole
        self.r0_ph      = r0_ph
        self.D_ph       = D_ph
        self.tag        = tag
        self.r0_tag     = r0_tag
        self.t0_tag     = t0_tag
        self.dt_tag     = dt_tag
        self.uid = hash_state(self.__dict__,
            include_keys = ['Natom', 'm', 'mu', 'vrecoil',
                            'max_init', 'Tt', 'Tl', 'width',
                            'r0', 'v0', 'pin_hole',
                            'r0_ph', 'D_ph', 'tag', 'r0_tag',
                            't0_tag', 'dt_tag'])
        self.fname = os.path.join(data_dir, self.uid)
        # convert to numpy arrays
        self.r0         = np.array(self.r0)
        self.r0_ph      = np.array(self.r0_ph)
        if not reinit:
            try:
                print('Attempting to load cloud...')
                self.load()
            except(FileNotFoundError):
                print('Cloud data not found. Generating now...')
                reinit = True

        if reinit:
            self.initialize_state()
            self.Ninit = self.init_profile[0] / self.Natom
            if save_cloud:
                self.save()

    def save(self):
        print('\nSaving cloud data to {}'.format(self.fname))
        file = open(self.fname,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        file = open(self.fname,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        self.r0         = np.array(self.r0)
        self.r0_ph      = np.array(self.r0_ph)
        print('Cloud data loaded from {}'.format(self.fname))

    def set_state(self, mask=None, xs=None, vs=None):
        if mask is None:
            mask = np.array([True]*self.Natom)
        N = np.sum(mask)
        if xs is None:
            for i in range(3):
                self.xs[mask, i] = np.random.normal(
                    self.r0[i], self.width[i], N)
        else:
            self.xs[mask] = xs
        if vs is None:
            for i in range(3):
                if i in (1, 2):
                    self.vs[mask, i] = maxwell_velocity(
                        self.Tt, self.m, nc=N) + self.v0[i]
                elif i == 0:
                    self.vs[mask, i] = maxwell_velocity(
                        self.Tl, self.m, nc=N) + self.v0[i]
        else:
            self.vs[mask] = vs

    def initialize_state(self):
        print('Initializing cloud...')
        self.xs           = np.zeros((self.Natom, 3))
        self.vs           = np.zeros((self.Natom, 3))
        self.spins        = np.random.choice([-1,1], size=self.Natom)
        self.drop_mask    = np.array([False]*self.Natom)
        self.keep_mask    = np.logical_not(self.drop_mask)
        self.set_state()
        self.init_profile = np.array([])
        for n in range(self.max_init):
            mask = np.array([False]*self.Natom)
            if self.tag:
                mask = np.logical_or(mask, self.tag_check())
            if self.pin_hole:
                mask = np.logical_or(mask, self.pin_hole_check())
            self.set_state(mask=mask)
            self.drop_mask = mask
            self.keep_mask = np.logical_not(self.drop_mask)
            N = self.get_number()
            self.init_profile = np.append(self.init_profile, [N])
            sys.stdout.write('{} atoms initialized (target of {}) \r'.format(
                    N, self.Natom))
            sys.stdout.flush()
            if self.get_number() == self.Natom:
                self.drop_mask = mask
                self.keep_mask = np.logical_not(self.drop_mask)
                return
        print('\nMaximum initialization iterations reached ({})'.format(
                self.max_init))

    def pin_hole_check(self):
        ts = ((self.r0_ph - self.xs)/self.vs)[:,0]
        ts = np.vstack([ts]*3).T
        xs_ph = self.xs + self.vs * ts
        r2 = xs_ph[::, 1]**2 + xs_ph[::, 2]**2
        drop_mask = r2 > self.D_ph**2 / 4.0
        return drop_mask

    # untested
    def slit_check(self):
        ts = ((self.r0_slit - self.xs)/self.vs)[:,0]
        ts = np.vstack([ts]*3).T
        xs_slit = self.xs + self.vs * ts
        y_slit, z_slit = np.abs(xs_slit[::, 1]) + np.abs(xs_slit[::, 2])
        drop_mask = np.logical_or(y_slit > self.W_slit, z_slit > self.L_slit)
        return drop_mask

    def tag_check(self):
        ts = ((self.r0_tag - self.xs)/self.vs)[:,0]
        t0 = np.mean(ts) + self.t0_tag
        drop_mask = np.logical_or(t0 - self.dt_tag/2.0 > ts,
                                  t0 + self.dt_tag/2.0 < ts)
        return drop_mask

    # time evolution
    def rk4(self, a, t, dt):
        mz = np.vstack([list(self.spins)]*3).T
        k1 = dt * mz *  a(self.xs, t)
        l1 = dt * self.vs
        k2 = dt * mz * a(self.xs + l1/2, t + dt/2)
        l2 = dt * (self.vs + k1/2)
        k3 = dt * mz * a(self.xs + l2/2, t + dt/2)
        l3 = dt * (self.vs + k2/2)
        k4 = dt * mz * a(self.xs + l3, t + dt)
        l4 = dt * (self.vs + k3)
        self.xs[::] = self.xs + 1/6 * (l1 + 2*l2 + 2*l3 + l4)
        self.vs[::] = self.vs + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        r2 = self.xs[::, 0]**2 + self.xs[::, 1]**2 + self.xs[::, 2]**2
        #self.drop_mask = r2 > 2.0
        #self.keep_mask = np.logical_not(self.drop_mask)

    # time evolution (no forces)
    def free_expand(self, expansion_time):
            self.xs = self.xs + expansion_time * self.vs

    # internal state preparation
    def optical_pump(self, mode):
        if mode in ('vs', 'vel', 'v', 'mop'):
            self.spins[::] = np.sign(self.vs[::, 2])
        elif mode in ('xs', 'pos', 'x'):
            self.spins[::] = np.sign(self.xs[::, 2])
        elif mode in (1, '1', 'up', 'HFS', 'hfs', 'minus'):
            self.spins = -np.ones_like(self.xs[::,0])
        elif mode in (0, '0', 'down', 'LFS', 'lfs', 'plus'):
            self.spins = np.random.choice([-1, 1], self.Natom, p=[0.0, 1.0])
        elif mode in ('none', 'None', None, 'thermal', 'therm'):
            self.spins = np.random.choice([-1, 1], self.Natom, p=[0.5, 0.5])
        else:
            raise ValueError('optical pumping mode {} not understood'.format(mode))
        self.recoil()

    # TODO enable and check recoils
    def recoil(self):
        recoils = maxwell_velocity(370.47e-6, self.m,
                nc=3*self.Natom)
        recoils = recoils.reshape(self.Natom, 3)
        #self.vs = self.vs + recoils

    def get_number(self):
        n = np.sum(self.keep_mask)
        return n

    def get_xs(self):
        return self.xs[self.keep_mask, ::]

    def get_vs(self):
        return self.vs[self.keep_mask, ::]

    def get_centers(self):
        Cx = np.mean(self.xs[self.keep_mask, 0])
        Cy = np.mean(self.xs[self.keep_mask, 1])
        Cz = np.mean(self.xs[self.keep_mask, 2])
        return Cx, Cy, Cz

    def get_sigmas(self):
        sigmax = np.std(self.xs[self.keep_mask, 0])
        sigmay = np.std(self.xs[self.keep_mask, 1])
        sigmaz = np.std(self.xs[self.keep_mask, 2])
        return sigmax, sigmay, sigmaz

    def get_skews(self):
        skewx = skew(self.xs[self.keep_mask, 0])
        skewy = skew(self.xs[self.keep_mask, 1])
        skewz = skew(self.xs[self.keep_mask, 2])
        return skewx, skewy, skewz

    def get_vcenters(self):
        Cx = np.mean(self.vs[self.keep_mask, 0])**2 
        Cy = np.mean(self.vs[self.keep_mask, 1])**2 
        Cz = np.mean(self.vs[self.keep_mask, 2])**2 
        return Cx, Cy, Cz

    def get_vvars(self):
        Vx = np.var(self.vs[self.keep_mask, 0])
        Vy = np.var(self.vs[self.keep_mask, 1])
        Vz = np.var(self.vs[self.keep_mask, 2])
        V  = (Vx  + Vy + Vz)/3
        return Vx, Vy, Vz, V

    def get_mean_kinetic(self):
        c = self.m * self.get_number() / 2
        vCs = np.array(self.get_vcenters())
        return c * vCs

    def get_temp(self):
        c = self.m/kB
        vvars = np.array(self.get_vvars())
        return  c * vvars 

    def get_thermal_kinetic(self):
        c = self.m * self.get_number() / 2
        vvars = np.array(self.get_vvars())
        return c * vvars

    def get_kinetic(self):
        mean2vx, mean2vy, mean2vz = self.get_mean_kinetic()
        varvx, varvy, varvz, var = self.get_thermal_kinetic()
        Kx, Ky, Kz = mean2vx + varvx, mean2vy + varvy, mean2vz + varvz
        K = Kx + Ky + Kz
        return Kx, Ky, Kz, K

    # TODO: review
    def get_density(self, correction=1):
        n = self.get_number()
        Dx = n/(2*pi*np.var(self.xs[self.keep_mask, 0]))**(3/2)
        Dy = n/(2*pi*np.var(self.xs[self.keep_mask, 1]))**(3/2)
        Dz = n/(2*pi*np.var(self.xs[self.keep_mask, 2]))**(3/2)
        D = n / ( 2/3 * pi *
            np.sum(np.var(self.xs[self.keep_mask,::], axis=0), axis=0))**(3/2)
        return Dx*correction, Dy*correction, Dz*correction, D*correction

    def get_psd(self, correction=1):
        Tx, Ty, Tz, T = self.get_temp()
        Dx, Dy, Dz, D = self.get_density(correction=correction)
        rho = D * (h**2/(2 * pi * self.m * kB * T))**(3/2)
        return rho


def hash_state(self_dict, include_keys):
    name_dict = copy.deepcopy(self_dict)
    for k, v in self_dict.items():
        if k not in include_keys:
            del name_dict[k]
        if type(v) == np.ndarray:
            name_dict[k] = list(v)
    uid = hashlib.sha1(
             json.dumps(
             name_dict,
             sort_keys=True).encode('utf-8')).hexdigest()
    return uid

# simulation class evolves a cloud instance
class Simulation():
    def __init__(self,
                 cloud_params,
                 sim_params,
                 reinit=True,
                 resimulate=True,
                 verbose=True,
                 load_fname=None,
                 save_simulation=True,
                 observation_idx=None,
                 data_dir='data'):

        self.cloud_params = cloud_params
        self.sim_params = sim_params
        self.uid = hash_state(self.__dict__,
            include_keys = ['sim_params', 'cloud_params'])
        self.fname = os.path.join(data_dir, self.uid)
        if not resimulate:
            try:
                print('Attempting to load simulation...')
                self.load(fname=load_fname)
            except(FileNotFoundError):
                print('Simulation data not found. Generating now...')
                resimulate = True

        if resimulate:
            self.process_sim_params(**self.sim_params)
            self.process_timing()
            self.cloud = Cloud(**cloud_params, reinit=reinit)
            self.init_cloud = copy.deepcopy(self.cloud)
            Ntsteps = len(self.ts)
            self.Ntsteps = Ntsteps
            self.Natom = cloud_params['Natom']
            if observation_idx == 'all':
                self.observation_idx = np.arange(0, Ntsteps, 1)
            elif observation_idx in (None, 'None', 'none'):
                self.observation_idx = [0 , 1, Ntsteps - 2, Ntsteps -1]
            else:
                self.observation_idx = observation_idx 
            self.measures_map = {
                                 'traj'     : self.cloud.get_xs,
                                 'vels'     : self.cloud.get_vs,
                                 'temps'    : self.cloud.get_temp,
                                 'dens'     : self.cloud.get_density,
                                 'psd'      : self.cloud.get_psd,
                                 'centers'  : self.cloud.get_centers,
                                 'skews'    : self.cloud.get_skews,
                                 'sigmas'   : self.cloud.get_sigmas,
                                 'meanKs'   : self.cloud.get_mean_kinetic,
                                 'thermKs'  : self.cloud.get_thermal_kinetic,
                                 'kinetics' : self.cloud.get_kinetic}
            self.init_sim()
            self.run_sim(verbose=verbose)
            if save_simulation:
                self.save()

    def save(self):
        print('\nSaving simulation data to {}'.format(self.fname))
        # don't save field data here
        fields = []
        for pulse in  self.pulses:
            field = pulse['field']
            fields += [field]
            del pulse['field']
        file = open(self.fname,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        # keep field as attribute for plotting later
        for pulse, field in  zip(self.pulses, fields):
            pulse['field'] = field

    def load(self, fname=None):
        from_file=True
        if fname == None:
            from_file = False
            fname = self.fname
        file = open(fname,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        print('Simulation data loaded from {}'.format(fname))
        if not from_file:
            self.process_sim_params(self.dt, self.delay, self.r0_detect, self.pulses)

    def process_sim_params(self, dt, delay, r0_detect, pulses):
        self.dt        = dt
        self.delay     = delay
        self.r0_detect = r0_detect
        self.pulses    = pulses
        fields     = {}
        field_num  = 1
        geometries = []
        for pulse in self.pulses:
            geometry = pulse['geometry']
            if geometry in geometries:
                pulse['field'] = fields[geometry['field_name']]
            else:
                recalc_B = pulse['recalc_B']
                field = mag.Field(geometry, recalc_B=recalc_B)
                pulse['field'] = field
                field_name = 'field' + str(field_num)
                fields[field_name] = field
                geometry['field_name'] = field_name
                geometries += [geometry]
                field_num += 1

    def process_timing(self):
        dt = self.dt
        ta = 0.0
        tb = self.delay
        pulse_ts_list = [[ta]]
        for pulse in self.pulses:
            ta = tb
            tb = ta + pulse['tau'] + pulse['tof'] + dt
            pulse['t0'] = ta
            pulse_ts = np.arange(ta, tb, dt)
            pulse['pulse_ts'] = pulse_ts
            pulse_ts_list += [pulse_ts]
        ts = np.concatenate(pulse_ts_list)
        ts = np.insert(ts, len(ts), 0.0)
        for pulse in self.pulses:
            pulse_Is = [mag.curr_pulse(t, **pulse) for t in ts ]
            pulse['Is'] = pulse_Is
        # create a slot for the final free exapansion, calculated from r0_detect
        self.ts = ts

    def init_sim(self):
        Ntsteps = self.Ntsteps
        Natom = self.Natom
        self.init_measures()
        # jump past any requested delay
        # if delay = 0, the initial data is repeated at time step 2
        self.cloud.free_expand(self.delay)
        self.update_measures(1)

    def init_measures(self):
        Ndetections = len(self.observation_idx)
        N = self.cloud.get_number()
        measures = dict(
             traj     = np.zeros((Ndetections, N, 3)),
             vels     = np.zeros((Ndetections, N, 3)),
             temps    = np.zeros((Ndetections, 4)),
             dens     = np.zeros((Ndetections, 4)),
             psd      = np.zeros( Ndetections),
             centers  = np.zeros((Ndetections, 3)),
             skews    = np.zeros((Ndetections, 3)),
             sigmas   = np.zeros((Ndetections, 3)),
             meanKs   = np.zeros((Ndetections, 3)),
             thermKs  = np.zeros((Ndetections, 4)),
             kinetics = np.zeros((Ndetections, 4)),
             )
        self.measures = measures
        self.update_measures(0)

    def update_measures(self, detection_idx):
        for measure_name in self.measures_map.keys():
            self.measures[measure_name][detection_idx] = self.measures_map[measure_name]()

    # experimental
    def tof_detection(self, rbeam=0.0635, dt=0.1):
        rbeam2 = rbeam**2
        vs = self.cloud.vs
        dzs = np.arange(-2.5, 2.5, rbeam)
        Ntsteps = 3000
        signals = np.zeros((len(dzs), Ntsteps))
        for zi, dz in enumerate(dzs):
            xs = self.params['r0_detect'] - self.cloud.xs
            for ti in range(Ntsteps):
                r2 = xs[::, 0]**2 + (dz - xs[::, 2])**2
                detectable = r2 <= rbeam2
                val = float(np.sum(detectable))
                signals[zi, ti] = val
                xs = xs + vs * dt
        return signals

    def make_acceleration(self, pulse):
        field = pulse['field']
        m  = self.cloud.m
        mu = self.cloud.mu
        xinterp, yinterp, zinterp = field.grad_norm_BXYZ_interp
        def a(xs, t):
            dBdx_interp = xinterp(xs)
            dBdy_interp = yinterp(xs)
            dBdz_interp = zinterp(xs)
            a_xyz =  -mu / m * mag.curr_pulse(t, **pulse) *\
                np.c_[dBdx_interp, dBdy_interp, dBdz_interp]
            return a_xyz
        return a

    def run_sim(self, verbose=True, fly=True):
            # step forward through pulse sequence
            ti = 2  # time step index
            Ndetections = len(self.observation_idx)
            for pulse_num, pulse in enumerate(self.pulses):
                a = self.make_acceleration(pulse)
                pulse_ts = pulse['pulse_ts']
                t0 = pulse_ts[0]
                pump = pulse['optical_pumping']
                self.cloud.optical_pump(pump)
                print('Pulse {} begins t = {:.2f}'.format(
                        pulse_num + 1, t0))
                print('optically pumped to {} at t = {}'.format(
                        pump, t0))
                for t in pulse_ts:
                    ts_remain = ((self.r0_detect - self.cloud.xs) / self.cloud.vs)[:,0]
                    t_remain = np.mean(ts_remain)
                    if t_remain >= 0.0 or not fly:
                        self.cloud.rk4(a, t, self.dt)
                        if ti in self.observation_idx:
                            self.update_measures(
                                list(self.observation_idx).index(ti))
                        if verbose:
                            sys.stdout.write(
                                ' '*43 + 't = {:.2f} of {:.2f}\r'.format(
                                    t, self.ts[-2]))
                            sys.stdout.flush()
                    elif t_remain < 0.0 and fly:
                        print('Detection truncates pulse')
                        print('Back propagating {} us'.format(t_remain))
                        self.cloud.free_expand(t_remain)
                        self.ts[-1] = self.ts[-2]
                        if Ndetections > 4:
                            for i in range(ti, self.Ntsteps - 1):
                                self.update_measures(i)
                        else:
                            self.update_measures(-2)
                        break
                    ti += 1

            # final free expansion expansion
            if not fly:
                tof = 0.0
                t_remain = tof

            if t_remain > 0.0:
                self.cloud.free_expand(t_remain)
                self.ts[-1] = self.ts[-2] + t_remain
                if verbose:
                    print('Cloud expanding for {:.2f} us'.format(t_remain))
            #signals = self.tof_detection()
            #plt.imshow(signals, aspect='auto', extent=(t_remain, t_remain+3000*0.1, -2.5, 2.5))
            #plt.show()
            self.update_measures(-1)

            if verbose:
                Nremain = np.sum(self.cloud.keep_mask)
                print('{}/{} atoms remain'.format(
                        Nremain, self.Natom))

    # plotting
    def plot_current(self, fig=None, ax=None, fignum=1, show=False):
        if fig is None:
            fig = plt.figure(fignum, figsize=(6, 2))
        if ax is None:
            ax = fig.add_subplot(1,1,1)
        ts = self.ts
        for n, pulse in enumerate(self.pulses):
            geometry = pulse['geometry']
            Idic = {}
            for k, v in geometry.items():
                if k[0] == 'I':
                    Idic[k] = v
            for Iname, I0 in Idic.items():
                label = 'pulse {} {}'.format(n, Iname)
                Is = I0 * np.array(pulse['Is'])
                ax.plot(ts, Is, label=label)
        plt.legend()
        if show:
            plt.show()

    def plot_temps(self, fig=None, ax=None, fignum=1, show=False,
            logy=False, include_names=['Tx','Ty','Tz','T']):
        if fig is None:
            fig = plt.figure(fignum, figsize=(3, 3))
        if ax is None:
            ax = fig.add_subplot(1,1,1)
        ts = np.take(self.ts, self.observation_idx)
        temps = self.measures['temps']
        temp_names = ['Tx', 'Ty', 'Tz', 'T ']
        colors     = ['C3', 'C7', 'k' , 'C2']
        lines      = ['--', '-.', '-', ':'  ]
        for label, temp, color, line in zip(temp_names, temps.T, colors, lines):
            if label in include_names:
                use_label = '$T_{}$'.format(label[1])
                if label == 'T':
                    use_label = '$T$'
                if logy:
                    ax.semilogy(ts, temp, label=use_label, c=color, ls=line)
                else:
                    ax.plot(ts, temp, label=use_label, c=color, ls=line)
        ax.set_xlabel('$t$' + units_map('t'))
        ax.set_ylabel('$T$' + units_map('T'))
        ax.legend(loc='upper right')
        if show:
            plt.show()

    def plot_psd(self, fig=None, ax=None, fignum=1, show=False):
        if fig is None:
            fig = plt.figure(fignum, figsize=(3, 3))
        if ax is None:
            ax = fig.add_subplot(1,1,1)
        ts = np.take(self.ts, self.observation_idx)
        psd = self.measures['psd']
        ax.plot(ts, psd, c='k')
        ax.set_xlabel('$t$' + units_map('t'))
        ax.set_ylabel('phase space density')
        if show:
            plt.show()

    def plot_kinetic_dist(self, fig=None, fignum=1, show=False):
        if fig is None:
            fig = plt.figure(fignum, figsize=(3, 3))
        ts = np.take(self.ts, self.observation_idx)
        mean2vx, mean2vy, mean2vz = self.measures['meanKs'].T
        varvx, varvy, varvz       = self.measures['thermKs'].T
        Kx, Ky, Kz, K             = self.measures['kinetics'].T
        vels                      = self.measures['vels']
        ax_mean2trans = fig.add_subplot(2,2,1)
        ax_mean2long  = fig.add_subplot(2,2,2)
        ax_varvxyz    = fig.add_subplot(2,2,3)
        ax_kin        = fig.add_subplot(2,2,4)
        ax_mean2trans.plot(ts, mean2vy, label='K ycom')
        ax_mean2trans.plot(ts, mean2vz, label='K zcom')
        ax_mean2trans.legend()
        ax_mean2long.plot(ts, mean2vx,  label='K xcom')
        ax_mean2long.legend()
        ax_varvxyz.plot(ts, varvx, label='K xvar')
        ax_varvxyz.plot(ts, varvy, label='K yvar')
        ax_varvxyz.plot(ts, varvz, label='K zvar')
        ax_varvxyz.legend()
        ax_kin.plot(ts, K, label='K')
        ax_kin.legend()
        if show:
            plt.show()


    def plot_measures(self, save_loc, fignum=1, save=True, show=False):
        if save or show:
            cloud = self.cloud
            for pulse in self.pulses:
                field = pulse['field']
                fignum = mag.plot_slices(fignum, field)
                fignum = mag.plot_slices(fignum, field, grad_norm=True)
                fignum = mag.plot_contour(fignum, field)
                fignum = mag.plot_contour(fignum, field, grad_norm=True)
            #fignum = plot_traj(fignum, self, cloud, field)
            #fignum = plot_integrated_density(fignum, self, cloud)
            #fignum = plot_integrated_density2(fignum, self, cloud)
            #fignum = plot_phase_space(fignum, self, cloud)
            #fignum = plot_phase_space2(fignum, self, cloud)
            #fignum = plot_temps(fignum, self, cloud,
            #        logy=True, include_names=['Tx','Ty','Tz'])
            #fignum = plot_psd(fignum, self, cloud)
            #plot_kinetic_dist(fignum, self, cloud)
            #fignum = plot_scalar_summary(fignum, self)
            # show or save plots
        if show:
            plt.show()
        if save:
            if show:
                print('Cannot show and save plots.')
            else:
                multipage(save_loc)
                print('\n Plots saved to {}'.format(save_loc))

# ANALYSIS
# ========
def units_map(param, mm=False):
    L = 'cm'
    if mm:
        L = 'mm'
    if param in ('temps', 'Tl', 'Tt'):
        unit = ' [mK]'
    elif param[0] =='I':
        unit = ' [A]'
    elif param[:2] == 'r0'\
         or param[0] in ('L', 'W', 'R', 'A')\
         or param in ('centers', 'sigmas', 'D_ph', 'width', 'd'):
        unit = ' ['+L+']'
    elif param[:2] in ('dt', 't0')\
         or param in ('tcharge', 'delay', 'tmax', 'tau'):
        unit = r' [$\mathrm{\mu s}$]'
    elif param in ('v0', 'vrecoil'):
        unit = r' [$\mathrm{'+L+'~\mu s^{-1}}$]'
    elif param in ('meanKs', 'thermKs', 'kinetics'):
        unit = r' [$\mathrm{kg~'+L+'^2~\mu s^{-2}}$]'
    elif param in ('ts', 't', 'times','time'):
        unit = r' [$\mathrm{\mu s}$]'
    else:
        unit = ''
    return unit

# process supplied params dictonary which may contain lists of parameters,
# signifying a parameter sweep.
def update_sweep(sweep, sub_sweep):
    for sk in sub_sweep:
        if sk in sweep:
            raise ValueError('parameter {} repeated'.format(sk))
        else:
            sweep.update(sub_sweep)

def get_sweep(dic, keychain=[], sweep = {},
        vectors=['r0_detect', 'width',
                 'r0_cloud', 'v0', 'r0_ph',
                 'r0_tag', 'n', 'r0', 'pulses']):
    if type(vectors) == str:
        vectors = [vectors]
    for k, v in dic.items():
        if k in vectors:
            if type(v[0]) == list:
                keychain_copy = keychain.copy()
                keychain += [k]
                kc = '/'.join(keychain)
                update_sweep(sweep, {kc:v})
                keychain = keychain_copy

            elif type(v[0]) == dict:
                keychain_copy = keychain.copy()
                keychain += [k]
                keychain_copy2 = keychain.copy()
                for pdici, pdic in enumerate(v):
                    keychain += str(pdici)
                    kc = '/'.join(keychain)
                    sub_sweep = get_sweep(pdic, sweep=sweep, keychain=keychain)
                    keychain = keychain_copy2
                keychain = keychain_copy

        elif type(v) == dict:
            keychain_copy = keychain.copy()
            keychain += [k]
            kc = '/'.join(keychain)
            sub_sweep = get_sweep(v, sweep=sweep, keychain=keychain)
            keychain = keychain_copy

        else:
            if type(v) == list:
                keychain_copy = keychain.copy()
                keychain += [k]
                kc = '/'.join(keychain)
                update_sweep(sweep, {kc:v})
                keychain = keychain_copy
    return sweep

def str2int(string):
    try:
        ret = int(string)
    except:
        ret = string
    return ret

#https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
def get_dict(dic, map_list):
    return reduce(operator.getitem, map_list, dic)

def set_dict(dic, map_list, value):
    get_dict(dic, map_list[:-1])[map_list[-1]] = value

def process_sweep(sweep, args_tmp):
    sweep_keys = []
    sweep_vals   = []
    sweep_shape  = []
    args_list    = []
    for k, vs in sweep.items():
        sweep_keys += [k]
        sweep_vals += [vs]
        sweep_shape += [(os.path.basename(k), len(vs))]
    nsims = np.product([l[1] for l in sweep_shape])
    for vs in product(*sweep_vals):
        new_args = copy.deepcopy(args_tmp)
        for v, k in zip(vs, sweep_keys):
            ks = [str2int(k) for k in k.split('/')]
            set_dict(new_args, ks, v)
        args_list.append(new_args)
    return args_list, sweep_vals, sweep_keys, sweep_shape

# partition list into n parts
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def experiment(cloud_params_tmp,
               sim_params_tmp,
               reinit=True,
               resimulate=True,
               save_simulations=True,
               verbose=True,
               to_record=-1,
               observation_idx=None):
    if to_record == 'all':
        to_record = slice(None)
        observation_idx='all'
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    if rank == 0:
        args_tmp = {'cloud_params'  : cloud_params_tmp,
                    'sim_params'    : sim_params_tmp}
        sweep = get_sweep(args_tmp)
        print(sweep)
        args_list, sweep_vals, sweep_keys, sweep_shape =\
                process_sweep(sweep, args_tmp)
        nsims = np.product([l[1] for l in sweep_shape])
        # p for 'partitioned'. Partioned list scattered to each core.
        nchunks = nsims // nprocs + (nsims % nprocs > rank)
        print(rank, nchunks)
        pargs_list = list(chunks(args_list, nchunks))
        print('\n Hello from rank 0! You are running {} jobs on {} cores...'\
            .format(nsims, nprocs))
    else:
        pargs_list   = None
        sweep_vals   = None
        sweep_shape  = None
        flat_records = None
    pargs_list = comm.scatter(pargs_list, root=0)
    num_sims = len(pargs_list)
    records = {'temps'   : [],
               'centers' : [],
               'sigmas'  : [],
               'skews'   : [],
               'meanKs'  : [],
               'thermKs' : [],
               'kinetics': [],
               'Ninit'   : [],
               'ts'      : []}
    # loop through all requested simulations
    for sim_num, args in enumerate(pargs_list):
        if verbose:
            print('\n RANK {} SIMULATION {} OF {}:\r'.format(
                    rank, sim_num + 1, num_sims))
        sim = Simulation(**args,
                save_simulation=save_simulations,
                verbose=verbose, reinit=reinit, resimulate=resimulate,
                observation_idx=observation_idx)
        records['Ninit'] += [sim.cloud.Ninit]
        records['ts'] += [sim.ts]
        for measure_name in records.keys():
            if measure_name not in ('ts', 'Ninit'):
                records[measure_name] += [sim.measures[measure_name][to_record]]
    # list of list of records, from each core
    all_records = comm.gather(records, root=0)
    if rank == 0:
        flat_records = all_records[0]
        for k, v in flat_records.items():
            for record in all_records[1:]:
                v.extend(record[k])
    sweep_vals_list=sweep_vals
    return sweep_shape, sweep_vals_list, flat_records, rank

# TODO: update
def scan_3d(sweep_shape, sweep_vals_list, records, plot_fname, unit='cm'):
    # number of elements of each sweep parameter
    shaper = [s[1] for s in sweep_shape]
    # names of each sweep parameter
    sweep_params = [s[0] for s in sweep_shape]
    # grab the first component of vectors to use as an axis
    # TODO: generalize, use norm? use 3 separate components?
    sweep_vals_list_flat = []
    for sweep_vals in sweep_vals_list:
        sv = [v if type(v) != list else v[0] for v in sweep_vals]
        sweep_vals_list_flat.append(sv)
    X, Y, Z = np.meshgrid(*sweep_vals_list_flat)
    xs, ys, zs = sweep_vals_list_flat
    xparam, yparam, zparam = sweep_params
    figs=[]
    fignum=100
    for coordi, coord_label in enumerate(['x', 'y', 'z']):
        fig  = plt.figure(fignum, figsize=(6,4))
        ax   = fig.add_subplot(1,1,1)
        record = np.asarray(records['sigmas'])[:, coordi]
        record = record.reshape(shaper)
        for yi in range(shaper[1]):
            ms = []
            for xi in range(shaper[0]):
                s = record[xi, yi, ::]
                m, b = np.polyfit(zs, s, 1)
                ms += [m]
            ax.plot(xs, ms, label=ys[yi])
        plt.legend()
        fignum += 1
        figs.append(fig)
    print('\nSaving plots 2d scan analysis to', plot_fname)
    multipage(plot_fname, figs=figs)

def scan_2d(sweep_shape, sweep_vals_list, records, plot_fname,
            to_load = {}, 
            labels = ['centers', 'sigmas', 'skews', 'temps',
                       'meanKs', 'thermKs', 'kinetics'],
            subtract_nokick = False):
    # number of elements of each sweep parameter
    shaper = [s[1] for s in sweep_shape]
    # names of each sweep parameter
    sweep_params = [s[0] for s in sweep_shape]
    # grab the first component of vectors to use as an axis
    # TODO: generalize, use norm? use 3 separate components?
    sweep_vals_list_flat = []
    for sweep_vals in sweep_vals_list:
        sv = [v if type(v) != list else v[0] for v in sweep_vals]
        sweep_vals_list_flat.append(sv)
    #X, Y = np.meshgrid(*sweep_vals_list_flat)
    xs, ys = sweep_vals_list_flat
    xparam, yparam = sweep_params
    xparam = ' '.join(xparam.split('_'))
    yparam = ' '.join(yparam.split('_'))
    figs=[]
    fignum=100

    prop_cycle = plt.rcParams['axes.prop_cycle']

    loaded_sims = {}
    for sim_name, load_fname in to_load.items():
        loaded_sims[sim_name] = Simulation({},{},
                load_fname=load_fname,
                reinit=False, resimulate=False)

    for label in labels:
        colors = cycle(prop_cycle.by_key()['color'])
        if label == 'Ninit':
            record = np.asarray(records[label])
            record = record.reshape(shaper)
            fig  = plt.figure(fignum, figsize=(6,4))
            ax   = fig.add_subplot(1,1,1)
            for line_label, line_data in zip(ys, record.T):
                ax.plot(xs, line_data, color=next(colors), label=yparam +'  = ' + str(line_label))
            ax.set_ylabel(label)
            ax.set_xlabel(xparam)
            plt.legend()
            fignum += 1
            figs.append(fig)
        elif label in ('centers', 'sigmas', 'skews',
                'temps', 'meanKs', 'thermKs', 'kinetics'):
            coord_labels = ['x', 'y', 'z']
            if label in ('temps', 'kinetics'):
                coord_labels = ['x', 'y', 'z', '']
            for coordi, coord_label in enumerate(coord_labels):
                record = np.asarray(records[label])[:, coordi]
                record = record.reshape(shaper)
                ylabel = ' '.join([label[:-1], coord_label])
                ylabel = ylabel + units_map(label)
                xlabel = xparam + units_map(xparam)
                fig  = plt.figure(fignum, figsize=(6,4))
                ax   = fig.add_subplot(1,1,1)
                for sim_name, sim in loaded_sims.items():
                    if not subtract_nokick:
                        ax.axhline(sim.measures[label][-1][coordi],
                                label = sim_name,color=next(colors))
                for line_label, line_data in zip(ys, record.T):
                    vals = line_data
                    if subtract_nokick:
                        for sim_name, sim in loaded_sims.items():
                            if yparam == 'r0 detect':
                                try:
                                    split_name = sim_name.split('no kick ')[1].split(' [')[0]
                                except:
                                    split_name = None
                                if str(line_label) == split_name:
                                    vals = line_data - sim.measures[label][-1][coordi]
                            else:
                                if sim_name == 'no kick':
                                    vals = line_data - sim.measures[label][-1][coordi]

                    line_label = yparam + ' = ' + str(line_label) + units_map(yparam)
                    ax.plot(xs, vals, label=line_label, color=next(colors))
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)
                plt.legend()
                fignum += 1
                figs.append(fig)
    print('\nSaving plots 2d scan analysis to', plot_fname)
    multipage(plot_fname, figs=figs)

# xaxis can be `time` or `param`
def scan_1d(sweep_shape, sweep_vals_list, records, plot_fname,
            xaxis = 'time',
            labels = ['centers', 'sigmas', 'skews', 'temps',
                       'meanKs', 'thermKs', 'kinetics']
            ):
        # number of elements of each sweep parameter
    shaper = [s[1] for s in sweep_shape]
    # names of each sweep parameter
    sweep_params = [s[0] for s in sweep_shape]
    # grab the first component of vectors to use as an axis
    # TODO: generalize, use norm? use 3 separate components?
    sweep_vals_list_flat = []
    for sweep_vals in sweep_vals_list:
        sv = [v if type(v) != list else v[0] for v in sweep_vals]
        sweep_vals_list_flat.append(sv)
    fignum=100
    figs=[]
    labels = ['centers', 'sigmas', 'skews', 'temps',
               'meanKs', 'thermKs', 'kinetics']
    if xaxis == 'param':
        xs = sweep_vals_list_flat[0]
        xparam, = sweep_params
        for label in labels:
            coord_labels = ['x', 'y', 'z']
            if label in ('temps', 'kinetics'):
                coord_labels = ['x', 'y', 'z', '']
            for coordi, coord_label in enumerate(coord_labels):
                record = np.asarray(records[label])[:, coordi]
                full_label = ' '.join([label[:-1], coord_label])
                record = record.reshape(shaper)
                fig  = plt.figure(fignum, figsize=(6,4))
                ax   = fig.add_subplot(1,1,1)
                ax.plot(xs, record)
                ax.set_ylabel(full_label)
                ax.set_xlabel(xparam)
                plt.legend()
                fignum += 1
                figs.append(fig)

    if xaxis == 'time':
        xparam = 'time'
        for label in labels:
            coord_labels = ['x', 'y', 'z']
            if label in ('temps', 'kinetics'):
                coord_labels = ['x', 'y', 'z']
            for coordi, coord_label in enumerate(coord_labels):
                fig  = plt.figure(fignum, figsize=(6,4))
                ax   = fig.add_subplot(1,1,1)
                full_label = ' '.join([label[:-1], coord_label])
                for param_id, sparam_val in enumerate(sweep_vals_list_flat[0]):
                    print(records[label][0])
                    record = np.asarray(records[label])[param_id][:, coordi]
                    ts = np.asarray(records['ts'])[param_id]
                    ax.plot(ts[1:-1], record[1:-1],
                        label=sweep_params[0] + ' = ' + str(sparam_val))
                    ax.set_ylabel(full_label)
                    ax.set_xlabel(xparam)
                plt.legend()
                fignum += 1
                figs.append(fig)
    print('\nSaving plots 1d scan analysis to', plot_fname)
    multipage(plot_fname, figs=figs)


if __name__ == '__main__':
    pass
