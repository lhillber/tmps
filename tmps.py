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
from itertools import product
import hashlib
import json
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

# lfs, 100 mK 10000 atoms
#no_kick_fname = "data/e04b3b69cc9e856a77598a0cb27f805fcc4a5b77" #pre stirap mods
no_kick_fname = "data/93e7413756d794dd0b1827281c6399f81a3048dd"
#typical_kick_fname = "data/75f03734da886f719f1c02b081982c5cf2329532"   #pre stirap mods poly-d600, lfs
typical_kick_fname = "data/0c458a3480a4b1f13be44fe211388d47328a82fe"
#typical_kick_fname = "data/0286b40a89ebcb568d94af24a4731b26e93e5cc8" #sin-100-730
def extract_geometry(params):
    geometry = {}
    del_list = []
    for k, v in params.items():
        if k not in (
                 'tag', 'r0_tag', 't0_tag', 'dt_tag',
                 'pin_hole', 'r0_ph', 'D_ph', 
                 'max_init', 'Tl', 'Tt', 'Natom', 'width',
                 'r0_cloud', 'v0', 'Npulse', 'shape', 'tau',
                 'tcharge', 'r0_detect', 'decay', 'dt',
                 'delay', 'm' , 'mu', 'vrecoil',
                 'optical_pumping', 'parity'):
            geometry[k] = v
            del_list += [k]
    for k in del_list:
        del params[k]
    return params, geometry

def format_params(params):
    if params['delay'] == None:
        if norm(params['v0']) == 0.0:
            params['delay'] = 0.0
        else:
            params['delay'] =\
                (norm(params['r0_cloud']) - norm(params['width'])) /\
                 norm(params['v0'])  # us
            params['delay'] = 730 + 50 - params['tau']/2
    if params['dt'] == None:
        params['dt'] = params['tau']/100  # us
    params['tmax'] = (params['Npulse']) * (
        params['tau'] + params['tcharge']) + params['delay'] # us
    return params


# PLOTTING
# ========
def plot_phase_space(fignum, sim, cloud, time_indicies=[0, -1]):
    print('plotting phase space slices...')
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
        time_indicies=[0, -1], remove_mean=True):
    print('plotting phase space slices...')
    for ti in time_indicies:
        xs = sim.measures['traj'][ti, cloud.keep_mask, ::]
        vs = sim.measures['vels'][ti, cloud.keep_mask, ::] * 1e6 * 1e-2 
        fig, axs = plt.subplots(4, 4, num=fignum, figsize=(7,7), sharex='col', sharey='row')
        coordi = [[(None, None), (1, 0), (2, 1), (0, 2)],
                  [   (3, 4),    (3, 0), (3, 1), (3, 2)],
                  [   (4, 5),    (4, 0), (4, 1), (4, 2)],
                  [   (5, 3),    (5, 0), (5, 1), (5, 2)]]
        coords = ['$x$', '$y$', '$z$', '$v_x$', '$v_y$', '$v_z$']
        ps = np.hstack([xs, vs])
        zmax = 0
        for i in range(4):
            for j in range(4):
                ax = axs[i, j]
                n, m = coordi[i][j]
                if (m, n) == (None, None):
                    ax.axis('off')
                    continue
                x = ps[:5000, m]
                y = ps[:5000, n]
                xname = coords[m]
                yname = coords[n]
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)
                #Sort the points by density
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]
                zmax_tmp = np.max(z)
                if zmax_tmp > zmax:
                    zmax = zmax_tmp
                if xname[1] == 'v':
                    ax.set_xlim(-20,20)
                else:
                    ax.set_xlim(-2,2)
                if yname[1] == 'v':
                    ax.set_ylim(-20,20)
                else:
                    ax.set_ylim(-2,2)
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
                if remove_mean:
                    x = x - np.mean(x)
                    y = y - np.mean(y)
                ax.scatter(x, y, c=z, vmax=zmax, s=0.25, rasterized=True,
                #ax.scatter(x, y, s=0.25, rasterized=True,
                        alpha=0.5)
        fignum += 1
    return fignum

def plot_integrated_density(fignum, sim, cloud):
    print('plotting real space slices...')
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

def plot_temps(fignum, sim, cloud, include_names=['Tx','Ty','Tz','T'], logy=False):
    ts = np.take(sim.ts, sim.detection_time_steps)
    temps = sim.measures['temps']
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
            if logy:
                axT.semilogy(ts, temp, label=use_label, c=color, ls=line)
            else:
                axT.plot(ts, temp, label=use_label, c=color, ls=line)

    #if logy:
    #    axT.semilogy(ts, 370.47e-6 * np.ones_like(ts),
    #            label='recoil', c=color, ls=line)
    #else:
    #    axT.plot(ts, temp, 370.47e-6 * np.ones_like(ts),
    #            label='recoil', c=color, ls=line)
    axT.set_xlabel('$t$ [$\mu$s]')
    axT.set_ylabel('$T$ [mK]')
    axT.legend(loc='upper right')
    fignum += 1
    #axT.text(-0.3, 1.01, '(d)', weight='bold', transform=axT.transAxes)
    return fignum

def plot_psd(fignum, sim, cloud):
    ts = np.take(sim.ts, sim.detection_time_steps)
    psd = sim.measures['psd']
    figPSD = plt.figure(fignum, figsize=(3,3))
    axPSD = figPSD.add_subplot(1,1,1)
    axPSD.plot(ts, psd, c='k')
    axPSD.set_xlabel('$t$ [$\mu$s]')
    axPSD.set_ylabel('phase space density')
    fignum += 1
    return fignum

def plot_kinetic_dist(fignum, sim, cloud):
    ts = np.take(sim.ts, sim.detection_time_steps)
    vels = sim.measures['vels']
    mean2vx, mean2vy, mean2vz = sim.measures['meanKs'].T
    varvx, varvy, varvz = sim.measures['thermKs'].T
    Kx, Ky, Kz, K = sim.measures['kinetics'].T
    figkin = plt.figure(fignum, figsize=(3,3))
    ax_mean2trans = figkin.add_subplot(2,2,1)
    ax_mean2long  = figkin.add_subplot(2,2,2)
    ax_varvxyz    = figkin.add_subplot(2,2,3)
    ax_kin        = figkin.add_subplot(2,2,4)

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

    #ax_kin.plot(ts, K, label='Kz')
    fignum += 1

def plot_scalar_summary(fignum, cloud, field):
    ts = cloud.ts
    temps = cloud.temps
    dens  = cloud.dens
    psd   = cloud.psd
    Is    = cloud.params['IAH'] * np.array([
                mag.curr_pulse(t, **cloud.params) for t in ts])
    temp_names = cloud.temp_names
    dens_names = cloud.dens_names
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
    print('plotting 3D trajectory...')
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
            print('only plotting trajectory for the first 100 atoms')
            break
    fignum+=1
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

# THERMODYNAMICS AND MECHANICS
# ============================
class Cloud():
    def __init__(self, Natom, max_init, Tt, Tl, width, r0_cloud, m, mu, vrecoil,
            v0, tag, r0_tag, t0_tag, dt_tag, pin_hole, r0_ph, D_ph,
            reinitialize=True, **kwargs):
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
        # names of variables returned by get_temp and get_density.
        self.temp_names = ('Tx', 'Ty', 'Tz', 'T ')
        self.dens_names = ('Dx', 'Dy', 'Dz', 'D ')
        data_dir = 'data/clouds'
        self.uid = hashlib.sha1(json.dumps(self.__dict__, sort_keys=True).encode(
                'utf-8')).hexdigest()
        self.fname = os.path.join(data_dir, self.uid)
        self.r0         = np.array(self.r0)
        self.r0_ph      = np.array(self.r0_ph)
        if not reinitialize:
            try:
                print('attempting to load cloud')
                self.load()
            except(FileNotFoundError):
                print('Cloud data not found')
                reinitialize = True

        if reinitialize:
            self.initialize_state() # creates init_profile
            self.Ninit = self.init_profile[0] / self.Natom
            self.save()

    def save(self):
        print('saving cloud data to {}'.format(self.fname))
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

    def initialize_state(self):
        print('initializing cloud...')
        self.xs        = np.zeros((self.Natom, 3))
        self.vs        = np.zeros((self.Natom, 3))
        self.spins     = np.zeros(self.Natom)
        self.drop_mask = np.array([False]*self.Natom)
        self.keep_mask = np.logical_not(self.drop_mask)
        self.init_profile = np.array([])
        self.spins[::] = np.random.choice([-1,1], size=self.Natom)
        for n in range(self.max_init):
            sys.stdout.write('{} atoms initialized (target of {}) \r'.format(
                    self.get_number(), self.Natom))
            sys.stdout.flush()
            if n == self.max_init - 1:
                print('maximum initialization iterations reached ({})'.format(
                        self.max_init))

            first=True
            Nkeep = self.get_number()
            Ndrop = self.Natom - Nkeep
            if n==0:
                N = Nkeep
                mask = self.keep_mask
                self.init_profile = np.append(self.init_profile, [Nkeep])
            elif n>0:
                N = Ndrop
                mask = self.drop_mask
                self.init_profile = np.append(self.init_profile, [Nkeep])
            for i in range(3):
                if i in (1, 2):
                    self.vs[mask, i] = maxwell_velocity(
                        self.Tt, self.m, nc=N) + self.v0[i]
                elif i == 0:
                    self.vs[mask, i] = maxwell_velocity(
                        self.Tl, self.m, nc=N) + self.v0[i]
                self.xs[mask, i] = np.random.normal(
                    self.r0[i], self.width[i], N)
            if N == 0:
                break
            else:
                if self.tag:
                    if first:
                        self.drop_mask = self.tag_check()
                        first = False
                    else:
                        self.drop_mask = np.logical_or(
                            self.drop_mask,
                            self.tag_check())
                if self.pin_hole:
                    if first:
                        self.drop_mask = self.pin_hole_check(
                                self.r0_ph, self.D_ph)
                        first = False
                    else:
                        self.drop_mask = np.logical_or(
                            self.drop_mask,
                            self.pin_hole_check(self.r0_ph, self.D_ph))
                self.keep_mask = np.logical_not(self.drop_mask)
        print()

    def pin_hole_check(self, r0, D):
        ts = ((r0 - self.xs)/self.vs)[:,0]
        ts = np.vstack([ts]*3).T
        xs_ph = self.xs + self.vs * ts
        r2 = xs_ph[::, 1]**2 + xs_ph[::, 2]**2
        drop_mask = r2 > D**2 / 4.0
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

    def free_expand(self, expansion_time):
            x1 = self.xs + expansion_time * self.vs
            self.xs[::] = x1

    def optical_pump(self, mode):
        if mode in ('vs', 'vel', 'v', 'mop'):
            self.spins[::] = np.sign(self.vs[::, 2])
        if mode in ('xs', 'pos', 'x'):
            self.spins[::] = np.sign(self.xs[::, 2])
        if mode in (1, '1', 'up', 'HFS', 'hfs', 'minus'):
            self.spins = -np.ones_like(self.xs[::,0])
        if mode in (0, '0', 'down', 'LFS', 'lfs', 'plus'):
            self.spins = np.random.choice([-1, 1], self.Natom, p=[0.0, 1.0])
        #else:
        #    raise('optical pumping mode {} not understood'.format(mode))
        #self.recoil()

    def recoil(self):
        recoils = maxwell_velocity(370.47e-6, self.m,
                nc=3*self.Natom)
        recoils = recoils.reshape(self.Natom, 3)
        self.vs = self.vs + recoils

    def get_xs(self):
        return self.xs[self.keep_mask, ::]

    def get_vs(self):
        return self.vs[self.keep_mask, ::]

    def get_temp(self):
        c = self.m/kB
        Tx = c*np.var(self.vs[self.keep_mask, 0])
        Ty = c*np.var(self.vs[self.keep_mask, 1])
        Tz = c*np.var(self.vs[self.keep_mask, 2])
        T  = (Tx  + Ty + Tz)/3
        return Tx, Ty, Tz, T

    def get_number(self):
        n = np.sum(self.keep_mask)
        return n

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

    def get_mean_kinetic(self):
        c = self.m * self.get_number() / 2
        mean2vx = c*np.mean(self.vs[self.keep_mask, 0])**2 
        mean2vy = c*np.mean(self.vs[self.keep_mask, 1])**2 
        mean2vz = c*np.mean(self.vs[self.keep_mask, 2])**2 
        return mean2vx, mean2vy, mean2vz

    def get_thermal_kinetic(self):
        c = self.m * self.get_number() / 2
        varvx = c*np.var(self.vs[self.keep_mask, 0]) 
        varvy = c*np.var(self.vs[self.keep_mask, 1]) 
        varvz = c*np.var(self.vs[self.keep_mask, 2])
        return varvx, varvy, varvz

    def get_kinetic(self):
        mean2vx, mean2vy, mean2vz = self.get_mean_kinetic()
        varvx, varvy, varvz = self.get_thermal_kinetic()
        Kx, Ky, Kz = mean2vx + varvx, mean2vy + varvy, mean2vz + varvz
        K = Kx + Ky + Kz
        return Kx, Ky, Kz, K


def maxwell_velocity(T, m, nc=3):
    # characteristic velocity of temperature T
    v_ = (kB * T / m) ** (1/2) # cm/us
    if nc == 1:
        return np.random.normal(0, v_)
    else:
        return np.random.normal(0, v_, nc)

    # process supplied params dictonary which may contain lists of parameters,
    # signifying a parameter sweep.
def process_experiment(params_tmp):
    params_list = []
    sweep_vals_list = []
    sweep_params=[]
    for k, v in params_tmp.items():
        # these parameters are a vectors, check if a list of them is supplied
        if k[0] == 'n' or k[:2] in ('r0', 'v0') or k == 'width':
            if type(v[0]) == list:
                sweep_params.append(k)
                sweep_vals_list.append(v)
        # Most parameters are scalars, check if a list of them is supplied
        else:
            if type(v) == list:
                sweep_params.append(k)
                sweep_vals_list.append(v)
    # list of 2-tuples with sweeped parameters first, their length second
    sweep_shape = [(sweep_param, len(sweep_vals)) for
        sweep_param, sweep_vals in zip(sweep_params, sweep_vals_list)]
    # Total number of simulations requested
    num_sims = np.product([s[1] for s in sweep_shape], dtype=int)
    # initialize final/initial ratios of temperatures, x, y, z, and average
    if num_sims > 1:
        sim_num = 0
        # ensure a set of parameters is generated for every combination of
        # simulation parameters
        sweep_vals_list_product = product(*sweep_vals_list)
        sweep_params_product = [sweep_params] * num_sims
        for sweep_vals, sweep_params in zip(
                sweep_vals_list_product, sweep_params_product):
            new_params = params_tmp.copy()
            for sweep_val, sweep_param in zip(sweep_vals, sweep_params):
                new_params[sweep_param] = sweep_val
            params_list.append(new_params)
            sim_num += 1
    else:
       params_list = [params_tmp]
    return num_sims, params_list, sweep_vals_list, sweep_params, sweep_shape

# partition list into n parts
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def experiment(params_tmp, recalc_B=False, reinitialize=True, resimulate=True,
            save_simulations=True, verbose=True, to_record=-1,
            detection_time_steps=None):
    if to_record == 'all':
        to_record = slice(None)
        detection_time_steps='all'
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    if rank == 0:
        num_sims, params_list, sweep_vals_list, sweep_params, sweep_shape =\
            process_experiment(params_tmp)
        pparams_list = list(chunks(params_list, 1+int(num_sims/nprocs)))
        print('\n Hello from rank 0! You are running {} jobs on {} cores...'\
            .format(num_sims, nprocs))
    else:
        pparams_list    = None
        sweep_shape     = None
        sweep_vals_list = None
        flat_records    = None
    pparams_list = comm.scatter(pparams_list, root=0)
    num_sims = len(pparams_list)
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
    for sim_num, params in enumerate(pparams_list):
        if verbose:
            print('\n RANK {} SIMULATION {} OF {}:\r'.format(
                    rank, sim_num+1, num_sims))
        sim = Simulation(params,
                recalc_B=recalc_B, save_simulations=save_simulations,
                verbose=verbose, reinitialize=reinitialize, resimulate=resimulate,
                detection_time_steps=detection_time_steps)
        records['Ninit'] += [sim.cloud.Ninit]
        records['ts'] += [sim.ts]
        for measure_name in records.keys():
            if measure_name not in ('ts', 'Ninit'):
                records[measure_name] += [sim.measures[measure_name][to_record]]
    all_records = comm.gather(records, root=0)
    if rank == 0:
        flat_records = all_records[0]
        for k, v in flat_records.items():
            for record in all_records[1:]:
                v.extend(record[k])
    return sweep_shape, sweep_vals_list, flat_records, rank

# Default behavior
class Simulation():
    def __init__(self, params, recalc_B=False, reinitialize=True, resimulate=True,
                 verbose=True, load_fname=None, save_simulations=True,
                 detection_time_steps=None, data_dir='data',
                 pulse2_params=None):
        if recalc_B:
            resimulate = True
        self.params = params
        self.pulse2_params = pulse2_params
        self.uid = hashlib.sha1(json.dumps(params, sort_keys=True).encode(
                'utf-8')).hexdigest()
        self.fname = os.path.join(data_dir, self.uid)
        if not resimulate:
            try:
                print('attempting to load simulation')
                self.load(fname=load_fname, recalc_B=recalc_B)
            except(FileNotFoundError):
                print('Simulation data not found')
                resimulate = True

        if resimulate:
            params, geometry = extract_geometry(params)
            self.geometry = geometry
            params = format_params(params)
            self.params = params
            self.field = mag.Field(geometry, recalc_B=recalc_B)
            self.cloud = Cloud(**params, reinitialize=reinitialize)
            if not pulse2_params == None:
                params2, geometry2 = extract_geometry(pulse2_params)
                params2['delay'] = params['delay']
                params2['m']     = params['m']
                params2['mu']    = params['mu']
                params2['Npulse']    = params['Npulse']
                self.geometry2 = geometry2
                self.params2 = params2
                self.field2 = mag.Field(geometry2, recalc_B=recalc_B)

            ts = np.arange(params['delay'],
                    params['tmax'] + params['dt'], params['dt'])
            ts = np.insert(ts, 0, 0.0)
            Ntsteps = len(ts) + 0 # add one for final state after expansion
            Natom = params['Natom']
            self.Ntsteps = Ntsteps
            self.Natom   = Natom
            self.ts      = ts
            if detection_time_steps == 'all':
                self.detection_time_steps = np.arange(0, Ntsteps, 1)
            elif detection_time_steps in (None, 'None', 'none'):
                self.detection_time_steps = [0 , 1, Ntsteps - 2, Ntsteps -1]
            else:
                self.detection_time_steps = detection_time_steps
            self.measures_map = {
                                 'traj'   : self.cloud.get_xs,
                                 'vels'   : self.cloud.get_vs,
                                 'temps'  : self.cloud.get_temp,
                                 'dens'   : self.cloud.get_density,
                                 'psd'    : self.cloud.get_psd,
                                 'centers': self.cloud.get_centers,
                                 'skews'  : self.cloud.get_skews,
                                 'sigmas' : self.cloud.get_sigmas,
                                 'meanKs' : self.cloud.get_mean_kinetic,
                                 'thermKs': self.cloud.get_thermal_kinetic,
                                 'kinetics': self.cloud.get_kinetic}
            self.init_sim()
            self.run_sim(verbose=verbose)
            if save_simulations:
                self.save()

    def save(self):
        print('saving simulation data to {}'.format(self.fname))
        # don't save field data here 
        field = self.__dict__['field']
        del self.__dict__['field']
        file = open(self.fname,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        # keep field as attribute for plotting later
        self.field = field

    def load(self, recalc_B=False, fname=None):
        params = self.params
        from_file=True
        if fname == None:
            from_file = False
            fname = self.fname
        file = open(fname,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        print('Simulation data loaded from {}'.format(self.fname))
        if from_file:
            params = self.params
            geometry = self.geometry
        elif not from_file:
            params, geometry = extract_geometry(params)
            self.geometry = geometry
            params = format_params(params)
            self.field = mag.Field(self.geometry, recalc_B=recalc_B)

    def update_measures(self, detection_idx):
        for measure_name in self.measures_map.keys():
            self.measures[measure_name][detection_idx] = self.measures_map[measure_name]()

    def init_measures(self):
        Ndetections = len(self.detection_time_steps)
        self.Ndetections = Ndetections
        Natom = self.Natom
        measures = dict(
             traj     = np.zeros((Ndetections, Natom, 3)),
             vels     = np.zeros((Ndetections, Natom, 3)),
             temps    = np.zeros((Ndetections, 4)),
             dens     = np.zeros((Ndetections, 4)),
             psd      = np.zeros( Ndetections),
             centers  = np.zeros((Ndetections, 3)),
             skews    = np.zeros((Ndetections, 3)),
             sigmas   = np.zeros((Ndetections, 3)),
             meanKs   = np.zeros((Ndetections, 3)),
             thermKs   = np.zeros((Ndetections, 3)),
             kinetics = np.zeros((Ndetections, 4)),
             )
        self.measures = measures
        self.update_measures(0)

    def init_sim(self):
        Ntsteps = self.Ntsteps
        Natom = self.Natom
        self.init_measures()
        # jump past any requested delay
        # if delay = 0, the initial data is repeated at time step 2
        self.cloud.free_expand(self.params['delay'])
        self.update_measures(1)

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

    def run_sim(self, verbose=True, fly=True):
            params = self.params
            a = mag.make_acceleration(self.field, **self.params)
            tau = params['tau']
            if not self.pulse2_params == None:
                a2 = mag.make_acceleration(self.field2, **self.params2)
            # step forward through pulse sequence
            ti = 2  # time step index
            ta = params['delay']
            tb = params['delay'] + tau + params['tcharge']
            ts_remain = ((self.params['r0_detect'] - self.cloud.xs)/self.cloud.vs)[:,0]
            t_remain = np.mean(ts_remain)
            for p in range(params['Npulse']):
                for t in np.arange(ta, tb, params['dt']):
                    if t_remain > 0 or not fly:
                        usea = a
                        tau = params['tau']
                        if not self.pulse2_params == None:
                            if p % 2 == self.params2['parity']:
                                usea = a2
                                tau = self.params['tau']
                            else:
                                tau = self.params2['tau']
                        self.cloud.rk4(usea, t, params['dt'])
                        if ti in self.detection_time_steps:
                            self.update_measures(
                                list(self.detection_time_steps).index(ti))
                        if verbose:
                            sys.stdout.write(
                                ' '*43 + 't = {:.2f} of {:.2f}\r'.format(
                                    t, params['tmax']))
                            sys.stdout.flush()
                        if  ta - params['dt']/2 < t < ta + params['dt']/2:
                            pump = params['optical_pumping']
                            if not self.pulse2_params == None:
                                if p % 2 == self.params2['parity']:
                                    pump = self.params2['optical_pumping']
                            self.cloud.optical_pump(pump)
                            if verbose:
                                print('pulse {} begins t = {:.2f}'.format(p+1, t))
                                print('optically pumped t = {:.2f}'.format(t))
                    elif t_remain < 0 and fly:
                        print('detection truncates pulse...')
                        self.cloud.free_expand(t_remain)
                        print('back propagating {}...'.format(t_remain))
                        if self.Ndetections > 4:
                            for i in range(ti, self.Ntsteps):
                                self.update_measures(i)
                        break
                    if fly:
                        ts_remain = ((self.params['r0_detect'] - self.cloud.xs)/self.cloud.vs)[:,0]
                        t_remain = np.mean(ts_remain)
                    #if ta + params['tau'] - params['dt'] < t + params['dt']/2 < ta + params['tau']:
                        # TO DO: print a better table. Show all measures
                        #if verbose:
                            #print()
                            #print('Pulse {} ends t = {}\r'.format(p+1, t))
                            #print( '  temp | initl | final | ratio')
                            #print( '----------------------------')
                            #for label, temp in zip(self.cloud.temp_names, self.cloud.temps.T):
                            #    print(  '  {}   | {:>5.3f} | {:>5.3f} | {:<5.3f}'.format(
                            #        label, temp[0], temp[ti], temp[ti]/temp[0]))
                            #print()
                    ti += 1
                ta = tb 
                tb = ta + tau + params['tcharge']
            # final free expansion expansion
            if not fly:
                tof = 0
                t_remain = tof
            if t_remain>0:
                self.cloud.free_expand(t_remain)
                if verbose:
                    print('Cloud expanding for {:.2f} us'.format(t_remain))
            #signals = self.tof_detection()
            #plt.imshow(signals, aspect='auto', extent=(t_remain, t_remain+3000*0.1, -2.5, 2.5))
            #plt.show()
            self.update_measures(-1)

            if verbose:
                Nremain = np.sum(self.cloud.keep_mask)
                print('{}/{} atoms remain'.format(
                        Nremain, params['Natom']))
    # plotting
    def plot_measures(self, save_loc, fignum=1, save=True, show=False):
        if save or show:
            cloud = self.cloud
            field=self.field
            fignum = mag.plot_slices(fignum, field)
            fignum = mag.plot_slices(fignum, field, grad_norm=True)
            fignum = mag.plot_contour(fignum, field)
            fignum = mag.plot_contour(fignum, field, grad_norm=True)
            fignum = plot_traj(fignum, self, cloud, field)
            #fignum = plot_integrated_density(fignum, self, cloud)
            fignum = plot_phase_space2(fignum, self, cloud)
            fignum = plot_temps(fignum, self, cloud,
                    logy=True, include_names=['Tx','Ty','Tz'])
            fignum = plot_psd(fignum, self, cloud)
            plot_kinetic_dist(fignum, self, cloud)
            #fignum = plot_scalar_summary(fignum, cloud, field)
            # show or save plots
        if show:
            plt.show()
        if save:
            if show:
                print('Cannot show and save plots.')
            else:
                multipage(save_loc)
                print('plots saved to {}'.format(save_loc))



# ANALYSIS
# ========
def units_map(param):
    if param in ('temps', 'Tl', 'Tt'):
        unit = ' [mK]'
    elif param in ('I1', 'I2', 'IHH', 'IAH', 'I'):
        unit = ' [A]'
    elif param[:2] == 'r0'\
         or param[0] in ('L', 'W', 'R', 'A')\
         or param in ('centers', 'sigmas', 'D_ph', 'width', 'd'):
        unit = ' [cm]'
    elif param[:2] in ('dt', 't0')\
         or param in ('tcharge', 'delay', 'tmax', 'tau'):
        unit = r' [$\mathrm{\mu s}$]'
    elif param in ('v0', 'vrecoil'):
        unit = r' [$\mathrm{cm~\mu s^{-1}}$]'
    elif param in ('meanKs', 'thermKs', 'kinetics'):
        unit = r' [$\mathrm{kg~cm^2~\mu s^{-2}}$]'
    elif param in ('ts', 't', 'times','time'):
        unit = r' [$\mathrm{\mu s}$]'
    else:
        unit = ''
    return unit

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
    print('saving plots 2d scan analysis to', plot_fname)
    multipage(plot_fname, figs=figs)


def scan_2d(sweep_shape, sweep_vals_list, records, plot_fname, unit='cm',
            typical_kick_hline=True, no_kick_hline=True):
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
    labels = ['centers', 'sigmas', 'skews', 'temps',
               'meanKs', 'thermKs', 'kinetics']
    if typical_kick_hline:
        typical_kick_sim = Simulation({},
                load_fname=typical_kick_fname,
                reinitialize=False, resimulate=False)
    if no_kick_hline:
        no_kick_sim = Simulation({},
                load_fname=no_kick_fname,
                reinitialize=False, resimulate=False)
    for label in labels:
        if label == 'Ninit':
            record = np.asarray(records[label])
            record = record.reshape(shaper)
            fig  = plt.figure(fignum, figsize=(6,4))
            ax   = fig.add_subplot(1,1,1)
            for line_label, line_data in zip(ys, record.T):
                ax.plot(xs, line_data, label=yparam +'  = ' + str(line_label))
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
                if typical_kick_hline:
                    ax.axhline(typical_kick_sim.measures[label][-1][coordi],
                            label = 'typical kick', c='b')
                if no_kick_hline:
                    ax.axhline(no_kick_sim.measures[label][-1][coordi],
                            label='no kick', c='r')
                for line_label, line_data in zip(ys, record.T):
                    vals = line_data
                    line_label = yparam + ' = ' + str(line_label) + units_map(yparam)
                    if label != 'temps':
                        if unit in ('inch', 'in'):
                            vals = line_data * 0.3937
                    ax.plot(xs, vals, label=line_label)
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)
                plt.legend()
                fignum += 1
                figs.append(fig)
    print('saving plots 2d scan analysis to', plot_fname)
    multipage(plot_fname, figs=figs)

        #tick_skip = 1
        #xticks = range(0, len(xs), tick_skip)
        #yticks = range(len(ys))
        #xticklabels = ["{:6.0f}".format(x) for x in xs[::tick_skip]]
        #yticklabels = ["{:6.0f}".format(y) for y in ys]
        ##yticklabels = ["{:6.0f}".format(y*1e-2*1e6) for y in ys]
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(xticklabels)
        #ax.set_yticks(yticks)
        #ax.set_yticklabels(yticklabels)
        #ax.set_title(label)
        #ax.set_xlabel(xparam)
        #ax.set_ylabel(yparam)
        ##ax.set_xlabel('delay [us]')
        ##ax.set_ylabel('V0 [m/s]')
        #fig.colorbar(im)

# xaxis can be `time` or `param`
def scan_1d(sweep_shape, sweep_vals_list, records, plot_fname,
            xaxis = 'time'):
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
                coord_labels = ['x', 'y', 'z', '']
            for coordi, coord_label in enumerate(coord_labels):
                fig  = plt.figure(fignum, figsize=(6,4))
                ax   = fig.add_subplot(1,1,1)
                full_label = ' '.join([label[:-1], coord_label])
                for param_id, sparam_val in enumerate(sweep_vals_list_flat[0]):
                    record = np.asarray(records[label])[param_id][:, coordi]
                    ts = np.asarray(records['ts'])[param_id]
                    ax.plot(ts[1:-1], record[1:-1],
                        label=sweep_params[0] + ' = ' + str(sparam_val))
                    ax.set_ylabel(full_label)
                    ax.set_xlabel(xparam)
                plt.legend()
                fignum += 1
                figs.append(fig)
    print('saving plots 1d scan analysis to', plot_fname)
    multipage(plot_fname, figs=figs)

def process_sweep(sweep_shape, sweep_vals_list, records):
    scan_2d(sweep_shape, sweep_vals_list, records)

if __name__ == '__main__':
    pass
