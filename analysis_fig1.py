# PARAMETERS
# ==========
# Input all numbers in uits of: cm, us, mK, A, kg
# Atomic beam travels along x, the longitudinal direction.
# Kick is along z, a transverse direction.
# The remaining transverse direction is y.

import tmps
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# The following two lines ensure type 1 fonts are used in saved pdfs
mpl.rcParams['pdf.fonttype'] = 42

# plotting defaults
plt_params = {
          'font.size'   : 14,
          }
plt.rcParams.update(plt_params)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

# Bias current
I1 = [I for I in np.arange(-1800.0, 1800+600, 600.0)]
I2 = 1500.0

# Final detection distance from coils
r0_detect= [[39.8+4.8, 0.0, 0.0],
            [57.8, 0.0, 0.0]]

r0_cloud = [-38.0, 0.0, 0.0]

r0_ph = [-5.4, 0.0, 0.0]

width = [0.5, 0.24, 0.26]


# Properties of the cloud
cloud_params = dict(
    # particle params (Li-7)
    m       = 1.16e-26,        # particle mass, kg
    mu      = 9.274e-24 * 1e4, # particle magnetic moment, A m^2 cm^2/ m^2
    vrecoil = 0,               # recoil velocity of transition, cm/us

    # cloud params
    max_init  = 10000,                      # max initialization cycles
    Tt        = 120.0,                      # initial transverse temperature, mK
    Tl        = 200.0,                      # initial longitudinal temperature, mK
    Natom     = 1000,                     # number of atoms to simulate
    width     = width,        # initial size (st. dev), cm
    r0_cloud = r0_cloud,          # initial center, cm
    v0        = [480.0*1e-6*1e2, 0.0, 0.0], # initial velocity, cm/us

    # pin hole params
    pin_hole = True,
    r0_ph    = r0_ph, # x,y,z location of pin hoe center, cm
    D_ph     = 0.5,              # diameter of pin hole, cm

    # tagging TODO: better represent laser beams throughout
    tag    = True,
    r0_tag = [-13.0, 0.0, 0.0],
    t0_tag = 0.0,
    dt_tag = 15.0)

geometry = dict(
    config = 'smop3', # 3 square rectangular coils in mop configuration

    # peak currents
    I2 = I2,
    I1 = I1,

    # coil orientation
    n  = [0, 0, 1],   # coil normal
    r0 = [0.0, 0.0, 0.0], # coil center
    ang = 0.0,            # angle between long axis and x axis

    # coil dimensions
    d  = 0.086,           # wire diameter, cm
    L1 = 4.75,            # long axis length of kick coil, cm
    W1 = 2.84,            # short axis width of kick coil, cm
    M1 = 6,               # turns / layer of kick coil, int
    N1 = 2,               # layers of kick coil, int

    A  = 2.34 / 2.0,      #

    L2 = 4.29,
    W2 = 2.03,
    M2 = 6,
    N2 = 2,

    # B-field solution grid
    Nzsteps  =  100,
    Nysteps  =  100,
    Nxsteps  =  200,
    xmin     = -5,
    xmax     =  5,
    ymin     = -3,
    ymax     =  3,
    zmin     = -3,
    zmax     =  3
    )

nokick_geometry = geometry.copy()
nokick_geometry['I1'] = 0.0
nokick_geometry['I2'] = 0.0

sim_params = dict(
    delay     =  None,
    dt        = 0.5,
    pulses = [
        dict(
              geometry        = geometry, # scanning I1 in geometry
              recalc_B        = False,
              shape           = 'sin',
              scale           = 1.0,
              tau             = 100.0,
              tof             = 0.0,
              optical_pumping = 'hfs'),
        ],
    r0_detect = r0_detect # scanning
    )

nokick_sim_params1 = copy.deepcopy(sim_params)
nokick_sim_params2 = copy.deepcopy(sim_params)
nokick_sim_params1['pulses'][0]['geometry'] = nokick_geometry
nokick_sim_params2['pulses'][0]['geometry'] = nokick_geometry
nokick_sim_params1['r0_detect'] = r0_detect[0]
nokick_sim_params2['r0_detect'] = r0_detect[1]

laser_params=None
# DO IT! (in parallel)
sweep_shape, sweep_vals_list, records, rank = tmps.experiment(
    cloud_params, sim_params, laser_params,
    to_record=-1,          # grab last point of sim measures for plotting
    reinit=False,          # reinitialize cloud? if False trys to load
    resimulate=False,      # resimulate all? if False, trys to load
    save_simulations=False, # save simulation class?
    verbose=True)          # print operations

# for parallel execution, only analyze data from the master process
if rank == 0:
    def analysis_fig1(sweep_shape, sweep_vals_list, records, plot_fname,
                labels        = ['centers', 'sigmas']):
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
        print(sweep_params)
        print(sweep_vals_list_flat)
        xs, ys = sweep_vals_list_flat
        xs = np.array(xs)
        xparam, yparam = sweep_params
        xparam = ' '.join(xparam.split('_'))
        yparam = ' '.join(yparam.split('_'))

        no_kick40_sim = tmps.Simulation(cloud_params, nokick_sim_params1, None,
                reinit=False, resimulate=False)

        no_kick60_sim = tmps.Simulation(cloud_params, nokick_sim_params2, None,
                reinit=False, resimulate=False)

        fig, axarr = plt.subplots(2, 2, sharex=True, sharey='row')
        coord_labels = [None, r'$\mathrm{coils}~@~90^\circ$',  r'$\mathrm{coils}~@~0^\circ$']
        ps = []
        results = {}
        for label in labels:
            print(label)
            key = label
            if label == 'centers':
                row = 0
            elif label in ('sigmas', 'temps'):
                row = 1
            for coordi, coord_label in enumerate(coord_labels):
                if coordi == 0:
                    continue
                elif coordi in (1, 2):
                    if coordi == 1:
                        c = 'g'
                        skey ='_90deg'
                    elif coordi == 2:
                        c = 'r'
                        skey = '_00deg'
                    else:
                        continue
                    no_kick40_val = no_kick40_sim.measures[label][-1][coordi]
                    no_kick60_val = no_kick60_sim.measures[label][-1][coordi]
                    record = np.asarray(records[label])[:, coordi]
                    record = record.reshape(shaper)
                    ylabel = label[:-1] + tmps.units_map(label, mm=True)
                    xlabel = 'bias current' + tmps.units_map(xparam, mm=True)
                for col, (line_label, line_data) in enumerate(zip(ys, record.T)):
                    if col == 0:
                        sskey = '_40cm'
                        if label in ('sigmas', 'temps'):
                            #vals = np.sqrt(line_data**2 - no_kick40_val**2)
                            vals = line_data
                        else:
                            vals = line_data
                        results[key+skey+sskey+'_nokick'] = 10*no_kick40_val
                    elif col == 1:
                        col = 1
                        sskey = '_60cm'
                        if label in ('sigmas', 'temps'):
                            #vals = np.sqrt(line_data**2 - no_kick60_val**2)
                            vals = line_data
                        else:
                            vals = line_data
                        results[key+skey+sskey+'_nokick'] = 10*no_kick60_val
                    ax = axarr[row, col]
                    if label != 'temps':
                        results[key+skey+sskey] = 10*vals[::-1]
                    if (row, col) == (0, 0):
                        p = ax.plot(xs, 10 * vals[::-1], label=coord_label, c=c)

                        ps += p
                    else:
                        fac = 10.0
                        if label == 'temps':
                            fac = 1.0
                        ax.plot(xs, fac * vals[::-1], c=c)

                    ax.set_ylabel(ylabel)
                    ax.set_xlabel(xlabel)
                    ax.grid(True)
                    plt.locator_params(axis='x', nbins=6)
        for ax in axarr.flat:
            ax.label_outer()
        axarr[0,0].set_ylim(-5, 5)
        #axarr[1,0].set_ylim(0.0, 1.0)
        axarr[0,0].set_title('40 cm from coils')
        axarr[0,1].set_title('60 cm from coils')
        axarr[0,0].legend(handles=ps, loc="lower left", handlelength=1)
        axarr[0,0].text(0.85, 0.85, '(a)', transform = axarr[0,0].transAxes)
        axarr[0,1].text(0.85, 0.85, '(b)', transform = axarr[0,1].transAxes)
        axarr[1,0].text(0.85, 0.85, '(c)', transform = axarr[1,0].transAxes)
        axarr[1,1].text(0.85, 0.85, '(d)', transform = axarr[1,1].transAxes)
        results['Vs'] = xs
        np.save('results/shifted_centers-sigmas_00-90deg_40-60cm', results)
        print(plot_fname)
        #plt.show()
        tmps.multipage(plot_fname, figs=[fig])

    #results = np.load('results/centers-sigmas_00-90deg_40-60cm_short-hard-kick.npz')
    analysis_fig1(sweep_shape, sweep_vals_list, records,
            plot_fname=os.path.join(
            'plots/analysis',
            'centers-sigmas_00-90deg_40-60cm.pdf'
            ))


    #tmps.scan_3d(sweep_shape, sweep_vals_list, records, unit='cm',
    #    plot_fname=os.path.join('plots/bias',
    #            'bias_scan_square3_poly_lfs_1M_tag-pinhole-adj2_100mK_delay600' + '.pdf'))

