import tmps
import os

# PARAMETERS
# ==========
# Input all numbers in uits of: cm, us, mK, A, kg
# Atomic beam travels along x, the longitudinal direction.
# Kick is along z, a transverse direction.
# The remaining transverse direction is y.

params = dict(
    # particle params (Li-7)
    m  = 1.16e-26,        # particle mass, kg
    mu = 9.274e-24 * 1e4, # particle magnetic moment, A m^2 cm^2/ m^2
    vrecoil = 0,          # recoil velocity of transition, cm/us

    # cloud params
    max_init = 10000,                    # max initialization cycles
    Tt       = 100.0,                    # initial transverse temperature, mK
    Tl       = 100.0,                    # initial longitudinal temperature, mK
    Natom    = 100000,                   # number of atoms to simulate
    width    = [ 0.35, 0.25, 0.25],      # initial size (st. dev), cm
    r0_cloud = [-37.0, 0.0, 0.0],        # initial center, cm
    v0       = [480*1e2*1e-6, 0.0, 0.0], # initial velocity, cm/us

    # state preparation
    optical_pumping = 'hfs',# state preparation before kick ('1' ,'0', 'vs', 'xs')

    # pin hole params
    pin_hole = True,    
    r0_ph    = [-6.0, 0.0, 0.0], # x,y,z location of pin hoe center, cm
    D_ph     = 0.5,        # diameter of pin hole, cm

    # tagging
    tag = True,
    r0_tag = [-13.0, 0.0, 0.0],
    t0_tag = 0.0,
    dt_tag = 15.0,

    # detection point
    r0_detect= [40.0, 0.0, 0.0],   # 1. longitudinal location of detection, cm
    #r0_detect= [60.0, 0.0, 0.0],  # 2. longitudinal location of detection, cm

    # current pulse params
    shape    = 'poly', # pulse shape, sin or square
    tau      = 800.0,  # discharge time, us
    Npulse   = 1,      # number of current pulses
    tcharge  = 0.0,    # recharge time,  us
    decay    = 1.5,    # factor to decrease I each pulse
    # `None` uses defualt calculated values in `format_params`
    dt       = 1.0,    # time step (see format_params), us
    delay    = 730.0,  # deleay before first pulse, us

    # coil params
    config = 'smop3',
    I1      = 1500.0,  # bias max current, A
    I2      = 1500.0,  # kick max current, A
    n = [0, 0, 1],
    r0 = [0.0, 0.0, 0.0],
    L1 = 4.75,
    W1 = 2.84,
    d  = 0.086,
    M1 = 6,
    N1 = 2,
    ang = 0.0,
    A = 2.34 / 2.0,
    L2 = 4.29,
    W2 = 2.03,
    M2 = 6,
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

# DO IT!
sim = tmps.Simulation(params,
    detection_time_steps='all',
    recalc_B=False,
    reinitialize=False,
    resimulate=False,
    save_simulations=True,
    verbose=True)

sim.plot_measures('plots/testing.pdf', show=False, save=True)
