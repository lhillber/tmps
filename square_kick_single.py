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
    optical_pumping = '0',# state preparation before kick ('1' ,'0', 'vs', 'xs')

    # pin hole params
    pin_hole = True,    
    r0_ph    = [-9.2, 0, 0], # x,y,z location of pin hoe center, cm
    D_ph     = 0.3,        # diameter of pin hole, cm

    # tagging
    tag = True,
    r0_tag = [-3, 0, 0],
    t0_tag = 0,
    dt_tag = 5,

    # tagging
    tag = True,
    r0_tag = [-3, 0, 0],
    t0_tag = 0,
    dt_tag = 5,

    # current pulse params
    IHH      = 0,  # max current, A
    IAH      = 0,  # max current, A
    Npulse   = 1,     # Number of current pulses
    shape    = 'sin', # pulse shape, sin or square
    tau      = 100,   # discharge time, us
    tcharge  = 0,     # recharge time,  us
    r0_detect= [52.7, 0.0, 0.0],  # longitudinal location of detection, cm
    decay    = 1.5,   # factor to decreasue I0 each pulse
    # `None` uses defualt calculated values in `format_params`
    tmax     = None,  # max time (see format_params), us
    dt       = None,  # time step (see format_params), us
    delay    = None,  # deleay before first pulse, us

    # cloud params
    Tt       = 100,                       # initial transverse temperature, mK
    Tl       = 100,                      # initial longitudinal temperature, mK
    Natom    = 10000,                   # number of atoms to simulate
    width    = [ 0.35, 0.5/2, 0.5/2],   # initial size (st. dev), cm
    r0_cloud = [-29.5, 0.0, 0.0],         # initial center, cm
    v0       = [480*1e2*1e-6, 0, 0.0], # initial velocity, cm/us

    # coil params
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

    # random number generator seed = None for random initialization
    seed     = None,
    )

# DO IT!
sim = tmps.Simulation(params,
    recalc_B=False,
    resimulate=True,
    save_simulations=True,
    verbose=True)

sim.plot_measures(show=False, save=True)
