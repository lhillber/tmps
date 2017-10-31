import tmps

# PARAMETERS
# ==========
# simulation definition
# Input allnumbers in uits of: cm, us, mK, A, kg
params = dict(
    # particle params
    m  = 1.16e-26,
        # particle mass, kg
    mu = 9.274e-24 * 1e4,
        # particle magnetic moment, A m^2 cm^2/ m^2

    # coil params
    RAH = 1.0795,
    RHH = 1.27,
    AAH = 1.18745,
    AHH = 1.18745,
    r0AH = [0.0, 0.0, 0.0],
    r0HH = [0.0, 0.0, 0.0],
    dAH = 0.04318,
    dHH = 0.04318,
    MAH = 5,
    MHH = 5,
    NAH = 2,
    NHH = 2,
    nAH = [0, 0, 1],
    nHH = [0, 0, 1],
    HH_scale = 1,    # I0_HH / I0_AH

    # cloud params
    T        = 200,                        # initial temperature, mK
    Natom    = 1000,                      # number of atoms to simulate
    width    = [ 0.25, 0.25, 0.25],        # initial size (st. dev), cm
    r0_cloud = [0.0, 0.0, 0.0],            # initial center, cm
    v0       = [0 * 1e2 * 1e-6, 0,0, 0.0], # initial velocity, cm/us

    # current pulse params
    I0       = 2000,                        # max current, A
    t0       = 0.0,                         # start time, us
    Npulse   = 1,                           # Number of current pulses
    shape    = 'sin',                       # pulse shape, sin or square
    tau      = 100,                         # discharge time, us
    tcharge  = 10,                          # recharge time,  us
    decay    = 1.5,                         # factor to decreasue I0 each pulse

    # time evolution params
    #  method = 'rk4' #TODO: try others (leapfrog)
    # `None` uses defualt calculated values in `format_params`
    tmax     = None,                        # max time (see format_params), us
    dt       = None,                        # time step (see format_params), us
    delay    = None,                        # deleay before first pulse, us

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

    # random number generator seed = None for random initialization
    seed     = None,
    # relative or absolute path to dir where plots are to be saved (must exist)
    plot_dir = 'plots',
    # suffix or version number appended to plot file names
    suffix   = ''
    )

# DO IT!
tmps.run_sim(params)
