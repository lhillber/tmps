import tmps
import os

# PARAMETERS
# ==========
# Input all numbers in uits of: cm, us, mK, A, kg
# Atomic beam travels along x, the longitudinal direction.
# Kick is along z, a transverse direction.
# The remaining transverse direction is y.
# 

from square3_geometry import geometry
from init_cloud import cloud_params

sim_params = dict(
    dt         = 1.0,
    delay      =  730.0,
    r0_detect  = [40.0, 0.0, 0.0],   # 1. longitudinal location of detection, cm
    pulses = [
        dict(
              geometry        = geometry,
              recalc_B        = False,
              shape           = 'sin',
              scale           = 1.0,
              tau             = 100.0,
              tof             = 0.0,
              optical_pumping = 'hfs'),
        ]
    )

# DO IT!
sim = tmps.Simulation(cloud_params, sim_params,
    observation_idx  = 'all',
    reinit            = False,
    resimulate       = False,
    save_simulation  = True,
    verbose          = True )
sim.plot_current(show=False)
sim.plot_temps(show=False)
sim.plot_psd(show=False)
sim.plot_kinetic_dist(show=False)
#sim.plot_measures('plots/testing.pdf', show=False, save=True)
