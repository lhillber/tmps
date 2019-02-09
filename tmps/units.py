#! user/bin/python3

from numpy import pi

# Input all numbers in uits of: cm, us, mK, A, kg

# Boltzmann's constant, kg m^2 s^-2 K^-1 cm^2/m^2 s^2/us^2 K/mk
kB = 1.381e-23 * 1e4 * 1e-12 * 1e-3
# Planck's Constant, m^2 kg s^-1 cm^2/m^2 s/us
h = 6.62607004e-34 * 1e4 * 1e-6
# Bohr magneton, A m^2 cm^2/ m^2
muB = 9.274e-24 * 1e4
# speed of light, m/s cm/m
cc = 299792458 * 1e2 * 1e-6
# electron charge, A s us/s
ee = 1.60217e-19 * 1e-6
# vac permittivity A^2 s^4 kg^-1 m^-3 us^4/s^4 m^3/cm^3
eps0 = 8.854187817e-12 * 1e24 * 1e-6
# vac permiability kg m A^-2 s^-2 cm/m s^2/us^2
u0 = 1.257e-6 * 1e2 * 1e-12

# Li7 propertied a and b are HFS params in MHz,
# outer layer is L, inner layer is J
Li7_props = {
             'name': 'Li7',
             'm': 1.16e-26,  # mass, kg
             'I': 3.0/2.0,   # nuclear spin
             0.0:
            {1.0/2.0:
                {
                   'name': 'ground',
                   'a': 401.752,
                   'b': None,
                   'gJ': 2.0023010,
                  }},
             1.0:
             {1.0/2.0:
                {
                   'name': 'D1',
                   'a': 45.914,
                   'b': None,
                   'gJ': 0.6668,
                   'wlen': 0.0000670976,                   # cm
                   'wnum': 2.0 * pi / 0.0000670976,        # rad/cm
                   'freq': cc / 0.0000670976,              # MHz
                   'omega': 2.0 * pi * cc / 0.0000670976,  # rad/us
                   'lifet': 0.0272,                        # lifetime, us
                   'gamma': 1 / 0.02701,                   # rad/us
                   'dnu': 1/(2 * pi * 0.02701)             # MHz
                  },
              3.0/2.0:
                  {
                   'name': 'D2',
                   'a': -3.055,
                   'b': -0.2221,
                   'gJ': 1.335,
                   'wlen': 0.0000670961,                   # cm
                   'wnum': 2.0 * pi / 0.0000670976,         # rad/cm
                   'freq': cc / 0.0000670976,              # MHz
                   'omega': 2.0 * pi * cc / 0.0000670976,  # rad/us
                   'lifet': 0.0272,                        # us
                   'gamma': 1 / 0.02701,                   # rad/us
                   'dnu': 1/(2 * pi * 0.02701)             # MHz
                   }}
               }
