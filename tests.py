# Testing functions for the tmps project ensuring correct physics and correct code
#
# By Logan Hillberry 2018 February 02
#

import tmps
import numpy as np
import matplotlib.pyplot as plt

# Rb 85 parameters
m  = 1.419226084e-25
    # particle mass, kg
mu = 9.274e-24 * 1e4
    # particle magnetic moment, A m^2 cm^2/ m^2
vrecoil = 6.023 * 1e-1 * 1e-6,
    # recoil velocity, mm/s * cm/mm * s/us
kB = 1.381e-23 * 1e4 * 1e-12 * 1e-3
    # Boltzmann's constant, kg m^2 s^-2 K^-1 cm^2/m^2 s^2/us^2 K/mk


# Temperature
# -----------

# analytic speed distribution
def maxwell_speed(v):
    C = (m / (2 * np.pi * kB * T))**(3/2) * 4 * np.pi * v*v
    return C * np.exp(- m * v*v / (2 * kB * T))

def most_prob_speed(T):
    return np.sqrt(2 * kB * T / m)

def check_vel_speed_dist(T):
    # speed distribution calculated from velocity component distribution
    vdata = tmps.maxwell_velocity(T, m, nc=int(3e6))
    sdata = np.sqrt(np.sum(vdata.reshape(int(1e6),3) ** 2, axis=1))
    vana = most_prob_speed(T)
    # Normalized histogram (sampled plot)
    n, bins, patches = plt.hist(sdata, bins=100, normed=1)
    # velocity bin spacing
    dv = bins[1] - bins[0]
    # analytic plot
    vs = np.linspace(0, 4*vana , 200)
    plt.plot(vs, maxwell_speed(vs))
    most_prob_samp = bins[np.argmax(n)] + dv/2

    print('Most probably speed:', 'sample:',
          most_prob_samp, 'analytic:', vana)
    plt.show()

#check_vel_speed_dist(T = 370000e-9)

cloud = tmps.Cloud(Natom=int(1e6),
                   T=300000,
                   width=[2,2,2],
                   r0_cloud=[0,0,0],
                   m=m,
                   mu=mu,
                   vrecoil=vrecoil,
                   suffix='tests',
                   v0=[0,0,0],
                   data_dir='data/tests',
                   plot_dir='plots/tests')

data=[]
for i in range(100):
    T = cloud.get_temp()[0]
    print(T)
    data.append(T)
    cloud.recoil()
    #print(np.var(cloud.vs))
plt.plot(data)
plt.show()
