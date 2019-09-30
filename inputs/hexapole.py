from tmps.fio import base_dir
from tmps.simulation import Simulation
from tmps.plotter import Plotter
import os

cloud_params = dict(
    N=10000,
    Natoms=5e9,
    T=[0.3, 0.3, 0.3],
    S=[1.0 / 4, 1.0 / 4, 1.0 / 4],
    R=[0.0, 0.0, 0.0],
    V=[0.0, 0.0, 0.0],
    constraints=[],
    recalc=False,
    save=True,
)

geometry = dict(
    # coil params
    config="mop3",
    I2=800.0,  # bias max current, A
    I1=600.0,  # kick max current, A
    R2=4.089,
    R1=3.157,
    A1=3.434/2.0,
    A2=3.160/2.0,
    d=0.086,
    M=7,
    N=2,
    n=[0.0, 0.0, 1.0],
    r0=[0.0, 0.0, 0.0],
    meshspec=[[-2.0, 2.0, 100]] * 3,
)

sim_params = dict(
    delay=0.0,  # delay between atom release and first pulse
    dt=1.0,  # simulation time step, us
    observe="all",
    pulses=[
        dict(
            geometry=geometry,
            recalc=False,  # recalculate B field?
            shape="sin",  # pulse shape
            approx=True,  # use approximate force coefficients?
            scale=1.0,  # scale magnetic fields
            tau=500.0,  # pulse length, us
            tof=1000.0,  # time of flight after pulse, us
            optical_pumping="sz",  # magnetic state
        ),
        dict(
            geometry=geometry,
            recalc=False,  # recalculate B field?
            shape="sin",  # pulse shape
            approx=True,  # use approximate force coefficients?
            scale=-1.0,  # scale magnetic fields
            tau=0.1,  # pulse length, us
            tof=0.0,  # time of flight after pulse, us
            optical_pumping=None,  # magnetic state
        ),
    ],
)

sim = Simulation(sim_params, cloud_params, recalc=False, save=True, nickname="temp")
p = Plotter()
field = sim.pulses[0]["field"]

lines = [[0.0, 0.0, "var"], [0.1, 0.1, "var"], [0.2, 0.2, "var"], [0.3, 0.3, "var"]]
planes = {"x": [-0.5, 0.0, 0.5], "y": [-0.5, 0.0, 0.5], "z": [-0.5, 0.0, 0.5]}

p.add_many(
    [
        field.plot_3d(),
        field.plot_linecuts(grad_norm=False),
        field.plot_linecuts(grad_norm=True),
        field.plot_linecut(grad_norm=True),
        field.plot_linecuts(grad_norm=True, components="xyz", lines=lines),
        field.plot_slices(planes=planes, Bclip=1.0),
        field.plot_slices(grad_norm=True, planes=planes, Bclip=0.3),
        sim.plot_traj(N=20, show_field=False, show_wires=False),
        sim.plot_trajprojections(N=20),
        # sim.cloud.atom.plot_spectrum(),
        # sim.cloud.plot_phasespace(),
    ]
)
p.add_many(sim.plot_phasespaces())
p.add(sim.plot_current())
p.add_many(sim.plot_measures())

#sim.animate_traj(N=750)

fname = os.path.join(base_dir, "plots", "tmp", "mop_z.pdf")
p.save(fname)
