from src.plotter import units_map, Plotter
from src.units import muB
from src.fio import IO, base_dir

import numpy as np
import os
import sys
import src.cloud as cloud
import src.magnetics as mag
import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d, Axes3D

mpl.rcParams["ps.fonttype"] = 42
plt_params = {"font.size": 12, "figure.max_open_warning": 0}
plt.rcParams.update(plt_params)
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]


class Simulation(IO):
    def __init__(
        self, sim_params, cloud_params, dir=None, nickname=None, recalc=False, save=True
    ):

        self.sim_params = sim_params
        self.cloud = cloud.Cloud(**cloud_params)
        self.to_measure = ["xs", "vs", "T", "n", "rho", "R", "S", "K"]
        super().__init__(
            dir=dir,
            nickname=nickname,
            recalc=recalc,
            uid_keys=["sim_params", "cloud_params"],
        )

        if self.recalc:
            self.process_sim_params(**sim_params)
            self.process_timing()
            self.init_sim()
            self.run_sim()
            if save and not self.loaded:
                self.save()

    def __getattr__(self, attr):
        if attr in self.to_measure:
            return self.measures[attr]
        else:
            return super().__getattr__(attr)

    def process_sim_params(self, dt, delay, observe, pulses, r0_detect=None):
        self.obsi = 0
        self.ti = 0
        self.dt = dt
        self.delay = delay
        self.observe = observe
        self.observe = observe
        self.pulses = pulses
        self.r0_detect = r0_detect
        if r0_detect is not None:
            self.r0_detect = r0_detect
        fields = {}
        field_num = 1
        geometries = []
        for pulse in self.pulses:
            geometry = pulse["geometry"]
            if geometry in geometries:
                pulse["field"] = fields[geometry["field_name"]]
            else:
                recalc = pulse["recalc"]
                field = mag.Field(geometry, recalc=recalc)
                pulse["field"] = field
                field_name = "field" + str(field_num)
                fields[field_name] = field
                geometry["field_name"] = field_name
                geometries += [geometry]
                field_num += 1

    def process_timing(self):
        if self.delay is None:
            self.delay = np.abs(self.r0[0] / self.v0[0]) - self.pulses[0]["tau"] / 2.0
            print("Best delay estimate [us]: ", self.delay)
        dt = self.dt
        ta = 0.0
        tb = self.delay
        pulse_ts_list = [[ta]]
        for pulse in self.pulses:
            ta = tb
            tb = ta + pulse["tau"] + pulse["tof"] + dt
            pulse["t0"] = ta
            pulse_ts = np.arange(ta, tb, dt)
            pulse["pulse_ts"] = pulse_ts
            pulse_ts_list += [pulse_ts]
        ts = np.concatenate(pulse_ts_list)
        # allocate for initial delay
        if self.delay > 0.0:
            ts = np.insert(ts, 0, 0.0)
        # allocate for final free exapansion
        if self.r0_detect is not None:
            ts = np.insert(ts, len(ts), 0.0)
        Ntsteps = len(ts)
        for pulse in self.pulses:
            pulse_Is = np.array([mag.curr_pulse(t, **pulse) for t in ts])
            pulse["Is"] = pulse_Is
        self.ts = ts
        self.Ntsteps = Ntsteps
        if self.observe == "all":
            self.observe = np.arange(0, Ntsteps, 1)
        elif self.observe in (None, "default"):
            self.observe = [0, 1, Ntsteps - 2, Ntsteps - 1]
        self.Ndetections = len(self.observe)

    def init_measures(self):
        self.measures = {
            measure_name: np.zeros(
                tuple(
                    [self.Ntsteps]
                    + [n for n in getattr(self.cloud, measure_name).shape]
                )
            )
            for measure_name in self.to_measure
        }

    def set_measures(self):
        for measure_name in self.to_measure:
            self.measures[measure_name][self.obsi] = getattr(self.cloud, measure_name)
        self.obsi += 1

    def init_sim(self):
        self.init_measures()
        if self.delay > 0.0:
            self.free_expand(self.delay)
            self.ti += 1
        self.set_measures()

    def make_acceleration(self, pulse, approx=True):
        field = pulse["field"]
        m = self.cloud.atom.props["m"]
        gJ = self.cloud.atom.props[0.0][1.0 / 2.0]["gJ"]
        xinterp, yinterp, zinterp = field.gradnormB_interp

        def a(xs, t):
            normB = field.normB_interp(xs)
            if approx:
                coeff = self.cloud.mJs * gJ * muB / m
            else:
                coeff = (
                    np.array(
                        [
                            self.dEdBfuncs[f][mf](b)
                            for f, mf, b in zip(self.cloud.Fs, self.cloud.mFs, normB)
                        ]
                    )
                    / m
                )

            coeff = coeff[..., np.newaxis]
            dBdx_interp = xinterp(xs)
            dBdy_interp = yinterp(xs)
            dBdz_interp = zinterp(xs)
            a_xyz = (
                -coeff
                * mag.curr_pulse(t, **pulse)
                * np.c_[dBdx_interp, dBdy_interp, dBdz_interp]
            )
            return a_xyz

        return a

    def run_sim(self):
        for pulse_num, pulse in enumerate(self.pulses):
            a = self.make_acceleration(pulse, approx=pulse["approx"])
            pulse_ts = pulse["pulse_ts"]
            t0 = pulse_ts[0]
            pump = pulse["optical_pumping"]
            self.cloud.sample_internal(F0=pump, mF0="p", inplace=True)
            print("Pulse {} begins t = {:.2f}".format(pulse_num + 1, t0))
            print("Optically pumped to {} at t = {}".format(pump, t0))

            run = True
            for t in pulse_ts:
                assert t == self.cloud.t
                if self.r0_detect is not None:
                    ts_remain = ((self.r0_detect - self.x) / self.vx)[:, 0]
                    t_remain = np.mean(ts_remain)
                    if t_remain > 0.0:
                        run = True
                        if self.ti == self.Ntsteps - 2:
                            run = False
                    elif t_remain <= 0.0:
                        print("Detection truncates pulse")
                        print("Back propagating {} us".format(t_remain))
                        run = False

                if run:
                    self.cloud.rk4(a, self.dt)
                    sys.stdout.write(
                        " " * 43 + "t = {:.2f} of {:.2f}\r".format(t, self.ts[-1])
                    )
                    sys.stdout.flush()
                if t == self.ts[self.observe[self.obsi]]:
                    self.set_measures()
                self.ti += 1

        if self.r0_detect is not None:
            self.free_expand(t_remain)
            self.ts[-1] = self.ts[-2] + t_remain
            self.set_measures(-1)
            print("Cloud expanding for {:.2f} us".format(t_remain))

    def plot_current(self, axs=None, **kwargs):
        print("Plotting current...")
        if axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(7, 2))
        else:
            fig = plt.gcf()
        ts = self.ts
        for n, pulse in enumerate(self.pulses):
            geometry = pulse["geometry"]
            Idic = {}
            for k, v in geometry.items():
                if k[0] == "I":
                    Idic[k] = v
            for Iname, I0 in Idic.items():
                label = "pulse {} {}".format(n, Iname)
                Is = I0 * pulse["Is"]
                axs.plot(ts, Is, label=label, **kwargs)
        axs.set_xlabel("t [$\mu$s]")
        axs.set_ylabel("I [A]")
        plt.legend()
        return fig, axs

    def plot_measure(
        self,
        measure_name,
        axs=None,
        fignum=1,
        logy=False,
        colors=["C3", "C7", "k", "C2"],
        lines=["--", "-.", "-", ":"],
    ):
        if axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(7, 2))
        else:
            fig = plt.gcf()
        ts = np.take(sim.ts, self.observe)
        data = self.measures[measure_name]
        if measure_name in ("n", "rho"):
            data = data[..., np.newaxis]
        names = [measure_name + x for x in "xyz"]
        for name, mdata, color, line in zip(names, data.T, colors, lines):
            if logy:
                axs.semilogy(ts, mdata, label=name, c=color, ls=line)
            else:
                axs.plot(ts, mdata, label=name, c=color, ls=line)
        axs.set_xlabel("$t$" + units_map("t"))
        axs.set_ylabel(measure_name + units_map(measure_name))
        axs.legend(loc="upper right")
        return fig, axs

    def plot_measures(self):
        for measure_name in self.to_measure:
            if measure_name not in ("xs", "vs"):
                yield self.plot_measure(measure_name)

    def plot_phasespaces(
        self, plot_idx=[0, -1], fignum=1, remove_mean=False, Nsample=10000
    ):

        for ti in plot_idx:
            xs = self.xs[ti]
            vs = self.vs[ti]
            t = self.ts[ti]
            figaxs = self.cloud.plot_phasespace(
                remove_mean=remove_mean, Nsample=Nsample, xs=xs, vs=vs
            )
            figaxs[0].text(
                0.12, 0.87, r"$t = {}~\mu$s".format(t), verticalalignment="center"
            )
            yield figaxs

    def plot_trajprojections(self, axs=None, N=10, fignum=1, color=None):
        print("Plotting trajectory projections...")
        style_dict = {
            1.0 / 2.0: {"marker": r"$\uparrow$", "color": "red"},
            -1.0 / 2.0: {"marker": r"$\downarrow$", "color": "black"},
        }

        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=(7, 2))
        else:
            fig = plt.gcf()
        ax1, ax2, ax3 = axs
        for j, mj in enumerate(self.cloud.mJs):
            if j < N:

                if color == "mJ":
                    color = style_dict[mj]["color"]

                x = self.xs[::, j, 0]
                y = self.xs[::, j, 1]
                z = self.xs[::, j, 2]
                ax1.plot(x, y, color=color, lw=0.8)
                ax2.plot(x, z, color=color, lw=0.8)
                ax3.plot(y, z, color=color, lw=0.8)
            else:
                break
        ax1.set_xlabel("x [cm]")
        ax1.set_ylabel("y [cm]")
        ax2.set_xlabel("x [cm]")
        ax2.set_ylabel("z [cm]")
        ax3.set_xlabel("y [cm]")
        ax3.set_ylabel("z [cm]")
        fig = plt.gcf()
        fig.tight_layout()
        return fig, axs

    def plot_traj(
        self,
        axs=None,
        N=10,
        fignum=1,
        color=None,
        show_wires=True,
        show_field=True,
        use_pulse=0,
    ):
        print("Plotting 3D trajectory...")
        if axs is None:
            fig = plt.figure(figsize=(7, 7))
            axs = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            fig = plt.gcf()
        pulse = self.pulses[use_pulse]
        field = pulse["field"]
        x = self.xs[::, ::, 0]
        y = self.xs[::, ::, 1]
        z = self.xs[::, ::, 2]
        dx = 3 * np.std(x)
        dy = 3 * np.std(y)
        dz = 3 * np.std(z)
        xm = np.mean(x)
        ym = np.mean(y)
        zm = np.mean(z)
        axs.set_xlim(xm - dx, xm + dx)
        axs.set_ylim(ym - dy, ym + dy)
        axs.set_zlim(zm - dz, zm + dz)
        axs.view_init(elev=10.0, azim=80)
        axs.set_ylabel("y [cm]")
        axs.set_xlabel("x [cm]")
        axs.set_zlabel("z [cm]")
        style_dict = {
            1.0 / 2.0: {"marker": r"$\uparrow$", "color": "red"},
            -1.0 / 2.0: {"marker": r"$\downarrow$", "color": "black"},
        }

        if show_field:
            field.plot_3d(axs=axs)

        if show_wires:
            field.viz(axs=axs)

        for j, mj in enumerate(self.cloud.mJs):
            if j < N:
                if color == "mJ":
                    color = style_dict[mj]["color"]
                x = self.xs[::, j, 0]
                y = self.xs[::, j, 1]
                z = self.xs[::, j, 2]
                axs.plot(x, y, z, color=color, lw=1.2)
                axs.plot(
                    [x[-1]],
                    [y[-1]],
                    [z[-1]],
                    color="k",
                    mew=0.0,
                    alpha=0.9,
                    marker=style_dict[mj]["marker"],
                    ms=10,
                )

            else:
                break
        return fig, axs




