# cloud.py

from src.fio import IO
from src.atom import Atom
from src.units import kB, h

import sys
import numpy as np
from scipy.stats import skew
from numpy import pi
from numpy.linalg import norm
import matplotlib.pyplot as plt
from fastkde import fastKDE


class Cloud(IO):
    def __init__(
        self,
        N=1,
        S=[0.0, 0.0, 0.0],
        T=[0.0, 0.0, 0.0],
        R=[0.0, 0.0, 0.0],
        V=[0.0, 0.0, 0.0],
        F0="thermal",
        mF0="thermal",
        Natoms=1.0,
        constraints=set(),
        dir=None,
        nickname=None,
        recalc=False,
        save=True,
        atom_nickname="Li7_B_0-2-300",
    ):
        self.t = 0.0
        self.Natoms = Natoms
        self.constraints = set(constraints)

        N0 = N
        S0 = np.array(S)
        T0 = np.array(T)
        R0 = np.array(R)
        V0 = np.array(V)

        self.init_macrostate = {"N": N0, "S": S0, "T": T0, "R": R0, "V": V0}
        self.N0 = int(self.init_macrostate["N"])

        self.atom = Atom(nickname=atom_nickname, recalc=False)
        super().__init__(
            dir=dir,
            nickname=nickname,
            recalc=recalc,
            uid_keys=["constraint_data", "init_macrostate", "Natoms"],
        )
        if self.recalc:
            self._mask = np.full(self.N0, True)
            self._state = self.sample_state()
            self._internal = self.sample_internal(F0, mF0)

        self.apply_constraints(10000)

        if save and not self.loaded:
            self.save()

    # acess to microstate
    @property
    def state(self):
        return self._state

    @property
    def internal(self):
        return self._internal

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, vals):
        self._mask = vals

    @property
    def Fs(self):
        return self.internal[::, 0]

    @Fs.setter
    def Fs(self, vals):
        self.internal[::, 0] = vals

    @property
    def mFs(self):
        return self.internal[::, 1]

    @mFs.setter
    def mFs(self, vals):
        self.internal[::, 1] = vals

    @property
    def mJs(self):
        return self.internal[::, 2]

    @mJs.setter
    def mJs(self, vals):
        self.internal[::, 1] = vals

    @property
    def xs(self):
        return self.state[::, 0:3]

    @xs.setter
    def xs(self, vals):
        self._state[::, 0:3] = vals

    @property
    def vs(self):
        return self._state[::, 3:6]

    @vs.setter
    def vs(self, vals):
        self._state[::, 3:6] = vals

    # Estimates of macrostate
    @property
    def macrostate(self):
        return [self.N, self.S, self.T, self.R, self.V]

    @property
    def N(self):
        return np.sum(self.mask)

    @property
    def S(self):
        return np.std(self.xs, axis=0)

    @property
    def T(self):
        c = self.atom.props["m"] / kB
        return c * np.var(self.vs, axis=0)

    @property
    def R(self):
        return np.mean(self.xs, axis=0)

    @property
    def V(self):
        return np.mean(self.vs, axis=0)

    @property
    def K(self):
        return skew(self.xs, axis=0)

    @property
    def n(self):
        denom = 2 ** 6 * np.product(self.S)
        if denom:
            return self.Natoms / denom
        else:
            return np.nan

    @property
    def rho(self):
        T = np.mean(self.T)
        n = self.n
        denom = np.sqrt((2 * pi * self.atom.props["m"] * kB * T))
        if denom:
            return n * (h / denom) ** 3.0
        else:
            return np.nan

    def sample_state(self, macrostate=None, mask=None, inplace=False):
        if mask is None:
            mask = self.mask

        if macrostate is None:
            macrostate = self.init_macrostate
        S, T, R, V = [macrostate[k] for k in ("S", "T", "R", "V")]

        if inplace:
            state = self.state
        else:
            state = np.zeros((self.N0, 6))
        N = int(np.sum(mask))
        m = self.atom.props["m"]
        for i in range(3):
            v = (kB * T[i] / m) ** (1 / 2)
            state[mask, i] = np.random.normal(R[i], S[i], N)
            state[mask, i + 3] = np.random.normal(V[i], v, N)
        return state

    def sample_internal(self, F0, mF0, inplace=False):
        if inplace:
            internal = self.internal
        else:
            internal = np.zeros((self.N0, 3))
        Fs = self.sample_Fs(F0)
        mFs = self.sample_mFs(mF0, Fs=Fs)
        mJs = self.sample_mJs(Fs=Fs, mFs=mFs)
        for i, ternal in enumerate([Fs, mFs, mJs]):
            internal[::, i] = ternal
        return internal

    def sample_mFs(self, mF0, Fs=None):
        if Fs is None:
            Fs = self.Fs
        if mF0 == "thermal":
            mFs = np.array([np.random.choice(np.arange(-f, f + 1)) for f in Fs])
        elif mF0 == "p":
            mFs = np.array([f for f in Fs])
        elif mF0 == "m":
            mFs = np.array([-f for f in Fs])
        else:
            mFs = float(mF0) * np.ones_like(Fs)
        return mFs

    # internal state preparation
    def sample_Fs(self, F0):
        if F0 is None:
            return self.Fs

        if F0[0] in ("v", "s"):
            if len(F0) == 2:
                mode, coord = F0
            coord = "xyz".index(coord)
            if mode == "v":
                dat = self.vs
            elif mode == "s":
                dat = self.xs
            Fs = (np.sign(dat[::, coord]) + 3.0) / 2.0
        elif F0 in (1, "1", "hfs"):
            Fs = 1.0 * np.ones(self.N0)
        elif F0 in (2, "2", "LFS", "lfs"):
            Fs = 2.0 * np.ones(self.N0)
        elif F0 == "thermal":
            Fs = np.random.choice([1, 2], self.N0)
        else:
            raise ValueError("F0 {} not understood".format(F0))
        return Fs

    def sample_mJs(self, Fs=None, mFs=None):
        if Fs is None:
            Fs = self.Fs
        if mFs is None:
            mFs = self.mFs
        return np.array([self._F2J(F, mF) for F, mF in zip(Fs, mFs)])

    @staticmethod
    def _F2J(f, mf):
        if f == 1.0:
            mj = -1.0 / 2.0
        elif f == 2.0:
            mj = 1.0 / 2.0
            if mf == -2.0:
                mj = -1.0 / 2.0
        return mj

    def add_constraints(self, constraints):
        constraints = set(constraints)
        self.constraints.union(constraints)
        self._make_constraint_data()

    def check_constraints(self):
        mask = np.full(self.N0, True)
        for constraint in self.constraints:
            mask = np.logical_and(mask, constraint.check(self.xs, self.vs))
        return mask

    def _make_constraint_data(self):
        self.constraint_data = [
            (
                type(c).__name__,
                {
                    k: v.tolist() if type(v) == np.ndarray else v
                    for k, v in c.__dict__.items()
                },
            )
            for c in self.constraints
        ]

    def apply_constraints(self, nmax=100000):
        nc = len(self.constraints)
        for n in range(nmax):
            mask = self.check_constraints()
            retry_mask = np.logical_not(mask)
            self.mask = mask
            self.sample_state(mask=retry_mask, inplace=True)
            if self.N > 1:
                msg = "{} atoms pass {} constraints (target: {}, trial {}) \r".format(
                    self.N, nc, self.N0, n
                )
            else:
                msg = "{} atom passes {} constraints (target: {}, trial {}) \r".format(
                    self.N, nc, self.N0, n
                )
            sys.stdout.write(msg)
            sys.stdout.flush()

            if self.N == self.N0:
                print()
                return
        print("\nMaximum initialization iterations reached ({})".format(nmax))

    # time evolution
    def rk4(self, a, dt):
        k1 = dt * a(self.xs, self.t)
        l1 = dt * self.vs
        k2 = dt * a(self.xs + l1 / 2, self.t + dt / 2)
        l2 = dt * (self.vs + k1 / 2)
        k3 = dt * a(self.xs + l2 / 2, self.t + dt / 2)
        l3 = dt * (self.vs + k2 / 2)
        k4 = dt * a(self.xs + l3, self.t + dt)
        l4 = dt * (self.vs + k3)
        self.xs = self.xs + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        self.vs = self.vs + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt

    # time evolution (no forces)
    def free_expand(self, Dt):
        self.xs = self.xs + Dt * self.vs
        self.t += Dt

    # TODO enable and check recoils
    # def recoil(self):
    #    recoils = self.maxwell_velocity(370.47e-6, nc=3 * self.N)
    #    recoils = recoils.reshape(self.N, 3)
    #    self.vs = self.vs + recoils

    def plot_phasespace(
        self, axs=None, remove_mean=False, Nsample=10000, xs=None, vs=None
    ):
        print("Plotting phase space projections...")
        if axs is None:
            fig, axs = plt.subplots(4, 4, figsize=(7, 7.5))
        else:
            fig = plt.figure()
        if xs is None:
            xs = self.xs
        if vs is None:
            vs = self.vs
        vs = vs * 1e6  # plot vs in cm/s
        coordi = [
            [(None, None), (1, 0), (2, 1), (0, 2)],
            [(3, 4), (3, 0), (3, 1), (3, 2)],
            [(4, 5), (4, 0), (4, 1), (4, 2)],
            [(5, 3), (5, 0), (5, 1), (5, 2)],
        ]
        coords = ["$x$", "$y$", "$z$", "$v_x$", "$v_y$", "$v_z$"]
        ps = np.hstack([xs, vs])
        for i in range(4):
            for j in range(4):
                ax = axs[i, j]
                ax.set_aspect("auto")
                n, m = coordi[i][j]
                if (m, n) == (None, None):
                    ax.axis("off")
                    continue
                x = ps[:, m]
                y = ps[:, n]
                xm = np.mean(x)
                ym = np.mean(y)
                dx = 3 * np.std(x)
                dy = 3 * np.std(y)
                if remove_mean:
                    x = x - xm
                    y = y - ym
                    xm = 0.0
                    ym = 0.0
                xname = coords[m]
                yname = coords[n]
                Z, [xax, yax] = fastKDE.pdf(x, y)
                X, Y = np.meshgrid(xax, yax)
                x, y, z = [c.flatten() for c in (X, Y, Z)]
                zm = np.mean(z) / 1.5
                x[z < zm] = np.nan
                y[z < zm] = np.nan
                z[z < zm] = np.nan
                # Sort the points by density
                idx = z.argsort()
                idx = np.random.choice(idx, Nsample)
                x, y, z = x[idx], y[idx], z[idx]
                ax.set_xlim(xm - dx, xm + dx)
                ax.set_ylim(ym - dy, ym + dy)
                if i == 0:
                    ax.set_ylabel(yname)
                if i == 3:
                    ax.set_xlabel(xname)
                if j == 0:
                    ax.set_xlabel(xname)
                    ax.set_ylabel(yname)
                if i != 3:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if j != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.scatter(
                    x, y, c=z, s=20, alpha=1, rasterized=False, edgecolors="none"
                )
                ax.contour(xax, yax, Z, 4, colors="k")
        fig = plt.gcf()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        return fig, axs


class Pinhole:
    def __init__(self, r0, n, D):
        self.r0 = np.array(r0)
        self.n = np.array(n) / norm(n)
        self.D = D

    def check(self, xs, vs):
        ts = (self.r0 - xs).dot(self.n) / vs.dot(self.n)
        mask = ts > 0.0
        ts = ts[..., np.newaxis]
        xs2 = xs + vs * ts
        dist = norm(xs2 - self.r0, axis=1)
        mask = np.logical_and(mask, dist < self.D / 2.0)
        return mask


class Tag:
    def __init__(self, r0, n, dt):
        self.r0 = np.array(r0)
        self.n = np.array(n) / norm(n)
        self.dt = dt

    def check(self, xs, vs):
        ts = (self.r0 - xs).dot(self.n) / vs.dot(self.n)
        t0 = np.mean(ts)
        mask = np.logical_and(t0 - self.dt / 2.0 < ts, t0 + self.dt / 2.0 > ts)
        return mask


if __name__ == "__main__":
    # atom = Atom()

    cloud_params = dict(
        N=100,
        S=[0.25, 0.25, 0.25],
        T=[0.3, 0.3, 0.3],
        R=[0.0, 0.0, 0.0],
        V=[200 * 1e2 * 1e-6, 0.0, 0.0],
        Natoms=1e9,
        F0="thermal",
        mF0="thermal",
        constraints=[Pinhole([10.0, 0.0, 0.0], [1, 0, 0], 0.5)],
    )

    # Recalc
    cloud = Cloud(**cloud_params, recalc=True)
    print()
    cloud = Cloud(**cloud_params, recalc=False)
    print()

    print([k for k in cloud.__dict__])
    print("N", cloud.N)
    print("K", cloud.n)
    print("S", cloud.S)
    print("T", cloud.T)
    print("R", cloud.R)
    print("V", cloud.V)
    print("n", cloud.n)
    print("rho", cloud.rho)
    cloud.atom.plot_spectrum()
    # cloud.plot_phasespace()
    # plt.show()
