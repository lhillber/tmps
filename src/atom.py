# atom.py

from src.fio import IO
from src.units import h, muB, Li7_props

import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import eigvals
from sympy.physics.wigner import clebsch_gordan
import matplotlib.pyplot as plt


class Atom(IO):
    def __init__(
        self,
        props=Li7_props,
        L=0.0,
        J=1.0 / 2.0,
        Bmin=0,
        Bmax=2 * 1e-12,
        NB=300,
        dir=None,
        nickname="Li7_B_0-2-300",
        recalc=False,
        save=True,
    ):
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.NB = NB
        self.props = props

        super().__init__(
            dir=dir,
            nickname=nickname,
            recalc=recalc,
            uid_keys=["props", "Bmin", "Bmax", "NB"])

        if self.recalc:
            Es, Bs = self.make_E_interp(L=L, J=J, Bmin=Bmin, Bmax=Bmax, NB=NB)
            self.make_dEdB_interp(Es, Bs)
            print(self.Efuncs)

        if save and not self.loaded:
            self.save()

    def hyper_fine_zeeman(self, B, L, J, gI=-0.001182213):
        a = self.props[L][J]["a"]
        b = self.props[L][J]["b"]
        gJ = self.props[L][J]["gJ"]
        I = self.props["I"]
        F = np.arange(np.abs(I - J), I + J + 1)
        # c basis: coupled states
        cstates = np.array([[f, mf] for f in F for mf in np.arange(-f, f + 1)])
        # u basis: uncoupled states
        ustates = np.array(
            [[mj, mi] for mj in np.arange(-J, J + 1) for mi in np.arange(-I, I + 1)]
        )
        # number of states
        n = int((2 * J + 1) * (2 * I + 1))
        # hfs proportional to I.J, easy in c basis
        IdotJ = (
            1.0
            / 2.0
            * np.array(
                [(f * (f + 1) - J * (J + 1) - I * (I + 1)) for f in cstates[::, 0]]
            )
        )
        # CG's to go from c to u basis
        CG = np.array(
            [
                [clebsch_gordan(J, I, f, mj, mi, mf).n() for f, mf in cstates]
                for mj, mi in ustates
            ]
        )
        # I.J in u basis.
        IdotJij = np.array([[IdotJ.dot(CGi * CGj) for CGi in CG] for CGj in CG])
        # Build hyper fine + zeeman hamiltonian
        H = np.zeros((n, n))
        for M, (mj, mi) in enumerate(ustates):
            for N, (mjp, mip) in enumerate(ustates):
                Hmn = h * a * IdotJij[M, N]
                if b is not None:
                    Hmn += (
                        h
                        * b
                        * 3
                        * IdotJij[M, N]
                        * (IdotJij[M, N] + 1)
                        / (2 * I * (2 * I - 1) * 2 * J * (2 * J - 1))
                    )
                if M == N:
                    Hmn += +(gJ * mj + gI * mi) * muB * B
                H[M, N] = Hmn
        E = sorted(eigvals(H))
        return E

    # diagonalize hamiltonian to get energy
    def make_E(self, L, J, Bmin, Bmax, NB):
        I = self.props["I"]
        Bs = np.linspace(Bmin, Bmax, NB)
        n = int((2 * J + 1) * (2 * I + 1))
        Es = np.zeros((len(Bs), n))
        for i, B in enumerate(Bs):
            E = self.hyper_fine_zeeman(B, L, J)
            Es[i, ::] = E
        return Bs, Es

    def make_dEdB_interp(self, Es, Bs):
        print("Calculating energy-field derivatives...")
        dB = Bs[1] - Bs[0]
        N = Es.shape[1]
        dEdBs = np.array([np.gradient(Es[::, i], dB) for i in range(N)]).T
        self.dEdBfuncs = {
            f: {
                mf: interp1d(Bs, dEdBs[::, i * (int(2 * (f - 1) + 1)) + j], kind=2)
                for j, mf in enumerate(np.arange(-f, f + 1, 1.0))
            }
            for i, f in enumerate([1.0, 2.0])
        }

    def make_E_interp(self, L=0.0, J=1.0 / 2.0, Bmin=0, Bmax=2 * 1e-12, NB=300):
        print("Calculating energies in magnetic fields...")
        I = self.props["I"]
        Bs, Es = self.make_E(L, J, Bmin, Bmax, NB)
        self.Efuncs = {
            f: {
                mf: interp1d(Bs, Es[::, i * (int(2 * (f - 1) + 1)) + j], kind=2)
                for j, mf in enumerate(np.arange(-f, f + 1, 1.0))
            }
            for i, f in enumerate([1.0, 2.0])
        }
        return Es, Bs

    def plot_spectrum(self, axs=None, Bmin=0.0, Bmax=0.1 * 1e-12, NB=100, logx=False):
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(7, 7))
        else:
            fig = plt.gcf()
        Efuncs = self.Efuncs
        dEdBfuncs = self.dEdBfuncs
        Bs = np.linspace(Bmin, Bmax, NB)
        ax1, ax2 = axs
        for f, c in zip([1.0, 2.0], ["r", "k"]):
            for mf in np.arange(-f, f + 1, 1.0):
                if logx:
                    ax1.semilogx(Bs * 1e12, Efuncs[f][mf](Bs) / h, c=c)
                else:
                    ax1.plot(Bs * 1e12, Efuncs[f][mf](Bs) / h, c=c)
        for f, c in zip([1.0, 2.0], ["r", "k"]):
            for mf in np.arange(-f, f + 1, 1.0):
                if logx:
                    ax2.semilogx(Bs * 1e12, 1e-12 * dEdBfuncs[f][mf](Bs) / h, c=c)
                else:
                    ax2.plot(Bs * 1e12, 1e-12 * dEdBfuncs[f][mf](Bs) / h, c=c)
        ax2.axhline(1e-12 * muB / h)
        ax2.axhline(-1e-12 * muB / h)
        ax1.set_xlabel("B [T]")
        ax1.set_ylabel("frequency shift [MHz]")
        ax2.set_xlabel("B [T]")
        ax2.set_ylabel("derivative [MHz / T]")
        return fig, axs


if __name__ == "__main__":
    atom = Atom(Li7_props, recalc=True)
    print()
    atom = Atom(Li7_props, recalc=False)
    atom.plot_spectrum()
    plt.show()
