from tmps.units import u0, pi
from tmps.fio import IO
from copy import deepcopy
from numpy.linalg import norm, inv
from scipy.special import ellipe, ellipk
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# polynomial fit to experimental current pulse
def curr_pulse(t, t0, tau, shape, scale, **kwargs):
    ps = [
        -3.416_293_843_922_860_1e-39,
        2.558_364_211_799_268_4e-35,
        -8.611_425_798_164_012_4e-32,
        1.719_516_896_498_202_6e-28,
        -2.263_221_579_299_263_5e-25,
        2.062_228_907_660_808_3e-22,
        -1.328_063_652_698_256_1e-19,
        6.047_326_783_024_095_5e-17,
        -1.905_993_655_908_489_8e-14,
        3.919_842_060_495_859_3e-12,
        -4.411_560_644_375_574_6e-10,
        3.592_871_942_104_985_8e-09,
        5.919_529_766_440_032_4e-06,
        -0.000_702_212_562_615_079_65,
        0.024_339_393_545_649_748,
        0.687_271_422_846_342_79,
    ]
    poly = np.poly1d(ps)
    shapes = {
        "sin": np.sin(pi * (t - t0) / tau),
        "square": 1.0,
        "ramp": (t - t0) / tau,
        "poly": poly(t - t0),
    }
    if t0 <= t <= t0 + tau:
        return scale * shapes[shape]
    else:
        return 0.0


# https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normalsu
def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.
    """
    sin_angle = norm(d)
    if sin_angle == 0.0:
        return np.identity(3)
    d = d / sin_angle
    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array(
        [[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64
    )
    M = ddt + np.sqrt(1 - sin_angle ** 2) * (eye - ddt) + sin_angle * skew
    return M


def local_coords(n):
    """
    Creates a coordinate system using n as new z axis
    """
    n = n / norm(n)
    if np.abs(n[2]) == 1.0:
        l = np.array([1.0, 0.0, 0.0])
    else:
        l = np.cross(n, np.array([0.0, 0.0, 1.0]))
    l = l / norm(l)
    m = np.cross(n, l)
    return l, m, n


def transform(r, r0, n, ang, func, *args, **kwargs):
    r0 = np.array(r0)
    l, m, n = local_coords(n)
    rot = rotation_matrix(n * np.sin(ang))
    l = np.dot(l, rot)
    m = np.dot(m, rot)
    trans = np.vstack((l, m, n))
    inv_trans = inv(trans)
    r = r - r0
    r = np.dot(r, inv_trans)
    f = func(r, *args, **kwargs)
    f = np.dot(f, trans)
    return f


def cyl_coords(r):
    x, y, z = r.T
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi, z


class Domain:
    def __init__(self, meshspec, spec="number"):
        if spec is "number":
            axes = [np.linspace(*gs) for gs in meshspec]
        elif spec is "delta":
            for gs in meshspec:
                gs[1] = gs[1] + gs[2]
            axes = [np.arange(*gs) for gs in meshspec]
        self.axes = axes
        self.grid = np.meshgrid(*axes, indexing="ij")

    @property
    def shapes(self):
        return [X.shape for X in self.grid]

    @property
    def deltas(self):
        return [x[1] - x[0] for x in self.axes]

    @property
    def points(self):
        return np.c_[[X.ravel() for X in self.grid]].T

    def interpolation(self, V):
        assert tuple(len(ax) for ax in self.axes) == V.shape
        interp = RegularGridInterpolator(
            self.axes, V, method="linear", bounds_error=False, fill_value=0
        )
        return interp

    def vector_interpolation(self, VXYZ):
        return tuple([self.interpolation(V) for V in VXYZ])


class Field(IO):
    def __init__(self, geometry, dir=None, nickname=None, recalc=False, save=True):
        if "ang" not in geometry:
            geometry["ang"] = 0.0
        if "d" not in geometry:
            geometry["d"] = 0.086
        if "method" not in geometry:
            geometry["method"] = "exact"
        if "meshspec" not in geometry:
            meshspec = [[-1.0, 1.0, 100]] * 3
        else:
            meshspec = geometry["meshspec"]

        geometry["n"] = np.array(geometry["n"], dtype=float)
        geometry["r0"] = np.array(geometry["r0"], dtype=float)
        if geometry["config"][0:2] == "sq":
            sq = True
            config = geometry["config"][3:]
        else:
            sq = False

        config = geometry["config"]
        sqoffset = 0
        if config[0:2] == "sq":
            sq = True
            sqoffset = 2

        self.meshspec = meshspec
        self.config = config
        self.base_args = []
        self.geometry = geometry
        getattr(self, "add_" + self.config[sqoffset:])(sq=sq, **self.geometry)

        super().__init__(
            dir=dir,
            nickname=nickname,
            recalc=recalc,
            uid_keys=["geometry"],
        )

        if self.recalc:
            self.make_interpolation(meshspec)

        if save and not self.loaded:
            self.save()

    def evaluate(self, r):
        B = np.zeros_like(r)
        for kwargs in self.base_args:
            config = kwargs["config"]
            func = getattr(self, "B" + config + "_zaxis")
            #func = getattr(self, "B" + config)
            B += transform(r, func=func, **kwargs)
        return B

    def viz(self, ax):
        for kwargs in self.base_args:
            config = kwargs["config"]
            getattr(self, config + "_viz")(ax, **kwargs)

    def make_interpolation(self, meshspec=[[-1.0, 1.0, 100]] * 3):
        domain = Domain(meshspec)
        B = self.evaluate(domain.points)
        BXYZ = [B[::, i].reshape(domain.shapes[i]) for i in range(3)]
        normBXYZ = np.sqrt(sum(BX ** 2 for BX in BXYZ))
        gradnormBXYZ = np.gradient(normBXYZ, *domain.deltas)  # kg A^-1 us^-2 cm^-1
        self.domain = domain
        self.B_interp = domain.vector_interpolation(BXYZ)
        self.normB_interp = domain.interpolation(normBXYZ)
        self.gradnormB_interp = domain.vector_interpolation(gradnormBXYZ)

    @staticmethod
    def Bline_zaxis(r, L, d, I=None, method=None, **kwargs):
        if method is "exact":
            rho, phi, z = cyl_coords(r)
            radm = 1.0 / np.sqrt(4 * rho ** 2 + (2 * z - L) ** 2)
            radp = 1.0 / np.sqrt(4 * rho ** 2 + (2 * z + L) ** 2)
            Bphi = (
                u0 * I / (4 * pi * rho ** 2) * (radm * (L - 2 * z) + radp * (L + 2 * z))
            )
            Bz = np.zeros_like(Bphi)
            Bphi[np.isnan(Bphi)] = np.nan
            Bphi[np.isinf(Bphi)] = np.nan
            B = np.c_[-np.sin(phi) * Bphi, np.cos(phi) * Bphi, Bz]
            return B

    @staticmethod
    def Bloop_zaxis(r, R=None, I=None, method="exact", **kwargs):
        rho, phi, z = cyl_coords(r)
        if method is "exact":
            E = ellipe(4 * R * rho / ((R + rho) ** 2 + z ** 2))
            K = ellipk(4 * R * rho / ((R + rho) ** 2 + z ** 2))
            Bz = (
                u0
                * I
                / (2 * pi * np.sqrt((R + rho) ** 2 + z ** 2))
                * (K + E * (R ** 2 - rho ** 2 - z ** 2) / ((R - rho) ** 2 + z ** 2))
            )

            Brho = (
                u0
                * I
                * z
                / (2 * pi * rho * np.sqrt((R + rho) ** 2 + z ** 2))
                * (-K + E * (R ** 2 + rho ** 2 + z ** 2) / ((R - rho) ** 2 + z ** 2))
            )

        if method is "approx":
            pass
        Brho[np.isnan(Brho)] = 0.0
        Brho[np.isinf(Brho)] = 0.0
        Bz[np.isnan(Bz)] = 0.0
        Bz[np.isinf(Bz)] = 0.0
        B = np.c_[np.cos(phi) * Brho, np.sin(phi) * Brho, Bz]
        return B

    def add_loop(self, sq=False, **kwargs):
        if sq:
            self.add_sqloop(**kwargs)
        else:
            base_arg = {"config": "loop"}
            base_arg.update(kwargs)
            self.base_args += [base_arg]

    def add_line(self, **kwargs):
        base_arg = {"config": "line"}
        base_arg.update(kwargs)
        self.base_args += [base_arg]

    def add_sqloop(self, r0, n, ang, d, L, W, sq=True, I=1.0, method="exact"):
        l, m, n = local_coords(n)
        rot = rotation_matrix(n * np.sin(ang))
        l = np.dot(l, rot)
        m = np.dot(m, rot)
        self.add_line(n=l, ang=ang, r0=r0 - m * W / 2, L=L, d=d, I=I, method=method)
        self.add_line(n=-l, ang=ang, r0=r0 + m * W / 2, L=L, d=d, I=I, method=method)
        self.add_line(n=-m, ang=ang, r0=r0 - l * L / 2, L=W, d=d, I=I, method=method)
        self.add_line(n=m, ang=ang, r0=r0 + l * L / 2, L=W, d=d, I=I, method=method)

    def add_hexapole(self, r0, n, d, L, a, I=1.0, method="exact", **kwargs):
        s = a * np.sin(pi / 6)
        c = a * np.cos(pi / 6)
        locs = [
            np.array([0, a, 0]),
            np.array([0, s, c]),
            np.array([0, -s, c]),
            np.array([0, -a, 0]),
            np.array([0, -s, -c]),
            np.array([0, s, -c]),
        ]

        for m, loc in enumerate(locs):
            self.add_line(n=(-1)**m*n, ang=0, r0=r0+loc, L=L, d=d, I=I, method=method)


    def add_coil(self, r0, n, ang, d, M, N, I=1.0, method="exact", sq=False, **kwargs):
        for k in range(N):
            for j in range(M):
                shift = n * (j + 1 / 2) * d
                expand = (k + 1 / 2) * d
                if sq:
                    L = kwargs["L"]
                    W = kwargs["W"]
                    self.add_sqloop(
                        r0 + shift,
                        n,
                        ang,
                        d,
                        L=L + expand,
                        W=W + expand,
                        I=I,
                        method=method,
                    )
                else:
                    R = kwargs["R"]
                    self.add_loop(
                        r0=r0 + shift,
                        n=n,
                        ang=ang,
                        d=d,
                        R=R + expand,
                        I=I,
                        method=method,
                    )

    def add_AH(self, r0, n, ang, d, M, N, A, I=1.0, method="exact", sq=False, **kwargs):
        r0a = r0 + n * A
        r0b = r0 - n * A
        if not sq:
            ang = pi / 2.0 - ang
        self.add_coil(r0a, n, ang, d, M, N, I=I, method=method, sq=sq, **kwargs)
        self.add_coil(r0b, -n, -ang, d, M, N, I=I, method=method, sq=sq, **kwargs)

    def add_HH(self, r0, n, ang, d, M, N, A, I=1.0, method="exact", sq=False, **kwargs):
        r0a = r0 + n * A
        r0b = r0 - n * A
        if not sq:
            ang = pi / 2.0 - ang
        self.add_coil(r0a, n, ang, d, M, N, I=I, method=method, sq=sq, **kwargs)
        self.add_coil(r0b, -n, -ang, d, M, N, I=-I, method=method, sq=sq, **kwargs)

    def add_mop(
        self,
        r0,
        n,
        ang,
        d,
        M,
        N,
        AAH,
        AHH,
        IAH=1.0,
        IHH=1.0,
        method="exact",
        sq=False,
        **kwargs,
    ):

        HH_kwargs = deepcopy(kwargs)
        AH_kwargs = deepcopy(kwargs)
        if sq:
            L = HH_kwargs.pop("LHH")
            W = HH_kwargs.pop("WHH")
            HH_kwargs["L"] = L
            HH_kwargs["W"] = W
            L = AH_kwargs.pop("LAH")
            W = AH_kwargs.pop("WAH")
            AH_kwargs["L"] = L
            AH_kwargs["W"] = W
        else:
            R = HH_kwargs.pop("RHH")
            HH_kwargs["R"] = R
            R = AH_kwargs.pop("RAH")
            AH_kwargs["R"] = R
        self.add_HH(r0, n, ang, d, M, N, AHH, I=IHH, method=method, sq=sq, **HH_kwargs)
        self.add_AH(r0, n, ang, d, M, N, AAH, I=IAH, method=method, sq=sq, **AH_kwargs)

    def add_mop3(
        self,
        r0,
        n,
        ang,
        d,
        M,
        N,
        A1,
        A2,
        I2=1.0,
        I1=1.0,
        method="exact",
        sq=False,
        **kwargs,
    ):

        HH_kwargs = deepcopy(kwargs)
        C_kwargs = deepcopy(kwargs)
        if sq:
            L = HH_kwargs.pop("L2")
            W = HH_kwargs.pop("W2")
            HH_kwargs["L"] = L
            HH_kwargs["W"] = W
            L = C_kwargs.pop("L1")
            W = C_kwargs.pop("W1")
            C_kwargs["L"] = L
            C_kwargs["W"] = W
        else:
            R = HH_kwargs.pop("R2")
            HH_kwargs["R"] = R
            R = C_kwargs.pop("R1")
            C_kwargs["R"] = R
        self.add_HH(r0, n, ang, d, M, N, A2, I=I2, method=method, sq=sq, **HH_kwargs)
        if I1 >= 0:
            r0a = r0 + n * A1
            self.add_coil(
                r0a, n, ang, d, M, N, I=I1, method=method, sq=sq, **C_kwargs
            )
        elif I1 < 0:
            r0b = r0 - n * A1
            self.add_coil(
                r0b, -n, -ang, d, M, N, I=-I1, method=method, sq=sq, **C_kwargs
            )

    def line_viz(self, ax, n, r0, L, d, color="k", **kwargs):
        a = [[0, -L / 2], [d / 2, -L / 2], [d / 2, L / 2], [0, L / 2]]
        a = np.array(a)
        r, theta = np.meshgrid(a[:, 0], np.linspace(0, 2 * np.pi, 30))
        z = np.tile(a[:, 1], r.shape[0]).reshape(r.shape)
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        l, m, n = local_coords(n)
        trans = np.vstack((l, m, n))
        rxyz = np.c_[x.ravel(), y.ravel(), z.ravel()]
        rxyz = np.dot(rxyz, trans)
        line_data = rxyz + r0
        x = line_data[:, 0].reshape(r.shape)
        y = line_data[:, 1].reshape(r.shape)
        z = line_data[:, 2].reshape(r.shape)
        ax.plot_surface(x, y, z, color=color)

    def loop_viz(self, ax, n, ang, r0, d, R, color="k", **kwargs):
        angle = np.linspace(0, pi, 16)
        theta, phi = np.meshgrid(3 * 2 * angle / 4, 2 * angle)
        x = (R + d / 2.0 * np.cos(phi)) * np.cos(theta)
        y = (R + d / 2.0 * np.cos(phi)) * np.sin(theta)
        z = d / 2.0 * np.sin(phi)
        l, m, n = local_coords(n)
        rot = rotation_matrix(n * np.sin(ang))
        l = np.dot(l, rot)
        m = np.dot(m, rot)
        trans = np.vstack((l, m, n))
        rxyz = np.c_[x.ravel(), y.ravel(), z.ravel()]
        rxyz = np.dot(rxyz, trans)
        loop_data = rxyz + r0
        x = loop_data[:, 0].reshape(theta.shape)
        y = loop_data[:, 1].reshape(theta.shape)
        z = loop_data[:, 2].reshape(theta.shape)
        ax.plot_surface(x, y, z, color=color)

    def plot_3d(self, axs=None, grad_norm=False, skip=10, Bclip=3.0, **kwargs):
        r = self.domain.points
        X, Y, Z = self.domain.grid

        fig = plt.gcf()
        if axs is None:
            fig = plt.figure(figsize=(7, 7))
            axs = fig.add_subplot(1, 1, 1, projection="3d")

        if grad_norm:
            print("Plotting 3D gradient vectors...")
            interps = self.gradnormB_interp
            title = r"$\nabla B$"
        else:
            print("Plotting 3D field vectors...")
            interps = self.B_interp
            title = r"$\bf{B}$"

        VX, VY, VZ = [1e12 * interp(r) for interp in interps]
        VX.shape = X.shape
        VY.shape = Y.shape
        VZ.shape = Z.shape
        B = np.sqrt(VX ** 2 + VY ** 2 + VZ ** 2)
        mask = B > Bclip
        print(sum(mask))
        B[mask] = np.nan
        VX[mask] = np.nan
        VY[mask] = np.nan
        VZ[mask] = np.nan
        skip_s = (slice(0, -1, skip), slice(0, -1, skip), slice(0, -1, skip))
        axs.quiver(
            X[skip_s],
            Y[skip_s],
            Z[skip_s],
            VX[skip_s],
            VY[skip_s],
            VZ[skip_s],
            length=0.6,
            pivot="middle",
            lw=0.5,
            color="k",
        )
        axs.set_ylabel("y [cm]")
        axs.set_xlabel("x [cm]")
        axs.set_zlabel("z [cm]")
        axs.set_title(title)
        self.viz(axs)
        return fig, axs

    def plot_slices(
        self,
        contours=4,
        label_contours=True,
        arrows=True,
        skip=10,
        planes={"x": [-0.5, 0.0, 0.5], "y": [-0.5, 0.0, 0.5], "z": [-0.5, 0.0, 0.5]},
        grad_norm=False,
        Bclip=3.0,
        **kwargs,
    ):

        xyz = self.domain.axes
        Xshape, Yshape, Zshape = self.domain.shapes
        if grad_norm:
            print("Plotting 2D contour slice of gradient...")
            interps = self.gradnormB_interp
            title = r"$\nabla B$"
        else:
            print("Plotting 2D contour slice of field...")
            interps = self.B_interp
            title = r"$\bf{B}$"

        consts = ["xyz".index(k) for k in planes.keys()]
        vars = ["xyz"[:c] + "xyz"[c + 1 :] for c in consts]
        vars = [["xyz".index(i) for i in vs] for vs in vars]

        nrow = len(planes)
        ncol = max([len(p) for p in planes.values()])
        fig, axs = plt.subplots(nrow, ncol, sharex=False, sharey="row", figsize=(7, 8))
        for i, (c, v) in enumerate(zip(consts, vars)):
            ordered_xyz = [0, 0, 0]
            ordered_xyz[v[0]] = xyz[v[0]]
            ordered_xyz[v[1]] = xyz[v[1]]
            consts = planes["xyz"[c]]
            for j, const in enumerate(consts):
                ordered_xyz[c] = const
                if nrow > 1 and ncol > 1:
                    ax = axs[i, j]
                elif nrow > 1 or ncol > 1:
                    ax = axs[max(i, j)]
                else:
                    ax = axs
                X, Y, Z = np.meshgrid(*ordered_xyz, indexing="ij")
                XYZ = [X, Y, Z]
                r = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
                VX, VY, VZ = [1e12 * interp(r) for interp in interps]
                VX.shape = X.shape
                VY.shape = Y.shape
                VZ.shape = Z.shape
                VXYZ = [VX, VY, VZ]
                B = np.sqrt(VX ** 2 + VY ** 2 + VZ ** 2)
                mask = B > Bclip
                B[mask] = np.nan
                VX[mask] = np.nan
                VY[mask] = np.nan
                VZ[mask] = np.nan
                s = [slice(None, None, None)] * 3
                skip_s = [slice(0, -1, skip), slice(0, -1, skip), slice(0, -1, skip)]
                for ii, n in enumerate(B.shape):
                    if n == 1:
                        s[ii] = 0
                        skip_s[ii] = 0
                s = tuple(s)
                skip_s = tuple(skip_s)
                x = xyz[v[0]]
                y = xyz[v[1]]
                z = B[s].T
                ax.imshow(
                    z,
                    origin="lower",
                    interpolation="None",
                    extent=[x[0], x[-1], y[0], y[-1]],
                )

                if contours:
                    cp = ax.contour(x, y, z, contours, colors="k", **kwargs)
                if contours and label_contours:
                    ax.clabel(cp, inline=1, fontsize=11, fmt="%1.2f")
                if arrows:
                    ax.quiver(
                        XYZ[v[0]][skip_s],
                        XYZ[v[1]][skip_s],
                        VXYZ[v[0]][skip_s],
                        VXYZ[v[1]][skip_s],
                        pivot="middle",
                        lw=0.5,
                        color="k",
                    )
                ax.set_xlabel("xyz"[v[0]] + " [cm]")
                ax.set_title("$%s = %.2f$" % ("xyz"[c], const), fontsize=12)
                if j == 0:
                    ax.set_ylabel("xyz"[v[1]] + " [cm]")
        fig.tight_layout()
        fig.suptitle(title)
        plt.subplots_adjust(
            hspace=0.5, wspace=0.1, top=0.93, bottom=0.08, left=0.10, right=0.9
        )
        return fig, axs

    def plot_linecut(
        self,
        axs=None,
        component="z",
        line=[0.0, 0.0, "var"],
        grad_norm=False,
        legend=True,
        **plot_kwargs,
    ):
        if axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(7, 3))

        else:
            fig = plt.gcf()

        if grad_norm:
            interps = self.gradnormB_interp
            ylabel = r"$\partial B / \partial %s$ [T/cm]"
        else:
            interps = self.B_interp
            ylabel = "$B_%s$ [T]"

        comp = "xyz".index(component)
        interp = interps[comp]
        xyz = self.domain.axes
        vari = line.index("var")
        vars = xyz[vari]
        label = ""
        points = np.zeros((len(vars), 3))
        for k, v in zip("xyz", line):
            idx = "xyz".index(k)
            if v == "var":
                points[::, idx] = vars
            else:
                points[::, idx] = v
                label += " ${} = {}$".format(k, v)

        axs.plot(vars, 1e12 * interp(points), label=label, **plot_kwargs)
        if legend:
            axs.legend(fontsize=10)
        axs.set_ylabel(ylabel % "xyz"[comp])
        axs.set_xlabel(r"$%s$ [cm]" % "xyz"[vari])
        axs.label_outer()
        return fig, axs

    def plot_linecuts(
        self,
        components="xyz",
        lines=[[0.0, 0.0, "var"], [0.0, "var", 0.0], ["var", 0.0, 0.0]],
        grad_norm=False,
    ):
        varinds = [[], [], []]
        for i, line in enumerate(lines):
            varinds[line.index("var")] += [i]
        varinds = [vinds for vinds in varinds if vinds != []]
        nrow = len(components)
        ncol = len(varinds)
        fig, axs = plt.subplots(nrow, ncol, sharex="col", sharey="row", figsize=(7, 7))
        for comp in components:
            i = components.index(comp)
            for j, vinds in enumerate(varinds):
                if nrow > 1 and ncol > 1:
                    ax = axs[i, j]
                elif nrow > 1 or ncol > 1:
                    ax = axs[max(i, j)]
                else:
                    ax = axs
                for idx in vinds:
                    line = lines[idx]
                    self.plot_linecut(
                        axs=ax, component=comp, line=lines[idx], grad_norm=grad_norm
                    )
        return fig, axs


if __name__ == "__main__":

    geometries = [
        dict(config="loop", I=1000, R=1.0, n=[0.0, 1.0, 1.0], r0=[0.0, 0.0, 0.0]),
        dict(config="line", I=1000, L=1.0, n=[0.0, 0.0, 1.0], r0=[0.0, 0.0, 0.0]),
        dict(
            config="sqloop",
            I=1000,
            L=1.5,
            W=1.5,
            d=0.086,
            ang=pi / 3.5,
            n=[0.0, 1.0, 1.0],
            r0=[0.0, 0.0, 0.0],
        ),
        dict(
            config="coil",
            I=1000,
            R=1.5,
            d=0.086,
            M=5,
            N=2,
            n=[0.0, 0.0, 1.0],
            r0=[0.0, 0.0, 0.0],
        ),
        dict(
            config="sqcoil",
            I=1000,
            L=1.5,
            W=1.5,
            d=0.086,
            M=5,
            N=2,
            ang=0.0,
            n=[0.0, 0.0, 1.0],
            r0=[0.0, 0.0, 0.0],
        ),
        dict(
            config="HH",
            I=1000,
            R=1.5,
            d=0.086,
            M=5,
            N=2,
            A=1.0,
            n=[0.0, 0.0, 1.0],
            r0=[0.0, 0.0, 0.0],
        ),
        dict(
            config="sqHH",
            I=1000,
            L=1.5,
            W=1.5,
            ang=pi / 4,
            d=0.086,
            M=5,
            N=2,
            A=1.0,
            n=[0.0, 0.0, 1.0],
            r0=[0.0, 0.0, 0.0],
        ),
        dict(
            config="AH",
            I=1000,
            R=1.5,
            d=0.086,
            M=5,
            N=2,
            A=1.0,
            n=[0.0, 0.0, 1.0],
            r0=[0.0, 0.0, 0.0],
        ),
        dict(
            config="sqAH",
            I=1000,
            L=1.5,
            W=1.5,
            ang=pi / 4,
            d=0.086,
            M=5,
            N=2,
            A=1.0,
            n=[0.0, 0.0, 1.0],
            r0=[0.0, 0.0, 0.0],
        ),
        dict(
            config="mop",
            IHH=1000,
            IAH=800,
            RAH=3.33,
            RHH=3.92,
            AAH=1.81,
            AHH=1.68,
            d=0.086,
            M=5,
            N=2,
            n=[0.0, 0.0, 1.0],
            r0=[0.0, 0.0, 0.0],
            meshspec=[[-3.0, 3.0, 100]] * 3,
        ),
        dict(
            config="hexapole",
            I=1000,
            d=0.086,
            a=1,
            L=8,
            n=[1.0, 0.0, 0.0],
            r0=[0.0, 0.0, 0.0],
            meshspec=[[-0.5, 0.5, 100]] * 3,
        ),
    ]

    for i, geometry in enumerate(geometries[-1:]):
        print(i, geometry["config"])
        field = Field(geometry, recalc=True, save=False)
        field.plot_linecuts(grad_norm=True, components="xz", lines=[[0, 0, "var"],[0.1, 0.1, "var"]])

        #field.plot_slices()
        #field.plot_3d()
        plt.show()
        plt.clf()
        print(field.normB_interp(np.array([0, 0, 0])))
