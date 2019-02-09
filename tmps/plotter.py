#! user/bin/python3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from tmps.fio import base_dir


def multipage(fname, fignums=None, **kwargs):
    """
    Save multi page pdfs, one figure per page
    """
    pp = PdfPages(fname)
    if fignums is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    else:
        figs = [plt.figure(fignum) for fignum in fignums]
    for fig in figs:
        fig.savefig(pp, format="pdf", **kwargs)
    pp.close()


class Plotter:
    def __init__(self, fignum=1):
        plt.close("all")
        self.fignum = fignum
        self.figs = {}

    def add(self, figaxs):
        fig, axs = figaxs
        self.figs[self.fignum] = fig
        self.fignum += 1
        return fig, axs

    def add_many(self, itter):
        for figaxs in itter:
            self.add(figaxs)

    def remove(self, fignum):
        fig, axs = self.figs.pop(fignum)
        return fig, axs

    def show(self, fignm):
        self.figs[self.fignum].show()

    def save(self, fname, fignums=None, **kwargs):
        print("Saving plots...")
        if fignums is None:
            fignums = self.figs.keys()
        multipage(fname, fignums=fignums, **kwargs)
        rel_fname = os.path.relpath(fname, base_dir)
        print("\n Plots saved to {}".format(rel_fname))


def units_map(param, mm=False):
    """
    Units of parameters
    """
    L = "cm"
    if mm:
        L = "mm"
    if param in ("temps", "Tl", "Tt", "T"):
        unit = " [mK]"
    elif param[0] == "I":
        unit = " [A]"
    elif (
        param[:2] == "r0"
        or param[0] in ("L", "W", "R", "A")
        or param in ("centers", "sigmas", "D_ph", "width", "d")
    ):
        unit = " [" + L + "]"
    elif param[:2] in ("dt", "t0") or param in ("tcharge", "delay", "tmax", "tau"):
        unit = r" [$\mathrm{\mu s}$]"
    elif param in ("v0", "vrecoil"):
        unit = r" [$\mathrm{" + L + "~\mu s^{-1}}$]"
    elif param in ("meanKs", "thermKs", "kinetics"):
        unit = r" [$\mathrm{kg~" + L + "^2~\mu s^{-2}}$]"
    elif param in ("ts", "t", "times", "time"):
        unit = r" [$\mathrm{\mu s}$]"
    else:
        unit = ""
    return unit
