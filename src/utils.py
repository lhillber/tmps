#! user/bin/python3
from copy import deepcopy
from numpy import ndarray
from hashlib import sha1
from json import dumps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from inspect import signature


# base directory of project
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

class TaggableType(type):
    def __init__(cls, name, bases, attrs):
        tagdata = {}
        for name, method in attrs.items():
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, "tagged"):
                tagdata[name] = method.args

        @property
        def tagged(self):
            tags = tagdata.copy()
            try:
                #tags.update(super(cls, self).tagged)
                pass
            except AttributeError:
                pass
            return tags

        cls.tagged = tagged

def tag(*args):
    def decorator(f):
        if isinstance(f, property):
            f.fget.tagged = True
            f.fget.args = args
        else:
            f.tagged = True
            f.args = args
        return f
    return decorator


class Parent(metaclass=TaggableType):
    def parent1(self):
        return "parent1"

    @tag()
    @property
    def parent2(self):
        return "parent2"


class Child1(Parent):
    @tag()
    @property
    def child1_1(self):
        return "return child1_1"

    @property
    @tag()
    def child1_2(self):
        return "return child1_2"


class Child2(Parent):
    @property
    def child2_1(self):
        return "return child2_1"

    @tag(1,2)
    def child2_2(self):
        return "return child2_2"

#p = Parent()
#c1 = Child1()
#c2 = Child2()
#print(p.tagged)
#
#print(c1.child1_1)
#print(c1.child1_2)
#print(c2.child2_1)
#print(c2.child2_2())
#print(c1.tagged)
#print(c2.tagged)


class IO:
    def save(self, fname, data):
        print('saving {}, {} to {}'.format(type(self).__name__, data, fname))

    def load(self, fname):
        print('loading {} from {}'.format(type(self).__name__, fname))
        print('saving {}'.format(fname))

class Parent(IO):
    def __init__(self, p1, p2, p3):
        self.pa = p1
        self.data = p2 + p3
        self.save(self.fname(), self.data)

    def fname(self):
        return str(self.data)+'.'+type(self).__name__


class Child(IO):
    def __init__(self, c1, c2, *args, **kwargs):
        self.parent = Parent(*args, *kwargs)
        self.data = c1
        self.cb = c2
        self.parent.data = 100
        self.parent.save(self.parent.fname(), self.parent.data)
        self.save(self.fname(), self.data)

    def fname(self):
        return str(self.data)+'.'+type(self).__name__

# parent = Parent(10, 10, 10)
# print()
# child = Child(2, 2, 10, 10, 10)


# class MetaRegistry(type):
#    def __init__(mcl):
#        mcl.registry = {}
#
#    @classmethod
#    def __prepare__(mcl, name, bases):
#        print(mcl.__dict__)
#        def register(*props):
#            def deco(f):
#                mcl.registry[name + "." + f.__name__] = props
#                return f
#            return deco
#        d = dict()
#        d['register'] = register
#        return d
#
#    def __new__(mcl, name, bases, dct):
#        del dct['register']
#        cls = super().__new__(mcl, name, bases, dct)
#        return cls
#
# def register(*args):
#    def decorator(f):
#        f.register = tuple(args)
#        return f
#    return decorator
#
# class my_class(object, metaclass=MetaRegistry):
#    @register('prop1','prop2')
#    def my_method( arg1,arg2 ):
#       pass # method code here...
#
#    @register('prop3','prop4')
#    def my_other_method( arg1,arg2 ):
#       pass # method code here...
#
# print(registry)


class Property(object):
    "Emulate PyProperty_Type() in Objects/descrobject.c"

    def __init__(self, fget=None, fset=None, fupdate=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fupdate = fupdate
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fupdate, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fupdate, self.__doc__)

    def updater(self, fupdate):
        return type(self)(self.fget, fset, self.fupdate, self.__doc__)


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


class Saver:
    def __init__(self, inst):
        self.__dict__
        self.include_keys
        self.exclude_keys


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


def dict_el_array2list(d):
    """
    Convert dict values to lists if they are arrays.
    """
    for k, v in d.items():
        if type(v) == ndarray:
            d[k] = list(v)
        if type(v) == dict:
            dict_el_array2list(v)
        if type(v) == list:
            for i, vel in enumerate(v):
                if type(vel) == dict:
                    dict_el_array2list(vel)
                if type(vel) == ndarray:
                    v[i] = list(vel)

def dict_key_to_string(d):
    """
    Convert dict values to lists if they are arrays.
    """
    for k, v in d.items():
        d[str(k)] = v
        if type(k) != str:
            del d[k]
        if type(v) == dict:
            dict_key_to_string(v)
        if type(v) == list:
            for vel in v:
                if type(vel) == dict:
                    dict_key_to_string(vel)

def hash_state(d, include_keys):
    """
    Create a unique ID for a d based on the values
    associated with include_keys.
    """
    print()
    name_dict = {}
    dc = deepcopy(d)
    for k, v in dc.items():
        if k in include_keys:
            name_dict[str(k)] = v
    dict_el_array2list(name_dict)
    dict_key_to_string(name_dict)
    uid = sha1(dumps(name_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return uid


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
