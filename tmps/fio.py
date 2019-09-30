#! user/bin/python3

# io.py
#
# 1) Hash a dictionary of parameters to generate a unique ID (uid).
# 2) Inherit from the IO class elsewhere to enable saving and loading of that
#    child class. Supports composition. Multiple inheritence untested.
#
#
#  By Logan Hillberry

from os import path, pardir
from copy import deepcopy
from numpy import ndarray, int64
from os import getcwd
from hashlib import sha1
import json
from pickle import dumps, loads


base_dir = path.join(getcwd(), pardir)

def hash_state(d, uid_keys):
    """
    Create a unique ID for a dict based on the values
    associated with uid_keys.
    """
    name_dict = {}
    dc = deepcopy(d)
    for k, v in dc.items():
        if k in uid_keys:
            name_dict[k] = v
    dict_el_array2list(name_dict)
    dict_el_int2float(name_dict)
    dict_key_to_string(name_dict)
    uid = sha1(json.dumps(name_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return uid


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


def dict_el_int2float(d):
    """
    Convert dict values to floats if they are ints.
    """
    for k, v in d.items():
        if type(v) in (int, int64) :
            d[k] = float(v)
        if type(v) == dict:
            dict_el_int2float(v)
        if type(v) == list:
            for i, vel in enumerate(v):
                if type(vel) == dict:
                    dict_el_int2float(vel)
                if type(vel) == int:
                    v[i] = float(vel)

def dict_key_to_string(d):
    """
    Convert dict keys to strings.
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


class IO:
    """
    Enable saving and loading of composed classes.
    """

    def __init__(self, recalc=False, nickname=None, dir=None, uid_keys=[]):
        print("enter")
        if nickname is None:
            self.nickname = self.uid(uid_keys)
        else:
            self.nickname = str(nickname)
        ext = type(self).__name__.lower()
        if dir is None:
            dir = path.join(base_dir, "data", ext)
        else:
            dir = str(dir)
        fname = path.join(dir, self.nickname) + "." + ext
        rel_fname = path.relpath(fname, base_dir)
        self._io = {'loaded': False,
                    'recalc': recalc,
                    'fname': fname,
                    'rel_fname': rel_fname,
                    'dir': dir,
                    'ext': ext}

        if not recalc:
            try:
                self.load()
                self.loaded = True
                self.recalc = False
            except FileNotFoundError:
                print(f"{ext} data not found")
                print("Generating now...")
                self.loaded = False
                self.recalc = True

    def __getattr__(self, attr):
        if attr == "__setstate__":
            raise AttributeError(attr)
        if attr in self._io:
            return self._io[attr]
        else:
            raise AttributeError(f"no attribute {attr}")

    @property
    def uid_values(self, uid_keys):
        return {k: v for k, v in self.__dict__ if k in uid_keys}

    def make_fname(self):
        return path.join(self.dir, self.nickname) + "." + self.ext

    def uid(self, uid_keys):
        return hash_state(self.__dict__, uid_keys)

    def save(self):
        print(f"Saving {self.ext} data to {self.rel_fname}")
        save_dict = deepcopy(self.__dict__)
        for k, v in self.__dict__.items():
            if IO in type(v).mro():
                del save_dict[k]
        with open(self.fname, "wb") as f:
            f.write(dumps(save_dict))

    def load(self):
        print(f"Loading {self.ext} data from {self.rel_fname}")
        with open(self.fname, "rb") as f:
            instance = f.read()
        self.__dict__.update(loads(instance))


if __name__ == "__main__":

    class Level1(IO):
        def __init__(self, params, dir=None, nickname=None, recalc=False, save=True):

            self.uid_keys = ["params"]
            self.params = params

            # adds IO funcatinality through save() and load()
            # Automatically calls load() unluess recalc = True
            # adds attributes recalc, loaded, and dir
            super().__init__(dir=dir, nickname=nickname, recalc=recalc)
            if recalc:
                self._data1 = self.make_data()

            if save and not self.loaded:
                self.save()

        @property
        def data(self):
            return self._data1

        def make_data(self):
            print(f"Making {type(self).__name__} data...")
            return sum(self.params.values())

    class Level2(IO):
        def __init__(
            self,
            params,
            level1_params,
            dir=None,
            nickname=None,
            recalc=False,
            save=True,
        ):
            self.params = params
            self.uid_keys = ["params"]
            self.level1 = Level1(level1_params, recalc=recalc)
            super().__init__(dir=dir, nickname=nickname, recalc=recalc)
            if recalc:
                self._data2 = self.create_data()

            if save and not self.loaded:
                self.save()

        @property
        def data(self):
            return self._data2

        def create_data(self):
            print("Creating {type(self).__name__} data...")
            return sum(self.params.values()) ** 2

    params1 = {"p1": 1.0, "p2": 2}
    params2 = {"p1": 10, "p2": 30.0}
    l2re = Level2(params2, params1, recalc=True)
    print("level2 loaded:")
    print(l2re.loaded)
    print("level1 loaded:")
    print(l2re.level1.loaded)
    print()
    print()
    l2 = Level2(params2, params1, recalc=False)
    print("level2 loaded:")
    print(l2.loaded)
    print("level1 loaded:")
    print(l2.level1.loaded)
    print(l2.level1.data)
    print(l2.data)
