from os.path import basename, isfile
import glob

from src.fio import base_dir

modules = glob.glob(base_dir + "/src/*.py")
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
