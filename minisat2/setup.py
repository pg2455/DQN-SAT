import os

from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext

MINISAT_SOURCE_ROOT = "/Users/pgupta/Workspace/SATConversion/explore/graph_sat/minisat2"
print __file__
include_dirs = []
include_dirs += ['/usr/include', '/usr/local/include']
include_dirs.append(MINISAT_SOURCE_ROOT)
include_dirs += filter(os.path.isdir,
                       [ os.path.join(MINISAT_SOURCE_ROOT, d)
                         for d in os.listdir(MINISAT_SOURCE_ROOT) ])



# library_dirs = ['/usr/local/lib']

ext = Extension("wrapper", sources=["wrapper.pyx", "core/Solver.cc" ], include_dirs= include_dirs ,  language="c++")
setup(ext_modules = [ext], cmdclass = {'build_ext':build_ext})
