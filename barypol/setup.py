#!/usr/bin/env python

import sys
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

Name = "barypol"
if len(sys.argv) == 2:
    Name = sys.argv.pop()
if len(sys.argv) == 1:
    sys.argv += "build_ext --inplace".split()

ext_modules = [Extension(
    name=Name,
    sources=[ Name + ".pyx" ],  # xx.cpp
    include_dirs=[ "h/", np.get_include() ],
    language="c++",
        extra_compile_args = "-Wno-format".split(),  # %d long
        # extra_link_args = "...".split()
        # libraries=
    )]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    # install_requires=['cython'],
    )
