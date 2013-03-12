# intergrid/test/util.py

from __future__ import division
import numpy as np

try:
    from bz.etc.numpyutil import str2g, str3g
except ImportError:
    str2g = str3g = str

def avmaxdiff( interpol, exact, query_points ):
    absdiff = np.fabs( interpol - exact )
    av = absdiff.mean()
    jmax = absdiff.argmax()  # flat
    print "av %.2g  max %.2g = %.3g - %.3g  at %s" % (
        av, absdiff[jmax], interpol[jmax], exact[jmax], query_points[jmax] )

