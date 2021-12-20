'''
Cross sections from FONLL
'''
import numpy as np
from coffea import hist
from coffea import lookup_tools
from coffea import util
import coffea.processor as processor
import time
from pprint import pprint
import json
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

# Job started on: Thu Nov 18 13:15:13 CET 2021 .
# FONLL heavy quark hadroproduction cross section, calculated on Thu Nov 18 13:15:15 CET 2021
# FONLL version and perturbative order: ## FONLL v1.3.4 fonll [ds/dpt^2dy (pb/GeV^2)]
# quark = bottom
# final state = meson. NP params (cm,lm,hm) = 24.2, 26.7, 22.2
# BR(q->meson) = 1
# ebeam1 = 6500, ebeam2 = 6500
# PDF set = NNPDF30_nlo_as_0118
# ptmin = 0.5
# ptmax = 45.5
# ymin  = -2.25
# ymax  = 2.25
# cross section is ds/dpt (pb/GeV)
# pt      central
fonll_pt = {
   0.5000: 1.0802e+07,
   1.5000: 2.6698e+07,
   2.5000: 3.3479e+07,
   3.5000: 3.3319e+07,
   4.5000: 2.9244e+07,
   5.5000: 2.3837e+07,
   6.5000: 1.8660e+07,
   7.5000: 1.4335e+07,
   8.5000: 1.0948e+07,
   9.5000: 8.3724e+06,
  10.5000: 6.4375e+06,
  11.5000: 4.9869e+06,
  12.5000: 3.8958e+06,
  13.5000: 3.0686e+06,
  14.5000: 2.4384e+06,
  15.5000: 1.9535e+06,
  16.5000: 1.5767e+06,
  17.5000: 1.2820e+06,
  18.5000: 1.0498e+06,
  19.5000: 8.6522e+05,
  20.5000: 7.1746e+05,
  21.5000: 5.9840e+05,
  22.5000: 5.0185e+05,
  23.5000: 4.2304e+05,
  24.5000: 3.5834e+05,
  25.5000: 3.0492e+05,
  26.5000: 2.6060e+05,
  27.5000: 2.2364e+05,
  28.5000: 1.9269e+05,
  29.5000: 1.6664e+05,
  30.5000: 1.4463e+05,
  31.5000: 1.2594e+05,
  32.5000: 1.1002e+05,
  33.5000: 9.6415e+04,
  34.5000: 8.4740e+04,
  35.5000: 7.4691e+04,
  36.5000: 6.6012e+04,
  37.5000: 5.8493e+04,
  38.5000: 5.1958e+04,
  39.5000: 4.6263e+04,
  40.5000: 4.1285e+04,
  41.5000: 3.6924e+04,
  42.5000: 3.3094e+04,
  43.5000: 2.9722e+04,
  44.5000: 2.6746e+04,
  45.5000: 2.4114e+04,
}

# Job started on: Mon May 24 22:18:20 CEST 2021 .
# FONLL heavy quark hadroproduction cross section, calculated on Mon May 24 22:18:21 CEST 2021
# FONLL version and perturbative order: ## FONLL v1.3.4 fonll [ds/dpt^2dy (pb/GeV^2)]
# quark = bottom
# final state = meson. NP params (cm,lm,hm) = 24.2, 26.7, 22.2
# BR(q->meson) = 1
# ebeam1 = 6500, ebeam2 = 6500
# PDF set = NNPDF30_nlo_as_0118
# ptmin = 8
# ptmax = 45
# ymin  = 0.125
# ymax  = 2.125
# cross section is ds/dy (pb)
# y       central
fonll_y = {
   0.1250: 1.2670e+07, 
   0.3750: 1.2580e+07, 
   0.6250: 1.2400e+07, 
   0.8750: 1.2140e+07, 
   1.1250: 1.1780e+07, 
   1.3750: 1.1350e+07, 
   1.6250: 1.0840e+07, 
   1.8750: 1.0250e+07, 
   2.1250: 9.6000e+06, 
}


# Make splines
fonll_pt_x =  np.fromiter(fonll_pt.keys(), dtype='float')
fonll_pt_x = np.sort(fonll_pt_x)
fonll_pt_y = np.array([fonll_pt[x] for x in fonll_pt_x])
fonll_pt_spline = UnivariateSpline(fonll_pt_x, fonll_pt_y)

fonll_y_x =  np.fromiter(fonll_y.keys(), dtype='float')
fonll_y_x = np.sort(fonll_y_x)
fonll_y_y = np.array([fonll_y[x] for x in fonll_y_x])
fonll_y_spline = UnivariateSpline(fonll_y_x, fonll_y_y)

# Functions to get integrals
def fonll_pt_integral(lo, hi):
    return fonll_pt_spline.integral(lo, hi)

def fonll_y_integral(lo, hi):
    return fonll_pt_spline.integral(lo, hi)

def fonll_absy_integral(lo, hi):
    if lo < 0 or hi < 0:
        raise ValueError(f"In fonll_absy_integral(), lo and hi should be nonnegative ({lo}, {hi})")
    return 2 * fonll_pt_spline.integral(lo, hi)

# Functions to get barycenters
def fonll_pt_barycenter(lo, hi):
    def integrand(x):
        return x * fonll_pt_spline(x)
    #print(quad(integrand, lo, hi))
    return quad(integrand, lo, hi)[0] / fonll_pt_spline.integral(lo, hi)

fonll_pt_barycenter_v = np.vectorize(fonll_pt_barycenter)

def fonll_y_barycenter(lo, hi):
    def integrand(x):
        return x * fonll_y_spline(x)
    return quad(integrand, lo, hi)[0] / fonll_y_spline.integral(lo, hi)

fonll_y_barycenter_v = np.vectorize(fonll_y_barycenter)
    