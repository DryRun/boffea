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



# Job started on: Mon Jan 10 23:59:55 CET 2022 .
# FONLL heavy quark hadroproduction cross section, calculated on Tue Jan 11 00:00:00 CET 2022
# FONLL version and perturbative order: ## FONLL v1.3.4 fonll [ds/dpt^2dy (pb/GeV^2)]
# quark = bottom
# final state = meson. NP params (cm,lm,hm) = 24.2, 26.7, 22.2
# BR(q->meson) = 1
# ebeam1 = 6500, ebeam2 = 6500
# PDF set = NNPDF30_nlo_as_0118
# ptmin = 0.5
# ptmax = 100.5
# ymin  = -2.25
# ymax  = 2.25
# cross section is ds/dpt (pb/GeV)
# pt      central
fonll_pt = {
    0.5000   : 1.0802e+07, 
    1.5000   : 2.6698e+07, 
    2.5000   : 3.3479e+07, 
    3.5000   : 3.3319e+07, 
    4.5000   : 2.9244e+07, 
    5.5000   : 2.3837e+07, 
    6.5000   : 1.8660e+07, 
    7.5000   : 1.4335e+07, 
    8.5000   : 1.0948e+07, 
    9.5000   : 8.3724e+06, 
    10.5000  : 6.4375e+06, 
    11.5000  : 4.9869e+06, 
    12.5000  : 3.8958e+06, 
    13.5000  : 3.0686e+06, 
    14.5000  : 2.4384e+06, 
    15.5000  : 1.9535e+06, 
    16.5000  : 1.5767e+06, 
    17.5000  : 1.2820e+06, 
    18.5000  : 1.0498e+06, 
    19.5000  : 8.6522e+05, 
    20.5000  : 7.1746e+05, 
    21.5000  : 5.9840e+05, 
    22.5000  : 5.0185e+05, 
    23.5000  : 4.2304e+05, 
    24.5000  : 3.5834e+05, 
    25.5000  : 3.0492e+05, 
    26.5000  : 2.6060e+05, 
    27.5000  : 2.2364e+05, 
    28.5000  : 1.9269e+05, 
    29.5000  : 1.6664e+05, 
    30.5000  : 1.4463e+05, 
    31.5000  : 1.2594e+05, 
    32.5000  : 1.1002e+05, 
    33.5000  : 9.6415e+04, 
    34.5000  : 8.4740e+04, 
    35.5000  : 7.4691e+04, 
    36.5000  : 6.6012e+04, 
    37.5000  : 5.8493e+04, 
    38.5000  : 5.1958e+04, 
    39.5000  : 4.6263e+04, 
    40.5000  : 4.1285e+04, 
    41.5000  : 3.6924e+04, 
    42.5000  : 3.3094e+04, 
    43.5000  : 2.9722e+04, 
    44.5000  : 2.6746e+04, 
    45.5000  : 2.4114e+04, 
    46.5000  : 2.1781e+04, 
    47.5000  : 1.9708e+04, 
    48.5000  : 1.7862e+04, 
    49.5000  : 1.6216e+04, 
    50.5000  : 1.4745e+04, 
    51.5000  : 1.3428e+04, 
    52.5000  : 1.2247e+04, 
    53.5000  : 1.1185e+04, 
    54.5000  : 1.0230e+04, 
    55.5000  : 9.3693e+03, 
    56.5000  : 8.5920e+03, 
    57.5000  : 7.8893e+03, 
    58.5000  : 7.2529e+03, 
    59.5000  : 6.6759e+03, 
    60.5000  : 6.1520e+03, 
    61.5000  : 5.6756e+03, 
    62.5000  : 5.2419e+03, 
    63.5000  : 4.8464e+03, 
    64.5000  : 4.4853e+03, 
    65.5000  : 4.1552e+03, 
    66.5000  : 3.8532e+03, 
    67.5000  : 3.5765e+03, 
    68.5000  : 3.3228e+03, 
    69.5000  : 3.0898e+03, 
    70.5000  : 2.8757e+03, 
    71.5000  : 2.6788e+03, 
    72.5000  : 2.4974e+03, 
    73.5000  : 2.3303e+03, 
    74.5000  : 2.1761e+03, 
    75.5000  : 2.0338e+03, 
    76.5000  : 1.9021e+03, 
    77.5000  : 1.7803e+03, 
    78.5000  : 1.6675e+03, 
    79.5000  : 1.5629e+03, 
    80.5000  : 1.4659e+03, 
    81.5000  : 1.3758e+03, 
    82.5000  : 1.2921e+03, 
    83.5000  : 1.2143e+03, 
    84.5000  : 1.1418e+03, 
    85.5000  : 1.0744e+03, 
    86.5000  : 1.0115e+03, 
    87.5000  : 9.5294e+02, 
    88.5000  : 8.9825e+02, 
    89.5000  : 8.4718e+02, 
    90.5000  : 7.9948e+02, 
    91.5000  : 7.5487e+02, 
    92.5000  : 7.1315e+02, 
    93.5000  : 6.7409e+02, 
    94.5000  : 6.3749e+02, 
    95.5000  : 6.0318e+02, 
    96.5000  : 5.7100e+02, 
    97.5000  : 5.4078e+02, 
    98.5000  : 5.1241e+02, 
    99.5000  : 4.8575e+02, 
    100.5000 : 4.6068e+02, 
}

# Job started on: Tue Jan 11 00:57:03 CET 2022 .
# FONLL heavy quark hadroproduction cross section, calculated on Tue Jan 11 00:57:04 CET 2022
# FONLL version and perturbative order: ## FONLL v1.3.4 fonll [ds/dpt^2dy (pb/GeV^2)]
# quark = bottom
# final state = meson. NP params (cm,lm,hm) = 24.2, 26.7, 22.2
# BR(q->meson) = 1
# ebeam1 = 6500, ebeam2 = 6500
# PDF set = NNPDF30_nlo_as_0118
# ptmin = 12
# ptmax = 45
# ymin  = 0.0
# ymax  = 2.5
# cross section is ds/dy (pb)
# y       central
fonll_y = {
    0.0000: 5.2140e+06,
    0.1250: 5.2090e+06,
    0.2500: 5.1930e+06,
    0.3750: 5.1660e+06,
    0.5000: 5.1280e+06,
    0.6250: 5.0800e+06,
    0.7500: 5.0210e+06,
    0.8750: 4.9520e+06,
    1.0000: 4.8730e+06,
    1.1250: 4.7830e+06,
    1.2500: 4.6850e+06,
    1.3750: 4.5770e+06,
    1.5000: 4.4600e+06,
    1.6250: 4.3340e+06,
    1.7500: 4.2000e+06,
    1.8750: 4.0580e+06,
    2.0000: 3.9090e+06,
    2.1250: 3.7530e+06,
    2.2500: 3.5910e+06,
    2.3750: 3.4220e+06,
    2.5000: 3.2490e+06,
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
    return fonll_y_spline.integral(lo, hi)

def fonll_absy_integral(lo, hi):
    if lo < 0 or hi < 0:
        raise ValueError(f"In fonll_absy_integral(), lo and hi should be nonnegative ({lo}, {hi})")
    return 2 * fonll_y_spline.integral(lo, hi)

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




'''
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
'''