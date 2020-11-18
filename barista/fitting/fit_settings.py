BU_FIT_WINDOW = [5.05, 5.5]
BD_FIT_WINDOW = [5.05, 5.5]
BS_FIT_WINDOW = [5.2, 5.52]

BU_FIT_WINDOW_MC = [5.1, 5.45]
BD_FIT_WINDOW_MC = [5.05, 5.5]
BS_FIT_WINDOW_MC = [5.2 + 0.05, 5.52 - 0.05]

BU_FIT_NBINS = 100
BD_FIT_NBINS = 100
BS_FIT_NBINS = 100

# Fit bins
#ptbins_coarse = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
ptbins_coarse = [8., 13., 18., 23., 28., 33.]
ptbins_fine = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0]
ybins = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.0, 2.25]

fit_cuts = {"tag": [], "probe": []}
cut_strings = {}
fit_text = {} # Text for plots

fit_cuts["tag"].append("inclusive")
fit_cuts["probe"].append("inclusive")
cut_strings["inclusive"] = "1"
fit_text["inclusive"] = "Inclusive"

cut_xvals = {}

for ipt in range(len(ptbins_coarse) - 1):
	cut_str = "(abs(y) < 2.25) && (pt > {}) && (pt < {})".format(ptbins_coarse[ipt], ptbins_coarse[ipt+1])
	cut_name = "ptbin_{}_{}".format(ptbins_coarse[ipt], ptbins_coarse[ipt+1]).replace(".", "p")
	fit_cuts["probe"].append(cut_name)
	#fit_cuts["tag"].append(cut_name)
	cut_strings[cut_name] = cut_str
	fit_text[cut_name] = "{}<pT<{}".format(ptbins_coarse[ipt], ptbins_coarse[ipt+1])
	cut_xvals[cut_name] = (ptbins_coarse[ipt], ptbins_coarse[ipt+1])

for iy in range(len(ybins) - 1):
	cut_str = "(pt > 10.0) && (pt < 30.0) && ({} < abs(y)) && (abs(y) < {})".format(ybins[iy], ybins[iy+1])
	cut_name = "ybin_{}_{}".format(ybins[iy], ybins[iy+1]).replace(".", "p")
	fit_cuts["probe"].append(cut_name)
	fit_cuts["tag"].append(cut_name)
	cut_strings[cut_name] = cut_str
	fit_text[cut_name] = "{}<|y|<{}".format(ybins[iy], ybins[iy+1])
	cut_xvals[cut_name] = (ybins[iy], ybins[iy+1])

# Fine pT binning for tag side only
for ipt in range(len(ptbins_fine) - 1):
	cut_str = "(abs(y) < 2.25) && (pt > {}) && (pt < {})".format(ptbins_fine[ipt], ptbins_fine[ipt+1])
	cut_name = "ptbin_{}_{}".format(ptbins_fine[ipt], ptbins_fine[ipt+1]).replace(".", "p")
	fit_cuts["tag"].append(cut_name)
	cut_strings[cut_name] = cut_str
	fit_text[cut_name] = "{}<pT<{}".format(ptbins_fine[ipt], ptbins_fine[ipt+1])
	cut_xvals[cut_name] = (ptbins_fine[ipt], ptbins_fine[ipt+1])
