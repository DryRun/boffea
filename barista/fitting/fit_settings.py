import ROOT

BU_FIT_WINDOW = [5.05, 5.5]
BD_FIT_WINDOW = [5.05, 5.4]
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

def MakeSymHypatia(ws, mass_range, tag="", rcache=None):
	name = "signal_hyp{}".format(tag)
	hyp_x      = ws.var("mass")
	hyp_lambda = ws.factory("hyp_lambda{}[ -1., -10., -0.00001]".format(tag))
	hyp_zeta   = ws.factory("hyp_zeta{}[0.]".format(tag))
	hyp_beta   = ws.factory("hyp_beta{}[0.]".format(tag))
	hyp_sigma  = ws.factory("hyp_sigma{}[0.01, 0.005, 0.3]".format(tag))
	hyp_mu     = ws.factory(f"hyp_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
	hyp_a      = ws.factory("hyp_a{}[3., 1.0, 4.0]".format(tag))
	hyp_n      = ws.factory("hyp_n{}[5., 0.5,  20.]".format(tag))
	#hyp_a2     = ws.factory("hyp_a2[10., 0., 100.]")
	#hyp_n2     = ws.factory("hyp_n2[5., 0., 100.]")
	hyp_pdf    = ROOT.RooHypatia2(name, name, hyp_x, hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a, hyp_n)
	if rcache:
		rcache.extend([hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n])
	return hyp_pdf

def MakeHypatia(ws, mass_range, tag="", rcache=None):
	name = "signal_hyp{}".format(tag)
	hyp_x      = ws.var("mass")
	hyp_lambda = ws.factory("hyp_lambda{}[ -1., -10., -0.00001]".format(tag))
	hyp_zeta   = ws.factory("hyp_zeta{}[0.]".format(tag))
	hyp_beta   = ws.factory("hyp_beta{}[0.]".format(tag))
	hyp_sigma  = ws.factory("hyp_sigma{}[0.01, 0.005, 0.3]".format(tag))
	hyp_mu     = ws.factory(f"hyp_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
	hyp_a      = ws.factory("hyp_a{}[3., 1.0, 4.0]".format(tag))
	hyp_n      = ws.factory("hyp_n{}[5., 0.75,  20.]".format(tag))
	hyp_a2     = ws.factory("hyp_a2{}[3.0, 1.0, 4.0]".format(tag))
	hyp_n2     = ws.factory("hyp_n2{}[5., 0.75,  20.]".format(tag))
	hyp_pdf    = ROOT.RooHypatia2(name, name, hyp_x, hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2)
	if rcache:
		rcache.extend([hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2])
	return hyp_pdf

def MakeHypatia2(ws, mass_range, tag="", rcache=None):
	'''
	Hypatia function with independent left and right tails, but implemented as a2 = r_a * a ; n2 = r_n * n, so the r can be frozen to 1 to be symmetric.
	'''
	name = "signal_hyp{}".format(tag)

	hyp_x      = ws.var("mass")
	hyp_lambda = ws.factory("hyp_lambda{}[ -1., -10., -0.00001]".format(tag))
	hyp_zeta   = ws.factory("hyp_zeta{}[0.]".format(tag))
	hyp_beta   = ws.factory("hyp_beta{}[0.]".format(tag))
	hyp_sigma  = ws.factory("hyp_sigma{}[0.01, 0.005, 0.3]".format(tag))
	hyp_mu     = ws.factory(f"hyp_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
	hyp_a      = ws.factory("hyp_a{}[5., 2.0, 100.]".format(tag))
	hyp_n      = ws.factory("hyp_n{}[5., 0.5,  20.]".format(tag))
	hyp_lr_a   = ws.factory("hyp_lr_a{}[1., 0.01, 100.]".format(tag))
	hyp_lr_n   = ws.factory("hyp_lr_n{}[1., 0.01, 100.]".format(tag))
	hyp_a2     = ws.factory("prod::hyp_a2{}(hyp_a{}, hyp_lr_a{})".format(tag, tag, tag))
	hyp_n2     = ws.factory("prod::hyp_n2{}(hyp_n{}, hyp_lr_n{})".format(tag, tag, tag))
	hyp_pdf    = ROOT.RooHypatia2(name, name, hyp_x, hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2)
	if rcache:
		rcache.extend([hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2])
	return hyp_pdf
