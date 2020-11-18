'''
Fit MC mass distributions
'''
import os
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)
import pickle

#ROOT.gROOT.ProcessLine(open('include/TripleGaussianPdf2.cc').read())
#ROOT.gROOT.ProcessLine(".x include/TripleGaussianPdf2.cc+")
ROOT.gSystem.Load("include/TripleGaussianPdf2.so")

#ROOT.gROOT.ProcessLine(open('include/TripleGaussian.cc').read())
ROOT.gROOT.ProcessLine(open('include/MyErfc.cc').read())
#from ROOT import TripleGaussian
from ROOT import TripleGaussianPdf2
from ROOT import MyErfc

use_mc_constraints = True

rcache = [] # Prevent RooFit objects from disappearing

figure_dir = "/home/dryu/BFrag/data/fits/data"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS

# Enums for PDFs
from enum import Enum
class BackgroundModel_t(Enum):
	kPolynomial = 1, 
	kChebychev = 2,
	kExp = 3

class SignalModel_t(Enum):
	kGaussian = 1,
	kDoubleGaussian = 2,
	kCB = 3

def pyerfc(x, par):
	erfc_arg = (x - par[0]) / par[1]
	return ROOT.TMath.Erfc(erfc_arg)

def plot_data(tree, mass_range=BU_FIT_WINDOW, cut="", tag=""):
	h_data = ROOT.TH1D("h_data", "h_data", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_data", cut)
	c = ROOT.TCanvas("c_data_{}".format(tag), "c_data_{}".format(tag), 800, 600)
	h_data.SetMarkerStyle(20)
	h_data.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{#pm}} [GeV]")
	h_data.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=BU_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, correct_eff=False, save_tag=None):

	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	if correct_eff:
		# Ignore very large weights, which are probably from 0 efficiency bins
		cut += " && (w_eff < 1.e6)"

	#tree.Print()
	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")

	if correct_eff:
		w_eff = ws.factory("w_eff[0.0, 1.e5]")
		rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y, w_eff), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut), ROOT.RooFit.WeightVar("w_eff"))
	else:
		rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Optional: bin data
	if binned:
		mass.setBins(BS_FIT_NBINS)
		rdatahist = ROOT.RooDataHist("fitDataBinned", "fitDataBinned", ROOT.RooArgSet(mass), rdataset)
		rdata = rdatahist
	else:
		rdata = rdataset

	# Signal: triple Gaussian
	mean = ws.factory(f"mean[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
	sigma1 = ws.factory("sigma1[0.005, 0.0005, 0.25]")
	sigma2 = ws.factory("sigma2[0.02, 0.0005, 0.25]")
	sigma3 = ws.factory("sigma3[0.05, 0.0005, 0.25]")
	ccore = ws.factory(f"ccore[0.8, 0.001, 1.0]")
	ctail = ws.factory(f"ctail[0.05, 0.001, 0.4]")
	#signal_tg = ws.factory(f"GenericPdf::signal_tg('TripleGaussian(mass, mean, sigma1, sigma2, sigma3, ccore, ctail)', {{mass, mean, sigma1, sigma2, sigma3, ccore, ctail}})")
	signal_tg = TripleGaussianPdf2("signal_tg", "signal_tg", mass, mean, sigma1, sigma2, sigma3, ccore, ctail)
	getattr(ws, "import")(signal_tg, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_tg, nsignal)

	# Background model: exponential + (Bu > Jpsi pi gaussian) + (partial reco ERF)
	bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha[-3.66, -100., -0.01])")
	nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.1}, 0.0, {ndata*2.0}]")
	bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)

	#bkgd_JpsiPi = ws.factory(f"Gaussian::bkgd_JpsiPi(mass, 5.37, 0.038)")
	#nJpsiPi = ROOT.RooFormulaVar("nJpsiPi", "nJpsiPi", "0.04 * nsignal", ROOT.RooArgList(nsignal))
	#print(nJpsiPi)
	#print(bkgd_JpsiPi)
	#bkgd_JpsiPi_model = ROOT.RooExtendPdf("bkgd_JpsiPi_model", "bkgd_JpsiPi_model", bkgd_JpsiPi, nJpsiPi)
	bkgd_jpsipi_model = make_JpsiPi(ws, cut_name)


	erfc_x0 = ws.factory(f"erfc_x0[5.12, 5.07, 5.2]")
	erfc_width = ws.factory(f"erfc_width[0.02, 0.012, 0.04]")
	bkgd_erfc = ws.factory(f"GenericPdf::bkgd_erfc('MyErfc(mass, erfc_x0, erfc_width)', {{mass, erfc_x0, erfc_width}})")
	#erfc_arg = ROOT.RooFormulaVar("erfc_arg", "erfc_arg", "(mass - erfc_x) / (erfc_width)", ROOT.RooArgList(mass, erfc_x, erfc_width))
	#erfc_tf1 = ROOT.TF1("erfc", pyerfc, BU_FIT_WINDOW[0], 5.2, 2)
	#erfc_pdf = ROOT.RooFit.bindPdf("Erfc", erfc_tf1, erfc_arg)
	#ROOT.gInterpreter.ProcessLine('RooAbsPdf* myerfc = RooFit::bindPdf("erfc_pdf", TMath::Erfc, erfc_arg);')
	#x = ROOT.x
	#erfc_pdf = ROOT.myerfc
	#erfc_pdf = ROOT.RooFit.bindPdf("Erfc", erf, erfc_arg)
	nbkgd_erfc = ws.factory(f"nbkgd_erfc[{ndata*0.1}, 0.0, {ndata*2.0}]")
	bkgd_erfc_model = ROOT.RooExtendPdf("bkgd_erfc_model", "bkgd_erfc_model", bkgd_erfc, nbkgd_erfc)

	# For ptbin 5-10, it looks like there's no partial background? Not resolvable, anyways.
	'''
	if cut_name == "ptbin_5p0_10p0":
		nbkgd_erfc.setVal(0)
		nbkgd_erfc.setConstant(True)
		erfc_x0.setConstant(True)
		erfc_width.setConstant(True)

		# Also impose a minimum on the background, otherwise the signal can consume all the background
		nbkgd_exp.setMin(ndata * 0.05)

	elif cut_name == "ptbin_13p0_14p0":
		# Problem with huge tail. Chop it off.
		nbkgd_exp.setMin(5.9e+03)

	elif cut_name == "ptbin_34p0_45p0" and binned:
		sigma3.setMax(0.09)
		nbkgd_exp.setMin(830)
	'''
	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model, bkgd_jpsipi_model, bkgd_erfc_model))

	# Perform fit
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	
	constraints = {}
	if use_mc_constraints:
		# Load constraints from MC fit
		with open("Bu/fitparams_MC_Bu_frozen.pkl", 'rb') as f:
			mc_fit_params = pickle.load(f)
		with open("Bu/fiterrs_MC_Bu_frozen.pkl", 'rb') as f:
			mc_fit_errors = pickle.load(f)

		'''
		err_multiplier = 1.0
		if cut_name == "ptbin_5p0_10p0":
			err_multiplier = 3.0
		elif cut_name == "ptbin_10p0_15p0":
			err_multiplier = 5.0
		elif cut_name == "ptbin_13p0_14p0":
			err_multiplier = 10.0
		'''

		print(f"{cut_name} : adding constraints")
		print("Constraint central values:")
		pprint(mc_fit_params[cut_name])
		print("Constraint widths:")
		pprint(mc_fit_errors[cut_name])
		for var in [sigma1, sigma2, sigma3, mean, ccore, ctail]:
			print("Adding constraint for {}".format(var.GetName()))
			varname = var.GetName()

			if varname == "nsignal":
				continue

			# Loose rectangular constraint on mean (via variable range)
			if varname == "mean":
				ws.var(varname).setMin(mc_fit_params[cut_name][varname] - 0.1)
				ws.var(varname).setMax(mc_fit_params[cut_name][varname] + 0.1)
				continue

			err_multiplier = 1.0
			param_val = mc_fit_params[cut_name][varname]
			param_err   = mc_fit_errors[cut_name][varname] * err_multiplier
			if param_err < 1.e-5:
				print("WARNING : Param {} has small error {}".format(varname, param_err))
				raise ValueError("Quitting")
				sys.exit(1)

			# For core width, set very loose constraint
			if "sigma1" in varname or "sigma2" in varname:
				param_err = mc_fit_params[cut_name][varname] / 2.

			constraints[varname] = ROOT.RooGaussian(
				"constr_{}".format(varname), 
				"constr_{}".format(varname), 
				var, 
				ROOT.RooFit.RooConst(param_val),
				ROOT.RooFit.RooConst(param_err))
			var.setVal(mc_fit_params[cut_name][varname])
			print(constraints[varname])

	# Probe: use tag fits to constrain erfc parameters
	if "probe" in save_tag and not "tag_probebins" in save_tag:
		tag_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag.replace("probe", "tag_probebins")), "READ")
		tag_file.ls()
		frname = "fitresult_model_fitData"
		if binned:
			frname += "Binned"
		tag_fitresult = tag_file.Get(frname)
		if not tag_fitresult:
			print("ERROR : Couldn't find fit result in file {}!".format(tag_file.GetPath()))
			raise ValueError("Quitting")

		tag_parameters = {}
		for var in [erfc_width, erfc_x0]:
			varname = var.GetName()
			param = tag_fitresult.floatParsFinal().find(varname)
			param_val = param.getVal()
			param_err = param.getError()
			var.setVal(param_val)
			var.setMin(max(param_val - 10.*param_err, var.getMin()))
			var.setMax(min(param_val + 10.*param_err, var.getMax()))
			constraints[varname] = ROOT.RooGaussian(
				"constr_{}".format(varname), 
				"constr_{}".format(varname), 
				var, 
				ROOT.RooFit.RooConst(param_val), 
				ROOT.RooFit.RooConst(3 * param_err)
				)


	if len(constraints):
		fit_args.append(ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(*(constraints.values()))))

	if correct_eff:
		if not binned:
			fit_args.append(ROOT.RooFit.SumW2Error(True)) # Unbinned + weighted needs special uncertainty treatment

	fit_result = model.fitTo(*fit_args)

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdata)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	getattr(ws, "import")(bkgd_jpsipi_model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def plot_fit(ws, tag="", subfolder="", text=None, binned=False, correct_eff=False):
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	model = ws.pdf("model")
	if binned:
		rdataset = ws.data("fitDataBinned")
	else:
		rdataset = ws.data("fitData")
	xvar = ws.var("mass")

	canvas = ROOT.TCanvas("c_datafit_{}".format(tag), "c_datafit_{}".format(tag), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"))
	model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.FillColor(ROOT.kGreen+2), ROOT.RooFit.FillStyle(3002), ROOT.RooFit.DrawOption("LF"))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_exp"), ROOT.RooFit.Components("bkgd_exp_model"), ROOT.RooFit.LineColor(ROOT.kRed+1))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_JpsiPi"), ROOT.RooFit.Components("bkgd_jpsipi_model"), ROOT.RooFit.LineColor(ROOT.kOrange-3), ROOT.RooFit.LineWidth(2), ROOT.RooFit.DrawOption("L"))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_erfc"), ROOT.RooFit.Components("bkgd_erfc_model"), ROOT.RooFit.LineColor(ROOT.kMagenta+1))
	rplot.GetXaxis().SetTitleSize(0)
	rplot.GetXaxis().SetLabelSize(0)
	rplot.GetYaxis().SetLabelSize(0.06)
	rplot.GetYaxis().SetTitleSize(0.06)
	rplot.GetYaxis().SetTitleOffset(0.8)
	rplot.Draw()

	l = ROOT.TLegend(0.7, 0.45, 0.88, 0.88)
	l.SetFillColor(0)
	l.SetBorderSize(0)
	l.AddEntry("data", "Data", "lp")
	l.AddEntry("fit", "Total fit", "l")
	l.AddEntry("signal", f"B^{{+}}#rightarrowJ/#psi K^{{#pm}} {ws.var('nsignal').getVal():.2f}", "lf")
	l.AddEntry("bkgd_exp", f"Comb. bkgd. {ws.var('nbkgd').getVal():.2f}", "l")
	l.AddEntry("bkgd_JpsiPi", f"B^{{+}}#rightarrowJ/#psi#pi^{{#pm}} {ws.function('n_jpsipi').getVal():.2f}", "l")
	l.AddEntry("bkgd_erfc", f"B^{{+}}#rightarrowJ/#psi+hadrons {ws.var('nbkgd_erfc').getVal():.2f}", "l")
	l.Draw()

	if text:
		textbox = ROOT.TLatex(0.15, 0.7, text)
		textbox.SetNDC()
		textbox.Draw()

	canvas.cd()
	bottom = ROOT.TPad("bottom", "bottom", 0., 0., 1., 0.5)
	bottom.SetTopMargin(0.02)
	bottom.SetBottomMargin(0.25)
	bottom.Draw()
	bottom.cd()

	binning = ROOT.RooBinning(BU_FIT_NBINS, BU_FIT_WINDOW[0], BU_FIT_WINDOW[1])
	if binned:
		data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar)
	else:
		data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	data_hist.Sumw2()
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -5
	for xbin in range(1, pull_hist.GetNbinsX()+1):
		data_val = data_hist.GetBinContent(xbin)
		data_unc = data_hist.GetBinError(xbin)
		fit_val = fit_hist.GetBinContent(xbin)
		if correct_eff:
			fit_unc = fit_hist.GetBinError(xbin) * math.sqrt(data_unc**2 / max(data_val, 1.e-10))
		else:
			fit_unc = math.sqrt(fit_val)
		pull_val = (data_val - fit_val) / max(fit_unc, 1.e-10)
		#print(f"Pull xbin {xbin} = ({data_val} - {fit_val}) / ({fit_unc}) = {pull_val}")
		pull_hist.SetBinContent(xbin, pull_val)
		pull_hist.SetBinError(xbin, 1)
		chi2 += pull_val**2
		ndf += 1
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}} [GeV]")
	pull_hist.GetYaxis().SetTitle("Pull w.r.t. fit [#sigma]")
	pull_hist.SetMarkerStyle(20)
	pull_hist.SetMarkerSize(1)
	pull_hist.GetXaxis().SetTitleSize(0.06)
	pull_hist.GetXaxis().SetLabelSize(0.06)
	pull_hist.GetYaxis().SetTitleSize(0.06)
	pull_hist.GetYaxis().SetLabelSize(0.06)
	pull_hist.GetYaxis().SetTitleOffset(0.6)
	pull_hist.SetMinimum(-5.)
	pull_hist.SetMaximum(5.)
	pull_hist.Draw("p")

	zero = ROOT.TLine(BU_FIT_WINDOW[0], 0., BU_FIT_WINDOW[1], 0.)
	zero.SetLineColor(ROOT.kGray)
	zero.SetLineStyle(3)
	zero.SetLineWidth(2)
	zero.Draw()

	canvas.cd()
	top.cd()
	chi2text = ROOT.TLatex(0.15, 0.6, f"#chi^{{2}}/NDF={round(chi2/ndf, 2)}")
	chi2text.SetNDC()
	chi2text.Draw()

	os.system(f"mkdir -pv {figure_dir}/{subfolder}")
	canvas.SaveAs("{}/{}/{}.png".format(figure_dir, subfolder, canvas.GetName()))
	canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))

	ROOT.SetOwnership(canvas, False)
	ROOT.SetOwnership(top, False)
	ROOT.SetOwnership(bottom, False)

def extract_yields(ws):
	return (ws.var("nsignal").getVal(), ws.var("nsignal").getError())

def make_JpsiPi(ws, cut_name):
	# Make PDF
	bkgd_jpsipi_cb = ws.factory(f"RooCBShape::bkgd_jpsipi_cb(mass, mean_jpsipi[5.15, 5.7], sigma_jpsipi_cb[0.15, 0.001, 0.4], alpha_jpsipi_cb[-1., -100., 0.], n_jpsipi_cb[1, 0,100000])");
	bkgd_jpsipi_gauss = ws.factory(f"Gaussian::bkgd_jpsipi_gauss(mass, mean_jpsipi, sigma_jpsipi_gauss[0.1, 0.01, 0.25])")
	c_cb_jpsipi = ws.factory("c_cb_jpsipi[0.9, 0.75, 1.0]")
	bkgd_jpsipi_pdf = ROOT.RooAddPdf("bkgd_jpsipi_pdf", "bkgd_jpsipi_pdf", ROOT.RooArgList(bkgd_jpsipi_cb, bkgd_jpsipi_gauss), ROOT.RooArgList(c_cb_jpsipi))
	getattr(ws, "import")(bkgd_jpsipi_pdf, ROOT.RooFit.RecycleConflictNodes())
	n_jpsipi = ROOT.RooFormulaVar("n_jpsipi", "n_jpsipi", "0.04 * nsignal", ROOT.RooArgList(ws.var("nsignal")))
	bkgd_jpsipi_model = ROOT.RooExtendPdf("bkgd_jpsipi_model", "bkgd_jpsipi_model", bkgd_jpsipi_pdf, n_jpsipi)
	getattr(ws, "import")(bkgd_jpsipi_model, ROOT.RooFit.RecycleConflictNodes())

	# Remap cut names for bad fits
	cut_name_remapped = cut_name
	if cut_name in ["ptbin_5p0_10p0", "ptbin_10p0_11p0", "ptbin_11p0_12p0", "ptbin_12p0_13p0", "ptbin_13p0_14p0", "ptbin_14p0_15p0"]:
		cut_name_remapped = "ptbin_10p0_15p0"
	elif cut_name in ["ybin_1p5_1p75", "ybin_1p75_2p0", "ybin_2p0_2p25"]:
		cut_name_remapped = "ybin_1p25_1p5"
	f_jpsipi = ROOT.TFile(f"Bu/fitws_mc_Bu2PiJpsi_{cut_name_remapped}.root")
	fit_result = f_jpsipi.Get("fitresult_bkgd_jpsipi_model_fitData")

	# Fix parameters
	params = fit_result.floatParsFinal()
	for i in range(params.getSize()):
		parname = params[i].GetName()
		if parname == "nbkgd_jpsipi":
			continue
		print(parname)
		val = params[i].getVal()
		error = params[i].getError()
		ws.var(parname).setVal(val)
		ws.var(parname).setConstant(True)

	rcache.extend([bkgd_jpsipi_model, bkgd_jpsipi_cb, bkgd_jpsipi_gauss, c_cb_jpsipi, bkgd_jpsipi_pdf, n_jpsipi])

	return bkgd_jpsipi_model



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Do Bu fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--some", type=str, help="Run select cut strings")
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	parser.add_argument("--binned", action="store_true", help="Do binned fit")
	parser.add_argument("--correct_eff", action="store_true", help="Apply efficiency correction before fitting")
	parser.add_argument("--fitparams", action="store_true", help="Print fit parameters")
	args = parser.parse_args()

	import glob
	data_files = glob.glob("/home/dryu/BFrag/data/histograms/Run2018*.root")

	if args.test: 
		cuts = {"tag": ["inclusive"], "probe": ["inclusive"]}
	elif args.all:
		cuts = fit_cuts
	elif args.some:
		cuts_list = args.some.split(",")
		if not set(cuts_list).issubset(set(fit_cuts["tag"] + fit_cuts["probe"])):
			raise ValueError("Unrecognized cuts: {}".format(args.some))
		cuts = {"tag": cuts_list, "probe": cuts_list}

	if args.fits:
		for side in ["tag", "tag_probebins", "probe"]:#, "tag"]:
			trigger_strategies = {
				"HLT_all": ["HLT_Mu7_IP4", "HLT_Mu9_IP5_only", "HLT_Mu9_IP6_only", "HLT_Mu12_IP6_only"],
				"HLT_Mu9": ["HLT_Mu9_IP5", "HLT_Mu9_IP6_only"],
				"HLT_Mu7": ["HLT_Mu7_IP4"],
			}
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				print("\n*** Fitting {} / {} ***".format(side, trigger_strategy))
				chain = ROOT.TChain()
				for trigger in trigger_strategies[trigger_strategy]:
					if side == "tag_probebins":
						tree_name = "Bcands_Bu_{}_{}".format("tag", trigger)
					else:
						tree_name = "Bcands_Bu_{}_{}".format(side, trigger)
					for data_file in data_files:
						chain.Add(f"{data_file}/{tree_name}")

				print("Total entries = {}".format(chain.GetEntries()))
				if side == "tag_probebins":
					cuts_thisside = cuts["probe"]
				else:
					cuts_thisside = cuts[side]
				for cut_name in cuts_thisside:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					cut_str = cut_strings[cut_name]
					plot_data(chain, cut=cut_str, tag="Bu_{}".format(save_tag))

					ws, fit_result = fit_data(chain, incut=cut_str, cut_name=cut_name, binned=args.binned, correct_eff=args.correct_eff, save_tag=save_tag)
					ws.Print()
					fit_result.Print()
					print("DEBUG : Saving to Bu/fitws_data_Bu_{}.root".format(save_tag))
					ws_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

	if args.plots:
		for side in ["probe", "tag"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag), "READ")
					ws = ws_file.Get("ws")
					if args.binned:
						subfolder = "binned"
					else:
						subfolder = "unbinned"
					if args.correct_eff:
						subfolder += "_correcteff"
					plot_fit(ws, tag="Bu_{}".format(save_tag), subfolder=subfolder, text=fit_text[cut_name], binned=args.binned, correct_eff=args.correct_eff)

	if args.tables and args.all:
		yields = {}
		for side in ["probe", "tag"]:
			yields[side] = {}
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				yields[side][trigger_strategy] = {}
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		pprint(yields)
		yields_file = "Bu/yields"
		if args.binned:
			yields_file += "_binned"
		if args.correct_eff:
			yields_file += "_correcteff"
		yields_file += ".pkl"
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)

	if args.fitparams:
		for side in ["probe", "tag"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"
					fitresult_file = "Bu/fitws_data_Bu_{}.root".format(save_tag)
					if not os.path.isfile(fitresult_file):
						print("No fit result for {}, skipping".format(save_tag))
						continue
					ws_file = ROOT.TFile(fitresult_file, "READ")
					#ws_file.ls()
					fit_result = ws_file.Get("fitresult_model_fitData")
					print("\n*** Printing fit results for {} ***".format(save_tag))
					fit_result.Print()

"""
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Do Bu fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	args = parser.parse_args()

	import glob
	data_files = glob.glob("/home/dryu/BFrag/data/histograms/Run2018*.root")

	if args.test: 
		cuts = ["inclusive"]
	elif args.all:
		cuts = sorted(fit_cuts.keys())

	if args.fits:
		for side in ["probe", "tag"]:
			for trigger in ["HLT_Mu7", "HLT_Mu9_only", "HLT_Mu12_only"]:
				chain = ROOT.TChain("Bcands_Bu_{}_HLT_Mu9_IP5|HLT_Mu9_IP6".format(side))
				#chain.Add("data_Run2018C_part3.root")
				#chain.Add("data_Run2018D_part1.root")
				#chain.Add("data_Run2018D_part2.root")
				for data_file in data_files:
					chain.Add(data_file)

				for cut_name in cuts:
					cut_str = fit_cuts[cut_name]
					plot_data(chain, cut=cut_str, tag="Bu_{}_{}".format(side, cut_name))

					ws, fit_result = fit_data(chain, cut=cut_str)
					ws.Print()
					fit_result.Print()
					ws_file = ROOT.TFile("fitws_data_Bu_{}_{}_{}.root".format(side, cut_name), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

	if args.plots:
		for side in ["probe", "tag"]:
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws_file.ls()
				ws = ws_file.Get("ws")
				plot_fit(ws, tag="Bu_{}_{}".format(side, cut_name), text=fit_text[cut_name])

	if args.tables:
		yields = {}
		for side in ["probe", "tag"]:
			yields[side] = {}
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws = ws_file.Get("ws")
				yields[side][cut_name] = extract_yields(ws)
		pprint(yields)
"""