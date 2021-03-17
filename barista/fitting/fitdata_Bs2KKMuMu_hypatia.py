'''
Fit MC mass distributions
'''
import os
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)
import pickle

#print("Loading TripleGaussianPdf...")
#ROOT.gROOT.ProcessLine(open('include/TripleGaussianPdf.cc').read())
#ROOT.gROOT.ProcessLine(".x include/TripleGaussianPdf.cc+")
#ROOT.gSystem.Load("include/TripleGaussianPdf.so")
#print("...done loading TripleGaussianPdf")
#from ROOT import TripleGaussianPdf

figure_dir = "/home/dryu/BFrag/data/fits/data"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	MakeSymHypatia

use_mc_constraints = True

rcache = [] # Prevent RooFit objects from disappearing

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

def plot_data(tree, mass_range=BS_FIT_WINDOW, cut="", tag=""):
	h_data = ROOT.TH1D("h_data", "h_data", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_data", cut)
	c = ROOT.TCanvas("c_data_{}".format(tag), "c_data_{}".format(tag), 800, 600)
	h_data.SetMarkerStyle(20)
	h_data.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{#pm}} [GeV]")
	h_data.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=BS_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, correct_eff=False):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	if correct_eff:
		# Ignore very large weights, which are probably from 0 efficiency bins
		cut += " && (w_eff < 1.e6)"

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")

	# Correct for efficiency with weights
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

	# Signal: hypatia
	signal_hyp = MakeSymHypatia(ws, mass_range, rcache=rcache)
	getattr(ws, "import")(signal_hyp, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_hyp, nsignal)

	# Background model: exponential
	bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha[-3.66, -100., -0.01])")
	nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.5}, 0.0, {ndata*2.0}]")
	bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model))

	# Perform fit
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	if use_mc_constraints:
		constraints = {}
		with open("Bs/fitparams_MC_Bs_hypatia_frozen.pkl", 'rb') as f:
			mc_fit_params = pickle.load(f)
		with open("Bs/fiterrs_MC_Bs_hypatia_frozen.pkl", 'rb') as f:
			mc_fit_errors = pickle.load(f)

		print(f"{cut_name} : adding constraints")
		print("Constraint central values:")
		pprint(mc_fit_params[cut_name])
		print("Constraint widths:")
		pprint(mc_fit_errors[cut_name])
		for param_name in ["hyp_lambda", "hyp_sigma", "hyp_mu", "hyp_a", "hyp_n"]:
			var = ws.var(param_name)
			print("Adding constraint for {}".format(param_name))
			var.setVal(mc_fit_params[cut_name][param_name])

			err_multiplier = 1.0
			param_val = mc_fit_params[cut_name][param_name]
			param_err   = mc_fit_errors[cut_name][param_name] * err_multiplier
			if param_err < 1.e-5:
				print("WARNING : Param {} has small error {}".format(param_name, param_err))
				raise ValueError("Quitting")
				sys.exit(1)

			# Loose rectangular constraint on mean (via variable range)
			if param_name == "hyp_mu":
				ws.var(param_name).setMin(param_val - 0.1)
				ws.var(param_name).setMax(param_val + 0.1)
				continue

			# Fix tails
			if "hyp_a" in param_name or "hyp_n" in param_name:
				ws.var(param_name).setVal(param_val)
				ws.var(param_name).setConstant(True)
				continue

			if param_name in ["hyp_lambda", "hyp_sigma"]:
				# For core width parameters, set very loose constraint
				param_err = max(abs(mc_fit_params[cut_name][param_name] / 2.), param_err * 10.)
			elif "hyp_n" in param_name:
				# Tail exponent n: restrict to max of n/10
				param_err = min(param_err, abs(param_val / 10.))
			elif "hyp_a" in param_name:
				# Tail distance from core: restrict to max of 0.5
				param_err = min(param_err, 0.5)

			# Adjust variable value and range to match constraints
			ws.var(param_name).setVal(param_val)
			param_min = max(ws.var(param_name).getMin(), param_val - 10. * param_err)
			param_max = min(ws.var(param_name).getMax(), param_val + 10. * param_err)
			if "hyp_lambda" in param_name:
				param_max = min(0., param_max)
			elif "hyp_a" in param_name or "hyp_n" in param_name or "hyp_sigma" in param_name:
				param_min = max(0., param_min)
			ws.var(param_name).setMin(param_min)
			ws.var(param_name).setMax(param_max)

			constraints[param_name] = ROOT.RooGaussian(
				"constr_{}".format(param_name), 
				"constr_{}".format(param_name), 
				var, 
				ROOT.RooFit.RooConst(param_val),
				ROOT.RooFit.RooConst(param_err))
			print(constraints[param_name])

	if len(constraints):
		fit_args.append(ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(*(constraints.values()))))
	if correct_eff:
		if not binned:
			fit_args.append(ROOT.RooFit.SumW2Error(True)) # Unbinned + weighted needs special uncertainty treatment

	if cut_name == "ptbin_18p0_23p0":
		ws.var("alpha").setVal(-3.0107e+00)
		ws.var("hyp_a").setVal(1.5000e+00)
		ws.var("hyp_lambda").setVal(-1.0902e+00)
		ws.var("hyp_mu").setVal(5.3667e+00)
		ws.var("hyp_n").setVal(6.3850e+00)
		ws.var("hyp_sigma").setVal(1.3932e-02)
	elif cut_name == "ybin_0p0_0p25":
		ws.var('alpha').setVal(-3.5850e+00)
		ws.var('hyp_a').setVal(1.0000e+01)
		ws.var('hyp_lambda').setVal(-2.8720e+00)
		ws.var('hyp_mu').setVal(5.3668e+00)
		ws.var('hyp_n').setVal(5.2368e+00)
		ws.var('hyp_sigma').setVal(1.7720e-02)

	fit_result = model.fitTo(*fit_args)

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdata)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def plot_fit(ws, fit_result, tag="", subfolder="", text=None, binned=False, correct_eff=False):
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
	l.AddEntry("signal", "B_{s}#rightarrowJ/#psi #phi", "lf")
	l.AddEntry("bkgd_exp", "Comb. bkgd.", "l")
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

	binning = ROOT.RooBinning(BS_FIT_NBINS, BS_FIT_WINDOW[0], BS_FIT_WINDOW[1])
	if binned:
		data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar)
	else:
		data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	data_hist.Sumw2()
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	fit_hist.Sumw2()
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -6
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
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}K^{#mp}} [GeV]")
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

	zero = ROOT.TLine(BS_FIT_WINDOW[0], 0., BS_FIT_WINDOW[1], 0.)
	zero.SetLineColor(ROOT.kGray)
	zero.SetLineStyle(3)
	zero.SetLineWidth(2)
	zero.Draw()

	canvas.cd()
	top.cd()
	chi2text = ROOT.TLatex(0.15, 0.6, f"#chi^{{2}}/NDF={round(chi2/ndf, 2)}")
	chi2text.SetNDC()
	chi2text.Draw()

	canvas.cd()
	os.system(f"mkdir -pv {figure_dir}/{subfolder}")
	canvas.SaveAs("{}/{}/{}.png".format(figure_dir, subfolder, canvas.GetName()))
	canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))

	ROOT.SetOwnership(canvas, False)
	ROOT.SetOwnership(top, False)
	ROOT.SetOwnership(bottom, False)

def extract_yields(ws):
	return (ws.var("nsignal").getVal(), ws.var("nsignal").getError())


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
		cuts = {"tag": args.some.split(","), "probe": args.some.split(",")}
		if not set(cuts).issubset(set(fit_cuts)):
			raise ValueError("Unrecognized cuts: {}".format(args.some))

	trigger_strategies_to_run = ["HLT_all", "HLT_Mu7", "HLT_Mu9"] # "HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"
	if args.fits:
		for side in ["tag", "probe"]:
			trigger_strategies = {
				"HLT_all": ["HLT_Mu7_IP4", "HLT_Mu9_IP5_only", "HLT_Mu9_IP6_only", "HLT_Mu12_IP6_only"],
				"HLT_Mu9": ["HLT_Mu9_IP5", "HLT_Mu9_IP6_only"],
				"HLT_Mu9_IP5": ["HLT_Mu9_IP5"],
				"HLT_Mu9_IP6": ["HLT_Mu9_IP6"],
				"HLT_Mu7": ["HLT_Mu7_IP4"],

			}

			for trigger_strategy in trigger_strategies_to_run:
				chain = ROOT.TChain()
				for trigger in trigger_strategies[trigger_strategy]:
					tree_name = "Bcands_Bs_{}_{}".format(side, trigger)
					for data_file in data_files:
						chain.Add(f"{data_file}/{tree_name}")

				print("Total entries = {}".format(chain.GetEntries()))
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					cut_str = cut_strings[cut_name]
					plot_data(chain, cut=cut_str, tag="Bs_{}".format(save_tag))

					ws, fit_result = fit_data(chain, incut=cut_str, cut_name=cut_name, binned=args.binned, correct_eff=args.correct_eff)
					ws.Print()
					fit_result.Print()
					print("Writing fit results to Bs/fitws_hyp_data_Bs_{}.root".format(save_tag))
					ws_file = ROOT.TFile("Bs/fitws_hyp_data_Bs_{}.root".format(save_tag), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

					# Clear cache
					del ws
					rcache = []


	if args.plots:
		for side in ["tag", "probe"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"]:
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bs/fitws_hyp_data_Bs_{}.root".format(save_tag), "READ")
					ws = ws_file.Get("ws")
					fit_result_name = "fitresult_model_fitData"
					if args.binned:
						fit_result_name += "Binned"
					fit_result = ws_file.Get(fit_result_name)
					if args.binned:
						subfolder = "binned"
					else:
						subfolder = "unbinned"
					if args.correct_eff:
						subfolder += "_correcteff"
					plot_fit(ws, fit_result, tag="Bs_hyp_{}".format(save_tag), subfolder=subfolder, text=fit_text[cut_name], binned=args.binned, correct_eff=args.correct_eff)

	if args.tables and args.all:
		yields = {}
		for side in ["tag", "probe"]:
			yields[side] = {}
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"]:
				yields[side][trigger_strategy] = {}
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bs/fitws_hyp_data_Bs_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		yields_file = "Bs/yields_hyp"
		if args.binned:
			yields_file += "_binned"
		if args.correct_eff:
			yields_file += "_correcteff"
		yields_file += ".pkl"
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)
		pprint(yields)

	if args.fitparams:
		for side in ["probe", "tag"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"
					fitresult_file = "Bs/fitws_hyp_data_Bs_{}.root".format(save_tag)
					if not os.path.isfile(fitresult_file):
						print("No fit result for {}, skipping. I was looking for {}.".format(save_tag, fitresult_file))
						continue
					ws_file = ROOT.TFile(fitresult_file, "READ")
					#ws_file.ls()
					fit_result = ws_file.Get(f"fitresult_model_fitData{'Binned' if args.binned else ''}")
					print("\n*** Printing fit results for {} ***".format(save_tag))
					fit_result.Print()
