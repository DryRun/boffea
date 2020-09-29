'''
Fit MC mass distributions
'''
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)
import pickle

print("Loading TripleGaussianPdf...")
#ROOT.gROOT.ProcessLine(open('include/TripleGaussianPdf.cc').read())
#ROOT.gROOT.ProcessLine(".x include/TripleGaussianPdf.cc+")
ROOT.gSystem.Load("include/TripleGaussianPdf.so")
print("...done loading TripleGaussianPdf")
from ROOT import TripleGaussianPdf

figure_dir = "/home/dryu/BFrag/data/fits/data"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS

use_mc_constraints = True

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


def fit_data(tree, mass_range=BS_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, eff_hist=None):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))

	# Correct for efficiency with weights
	if eff_hist:
		# Load efficiencies
		f_eff = ROOT.TFile("/home/dryu/BFrag/data/efficiency/efficiency2D.pkl", "READ")
		h_eff = f_eff.Get("h_trigeff2D_{}_{}_{}".format(btype, side, trigger_strategy)

	ndata = rdataset.sumEntries()

	# Optional: bin data
	if binned:
		mass.setBins(BS_FIT_NBINS)
		rdatahist = ROOT.RooDataHist("fitDataBinned", "fitDataBinned", ROOT.RooArgSet(mass), rdataset)
		rdata = rdatahist
	else:
		rdata = rdataset

	# Signal: double Gaussian
	'''
	g1 = ws.factory(f"Gaussian::g1(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma1[0.034, 0.001, 0.2])")
	g2 = ws.factory(f"Gaussian::g2(mass, mean, sigma2[0.15, 0.001, 0.2])")
	signal_dg = ws.factory(f"SUM::signal_dg(f1[0.95, 0.01, 0.99]*g1, g2)")
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_dg, nsignal)
	'''
	# Signal: triple Gaussian
	mean = ws.factory(f"mean[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
	sigma1 = ws.factory("sigma1[0.008, 0.0005, 1.0]")
	sigma2 = ws.factory("sigma2[0.02, 0.0005, 1.0]")
	sigma3 = ws.factory("sigma3[0.04, 0.0005, 1.0]")
	aa = ws.factory(f"aa[0.5, 0.001, 1.0]")
	bb = ws.factory(f"bb[0.8, 0.001, 1.0]")
	#signal_tg = ws.factory(f"GenericPdf::signal_tg('TripleGaussian(mass, mean, sigma1, sigma2, sigma3, aa, bb)', {{mass, mean, sigma1, sigma2, sigma3, aa, bb}})")
	signal_tg = TripleGaussianPdf("signal_tg", "signal_tg", mass, mean, sigma1, sigma2, sigma3, aa, bb)
	getattr(ws, "import")(signal_tg, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_tg, nsignal)

	# Background model: exponential
	bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha[-3.66, -100., -0.01])")
	nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.5}, 0.0, {ndata*2.0}]")
	bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model))

	# Perform fit
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	if use_mc_constraints:
		constraints = {}
		with open("Bs/fitparams_MC_Bs_frozen.pkl", 'rb') as f:
			mc_fit_params = pickle.load(f)
		with open("Bs/fiterrs_MC_Bs_frozen.pkl", 'rb') as f:
			mc_fit_errors = pickle.load(f)

		constraints = {}
		for var in [sigma1, sigma2, sigma3, mean, aa, bb]:
			varname = var.GetName()

			param_value = mc_fit_params[cut_name][varname]
			param_err   = mc_fit_errors[cut_name][varname]
			# Loosely constrain width
			if "sigma" in varname:
				param_err = param_value / 3.
			elif varname == "mean":
				param_err = 0.05

			constraints[varname] = ROOT.RooGaussian(
				"constr_{}".format(varname), 
				"constr_{}".format(varname), 
				var, 
				ROOT.RooFit.RooConst(param_value),
				ROOT.RooFit.RooConst(param_err))
			var.setVal(mc_fit_params[cut_name][varname])
		fit_args.append(ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(*(constraints.values()))))
	fit_result = model.fitTo(*fit_args)

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdata)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def plot_fit(ws, tag="", text=None, binned=False):
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
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"))
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

	binning = ROOT.RooBinning(100, BS_FIT_WINDOW[0], BS_FIT_WINDOW[1])
	data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -3
	for xbin in range(1, pull_hist.GetNbinsX()+1):
		data_val = data_hist.GetBinContent(xbin)
		fit_val = fit_hist.GetBinContent(xbin)
		pull_val = (data_val - fit_val) / max(math.sqrt(fit_val), 1.e-10)
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
	canvas.SaveAs("{}/{}.png".format(figure_dir, canvas.GetName()))
	canvas.SaveAs("{}/{}.pdf".format(figure_dir, canvas.GetName()))

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
	args = parser.parse_args()

	import glob
	data_files = glob.glob("/home/dryu/BFrag/data/histograms/Run2018*.root")

	if args.test: 
		cuts = {"tag": ["inclusive"], "probe": ["inclusive"]}
	elif args.all:
		cuts = fit_cuts
	elif args.some:
		cuts = {"tag": args.some.split(","), "probe": args.some.split(",")}
		if not set(cuts).issubset(set(fit_cuts + fit_cuts_fine)):
			raise ValueError("Unrecognized cuts: {}".format(args.some))

	if args.fits:
		for side in ["tag", "probe"]:
			trigger_strategies = {
				"HLT_all": ["HLT_Mu7_IP4", "HLT_Mu9_IP5_only", "HLT_Mu9_IP6_only", "HLT_Mu12_IP6_only"],
				"HLT_Mu9": ["HLT_Mu9_IP5", "HLT_Mu9_IP6_only"],
				"HLT_Mu7": ["HLT_Mu7_IP4"],
			}
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
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
					cut_str = cut_strings[cut_name]
					plot_data(chain, cut=cut_str, tag="Bs_{}".format(save_tag))

					ws, fit_result = fit_data(chain, incut=cut_str, cut_name=cut_name, binned=args.binned)
					ws.Print()
					fit_result.Print()
					ws_file = ROOT.TFile("Bs/fitws_data_Bs_{}.root".format(save_tag), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

	if args.plots:
		for side in ["tag", "probe"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					ws_file = ROOT.TFile("Bs/fitws_data_Bs_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					plot_fit(ws, tag="Bs_{}".format(save_tag), text=fit_text[cut_name], binned=args.binned)

	if args.tables and args.all:
		yields = {}
		for side in ["tag", "probe"]:
			yields[side] = {}
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				yields[side][trigger_strategy] = {}
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					ws_file = ROOT.TFile("Bs/fitws_data_Bs_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		if args.binned:
			yields_file = "Bs/yields_binned.pkl"
		else:
			yields_file = "Bs/yields.pkl"
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)
		pprint(yields)