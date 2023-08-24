'''
Fit MC mass distributions
'''
import os
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)
import pickle

from brazil.tdrstyle import SetTDRStyle
tdr_style = SetTDRStyle()

from brazil.cmslabel import CMSLabel, LuminosityLabel

#print("Loading TripleGaussianPdf...")
#ROOT.gROOT.ProcessLine(open('include/TripleGaussianPdf.cc').read())
#ROOT.gROOT.ProcessLine(".x include/TripleGaussianPdf.cc+")
#ROOT.gSystem.Load("include/TripleGaussianPdf.so")
#print("...done loading TripleGaussianPdf")
#from ROOT import TripleGaussianPdf
ROOT.gSystem.Load("include/TripleGaussianPdf_cc.so")
from ROOT import TripleGaussianPdf

figure_dir = "/home/dyu7/BFrag/data/fits/data"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	get_Bcands_name_data, get_MC_fit_params, get_MC_fit_errs, \
	MakeSymHypatia, MakeJohnson, MakeDoubleGaussian, MakeTripleGaussian, MakeTripleGaussianConstrained

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
	c.SaveAs("/home/dyu7/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=BS_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, correct_eff=False, side=None, selection="nominal", fitfunc=""):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	if correct_eff:
		# Ignore very large weights, which are probably from 0 efficiency bins
		cut += " && (w_eff < 1.e6)"

	if fitfunc == "poly":
		fitfunc = "johnson"
		bkgdfunc = "poly"
	else:
		bkgdfunc = "exp"

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

	# Signal
	if fitfunc == "hypatia":
		signal_pdf = MakeSymHypatia(ws, mass_range, rcache=rcache)
	elif fitfunc == "johnson":
		signal_pdf = MakeJohnson(ws, mass_range, rcache=rcache)
	elif fitfunc == "2gauss":
		signal_pdf = MakeDoubleGaussian(ws, mass_range, rcache=rcache)
	elif fitfunc == "3gauss":
		signal_pdf = MakeTripleGaussianConstrained(ws, mass_range, rcache=rcache)
	getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)

	# Background model: exponential
	if bkgdfunc == "exp":
		bkgd_comb = ws.factory(f"Exponential::bkgd_comb(mass, alpha[-3.66, -100., -0.01])")
		nbkgd_comb = ws.factory(f"nbkgd[{ndata*0.5}, 0.0, {ndata*2.0}]")
		bkgd_comb_model = ROOT.RooExtendPdf("bkgd_comb_model", "bkgd_comb_model", bkgd_comb, nbkgd_comb)
	elif bkgdfunc == "poly":
		bkgd_comb = ws.factory(f"Chebychev::bkgd_comb(mass, {{p1[0., -10., 10.]}})")
		nbkgd_comb = ws.factory(f"nbkgd[{ndata*0.1}, 0.0, {ndata*2.0}]")
		bkgd_comb_model = ROOT.RooExtendPdf("bkgd_comb_model", "bkgd_comb_model", bkgd_comb, nbkgd_comb)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_comb_model))

	# Perform fit
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	if use_mc_constraints:
		if selection == "nominal":
			mcside = f"{side}match"
		elif selection == "HiTrkPt":
			mcside = f"{side}HiTrkPtmatch"
		elif "MuonPt" in selection:
			mcside = f"{side}{selection}match"
		elif selection == "MediumMuon":
			mcside = f"{side}{selection}match"
		else:
			raise ValueError("asdfg Don't know what to do with selection {}".format(selection))

		# For tagx, just use tag shapes
		if "tagx" in mcside:
			mcside = mcside.replace("tagx", "tag")

		constraints = {}
		with open(get_MC_fit_params("Bs", selection=selection, fitfunc=fitfunc, frozen=True), "rb") as f:
			mc_fit_params = pickle.load(f)
		with open(get_MC_fit_errs("Bs", selection=selection, fitfunc=fitfunc, frozen=True), "rb") as f:
			mc_fit_errors = pickle.load(f)

		#print(f"{cut_name} : adding constraints")
		#print("Constraint central values:")
		#pprint(mc_fit_params[cut_name])
		#print("Constraint widths:")
		#pprint(mc_fit_errors[cut_name])
		if fitfunc == "hypatia":
			for param_name in ["hyp_lambda", "hyp_sigma", "hyp_mu", "hyp_a", "hyp_n"]:
				var = ws.var(param_name)
				print("Adding constraint for {}".format(param_name))
				var.setVal(mc_fit_params[mcside][cut_name][param_name])

				err_multiplier = 1.0
				param_val = mc_fit_params[mcside][cut_name][param_name]
				param_err   = mc_fit_errors[mcside][cut_name][param_name] * err_multiplier

				# Loose rectangular constraint on mean (via variable range)
				if param_name == "hyp_mu":
					ws.var(param_name).setMin(param_val - 0.1)
					ws.var(param_name).setMax(param_val + 0.1)
					continue

				if param_err < 1.e-5:
					print("WARNING : Param {} has small error {}".format(param_name, param_err))
					raise ValueError("Quitting")
					sys.exit(1)

				# Fix tails
				if "hyp_a" in param_name or "hyp_n" in param_name:
					ws.var(param_name).setVal(param_val)
					ws.var(param_name).setConstant(True)
					continue

				if param_name in ["hyp_lambda", "hyp_sigma"]:
					# For core width parameters, set very loose constraint
					param_err = max(abs(param_val / 2.), param_err * 10.)
				elif "hyp_n" in param_name:
					# Tail exponent n: restrict to max of n/10
					param_err = min(param_err, abs(param_val / 10.))
				elif "hyp_a" in param_name:
					# Tail distance from core: restrict to max of 0.5
					param_err = min(param_err, 0.5)

				# Adjust variable value and range to match constraints
				var.setVal(param_val)
				param_min = max(var.getMin(), param_val - 10. * param_err)
				param_max = min(var.getMax(), param_val + 10. * param_err)
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
				#print(constraints[param_name])

			# End loop over parameters
		elif fitfunc == "johnson":
			for param_name in ["j_mu", "j_lambda", "j_delta", "j_gamma"]:
				err_multiplier = 1.0
				pprint(mc_fit_params[mcside][cut_name])
				param_val = mc_fit_params[mcside][cut_name][param_name]
				param_err   = mc_fit_errors[mcside][cut_name][param_name] * err_multiplier

				if param_err < 1.e-5 and param_name != "j_mu":
					print("WARNING : Param {} has small error {}".format(param_name, param_err))
					raise ValueError("Quitting")
					sys.exit(1)

				var = ws.var(param_name)
				var.setVal(param_val)

				if param_name == "j_mu":
					# Loose rectangular constraint on mean (via variable range)
					ws.var(param_name).setMin(param_val - 0.1)
					ws.var(param_name).setMax(param_val + 0.1)
					continue
				elif param_name == "j_lambda":
					# For core width parameters, set very loose constraint
					param_err = max(abs(param_val / 2.), param_err * 10.)
				elif param_name == "j_delta" or param_name == "j_gamma":
					# For gamma and delta, constrain to MC error
					pass

				# Adjust variable value and range to match constraints
				var.setVal(param_val)
				param_min = max(var.getMin(), param_val - 10. * param_err)
				param_max = min(var.getMax(), param_val + 10. * param_err)
				ws.var(param_name).setMin(param_min)
				ws.var(param_name).setMax(param_max)

				constraints[param_name] = ROOT.RooGaussian(
					"constr_{}".format(param_name), 
					"constr_{}".format(param_name), 
					var, 
					ROOT.RooFit.RooConst(param_val),
					ROOT.RooFit.RooConst(param_err))
				print(constraints[param_name])
			# End loop over parameter names
		elif fitfunc == "2gauss":
			raise NotImplementedError("2gauss not implemented!")
		elif fitfunc == "3gauss":
			for param_name in mc_fit_params[mcside][cut_name].keys():
				if "nsignal" in param_name:
					continue

				err_multiplier = 1.0
				param_val = mc_fit_params[mcside][cut_name][param_name]
				param_err = mc_fit_errors[mcside][cut_name][param_name] * err_multiplier
				if param_err < 1.e-6 and not "mean" in param_name:
					print("WARNING : Param {} has small error {} ; cut_name = {}".format(param_name, param_err, cut_name))
					raise ValueError("Quitting")
					sys.exit(1)

				# Fix sigma_3, aa, bb
				constant_params = ["tg_aa", "tg_bb", "tg_sigma1", "tg_sigma2", "tg_sigma3"]
				isConstant = False
				for name2 in constant_params:
					if name2 in param_name:
						print(param_name)
						ws.Print()
						ws.var(param_name).setVal(param_val)
						ws.var(param_name).setConstant(True)
						isConstant = True
				if isConstant:
					continue

				# Loose rectangular range for mean, no constraint
				if "tg_mean" in param_name:
					ws.var(param_name).setMin(param_val - 0.1)
					ws.var(param_name).setMax(param_val + 0.1)
					continue

				elif "tg_cs" in param_name:
					# For core width parameters, set very loose constraint
					param_err = 1.0
					continue

				else:
					# For gamma and delta, constrain to MC error
					pass

				# Adjust variable value and range to match constraints
				ws.var(param_name).setVal(param_val)
				param_min = max(ws.var(param_name).getMin(), param_val - 10. * param_err)
				param_max = min(ws.var(param_name).getMax(), param_val + 10. * param_err)
				ws.var(param_name).setMin(param_min)
				ws.var(param_name).setMax(param_max)

				# Add constraint
				constraints[param_name] = ROOT.RooGaussian(
					f"constr_{param_name}", f"constr_{param_name}",
					ws.var(param_name),
					ROOT.RooFit.RooConst(param_val),
					ROOT.RooFit.RooConst(param_err),
				)
				print("Added constraint for {} (range [{}, {}])".format(param_name, param_min, param_max))
				print(constraints[param_name])
		# End if hypatia or johnson
	# End if use MC constrains

	if len(constraints):
		fit_args.append(ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(*(constraints.values()))))
	if correct_eff:
		if not binned:
			fit_args.append(ROOT.RooFit.SumW2Error(True)) # Unbinned + weighted needs special uncertainty treatment

	# Tweaks
	if fitfunc == "hypatia":
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

	canvas = ROOT.TCanvas("c_datafit_{}".format(tag), "c_datafit_{}".format(tag), 1400, 1400)

	top = ROOT.TPad("top", "top", 0., 0.3, 1., 1.)
	top.SetBottomMargin(0.025)
	top.Draw()
	top.cd()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data2"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), 
					ROOT.RooFit.MarkerColor(ROOT.kBlack), 
					ROOT.RooFit.MarkerSize(0.0), 
					ROOT.RooFit.MarkerStyle(20))
	model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"),
					ROOT.RooFit.LineColor(ROOT.kRed), 
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.FillColor(ROOT.kRed), 
					ROOT.RooFit.FillStyle(3007), 
					ROOT.RooFit.DrawOption("LF"))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_comb"), ROOT.RooFit.Components("bkgd_comb_model"), 
					ROOT.RooFit.LineColor(ROOT.kBlue+2), 
					ROOT.RooFit.LineWidth(1))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"), 
					ROOT.RooFit.LineColor(ROOT.kBlue), 
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.LineStyle(1))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), 
					ROOT.RooFit.MarkerColor(ROOT.kBlack), 
					ROOT.RooFit.MarkerSize(1.0), 
					ROOT.RooFit.MarkerStyle(20), 
					ROOT.RooFit.DrawOption("p e same"))
	rplot.GetXaxis().SetTitleSize(0)
	rplot.GetXaxis().SetLabelSize(0)
	rplot.GetYaxis().SetLabelSize(40)
	rplot.GetYaxis().SetTitleSize(40)
	rplot.GetYaxis().SetTitleOffset(2.25)
	rplot.SetMaximum(1.3 * rplot.GetMaximum())
	rplot.Draw()

	legend = ROOT.TLegend(0.61, 0.49, 0.95, 0.9)
	legend.SetFillColor(0)
	legend.SetFillStyle(0)
	legend.SetBorderSize(0)
	legend.SetTextFont(43)
	legend.SetTextSize(40)
	legend.AddEntry("data", "Data", "lp")
	legend.AddEntry("fit", "Signal + bkgd. fit", "l")
	legend.AddEntry("signal", "B_{s}#rightarrowJ/#psi #phi", "lf")
	legend.AddEntry("bkgd_comb", "Comb. bkgd.", "l")
	legend.Draw()

	canvas.cd()
	bottom = ROOT.TPad("bottom", "bottom", 0., 0., 1., 0.3)
	bottom.SetTopMargin(0.03)
	#bottom.SetBottomMargin(0.25)
	bottom.Draw()
	bottom.cd()

	#binning = ROOT.RooBinning(BS_FIT_NBINS, BS_FIT_WINDOW[0], BS_FIT_WINDOW[1])
	binning = xvar.getBinning()
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

	# Count fit parameters (free only)
	nparams = 0
	for param in fit_result.floatParsFinal(): 
		if  param.isConstant():
			continue
		if param.GetName() in ["j_delta", "j_gamma"]:
			print(f"Skipping {param.GetName()} because it is constrained to MC")
			continue
		print(f"DEBUGG : Counting param {param.GetName()} as -1df")
		nparams += 1

	ndf = -1 * nparams
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
		pull_hist.SetBinError(xbin, 0)
		chi2 += pull_val**2
		ndf += 1
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}K^{#mp}} [GeV]")
	pull_hist.GetYaxis().SetTitle("Pull w.r.t. fit [#sigma]")
	pull_hist.GetXaxis().SetLabelSize(40)
	pull_hist.GetXaxis().SetTitleSize(40)
	pull_hist.GetYaxis().SetLabelSize(40)
	pull_hist.GetYaxis().SetTitleSize(40)
	pull_hist.SetMarkerStyle(20)
	pull_hist.SetMarkerSize(1.0)
	#pull_hist.SetMarkerStyle(20)
	#pull_hist.SetMarkerSize(1)
	#pull_hist.GetXaxis().SetTitleSize(0.06)
	#pull_hist.GetXaxis().SetLabelSize(0.06)
	#pull_hist.GetYaxis().SetTitleSize(0.06)
	#pull_hist.GetYaxis().SetLabelSize(0.06)
	#pull_hist.GetYaxis().SetTitleOffset(0.6)
	pull_hist.SetMinimum(-4.)
	pull_hist.SetMaximum(4.)
	pull_hist.Draw("p")

	zero = ROOT.TLine(BS_FIT_WINDOW[0], 0., BS_FIT_WINDOW[1], 0.)
	zero.SetLineColor(ROOT.kBlack)
	zero.SetLineStyle(1)
	zero.SetLineWidth(2)
	zero.Draw()

	canvas.cd()
	top.cd()
	if text:
		chi2text = f"#chi^{{2}}/ndof={round(chi2/ndf, 3):.3f}"
		totaltext = f"#splitline{{{text}}}{{{chi2text}}}"
		#print(totaltext)
		textbox = ROOT.TLatex(0.22, 0.55, totaltext)
		textbox.SetNDC()
		textbox.SetTextSize(0.047)
		textbox.SetTextFont(42)
		textbox.Draw()

	cmslabel = CMSLabel()
	cmslabel.sublabel.text = "Preliminary"
	cmslabel.scale = 0.9
	cmslabel.draw()

	lumilabel = LuminosityLabel("34.7 fb^{-1} (13 TeV)")	
	lumilabel.scale = 0.75
	lumilabel.draw()

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
	parser = argparse.ArgumentParser(description="Do Bs fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--some", type=str, help="Run select cut strings")
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	parser.add_argument("--binned", action="store_true", help="Do binned fit")
	parser.add_argument("--correct_eff", action="store_true", help="Apply efficiency correction before fitting")
	parser.add_argument("--fitparams", action="store_true", help="Print fit parameters")
	parser.add_argument("--selection", type=str, default="nominal", help="Selection name (nominal, HiTrkPt, ...)")
	parser.add_argument("--fitfunc", type=str, default="johnson", help="Fit function name (hypatia, johnson)")
	parser.add_argument("--trigger_strategies", type=str, default="HLT_all,HLT_Mu7,HLT_Mu9,HLT_Mu9_IP5,HLT_Mu9_IP6", help="Trigger strategies to run")
	parser.add_argument("--sides", type=str, default="tag,probe,tagx", help="Sides to run")
	args = parser.parse_args()

	import glob
	data_files = glob.glob("/home/dyu7/BFrag/data/histograms/Run2018*.root")

	if args.test: 
		cuts = {"tag": ["inclusive"], "probe": ["inclusive"]}
	elif args.all:
		cuts = fit_cuts
	elif args.some:
		cuts = {"tag": args.some.split(","), "probe": args.some.split(",")}
		if not set(cuts).issubset(set(fit_cuts)):
			raise ValueError("Unrecognized cuts: {}".format(args.some))
	cuts["tagx"] = cuts["tag"]

	print("jkl; Printing fit_cuts")
	pprint(fit_cuts)

	trigger_strategies_to_run = args.trigger_strategies.split(",") #["HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"] # "HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"
	sides_to_run = args.sides.split(",")

	if args.fits:
		for side in sides_to_run:
		#for side in ["tagMaxPt", "probeMaxPt"]:
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
					tree_name = get_Bcands_name_data(btype="Bs", trigger=trigger, side=side, selection=args.selection)
					for data_file in data_files:
						chain.Add(f"{data_file}/{tree_name}")

				print("Total entries = {}".format(chain.GetEntries()))
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					cut_str = cut_strings[cut_name]
					plot_data(chain, cut=cut_str, tag="Bs_{}".format(save_tag))

					ws, fit_result = fit_data(chain, 
										incut=cut_str, 
										cut_name=cut_name, 
										binned=args.binned, 
										correct_eff=args.correct_eff, 
										side=side,
										selection=args.selection, 
										fitfunc=args.fitfunc)
					ws.Print()
					fit_result.Print()
					print("Writing fit results to Bs/fitws_data_Bs_{}.root".format(save_tag))
					ws_file = ROOT.TFile("Bs/fitws_data_Bs_{}.root".format(save_tag), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

					# Clear cache
					del ws
					rcache = []


	if args.plots:
		for side in sides_to_run:
		#for side in ["tagMaxPt", "probeMaxPt"]:
			for trigger_strategy in trigger_strategies_to_run:
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bs/fitws_data_Bs_{}.root".format(save_tag), "READ")
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

					if "pt" in cut_name:
						pt_line = fit_text[cut_name]
						if "probe" in side:
							y_line = r"|y|#in[0, 2.25)"
						else:
							y_line = r"|y|#in[0, 1.5)"
						plot_text = f"#splitline{pt_line}{y_line}"
					else:
						pt_line = "p_{T}#in(13, 50)"
						y_line = fit_text[cut_name]
						plot_text = f"#splitline{{{pt_line}}}{{{y_line}}}"

					plot_fit(ws, fit_result, tag="Bs_{}".format(save_tag), subfolder=subfolder, text=plot_text, binned=args.binned, correct_eff=args.correct_eff)

	if args.tables and args.all:
		yields = {}
		for side in sides_to_run:
		#for side in ["tagMaxPt", "probeMaxPt"]:
			yields[side] = {}
			for trigger_strategy in trigger_strategies_to_run:
				yields[side][trigger_strategy] = {}
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bs/fitws_data_Bs_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		yields_file = f"Bs/yields_{args.fitfunc}_{args.selection}"
		#yields_file = "Bs/yields_maxPt_hyp"
		if args.binned:
			yields_file += "_binned"
		if args.correct_eff:
			yields_file += "_correcteff"

		yields_file += ".pkl"
		print("Saving yields to {}".format(yields_file))
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)
		pprint(yields)

	if args.fitparams:
		for side in ["probe", "tag", "tagx"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"
					fitresult_file = "Bs/fitws_data_Bs_{}.root".format(save_tag)
					if not os.path.isfile(fitresult_file):
						print("No fit result for {}, skipping. I was looking for {}.".format(save_tag, fitresult_file))
						continue
					ws_file = ROOT.TFile(fitresult_file, "READ")
					#ws_file.ls()
					fit_result = ws_file.Get(f"fitresult_model_fitData{'Binned' if args.binned else ''}")
					print("\n*** Printing fit results for {} ***".format(save_tag))
					fit_result.Print()
