'''
Fit MC mass distributions
'''
import os
from pprint import pprint
import ROOT
from brazil.aguapreta import *
import pickle
ROOT.gROOT.SetBatch(True)

import sys
sys.path.append(".")
from fitmc_Bd2KPiMuMu import make_signal_pdf_main, make_signal_pdf_swap
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	MakeHypatia

# hack
#BD_FIT_WINDOW = [5.05, 5.4]
figure_dir = "/home/dryu/BFrag/data/fits/data"

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

def plot_data(tree, mass_range=BD_FIT_WINDOW, cut="", tag=""):
	h_data = ROOT.TH1D("h_data", "h_data", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_data", cut)
	c = ROOT.TCanvas("c_data_{}".format(tag), "c_data_{}".format(tag), 800, 600)
	h_data.SetMarkerStyle(20)
	h_data.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{#pm}} [GeV]")
	h_data.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=BD_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, correct_eff=False):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	if correct_eff:
		# Ignore very large weights, which are probably from 0 efficiency bins
		cut += " && (w_eff < 1.e6)"

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

	# Signal: double Gaussian
	'''
	g1 = ws.factory(f"Gaussian::g1(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma1[0.034, 0.001, 0.2])")
	g2 = ws.factory(f"Gaussian::g2(mass, mean, sigma2[0.15, 0.001, 0.2])")
	signal_dg = ws.factory(f"SUM::signal_dg(f1[0.95, 0.01, 0.99]*g1, g2)")
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_dg, nsignal)
	'''
	# Signal
	signal_pdf_main = MakeHypatia(ws, mass_range, tag="_main") # make_signal_pdf_main(ws, mass_range)
	signal_pdf_swap = MakeHypatia(ws, mass_range, tag="_swap") # make_signal_pdf_swap(ws, mass_range)
	csig_main = ws.factory(f"csig_main[0.9, 0.7, 1.0]")
	signal_pdf = ROOT.RooAddPdf("signal_pdf", "signal_pdf", 
								ROOT.RooArgList(signal_pdf_main, signal_pdf_swap), 
								ROOT.RooArgList(csig_main))
	#getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)

	# Background model: exponential
	bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha_bkgd[-3.66, -100., -0.01])")
	nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.5}, 0.0, {ndata*2.0}]")
	bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model))

	# Load prefit results and add constraints
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	if use_mc_constraints:
		constraints = {}
		with open("Bd/prefitparams_MC_Bd_hypatia_frozen.pkl", 'rb') as f:
			prefit_params = pickle.load(f)
		with open("Bd/prefiterrs_MC_Bd_hypatia_frozen.pkl", 'rb') as f:
			prefit_errs = pickle.load(f)

		# Remap some cuts due to low stats/bad MC fits
		cut_name_remapped = cut_name
		cut_name_remapped_swap = cut_name
		if cut_name in ["ybin_1p5_1p75", "ybin_1p75_2p0", "ybin_2p0_2p25"]:
			cut_name_remapped_swap = "ybin_1p25_1p5"
		#elif cut_name in ["ptbin_10p0_11p0", "ptbin_11p0_12p0", "ptbin_12p0_13p0"]:
		#	cut_name_remapped_swap = "ptbin_8p0_13p0"
		#elif cut_name in ["ptbin_13p0_14p0", "ptbin_14p0_15p0"]:
		#	cut_name_remapped_swap = "ptbin_13p0_18p0"

		for side in ["recomatch", "recomatchswap"]:
			for param_name in prefit_params[side][cut_name].keys():
				if param_name == "nsignal":
					continue

				# Loose rectangular constraint on mean (via variable range)
				if param_name == "mean":
					ws.var(param_name).setMin(prefit_params[side][cut_name][param_name] - 0.1)
					ws.var(param_name).setMax(prefit_params[side][cut_name][param_name] + 0.1)
					continue

				# Loose Gaussian constraint on sigmas
				if "sigma" in param_name:
					prefit_error = prefit_value / 2.

				if side == "recomatch":
					prefit_value = prefit_params[side][cut_name_remapped][param_name]
					prefit_error = prefit_errs[side][cut_name_remapped][param_name]
					#if cut_name_remapped != cut_name:
					#	prefit_error = prefit_error * 3
				else:
					prefit_value = prefit_params[side][cut_name_remapped_swap][param_name]
					prefit_error = prefit_errs[side][cut_name_remapped_swap][param_name]
					#if cut_name_remapped_swap != cut_name:
					#	prefit_error = prefit_error * 3

				ws.var(param_name).setVal(prefit_value)
				constraints[param_name] = ROOT.RooGaussian(
					f"constr_{param_name}", f"constr_{param_name}",
					ws.var(param_name),
					ROOT.RooFit.RooConst(prefit_value),
					ROOT.RooFit.RooConst(prefit_error),
				)
		prefit_nmain = prefit_params["recomatch"][cut_name]["nsignal"]
		prefit_nswap = prefit_params["recomatchswap"][cut_name]["nsignal"]
		if prefit_nmain == 0:
			raise ValueError("Skipping this point due to prefit_nmain == 0")
		print("Constraint on main fraction = {}".format(prefit_nmain / max(prefit_nmain + prefit_nswap, 1.e-20)))
		csig_main.setVal(prefit_nmain / (prefit_nmain + prefit_nswap))
		csig_main.setConstant(True)
		#constraints["csig_main"] = ROOT.RooGaussian(
		#		f"constr_csig_main", f"constr_csig_main",
		#		csig_main,
		#		ROOT.RooFit.RooConst(prefit_nmain / (prefit_nmain + prefit_nswap)),
		#		ROOT.RooFit.RooConst(0.1),
		#)
		constraints_set = ROOT.RooArgSet()
		for constraint in constraints.values():
			constraints_set.add(constraint)
		fit_args.append(ROOT.RooFit.ExternalConstraints(constraints_set))

	if correct_eff:
		if not binned:
			fit_args.append(ROOT.RooFit.SumW2Error(True)) # Unbinned + weighted needs special uncertainty treatment

	# Perform fit
	fit_result = model.fitTo(*fit_args)

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdata)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
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
	model.plotOn(rplot, ROOT.RooFit.Name("signal_total"), ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.FillColor(ROOT.kGreen+2), ROOT.RooFit.FillStyle(3002), ROOT.RooFit.DrawOption("LF"))
	model.plotOn(rplot, ROOT.RooFit.Name("signal_main"), ROOT.RooFit.Components("signal_pdf_main"), ROOT.RooFit.LineColor(ROOT.kAzure+1))
	model.plotOn(rplot, ROOT.RooFit.Name("signal_swap"), ROOT.RooFit.Components("signal_pdf_swap"), ROOT.RooFit.LineColor(ROOT.kOrange+2))
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
	l.AddEntry("signal_total", "B_{d}#rightarrowJ/#psi K^{*} (total)", "lf")
	l.AddEntry("signal_main", "B_{d}#rightarrowJ/#psi K^{*} (main)", "lf")
	l.AddEntry("signal_swap", "B_{d}#rightarrowJ/#psi K^{*} (swap)", "lf")
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

	binning = ROOT.RooBinning(BD_FIT_NBINS, BD_FIT_WINDOW[0], BD_FIT_WINDOW[1])
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
	ndf = -3
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
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}#pi^{#mp}} [GeV]")
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

	zero = ROOT.TLine(BD_FIT_WINDOW[0], 0., BD_FIT_WINDOW[1], 0.)
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
	#canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))

	top.SetLogy()
	rplot.SetMaximum(rplot.GetMaximum() * 10.)
	rplot.SetMinimum(1.)
	canvas.SaveAs("{}/{}/{}_logy.png".format(figure_dir, subfolder, canvas.GetName()))
	#canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))

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

	trigger_strategies_to_run = ["HLT_Mu9_IP5", "HLT_Mu9_IP6"] # "HLT_all", "HLT_Mu7", "HLT_Mu9", 
	if args.fits:
		for side in ["probe", "tag"]:
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
					tree_name = "Bcands_Bd_{}_{}".format(side, trigger)
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
					plot_data(chain, cut=cut_str, tag="Bd_{}".format(save_tag))

					try:
						ws, fit_result = fit_data(chain, incut=cut_str, cut_name=cut_name, binned=args.binned, correct_eff=args.correct_eff)
					except ValueError as err:
						print(err)						
					ws.Print()
					fit_result.Print()
					ws_file = ROOT.TFile("Bd/fitws_hyp_data_Bd_{}.root".format(save_tag), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

	if args.plots:
		for side in ["probe", "tag"]:
			for trigger_strategy in trigger_strategies_to_run:
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"					
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bd/fitws_hyp_data_Bd_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")

					if args.binned:
						subfolder = "binned"
					else:
						subfolder = "unbinned"
					if args.correct_eff:
						subfolder += "_correcteff"

					plot_fit(ws, tag="Bd_hyp_{}".format(save_tag), subfolder=subfolder, text=fit_text[cut_name], binned=args.binned, correct_eff=args.correct_eff)

	if args.tables and args.all:
		yields = {}
		for side in ["probe", "tag"]:
			yields[side] = {}
			for trigger_strategy in trigger_strategies_to_run:
				yields[side][trigger_strategy] = {}
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"					
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bd/fitws_hyp_data_Bd_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		yields_file = "Bd/yields_hyp"
		if args.binned:
			yields_file += "_binned"
		if args.correct_eff:
			yields_file += "_correcteff"
		yields_file += ".pkl"
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)
		pprint(yields)