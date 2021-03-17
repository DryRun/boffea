'''
Fit MC mass distributions
'''
import re
import pickle
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)

ROOT.gROOT.ProcessLine(open('include/TripleGaussian.cc').read())
#ROOT.gROOT.ProcessLine(open('include/DoubleGaussian.cc').read())
from ROOT import TripleGaussian#, DoubleGaussian

rcache = [] # Prevent RooFit objects from disappearing

figure_dir = "/home/dryu/BFrag/data/fits/mc"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	MakeHypatia

re_ptcut = re.compile("\(pt \> (?P<ptmin>\d+\.\d+)\) && \(pt < (?P<ptmax>\d+\.\d+)\)")

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

def plot_mc(tree, mass_range=BD_FIT_WINDOW, cut="", tag=""):
	h_mc = ROOT.TH1D("h_mc", "h_mc", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_mc", cut)
	c = ROOT.TCanvas("c_mc_prefit_{}".format(tag), "c_mc_prefit_{}".format(tag), 800, 600)
	h_mc.SetMarkerStyle(20)
	h_mc.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{+}K^{-}} [GeV]")
	h_mc.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/mc/{}.pdf".format(c.GetName()))

def make_signal_pdf_main(ws, mass_range, tag=""):
	mass = ws.var("mass")
	mean = ws.factory(f"mean[{BD_MASS}, {BD_MASS-0.2}, {BD_MASS+0.2}]")
	sigma_gauss1 = ws.factory("sigma_gauss1[0.01, 0.005, 0.2]")
	sigma_gauss2 = ws.factory("sigma_gauss2[0.003, 0.005, 0.2]")
	signal_gauss1 = ws.factory(f"Gaussian::signal_gauss1(mass, mean, sigma_gauss1)")
	signal_gauss2 = ws.factory(f"Gaussian::signal_gauss2(mass, mean, sigma_gauss2)")
	c_dg_g1 = ws.factory("c_dg_g1[0.8, 0.0, 1.0]")
	signal_dg = ROOT.RooAddPdf("signal_dg", "signal_dg", ROOT.RooArgList(signal_gauss1, signal_gauss2), ROOT.RooArgList(c_dg_g1))
	getattr(ws, "import")(signal_dg)

	sigma_cb = ws.factory(f"sigma_cb[0.08, 0.005, 0.2]")
	alpha = ws.factory(f"alpha[1.7, 0., 10.]")
	n = ws.factory(f"n[2.2, 0.1, 6.0]")
	signal_cb = ROOT.RooCBShape("signal_cb", "signal_cb", mass, mean, sigma_cb, alpha, n)
	getattr(ws, "import")(signal_cb)

	c_dg = ws.factory("c_dg[0.7, 0.0, 1.0]")
	signal_pdf_main = ROOT.RooAddPdf("signal_pdf_main", "signal_pdf_main", ROOT.RooArgList(signal_dg, signal_cb), ROOT.RooArgList(c_dg))
	#getattr(ws, "import")(signal_pdf_main, ROOT.RooFit.RecycleConflictNodes())

	rcache.extend([mass, mean, sigma_gauss1, sigma_gauss2, signal_gauss1, signal_gauss2, c_dg_g1, signal_dg, sigma_cb, alpha, 
		n, signal_cb, c_dg, signal_pdf_main])
	return signal_pdf_main

def make_signal_pdf_swap(ws, mass_range, tag=""):
	mass = ws.var("mass")
	mean_swap = ws.factory(f"mean_swap[{BD_MASS}, {BD_MASS-0.2}, {BD_MASS+0.2}]")
	sigma_cb_swap = ws.factory(f"sigma_cb_swap[0.02, 0.005, 0.3]")
	alpha_swap = ws.factory(f"alpha_swap[1.23, 0., 10.]")
	n_swap = ws.factory(f"n_swap[1.5, 0.1, 6.0]")
	signal_cb_swap = ROOT.RooCBShape("signal_cb_swap", "signal_cb_swap", mass, mean_swap, sigma_cb_swap, alpha_swap, n_swap)
	getattr(ws, "import")(signal_cb_swap)

	sigma_gauss_swap = ws.factory(f"sigma_gauss_swap[0.005, 0.005, 0.3]")
	signal_gauss_swap = ws.factory(f"Gaussian::signal_gauss_swap(mass, mean_swap, sigma_gauss_swap)")

	c_g_swap = ws.factory("c_g_swap[0.7, 0.0, 1.0]")
	signal_pdf_swap =ROOT.RooAddPdf("signal_pdf_swap", "signal_pdf_swap", ROOT.RooArgList(signal_gauss_swap, signal_cb_swap), ROOT.RooArgList(c_g_swap))
	#getattr(ws, "import")(signal_pdf_swap, ROOT.RooFit.RecycleConflictNodes())

	rcache.extend([mean_swap, sigma_cb_swap, alpha_swap, n_swap, signal_cb_swap, sigma_gauss_swap, signal_gauss_swap, c_g_swap, signal_pdf_swap])
	return signal_pdf_swap

def prefit_swap(tree, mass_range=BD_FIT_WINDOW, incut="1", cut_name=""):
	ws = ROOT.RooWorkspace('ws')

	# Slightly widen pT window, to min of 5 GeV
	if "ptbin" in cut_name:
		re_match = re_ptcut.search(incut)
		ptmin = float(re_match.group("ptmin"))
		ptmax = float(re_match.group("ptmax"))
		if ptmax - ptmin < 5.0:
			extra_pt = 5.0 - (ptmax - ptmin)
			incut = incut.replace("pt > {}".format(ptmin), "pt > {}".format(ptmin - extra_pt / 2.))
			incut = incut.replace("pt < {}".format(ptmax), "pt < {}".format(ptmax + extra_pt / 2.))

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	rdataset = ROOT.RooDataSet("fitMC", "fitMC", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Signal: crystal ball plus gaussian
	#signal_sumpdf = make_signal_pdf_swap(ws, mass_range)
	signal_pdf = MakeHypatia(ws, mass_range, tag="_swap", rcache=rcache)

	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model))

	# Tweaks
	if cut_name == "ybin_2p0_2p25":
		ws.var('hyp_a_swap').setVal(2.000000013543677)
		ws.var('hyp_a2_swap').setVal(3.875191685296117)
		ws.var('hyp_lambda_swap').setVal(-0.6759677950257732)
		ws.var('hyp_mu_swap').setVal(5.267436842375455)
		ws.var('hyp_n_swap').setVal(0.5000000001197751)
		ws.var('hyp_n2_swap').setVal(0.5000122199833117)
		ws.var('hyp_sigma_swap').setVal(0.04467854433621559)



	# Perform fit
	##nll = model.createNLL(rdataset, ROOT.RooFit.NumCPU(8))
	#minimizer = ROOT.RooMinuit(nll)
	#minimizer.migrad()
	#minimizer.minos()
	fit_result = model.fitTo(rdataset, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdataset)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def prefit_main(tree, mass_range=BD_FIT_WINDOW, incut="1", cut_name=""):
	ws = ROOT.RooWorkspace('ws')

	# Slightly widen pT window, to min of 5 GeV
	if "ptbin" in cut_name:
		re_match = re_ptcut.search(incut)
		ptmin = float(re_match.group("ptmin"))
		ptmax = float(re_match.group("ptmax"))
		if ptmax - ptmin < 5.0:
			extra_pt = 5.0 - (ptmax - ptmin)
			incut = incut.replace("pt > {}".format(ptmin), "pt > {}".format(ptmin - extra_pt / 2.))
			incut = incut.replace("pt < {}".format(ptmax), "pt < {}".format(ptmax + extra_pt / 2.))
	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	rdataset = ROOT.RooDataSet("fitMC", "fitMC", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	signal_pdf = MakeHypatia(ws, mass_range, tag="_main", rcache=rcache)
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model))

	# Tweaks
	if cut_name == "ptbin_11p0_12p0":
		ws.var('hyp_a_main').setVal(6.174781687368999)
		ws.var('hyp_a2_main').setVal(2.0000022412537932)
		ws.var('hyp_lambda_main').setVal(-1.3094224965625045)
		ws.var('hyp_mu_main').setVal(5.280238411047284)
		ws.var('hyp_n_main').setVal(0.500000710886777)
		ws.var('hyp_n2_main').setVal(6.6574429821435785)
		ws.var('hyp_sigma_main').setVal(0.028357370691639856)
	elif cut_name == "ptbin_12p0_13p0":
		ws.var('hyp_a2_main').setVal(2.000000003386235)
		ws.var('hyp_a_main').setVal(6.682127051344376)
		ws.var('hyp_lambda_main').setVal(-1.246657765974831)
		ws.var('hyp_mu_main').setVal(5.280051831384689)
		ws.var('hyp_n2_main').setVal(9.914180101997488)
		ws.var('hyp_n_main').setVal(0.5000000000024518)
		ws.var('hyp_sigma_main').setVal(0.026083238573931192)
	elif cut_name == "ptbin_13p0_14p0":
		ws.var('hyp_a2_main').setVal(2.000005451554139)
		ws.var('hyp_a_main').setVal(25.19944763922372)
		ws.var('hyp_lambda_main').setVal(-1.1246079990270275)
		ws.var('hyp_mu_main').setVal(5.280150228957284)
		ws.var('hyp_n2_main').setVal(11.428457263395346)
		ws.var('hyp_n_main').setVal(1.514297504786334)
		ws.var('hyp_sigma_main').setVal(0.021623492155214086)

	elif cut_name in ["ptbin_13p0_18p0", "ptbin_14p0_15p0", "ptbin_15p0_16p0"]:
		ws.var('hyp_a2_main').setVal(2.000019978561138)
		ws.var('hyp_a_main').setVal(13.736644780905044)
		ws.var('hyp_lambda_main').setVal(-1.1025031727755312)
		ws.var('hyp_mu_main').setVal(5.2802269906904025)
		ws.var('hyp_n2_main').setVal(9.054112109703674)
		ws.var('hyp_n_main').setVal(0.5077262400473249)
		ws.var('hyp_sigma_main').setVal(0.02074421353704371)

	elif cut_name == "ybin_1p0_1p25":
		ws.var('hyp_a2_main').setVal(2.006552865613159)
		ws.var('hyp_a_main').setVal(2.000000350392576)
		ws.var('hyp_lambda_main').setVal(-2.1089489439880795)
		ws.var('hyp_mu_main').setVal(5.280279585561485)
		ws.var('hyp_n2_main').setVal(3.324680960391312)
		ws.var('hyp_n_main').setVal(2.6330046498400064)
		ws.var('hyp_sigma_main').setVal(0.031119329497233214)

	# Perform fit
	fit_result = model.fitTo(rdataset, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

	# Add everything to the workspace
	getattr(ws, "import")(rdataset)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	print("DEBUG : Print ws at the end of prefit_main")
	ws.Print()
	return ws, fit_result


def fit_mc(tree, mass_range=BD_FIT_WINDOW, incut="1", cut_name="inclusive"):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	rdataset = ROOT.RooDataSet("fitMC", "fitMC", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	signal_pdf_main = make_signal_pdf_main(ws, mass_range)
	signal_pdf_swap = make_signal_pdf_swap(ws, mass_range)
	csig_main = ws.factory(f"csig_main[0.9, 0.7, 1.0]")
	signal_pdf = ROOT.RooAddPdf("signal_pdf", "signal_pdf", 
								ROOT.RooArgList(signal_pdf_main, signal_pdf_swap), 
								ROOT.RooArgList(csig_main))
	#getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)
	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model))

	# Load prefit results and add constraints
	constraints = {}
	with open("Bd/prefitparams_MC_Bd_hypatia_frozen.pkl", 'rb') as f:
		prefit_params = pickle.load(f)
	with open("Bd/prefiterrs_MC_Bd_hypatia_frozen.pkl", 'rb') as f:
		prefit_errs = pickle.load(f)
	for side in ["recomatch", "recomatchswap"]:
		for param_name in prefit_params[side][cut_name].keys():
			ws.var(param_name).setVal(prefit_params[side][cut_name][param_name])
			if param_name == "nsignal":
				continue
			constraints[param_name] = ROOT.RooGaussian(
				f"constr_{param_name}", f"constr_{param_name}",
				ws.var(param_name),
				ROOT.RooFit.RooConst(prefit_params[side][cut_name][param_name]),
				ROOT.RooFit.RooConst(3. * prefit_errs[side][cut_name][param_name]),
			)
	prefit_nmain = prefit_params["recomatch"][cut_name]["nsignal"]
	prefit_nswap = prefit_params["recomatchswap"][cut_name]["nsignal"]
	print("Constraint on main fraction = {}".format(prefit_nmain / (prefit_nmain + prefit_nswap)))
	constraints["csig_main"] = ROOT.RooGaussian(
			f"constr_csig_main", f"constr_csig_main",
			csig_main,
			ROOT.RooFit.RooConst(prefit_nmain / (prefit_nmain + prefit_nswap)),
			ROOT.RooFit.RooConst(0.1),
		)
	constraints_set = ROOT.RooArgSet()
	for constraint in constraints.values():
		constraints_set.add(constraint)
	fit_args = [rdataset, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save(), ROOT.RooFit.ExternalConstraints(constraints_set)]

	# Perform fit
	fit_result = model.fitTo(*fit_args)

	# Add everything to the workspace
	getattr(ws, "import")(rdataset)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def plot_fit(ws, tag="", text=None):
	print("\n*** In plot_fit ***")
	ws.Print()
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	model = ws.pdf("model")
	rdataset = ws.data("fitMC")
	xvar = ws.var("mass")

	canvas = ROOT.TCanvas("c_mcfit_{}".format(tag), "c_mcfit_{}".format(tag), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()
	top.SetLogy()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"))
	model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.FillColor(ROOT.kGreen+2), ROOT.RooFit.FillStyle(3002), ROOT.RooFit.DrawOption("LF"))
	rplot.GetXaxis().SetTitleSize(0)
	rplot.GetXaxis().SetLabelSize(0)
	rplot.GetYaxis().SetLabelSize(0.06)
	rplot.GetYaxis().SetTitleSize(0.06)
	rplot.GetYaxis().SetTitleOffset(0.8)
	rplot.SetMinimum(0.1)
	rplot.SetMaximum(rplot.GetMaximum() * 10.)
	rplot.Draw()

	l = ROOT.TLegend(0.6, 0.45, 0.88, 0.88)
	l.SetFillColor(0)
	l.SetBorderSize(0)
	l.AddEntry("data", "Data", "lp")
	l.AddEntry("fit", "Total fit", "l")
	l.AddEntry("signal", "B_{d}^{0} #rightarrow J/#psi(#mu#mu) K^{*}(892)(K^{#pm}#pi^{#mp})", "lf")
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

	binning = ROOT.RooBinning(100, BD_FIT_WINDOW[0], BD_FIT_WINDOW[1])
	data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -7
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

	zero = ROOT.TLine(BD_FIT_WINDOW[0], 0., BD_FIT_WINDOW[1], 0.)
	zero.SetLineColor(ROOT.kGray)
	zero.SetLineStyle(3)
	zero.SetLineWidth(2)
	zero.Draw()

	canvas.cd()
	top.cd()
	chi2text = ROOT.TLatex(0.15, 0.65, f"#chi^{{2}}/NDF={round(chi2/ndf, 2)}")
	chi2text.SetNDC()
	chi2text.Draw()

	canvas.cd()
	canvas.SaveAs("{}/{}.png".format(figure_dir, canvas.GetName()))
	canvas.SaveAs("{}/{}.pdf".format(figure_dir, canvas.GetName()))

	ROOT.SetOwnership(canvas, False)
	ROOT.SetOwnership(top, False)
	ROOT.SetOwnership(bottom, False)

def extract_yields(ws):
	return ws.var("nsignal").getVal()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Do Bu fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--some", type=str, help="Run select cut strings")
	parser.add_argument("--prefits", action="store_true", help="Do fits")
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	args = parser.parse_args()

	import glob
	mc_files = glob.glob("MCEfficiency_Bd.root")

	if args.test: 
		cuts = ["inclusive"]
	elif args.all:
		cuts = list(set(fit_cuts["tag"] + fit_cuts["probe"]))
	elif args.some:
		cuts = args.some.split(",")
		if not set(cuts).issubset(set(fit_cuts["tag"] + fit_cuts["probe"])):
			raise ValueError("Unrecognized cuts: {}".format(args.some))
	
	if args.prefits:
		final_params = {}
		final_errors = {}
		for side in ["recomatch", "recomatchswap"]:
			final_params[side] = {}
			final_errors[side] = {}
			chain = ROOT.TChain("Bcands_{}_Bd2KsJpsi2KPiMuMu_probefilter".format(side))
			#chain.Add("data_Run2018C_part3.root")
			#chain.Add("data_Run2018D_part1.root")
			#chain.Add("data_Run2018D_part2.root")
			for mc_file in mc_files:
				chain.Add(mc_file)

			for cut_name in cuts:
				print(f"\n*** Prefitting {side} {cut_name} ***")
				cut_str = cut_strings[cut_name]
				plot_mc(chain, cut=cut_str, tag="Bd_{}_{}".format(side, cut_name))

				if side == "recomatch":
					ws, fit_result = prefit_main(chain, incut=cut_str, cut_name=cut_name)
				else:
					ws, fit_result = prefit_swap(chain, incut=cut_str, cut_name=cut_name)
				ws.Print()
				fit_result.Print()

				plot_fit(ws, tag="prefit_Bd_hyp_{}_{}".format(side, cut_name), text=fit_text[cut_name])

				ws_file = ROOT.TFile("Bd/prefitws_hyp_mc_Bd_{}_{}.root".format(side, cut_name), "RECREATE")
				ws.Write()
				fit_result.Write()
				ws_file.Close()

				# Save parameters to dictionary
				
				this_final_params = fit_result.floatParsFinal()
				final_params[side][cut_name] = {}
				final_errors[side][cut_name] = {}
				for i in range(this_final_params.getSize()):
					parname = this_final_params[i].GetName()
					final_params[side][cut_name][parname] = this_final_params[i].getVal()
					final_errors[side][cut_name][parname] = this_final_params[i].getError()
		pprint(final_params)
		pprint(final_errors)
		if args.all:
			with open("Bd/prefitparams_MC_Bd_hypatia.pkl", "wb") as f:
				pickle.dump(final_params, f)
			with open("Bd/prefiterrs_MC_Bd_hypatia.pkl", "wb") as f:
				pickle.dump(final_errors, f)

		# Print individual parameters vs pT and y
		print("Main fraction vs. cut")
		for cut_name in cuts:
			n_main = final_params["recomatch"][cut_name]["nsignal"]
			n_swap = final_params["recomatchswap"][cut_name]["nsignal"]
			print(f"{cut_name}\t=>\t{n_main} / ({n_main}+{n_swap}) = {n_main / max((n_main + n_swap), 1.e-20)}")


	if args.fits:
		chain = ROOT.TChain()
		#chain = ROOT.TChain("Bcands_{}_Bd2KsJpsi2KPiMuMu_probefilter".format(side))
		#chain.Add("data_Run2018C_part3.root")
		#chain.Add("data_Run2018D_part1.root")
		#chain.Add("data_Run2018D_part2.root")
		for mc_file in mc_files:
			chain.Add(f"{mc_file}/Bcands_recomatch_Bd2KsJpsi2KPiMuMu_probefilter")
			chain.Add(f"{mc_file}/Bcands_recomatchswap_Bd2KsJpsi2KPiMuMu_probefilter")

		for cut_name in cuts:
			cut_str = cut_strings[cut_name]
			plot_mc(chain, cut=cut_str, tag="Bd_hyp_{}".format(cut_name))

			ws, fit_result = fit_mc(chain, incut=cut_str, cut_name=cut_name)
			ws.Print()
			fit_result.Print()
			ws_file = ROOT.TFile("Bd/fitws_hyp_mc_Bd_{}.root".format(cut_name), "RECREATE")
			ws.Write()
			fit_result.Write()
			ws_file.Close()

	if args.plots:
		for cut_name in cuts:
			ws_file = ROOT.TFile("Bd/fitws_hyp_mc_Bd_{}.root".format(cut_name), "READ")
			#ws_file.ls()
			ws = ws_file.Get("ws")
			plot_fit(ws, tag="Bd_hyp_{}".format(cut_name), text=fit_text[cut_name])


	if args.tables:
		print("\n\n*** Printing tables ***\n")
		yields = {}
		for cut_name in cuts:
			ws_file = ROOT.TFile("Bd/fitws_hyp_mc_Bd_{}.root".format(cut_name), "READ")
			#ws_file.ls()
			ws = ws_file.Get("ws")
			yields[cut_name] = extract_yields(ws)
		pprint(yields)

		print("\n\n*** Printing fit results ***\n")
		final_params = {}
		final_errors = {}
		parnames = []
		for cut_name in cuts:
			final_params[cut_name] = {}
			final_errors[cut_name] = {}
			print("{}".format(cut_name))
			ws_file = ROOT.TFile("Bd/fitws_hyp_mc_Bd_{}.root".format(cut_name), "READ")
			#ws = ws_file.Get("ws")
			#ws.Print()
			fit_result = ws_file.Get("fitresult_model_fitMC")
			this_final_params = fit_result.floatParsFinal()
			covm = fit_result.covarianceMatrix()
			iaa = -1
			ibb = -1
			for i in range(this_final_params.getSize()):
				parname = this_final_params[i].GetName()
				if not parname in parnames:
					parnames.append(parname)
				final_params[cut_name][parname] = this_final_params[i].getVal()
				final_errors[cut_name][parname] = this_final_params[i].getError()

		'''
		for parname in parnames:
			print(parname)
			pargraph = ROOT.TGraphErrors(len(cuts))
			for i, cut_name in enumerate(cuts):
				pargraph.SetPoint(i, i, final_params[side][cut_name][parname])
				pargraph.SetPointError(i, i, final_params[side][cut_name][parname])
			constant = ROOT.TF1("constant", "[0]", -0.5, i+0.5)
			pargraph.Fit(constant, "R0")
			pargraph.Print("all")
		'''
		pprint(final_params)
		pprint(final_errors)
		if args.all:
			with open("Bd/fitparams_hyp_MC_Bd.pkl", "wb") as f:
				pickle.dump(final_params, f)
			with open("Bd/fiterrs_hyp_MC_Bd.pkl", "wb") as f:
				pickle.dump(final_errors, f)

		print("Main fraction vs. cut")
		for cut_name in cuts:
			csig_main = final_params[cut_name]["csig_main"]
			print(f"{cut_name}\t=>\t{csig_main}")


